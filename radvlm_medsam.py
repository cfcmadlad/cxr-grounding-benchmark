"""
radvlm_medsam.py
----------------
Self-contained script: RadVLM + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements (install in a dedicated conda env):
    pip install torch torchvision
    pip install transformers==4.46.0
    pip install accelerate pydicom Pillow numpy scipy matplotlib pandas pyyaml

MedSAM is loaded from HuggingFace (wanglab/medsam-vit-base).

RadVLM weights must be downloaded from PhysioNet:
    https://physionet.org/content/radvlm-model/1.0.0/
The weights folder is read from config.yaml (RADVLM_PATH) — nothing hardcoded.

RadVLM emits bounding boxes as NORMALIZED [x1, y1, x2, y2] in [0, 1] in free
text. parse_box_from_response() therefore: parses 4 numbers with several
fallback patterns, clamps every value to [0, 1], then denormalizes to pixel
coordinates (x * img_w, y * img_h) before MedSAM. If no box parses the region
is marked not-detected and MedSAM is skipped.
"""

import os
import sys
import re
import json
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG + HF cache (must precede heavy imports) ────────────────────────────
from cxr_common import (
    load_config, setup_hf_home, load_image, find_image_path,
    load_gold_annotations, validate_box, clamp_box, compute_iou,
    result_row, write_result_row, init_results_csv, BOX_FIELDS, get_logger,
)

log = get_logger("radvlm")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR", "RADVLM_PATH"])
setup_hf_home(cfg)

RADVLM_PATH = cfg["RADVLM_PATH"]
GOLD_CSV    = cfg["GOLD_CSV"]
IMAGE_DIR   = cfg["IMAGE_DIR"]
OUTPUT_DIR  = os.path.join(cfg["OUTPUT_DIR"], "radvlm")
MAX_IMAGES  = cfg.get("MAX_IMAGES")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from transformers import SamModel, SamProcessor

# ── 15 TARGET REGIONS ─────────────────────────────────────────────────────────
REGIONS = [
    "right lung",
    "left lung",
    "cardiac silhouette",
    "mediastinum",
    "trachea",
    "right upper lung zone",
    "right mid lung zone",
    "right lower lung zone",
    "left upper lung zone",
    "left mid lung zone",
    "left lower lung zone",
    "right hilar region",
    "left hilar region",
    "right costophrenic angle",
    "left costophrenic angle",
]

# Map from our region names to Chest ImaGenome bbox_name field values.
REGION_TO_BBOX_NAME = {
    "right lung":               "right lung",
    "left lung":                "left lung",
    "cardiac silhouette":       "cardiac silhouette",
    "mediastinum":              "mediastinum",
    "trachea":                  "trachea",
    "right upper lung zone":    "right upper lung zone",
    "right mid lung zone":      "right mid lung zone",
    "right lower lung zone":    "right lower lung zone",
    "left upper lung zone":     "left upper lung zone",
    "left mid lung zone":       "left mid lung zone",
    "left lower lung zone":     "left lower lung zone",
    "right hilar region":       "right hilar structures",
    "left hilar region":        "left hilar structures",
    "right costophrenic angle": "right costophrenic angle",
    "left costophrenic angle":  "left costophrenic angle",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

print("Loading RadVLM...")
radvlm_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    RADVLM_PATH,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(DEVICE)
radvlm_model.eval()
radvlm_processor = AutoProcessor.from_pretrained(RADVLM_PATH)
print("RadVLM loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── RADVLM INFERENCE (from official model card) ───────────────────────────────

def inference_radvlm(model, processor, image, prompt, chat_history=None, max_new_tokens=256):
    if chat_history is None:
        chat_history = []

    conversation = []
    for idx, (user_text, assistant_text) in enumerate(chat_history):
        if idx == 0:
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}, {"type": "image"}],
            })
        else:
            conversation.append({
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            })
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": assistant_text}],
        })

    if len(chat_history) == 0:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}, {"type": "image"}],
        })
    else:
        conversation.append({
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        })

    full_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        images=image, text=full_prompt, return_tensors="pt", padding=True
    ).to(model.device, torch.float16)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    full_response = processor.decode(output[0], skip_special_tokens=True)
    response = re.split(r"(user|assistant)", full_response)[-1].strip()
    chat_history.append((prompt, response))
    return response, chat_history

# ── BOX PARSING ───────────────────────────────────────────────────────────────

_NUM = r"(\d*\.?\d+)"
_SEP = r"[\s,]+"

# Ordered by specificity. Every pattern captures exactly 4 groups.
_BOX_PATTERNS = [
    # [x1, y1, x2, y2]  (primary RadVLM format)
    r"\[\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*\]",
    # (x1, y1, x2, y2)
    r"\(\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*\)",
    # <box>x1, y1, x2, y2</box>  (fallback 1: tagged output)
    r"<box>\s*" + _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM + r"\s*</box>",
    # x1: .. y1: .. x2: .. y2: ..  (fallback 2: labeled fields)
    r"x1\s*[:=]\s*" + _NUM + r".*?y1\s*[:=]\s*" + _NUM +
    r".*?x2\s*[:=]\s*" + _NUM + r".*?y2\s*[:=]\s*" + _NUM,
    # bare  x1, y1, x2, y2  (last resort)
    _NUM + _SEP + _NUM + _SEP + _NUM + _SEP + _NUM,
]


def parse_box_from_response(response, img_w, img_h):
    """
    RadVLM outputs boxes as NORMALIZED [0,1] coordinates in its text response.
    Parse with several fallback patterns, clamp each value to [0, 1], then
    denormalize to pixel coordinates. Returns [x1, y1, x2, y2] in pixels, or
    None if nothing usable parses.
    """
    if not response:
        return None
    for pattern in _BOX_PATTERNS:
        match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        try:
            vals = [float(match.group(i)) for i in range(1, 5)]
        except (ValueError, IndexError):
            continue
        # Clamp normalized coordinates to [0, 1] before denormalizing.
        vals = [min(max(v, 0.0), 1.0) for v in vals]
        x1 = int(round(vals[0] * img_w))
        y1 = int(round(vals[1] * img_h))
        x2 = int(round(vals[2] * img_w))
        y2 = int(round(vals[3] * img_h))
        if x2 > x1 and y2 > y1:
            return [x1, y1, x2, y2]
    return None

# ── MEDSAM SEGMENTATION ───────────────────────────────────────────────────────

def get_medsam_mask(pil_image, box_xyxy):
    """Segment inside box_xyxy (pixel x1y1x2y2 at original resolution).

    SamProcessor resizes to 1024x1024 and rescales the box; the mask is resized
    back to original dimensions by post_process_masks(original_sizes=...).
    """
    inputs = medsam_processor(
        pil_image, input_boxes=[[box_xyxy]], return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = medsam_model(**inputs, multimask_output=False)
    masks = medsam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    return masks[0].squeeze().numpy().astype(bool)

# ── PER-REGION VISUALIZATION ──────────────────────────────────────────────────

def save_region_png(pil_image, region, pred_box, gold_box, mask, iou, image_id, out_dir):
    """
    3-panel figure for one region:
      Left:   original CXR + gold box (green) + predicted box (red)
      Middle: CXR + MedSAM mask overlay + gold box outline
      Right:  stats text panel
    """
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"RadVLM + MedSAM  |  {image_id}  |  {region}", fontsize=11, y=1.01)

    # Panel 1: boxes
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Predicted (red) vs Gold (green)")
    if gold_box:
        gx1, gy1, gx2, gy2 = [int(v) for v in gold_box]
        axes[0].add_patch(mpatches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor="lime", facecolor="none", label="Gold"))
    if pred_box:
        px1, py1, px2, py2 = [int(v) for v in pred_box]
        axes[0].add_patch(mpatches.Rectangle(
            (px1, py1), px2 - px1, py2 - py1,
            linewidth=2, edgecolor="red", facecolor="none", label="Predicted"))
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].axis("off")

    # Panel 2: mask
    axes[1].imshow(img_np, cmap="gray")
    axes[1].set_title("MedSAM mask + gold outline")
    if mask is not None:
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [0.2, 0.6, 1.0, 0.45]
        axes[1].imshow(overlay)
    if gold_box:
        gx1, gy1, gx2, gy2 = [int(v) for v in gold_box]
        axes[1].add_patch(mpatches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor="lime", facecolor="none"))
    axes[1].axis("off")

    # Panel 3: stats
    axes[2].axis("off")
    stats_lines = [
        "Model:   RadVLM + MedSAM",
        f"Image:   {image_id}",
        f"Region:  {region}",
        "",
        f"Box IoU: {iou:.3f}" if iou is not None else "Box IoU: N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
    ]
    axes[2].text(0.05, 0.95, "\n".join(stats_lines),
                 transform=axes[2].transAxes,
                 fontsize=9, verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    plt.tight_layout()
    region_slug = region.replace(" ", "_")
    out_path = Path(out_dir) / image_id / f"{region_slug}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = str(Path(OUTPUT_DIR) / "results_summary.csv")
    init_results_csv(summary_path)

    print(f"\nLoading gold annotations from {GOLD_CSV}...")
    gold_annotations = load_gold_annotations(GOLD_CSV)
    image_ids = list(gold_annotations.keys())
    if MAX_IMAGES:
        image_ids = image_ids[:MAX_IMAGES]
    print(f"{len(image_ids)} images to process.")

    all_results = []
    images_processed = 0
    regions_done = 0
    failed_count = 0

    def _process_image(img_idx, image_id):
        # Entire per-image body. Wrapped by the caller in try/except so a single
        # bad image can never stop the whole run.
        nonlocal images_processed, regions_done, failed_count
        print(f"\n[{img_idx+1}/{len(image_ids)}] {image_id}")

        img_path = find_image_path(IMAGE_DIR, image_id)
        if img_path is None:
            log.error("image_id=%s: image file not found, skipping.", image_id)
            failed_count += 1
            return

        pil_img = load_image(img_path)
        if pil_img is None:
            log.error("image_id=%s: load_image returned None, skipping.", image_id)
            failed_count += 1
            return

        images_processed += 1
        W, H = pil_img.size
        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)

            pred_box = None
            mask = None
            iou = None
            response = ""

            try:
                prompt = (
                    f"Locate the {region} in this chest X-ray. "
                    f"Provide the bounding box as [x1, y1, x2, y2] "
                    f"with coordinates normalized between 0 and 1."
                )
                response, _ = inference_radvlm(
                    radvlm_model, radvlm_processor, pil_img, prompt
                )
                pred_box = parse_box_from_response(response, W, H)
                if pred_box is None:
                    log.info("image_id=%s region=%s: no box parsed from: %s",
                             image_id, region, response[:120])

                # Clamp + validate before MedSAM.
                if pred_box is not None:
                    pred_box = clamp_box(pred_box, W, H)
                    if not validate_box(pred_box, W, H):
                        pred_box = None

                if pred_box is not None:
                    try:
                        mask = get_medsam_mask(pil_img, pred_box)
                    except Exception:
                        log.exception("image_id=%s region=%s: MedSAM failed",
                                      image_id, region)
                        mask = None
                    iou = compute_iou(pred_box, gold_box)

                detected = pred_box is not None
                write_result_row(
                    summary_path,
                    result_row(image_id, region, iou, detected, gold_box, pred_box),
                    BOX_FIELDS,
                )
                regions_done += 1

                per_image_boxes[region] = {
                    "pred_box": pred_box, "raw_response": response[:200]
                }
                per_image_masks[region] = mask
                all_results.append({
                    "image_id": image_id, "region": region,
                    "iou": iou, "detected": detected,
                })

                print(f"  {region}: pred={pred_box}  gold={gold_box}  "
                      f"IoU={f'{iou:.3f}' if iou is not None else 'N/A'}")

                try:
                    save_region_png(pil_img, region, pred_box, gold_box, mask,
                                    iou, image_id, out_dir=OUTPUT_DIR)
                except Exception:
                    log.exception("image_id=%s region=%s: visualization failed",
                                  image_id, region)

            except Exception:
                failed_count += 1
                log.exception("image_id=%s region=%s: inference block failed",
                              image_id, region)
                write_result_row(
                    summary_path,
                    result_row(image_id, region, None, False, gold_box, None),
                    BOX_FIELDS,
                )
                continue

        # Per-image JSON + masks NPZ (best-effort)
        try:
            json_out = Path(OUTPUT_DIR) / image_id / "boxes.json"
            json_out.parent.mkdir(parents=True, exist_ok=True)
            with open(json_out, "w") as f:
                json.dump(per_image_boxes, f, indent=2)
            masks_to_save = {
                r.replace(" ", "_"): m
                for r, m in per_image_masks.items() if m is not None
            }
            if masks_to_save:
                np.savez_compressed(
                    str(Path(OUTPUT_DIR) / image_id / "masks.npz"), **masks_to_save
                )
        except Exception:
            log.exception("image_id=%s: failed to write per-image JSON/NPZ", image_id)

    for img_idx, image_id in enumerate(image_ids):
        try:
            _process_image(img_idx, image_id)
        except Exception:
            failed_count += 1
            log.exception("image_id=%s: unhandled per-image error, skipping.", image_id)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Incremental results written to {summary_path}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        region_summary = (
            results_df.groupby("region")
            .agg(
                detected=("detected", "sum"),
                total=("region", "count"),
                mean_iou=("iou", "mean"),
            )
            .reset_index()
        )
        region_summary["detection_rate"] = (
            region_summary["detected"] / region_summary["total"] * 100
        ).round(1)
        region_summary["mean_iou"] = region_summary["mean_iou"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_iou", ascending=False).to_string(index=False))

        overall_iou = results_df["iou"].dropna().mean()
        overall_det = results_df["detected"].mean() * 100
        print(f"\nOverall mean IoU (detected only): {overall_iou:.3f}")
        print(f"Overall detection rate:           {overall_det:.1f}%")

    print(f"\nCompleted: {images_processed} images, {regions_done} regions, "
          f"{failed_count} failures")

if __name__ == "__main__":
    main()
