"""
medgemma15_medsam.py
---------------------
Self-contained script: MedGemma 1.5 (4B) + MedSAM for anatomical region
grounding on the Chest ImaGenome gold subset.

Model: google/medgemma-1.5-4b-it (gated; requires an HF token with approved
access -- reuses the same HF_TOKEN already used by job_maira2.sh / this repo).
Architecture: Gemma3ForConditionalGeneration, loaded via
AutoModelForImageTextToText. Verified on this cluster (2026-09-12 smoke test,
slurm_medgemma_smoke.340280.out) that the `gdino` env's transformers 5.13.0
loads this model directly -- no separate conda env was needed.

>>> CRITICAL BOX FORMAT (verified against REAL model output, not assumed) <<<
MedGemma follows the Gemini-family "box_2d" convention: a JSON list of
objects, each ``{"box_2d": [y0, x0, y1, x1], "label": ...}``, with all four
values integers on a 0-1000 NORMALIZED grid (NOT 0-1, NOT pixels) relative to
the ORIGINAL image dimensions (not the resized 896x896 model input -- the
896x896 resize is a non-aspect-preserving squash, so a fractional position
along either axis is preserved regardless of which space it's expressed in).
Order is Y BEFORE X -- opposite of every other model in this benchmark, which
use [x1,y1,x2,y2]. Real example captured on this cluster for a 3056x2544
image, region="right lung":
    RAW RESPONSE: '```json\\n[{"box_2d": [160, 150, 830, 900], "label": "right lung"}]\\n```'
Converted: x1=150/1000*3056=458, y1=160/1000*2544=407,
           x2=900/1000*3056=2750, y2=830/1000*2544=2112.
This particular example is a wide, imprecise box spanning most of the chest
rather than tightly isolating the right lung -- MedGemma was not fine-tuned
for anatomical-region grounding on CXRs, so wide/imprecise (but not
nonsensical) boxes are an expected real result, not a parsing bug. This is
exactly the kind of result that must be sanity-checked by eye per the smoke
test requirement below, rather than trusted blindly from a single sample.
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

log = get_logger("medgemma15")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR"])
setup_hf_home(cfg)

GOLD_CSV   = cfg["GOLD_CSV"]
IMAGE_DIR  = cfg["IMAGE_DIR"]
OUTPUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "medgemma15")
MAX_IMAGES = cfg.get("MAX_IMAGES")
PRINT_RAW_FIRST_N = 3   # print raw model output for the first N images for manual verification

MODEL_ID = "google/medgemma-1.5-4b-it"
BOX_GRID = 1000.0  # MedGemma/Gemini box_2d normalization grid

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

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
    "right hilar region":       "right hilar region",
    "left hilar region":        "left hilar region",
    "right costophrenic angle": "right costophrenic angle",
    "left costophrenic angle":  "left costophrenic angle",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── LOAD MODELS ───────────────────────────────────────────────────────────────

print(f"Loading {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
medgemma_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map=DEVICE,
)
medgemma_model.eval()
print(f"{MODEL_ID} loaded.")

print("Loading MedSAM...")
from transformers import SamModel, SamProcessor
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── MEDGEMMA GROUNDING ────────────────────────────────────────────────────────

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def medgemma_ground_region(pil_image, region, print_raw=False):
    """
    Prompt MedGemma for a bounding box of one named anatomical region.
    Returns (pixel-coordinate box [x1,y1,x2,y2] or None, raw response text).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": (
                    f'Detect the "{region}" in this chest X-ray image. '
                    f'Return the bounding box as a JSON list with one object: '
                    f'[{{"box_2d": [y0, x0, y1, x1], "label": "{region}"}}]. '
                    f'Output ONLY the JSON, nothing else.'
                )},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(medgemma_model.device, dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        gen = medgemma_model.generate(**inputs, max_new_tokens=200, do_sample=False)

    gen_tokens = gen[0][input_len:]
    response = processor.decode(gen_tokens, skip_special_tokens=True)

    if print_raw:
        print(f"    RAW[{region}]: {response[:300]!r}")

    W, H = pil_image.size
    box = parse_box_from_response(response, W, H)
    return box, response


def parse_box_from_response(response, img_w, img_h):
    """
    Extract [x1,y1,x2,y2] pixel coordinates from MedGemma's box_2d JSON.

    box_2d is [y0,x0,y1,x1] on a 0-1000 grid relative to the ORIGINAL image
    dimensions (verified on real output -- see module docstring). Handles a
    ```json fenced block or bare JSON. Returns None if nothing usable found.
    """
    if not response:
        return None
    cleaned = response.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except Exception:
        m = _JSON_ARRAY_RE.search(cleaned)
        if not m:
            return None
        try:
            parsed = json.loads(m.group(0))
        except Exception:
            return None

    if not isinstance(parsed, list) or len(parsed) == 0:
        return None
    entry = parsed[0]
    if not isinstance(entry, dict) or "box_2d" not in entry:
        return None
    box_2d = entry["box_2d"]
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        return None

    try:
        y0, x0, y1, x1 = [float(v) for v in box_2d]
    except (TypeError, ValueError):
        return None

    # Normalized 0-1000 grid, Y BEFORE X -> pixel [x1,y1,x2,y2] in original image space.
    px1 = x0 / BOX_GRID * img_w
    py1 = y0 / BOX_GRID * img_h
    px2 = x1 / BOX_GRID * img_w
    py2 = y1 / BOX_GRID * img_h

    px1, px2 = min(px1, px2), max(px1, px2)
    py1, py2 = min(py1, py2), max(py1, py2)
    if px2 <= px1 or py2 <= py1:
        return None
    return [px1, py1, px2, py2]

# ── MEDSAM SEGMENTATION ───────────────────────────────────────────────────────

def get_medsam_mask(pil_image, box_xyxy):
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
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"MedGemma 1.5 (4B) + MedSAM  |  {image_id}  |  {region}", fontsize=11, y=1.01)

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

    axes[2].axis("off")
    stats_lines = [
        "Model:    MedGemma 1.5 (4B) + MedSAM",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Box IoU:  {iou:.3f}" if iou is not None else "Box IoU:  N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
    ]
    axes[2].text(
        0.05, 0.95, "\n".join(stats_lines),
        transform=axes[2].transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8)
    )

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
        print_raw = img_idx < PRINT_RAW_FIRST_N

        for region in REGIONS:
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)

            pred_box = None
            mask = None
            iou = None
            raw_resp = ""

            try:
                pred_box, raw_resp = medgemma_ground_region(pil_img, region, print_raw=print_raw)

                if pred_box is not None:
                    pred_box = clamp_box(pred_box, W, H)
                    if not validate_box(pred_box, W, H):
                        pred_box = None

                if pred_box is not None:
                    try:
                        mask = get_medsam_mask(pil_img, pred_box)
                    except Exception:
                        log.exception("image_id=%s region=%s: MedSAM failed", image_id, region)
                        mask = None
                    iou = compute_iou(pred_box, gold_box)

                detected = pred_box is not None
                write_result_row(
                    summary_path,
                    result_row(image_id, region, iou, detected, gold_box, pred_box),
                    BOX_FIELDS,
                )
                regions_done += 1

                per_image_boxes[region] = {"pred_box": pred_box, "raw_response": raw_resp[:300]}
                per_image_masks[region] = mask
                all_results.append({
                    "image_id": image_id, "region": region,
                    "iou": iou, "detected": detected,
                })

                print(f"  {region}: IoU={f'{iou:.3f}' if iou is not None else 'N/A'}"
                      f"  pred={pred_box}")

                try:
                    save_region_png(pil_img, region, pred_box, gold_box, mask,
                                    iou, image_id, out_dir=OUTPUT_DIR)
                except Exception:
                    log.exception("image_id=%s region=%s: visualization failed", image_id, region)

            except Exception:
                failed_count += 1
                log.exception("image_id=%s region=%s: inference block failed", image_id, region)
                write_result_row(
                    summary_path,
                    result_row(image_id, region, None, False, gold_box, None),
                    BOX_FIELDS,
                )
                continue

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
                np.savez_compressed(str(Path(OUTPUT_DIR) / image_id / "masks.npz"), **masks_to_save)
        except Exception:
            log.exception("image_id=%s: failed to write per-image JSON/NPZ", image_id)

    for img_idx, image_id in enumerate(image_ids):
        try:
            _process_image(img_idx, image_id)
        except Exception:
            failed_count += 1
            log.exception("image_id=%s: unhandled per-image error, skipping.", image_id)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Incremental results written to {summary_path}")

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        region_summary = (
            results_df.groupby("region")
            .agg(detected=("detected", "sum"), total=("region", "count"), mean_iou=("iou", "mean"))
            .reset_index()
        )
        region_summary["detection_rate"] = (region_summary["detected"] / region_summary["total"] * 100).round(1)
        region_summary["mean_iou"] = region_summary["mean_iou"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_iou", ascending=False).to_string(index=False))

        overall_iou = results_df["iou"].dropna().mean()
        overall_det = results_df["detected"].mean() * 100
        print(f"\nOverall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

    print(f"\nCompleted: {images_processed} images, {regions_done} regions, "
          f"{failed_count} failures")

if __name__ == "__main__":
    main()
