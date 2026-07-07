"""
maira2_medsam.py
----------------
Self-contained script: MAIRA-2 + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements: see requirements_maira2.txt

IMPORTANT — Gated model access:
  MAIRA-2 requires accepting a one-time disclaimer on HuggingFace before
  weights can be downloaded. Go to https://huggingface.co/microsoft/maira-2
  and click "Agree and access repository". Then log in from the terminal:
      huggingface-cli login

All paths are read from config.yaml at the repo root — nothing is hardcoded.

Notes on MAIRA-2's phrase grounding:
  - Designed to ground pathology findings (e.g. "pleural effusion"), not
    normal anatomy. Anatomical grounding is not its primary use case, so
    mAP will likely be lower than RadVLM's 85.3% on this task.
  - Box coordinates are output NORMALIZED and relative to MAIRA-2's internally
    cropped 518x518 view. Mapping them back to original pixels is NOT a plain
    W/518, H/518 scale: MAIRA-2 pads the image to a square and centre-crops
    before resizing to 518, so the inverse must also undo that pad/crop offset.
    processor.adjust_box_for_original_image_size() performs exactly this
    crop-aware inverse (returns pixel (x1, y1, x2, y2) in the original image),
    which is why we use it rather than reimplementing a naive scale.
  - transformers>=4.48.0,<4.52 is required (tested up to 4.51.3 per model card).
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG + HF cache (must precede heavy imports) ────────────────────────────
from cxr_common import (
    load_config, setup_hf_home, load_image, find_image_path,
    load_gold_annotations, validate_box, clamp_box, compute_iou,
    result_row, write_result_row, init_results_csv, BOX_FIELDS, get_logger,
)

log = get_logger("maira2")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR"])
setup_hf_home(cfg)

GOLD_CSV   = cfg["GOLD_CSV"]
IMAGE_DIR  = cfg["IMAGE_DIR"]
OUTPUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "maira2")
MAX_IMAGES = cfg.get("MAX_IMAGES")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
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

print("Loading MAIRA-2...")
maira_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/maira-2",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)
maira_model = maira_model.eval().to(DEVICE)
maira_processor = AutoProcessor.from_pretrained(
    "microsoft/maira-2",
    trust_remote_code=True,
)
print("MAIRA-2 loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── MAIRA-2 PHRASE GROUNDING ──────────────────────────────────────────────────

def maira_ground_phrase(pil_image, phrase):
    """
    Run MAIRA-2 phrase grounding for a single phrase on a single image.
    Returns (pixel-coordinate box [x1, y1, x2, y2] or None, raw decoded text).

    MAIRA-2 emits normalized coords relative to its cropped 518x518 view;
    adjust_box_for_original_image_size() maps them back to original pixels
    (accounting for the pad/centre-crop, not just a linear W/518, H/518 scale).
    """
    processed_inputs = maira_processor.format_and_preprocess_phrase_grounding_input(
        frontal_image=pil_image,
        phrase=phrase,
        return_tensors="pt",
    ).to(DEVICE, torch.float16)

    with torch.no_grad():
        output_decoding = maira_model.generate(
            **processed_inputs,
            max_new_tokens=150,
            use_cache=True,
        )

    prompt_length = processed_inputs["input_ids"].shape[-1]
    decoded_text = maira_processor.decode(
        output_decoding[0][prompt_length:],
        skip_special_tokens=True,
    )

    prediction = maira_processor.convert_output_to_plaintext_or_grounded_sequence(decoded_text)

    # prediction is a list of (text, boxes_or_None) tuples.
    if not prediction:
        return None, decoded_text

    _, boxes = prediction[0]
    if not boxes:
        return None, decoded_text

    # Take the first (usually only) box and adjust for original image size.
    raw_box = boxes[0]  # normalized coords relative to MAIRA-2's cropped view
    adjusted = maira_processor.adjust_box_for_original_image_size(
        box=raw_box,
        original_image=pil_image,
    )
    # adjusted is (x1, y1, x2, y2) in pixel coords of the original image.
    x1, y1, x2, y2 = adjusted
    return [int(x1), int(y1), int(x2), int(y2)], decoded_text

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
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"MAIRA-2 + MedSAM  |  {image_id}  |  {region}",
        fontsize=11, y=1.01
    )

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

    # Panel 2: MedSAM mask + gold outline
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
        "Model:    MAIRA-2 + MedSAM",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Box IoU:  {iou:.3f}" if iou is not None else "Box IoU:  N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
        "",
        "Note: MAIRA-2 is trained on pathology",
        "grounding, not normal anatomy.",
        "Lower IoU on anatomy is expected.",
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
        # Entire per-image body. Wrapped by the caller in try/except so a single
        # bad image can never stop the whole run (BUG 4: MAIRA-2 stopped after
        # ~4 images on the HPC run because an unhandled per-image error escaped).
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
            raw_resp = ""

            try:
                pred_box, raw_resp = maira_ground_phrase(pil_img, region)

                # Clamp mapped-back box to image bounds and validate before MedSAM.
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
                    "pred_box": pred_box, "raw_response": raw_resp
                }
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
        print(f"\nOverall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

    print(f"\nCompleted: {images_processed} images, {regions_done} regions, "
          f"{failed_count} failures")

if __name__ == "__main__":
    main()
