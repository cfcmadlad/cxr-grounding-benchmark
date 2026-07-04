"""
biomedparse_seg.py
------------------
Self-contained script: BiomedParse for direct text-prompted mask segmentation
on the Chest ImaGenome gold subset.

IMPORTANT — this script is structurally different from the others.
BiomedParse uses custom model classes (BaseModel, build_model, etc.) that are
not available as a pip package. The script MUST be placed and run from inside
the cloned BiomedParse repository directory (BIOMEDPARSE_REPO in config.yaml):

    git clone https://github.com/microsoft/BiomedParse.git
    cd BiomedParse
    # place this script here, then run it (the Slurm job does this for you)

Because it runs from inside the BiomedParse repo, the Slurm job exports
PYTHONPATH and CXR_CONFIG so that ``cxr_common`` and ``config.yaml`` (which live
at the benchmark repo root) are still importable / findable.

Key difference from other scripts:
  - BiomedParse outputs MASKS directly from text prompts. No MedSAM step.
  - A bounding box is derived from the predicted mask (smallest enclosing box)
    for cross-model comparison, but mask/Dice is the primary metric.
  - Output scores are sigmoid logits; threshold at 0.5 for a binary mask.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG + shared utils (must precede heavy imports) ────────────────────────
try:
    from cxr_common import (
        load_config, setup_hf_home, load_image, find_image_file,
        load_gold_annotations, validate_box, compute_iou, compute_dice,
        result_row, write_result_row, init_results_csv, BOX_FIELDS_DICE, get_logger,
    )
except ImportError:
    # Running from inside the BiomedParse repo: the Slurm job sets PYTHONPATH to
    # the benchmark root. As a fallback for interactive runs, honor CXR_REPO_ROOT.
    _root = os.environ.get("CXR_REPO_ROOT")
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)
    from cxr_common import (
        load_config, setup_hf_home, load_image, find_image_file,
        load_gold_annotations, validate_box, compute_iou, compute_dice,
        result_row, write_result_row, init_results_csv, BOX_FIELDS_DICE, get_logger,
    )

log = get_logger("biomedparse")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR", "BIOMEDPARSE_REPO"])
setup_hf_home(cfg)

GOLD_CSV       = cfg["GOLD_CSV"]
IMAGE_DIR      = cfg["IMAGE_DIR"]
OUTPUT_DIR     = os.path.join(cfg["OUTPUT_DIR"], "biomedparse")
MAX_IMAGES     = cfg.get("MAX_IMAGES")
MASK_THRESHOLD = 0.5   # sigmoid threshold for binary mask

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch

# BiomedParse custom classes — only importable when running from inside the repo.
try:
    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.distributed import init_distributed
    from utilities.arguments import load_opt_from_config_files
    from utilities.constants import BIOMED_CLASSES
    from inference_utils.inference import interactive_infer_image
except ImportError as e:
    raise ImportError(
        f"BiomedParse custom classes not found: {e}\n"
        "Make sure you are running this script from inside the cloned "
        "BiomedParse repo directory (BIOMEDPARSE_REPO in config.yaml):\n"
        "  git clone https://github.com/microsoft/BiomedParse.git\n"
        "  cd BiomedParse\n"
        "  python biomedparse_seg.py"
    )

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

# ── LOAD BIOMEDPARSE ──────────────────────────────────────────────────────────

print("Loading BiomedParse...")
opt = load_opt_from_config_files(["configs/biomedparse_inference.yaml"])
opt = init_distributed(opt)

biomedparse_model = (
    BaseModel(opt, build_model(opt))
    .from_pretrained("hf_hub:microsoft/BiomedParse")
    .eval()
    .to(DEVICE)
)
with torch.no_grad():
    biomedparse_model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
        BIOMED_CLASSES + ["background"], is_eval=True
    )
print("BiomedParse loaded.")

# ── MASK UTILITIES ────────────────────────────────────────────────────────────

def mask_to_box(binary_mask):
    """Derive the smallest enclosing box from a binary mask.

    np.where returns (row_indices, col_indices) = (ys, xs). The box is built as
    [col_min, row_min, col_max, row_max] = [x1, y1, x2, y2] — NOT
    [row_min, col_min, ...]. Returns None for an empty mask.
    """
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def gold_box_to_mask(gold_box, img_h, img_w):
    """Convert gold box [x1,y1,x2,y2] to a binary mask for Dice scoring."""
    mask = np.zeros((img_h, img_w), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in gold_box]
    mask[y1:y2, x1:x2] = True
    return mask

# ── PER-REGION VISUALIZATION ──────────────────────────────────────────────────

def save_region_png(pil_image, region, pred_mask, pred_box, gold_box,
                    dice, iou, image_id, out_dir):
    img_np = np.array(pil_image)
    H, W = img_np.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"BiomedParse  |  {image_id}  |  {region}",
        fontsize=11, y=1.01
    )

    # Panel 1: boxes
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title("Mask-derived box (red) vs Gold (green)")
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

    # Panel 2: mask overlay + gold outline
    axes[1].imshow(img_np, cmap="gray")
    axes[1].set_title("BiomedParse mask + gold outline")
    if pred_mask is not None:
        overlay = np.zeros((H, W, 4))
        overlay[pred_mask] = [0.2, 0.6, 1.0, 0.45]
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
        "Model:    BiomedParse (direct mask)",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Mask Dice: {dice:.3f}" if dice is not None else "Mask Dice: N/A",
        f"Box IoU:   {iou:.3f}" if iou is not None else "Box IoU:   N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: no mask detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
        "",
        "Note: BiomedParse outputs masks directly",
        "from text (no MedSAM step needed).",
        "Box is derived from mask for comparison.",
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

    for img_idx, image_id in enumerate(image_ids):
        print(f"\n[{img_idx+1}/{len(image_ids)}] {image_id}")

        img_path = find_image_file(IMAGE_DIR, image_id)
        if img_path is None:
            log.error("image_id=%s: image file not found, skipping.", image_id)
            failed_count += 1
            continue

        pil_img = load_image(img_path)
        if pil_img is None:
            log.error("image_id=%s: load_image returned None, skipping.", image_id)
            failed_count += 1
            continue

        images_processed += 1
        W, H = pil_img.size

        # BiomedParse: all regions in one batched call. On failure, record a
        # failure row for every region and move on.
        try:
            with torch.no_grad():
                pred_masks_raw = interactive_infer_image(
                    biomedparse_model, pil_img, REGIONS
                )
        except Exception:
            log.exception("image_id=%s: BiomedParse inference failed for all regions",
                          image_id)
            for region in REGIONS:
                bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
                gold_box = gold_annotations[image_id].get(bbox_name)
                write_result_row(
                    summary_path,
                    result_row(image_id, region, None, False, gold_box, None, dice=None),
                    BOX_FIELDS_DICE,
                )
                failed_count += 1
            continue

        per_image_masks = {}
        per_image_boxes = {}

        for region, pred_raw in zip(REGIONS, pred_masks_raw):
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)

            pred_mask = None
            pred_box = None
            dice = None
            iou = None

            try:
                pred_raw = np.asarray(pred_raw)
                # Threshold sigmoid output to a binary mask.
                pred_mask = (pred_raw > MASK_THRESHOLD).astype(bool)

                # Resize mask to original image size if the model returned a
                # different resolution.
                if pred_mask.shape != (H, W):
                    pred_mask_img = Image.fromarray(pred_raw.astype(np.float32))
                    pred_mask_img = pred_mask_img.resize((W, H), Image.BILINEAR)
                    pred_mask = (np.array(pred_mask_img) > MASK_THRESHOLD)

                per_image_masks[region] = pred_mask

                # Derive bounding box [x1,y1,x2,y2] from the mask.
                pred_box = mask_to_box(pred_mask) if pred_mask.any() else None
                if pred_box is not None:
                    validate_box(pred_box, W, H)  # logs a warning if degenerate
                per_image_boxes[region] = {"pred_box": pred_box}

                # Dice (mask vs gold-box-as-mask).
                if pred_mask.any() and gold_box:
                    gold_mask = gold_box_to_mask(gold_box, H, W)
                    dice = compute_dice(pred_mask, gold_mask)

                # Box IoU (derived pred box vs gold box).
                iou = compute_iou(pred_box, gold_box)

                detected = bool(pred_mask.any())
                write_result_row(
                    summary_path,
                    result_row(image_id, region, iou, detected, gold_box, pred_box,
                               dice=dice),
                    BOX_FIELDS_DICE,
                )
                regions_done += 1

                all_results.append({
                    "image_id": image_id, "region": region,
                    "dice": dice, "iou": iou, "detected": detected,
                })

                print(
                    f"  {region}: Dice={f'{dice:.3f}' if dice is not None else 'N/A'}"
                    f"  IoU={f'{iou:.3f}' if iou is not None else 'N/A'}"
                    f"  detected={detected}"
                )

                try:
                    save_region_png(
                        pil_img, region, pred_mask if detected else None,
                        pred_box, gold_box, dice, iou, image_id, out_dir=OUTPUT_DIR
                    )
                except Exception:
                    log.exception("image_id=%s region=%s: visualization failed",
                                  image_id, region)

            except Exception:
                failed_count += 1
                log.exception("image_id=%s region=%s: post-processing failed",
                              image_id, region)
                write_result_row(
                    summary_path,
                    result_row(image_id, region, None, False, gold_box, None, dice=None),
                    BOX_FIELDS_DICE,
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
                mean_dice=("dice", "mean"),
                mean_iou=("iou", "mean"),
            )
            .reset_index()
        )
        region_summary["detection_rate"] = (
            region_summary["detected"] / region_summary["total"] * 100
        ).round(1)
        region_summary["mean_dice"] = region_summary["mean_dice"].round(3)
        region_summary["mean_iou"] = region_summary["mean_iou"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_dice", ascending=False).to_string(index=False))

        overall_dice = results_df["dice"].dropna().mean()
        overall_iou = results_df["iou"].dropna().mean()
        overall_det = results_df["detected"].mean() * 100
        print(f"\nOverall mean Dice:      {overall_dice:.3f}")
        print(f"Overall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

    print(f"\nCompleted: {images_processed} images, {regions_done} regions, "
          f"{failed_count} failures")

if __name__ == "__main__":
    main()
