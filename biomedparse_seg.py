"""
biomedparse_seg.py
------------------
Self-contained script: BiomedParse for direct text-prompted mask segmentation
on the Chest ImaGenome gold subset.

IMPORTANT — this script is structurally different from the others.
BiomedParse uses custom model classes (BaseModel, build_model, etc.) that are
not available as a pip package. The script MUST be placed and run from inside
the cloned BiomedParse repository directory:

    git clone https://github.com/microsoft/BiomedParse.git
    cd BiomedParse
    # place this script here, then run it

Requirements: see requirements_biomedparse.txt (uses conda env 'biomedparse',
Python 3.9.19 as specified in the official BiomedParse environment.yml)

Key difference from other scripts:
  - BiomedParse outputs MASKS directly from text prompts.
  - No MedSAM step needed — this model IS the segmenter.
  - We derive a bounding box from the predicted mask (smallest enclosing box)
    for comparison with box-based models, but mask/Dice is the primary metric.
  - Output scores are sigmoid logits; threshold at 0.5 for binary mask.
"""

# ── CONFIG (edit these before running) ────────────────────────────────────────
GOLD_CSV    = "/path/to/gold_bbox.csv"      # Chest ImaGenome gold annotations
IMAGE_DIR   = "/path/to/mimic_cxr_images"   # root folder with DICOM or JPG/PNG
OUTPUT_DIR  = "./outputs/biomedparse"       # where results are saved
MAX_IMAGES  = None                          # set to e.g. 10 for a quick test run
MASK_THRESHOLD = 0.5                        # sigmoid threshold for binary mask
# ──────────────────────────────────────────────────────────────────────────────

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

import torch

# BiomedParse custom classes — these only work when the script is run
# from inside the cloned BiomedParse repo directory.
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
        "BiomedParse repo directory:\n"
        "  git clone https://github.com/microsoft/BiomedParse.git\n"
        "  cd BiomedParse\n"
        "  python biomedparse_seg.py"
    )

# ── 15 TARGET REGIONS ─────────────────────────────────────────────────────────
# BiomedParse uses free-text prompts — these are our best-effort phrasings.
# Chest X-ray is one of the 9 supported modalities (X-Ray-Chest).
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
    """Derive smallest enclosing bounding box from binary mask."""
    ys, xs = np.where(binary_mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def mask_dice(pred_mask, gold_mask):
    """Dice coefficient between two boolean masks."""
    pred_bool = pred_mask.astype(bool)
    gold_bool = gold_mask.astype(bool)
    intersection = (pred_bool & gold_bool).sum()
    denom = pred_bool.sum() + gold_bool.sum()
    if denom == 0:
        return 0.0
    return 2 * intersection / denom

def box_iou(pred, gold):
    """IoU between two [x1,y1,x2,y2] boxes."""
    ix1 = max(pred[0], gold[0])
    iy1 = max(pred[1], gold[1])
    ix2 = min(pred[2], gold[2])
    iy2 = min(pred[3], gold[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
    area_gold = (gold[2] - gold[0]) * (gold[3] - gold[1])
    return inter / (area_pred + area_gold - inter)

def gold_box_to_mask(gold_box, img_h, img_w):
    """Convert gold box [x1,y1,x2,y2] to a binary mask for Dice scoring."""
    mask = np.zeros((img_h, img_w), dtype=bool)
    x1, y1, x2, y2 = [int(v) for v in gold_box]
    mask[y1:y2, x1:x2] = True
    return mask

# ── IMAGE LOADING ─────────────────────────────────────────────────────────────

def load_image(path):
    path = str(path)
    if path.lower().endswith((".dcm", ".dicom")):
        try:
            import pydicom
        except ImportError:
            raise ImportError("pip install pydicom")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(float)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return Image.open(path).convert("RGB")

def find_image_file(image_dir, image_id):
    image_dir = Path(image_dir)
    for ext in [".jpg", ".jpeg", ".png", ".dcm", ".dicom"]:
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
        matches = list(image_dir.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None

# ── PER-REGION VISUALIZATION ──────────────────────────────────────────────────

def save_region_png(pil_image, region, pred_mask, pred_box, gold_box,
                    dice, iou, image_id, out_dir):
    """
    3-panel figure for one region:
      Left:   original CXR + derived pred box (red) + gold box (green)
      Middle: CXR + predicted mask overlay + gold box outline
      Right:  stats panel
    """
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

# ── GOLD ANNOTATION LOADER ────────────────────────────────────────────────────

def load_gold_annotations(csv_path):
    df = pd.read_csv(csv_path)
    required = {"image_id", "bbox_name", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Gold CSV is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            "Please update REGION_TO_BBOX_NAME and column names in load_gold_annotations()."
        )
    gold = {}
    for _, row in df.iterrows():
        iid  = str(row["image_id"])
        name = str(row["bbox_name"]).lower().strip()
        box  = [int(row["x"]), int(row["y"]),
                int(row["x"]) + int(row["w"]),
                int(row["y"]) + int(row["h"])]
        gold.setdefault(iid, {})[name] = box
    return gold

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nLoading gold annotations from {GOLD_CSV}...")
    gold_annotations = load_gold_annotations(GOLD_CSV)
    image_ids = list(gold_annotations.keys())
    if MAX_IMAGES:
        image_ids = image_ids[:MAX_IMAGES]
    print(f"{len(image_ids)} images to process.")

    all_results = []

    for img_idx, image_id in enumerate(image_ids):
        print(f"\n[{img_idx+1}/{len(image_ids)}] {image_id}")

        img_path = find_image_file(IMAGE_DIR, image_id)
        if img_path is None:
            print(f"  Image file not found, skipping.")
            continue

        try:
            pil_img = load_image(img_path)
        except Exception as e:
            print(f"  Failed to load image: {e}, skipping.")
            continue

        W, H = pil_img.size

        # BiomedParse: run ALL regions in one call (batched by design).
        # interactive_infer_image returns a list of pred masks (one per prompt),
        # each as a float numpy array of shape (H, W) with values in [0, 1].
        try:
            with torch.no_grad():
                pred_masks_raw = interactive_infer_image(
                    biomedparse_model, pil_img, REGIONS
                )
        except Exception as e:
            print(f"  BiomedParse inference error: {e}, skipping image.")
            continue

        per_image_masks = {}
        per_image_boxes = {}

        for region, pred_raw in zip(REGIONS, pred_masks_raw):
            # Threshold sigmoid output to get binary mask
            pred_mask = (pred_raw > MASK_THRESHOLD).astype(bool)

            # Resize mask to original image size if needed
            if pred_mask.shape != (H, W):
                pred_mask_img = Image.fromarray(pred_raw.astype(np.float32))
                pred_mask_img = pred_mask_img.resize((W, H), Image.BILINEAR)
                pred_mask = (np.array(pred_mask_img) > MASK_THRESHOLD)

            per_image_masks[region] = pred_mask

            # Derive bounding box from mask for comparison
            pred_box = mask_to_box(pred_mask) if pred_mask.any() else None
            per_image_boxes[region] = {"pred_box": pred_box}

            # Gold box and derived gold mask
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box  = gold_annotations[image_id].get(bbox_name)

            # Dice (mask vs gold-box-as-mask)
            dice = None
            if pred_mask.any() and gold_box:
                gold_mask = gold_box_to_mask(gold_box, H, W)
                dice = mask_dice(pred_mask, gold_mask)

            # Box IoU (derived pred box vs gold box)
            iou = None
            if pred_box and gold_box:
                iou = box_iou(pred_box, gold_box)

            detected = pred_mask.any()
            print(
                f"  {region}: Dice={f'{dice:.3f}' if dice is not None else 'N/A'}"
                f"  IoU={f'{iou:.3f}' if iou is not None else 'N/A'}"
                f"  detected={detected}"
            )

            save_region_png(
                pil_img, region, pred_mask if detected else None,
                pred_box, gold_box, dice, iou, image_id, out_dir=OUTPUT_DIR
            )

            all_results.append({
                "image_id": image_id,
                "region":   region,
                "pred_box": pred_box,
                "gold_box": gold_box,
                "dice":     dice,
                "iou":      iou,
                "detected": detected,
            })

        # Per-image JSON
        json_out = Path(OUTPUT_DIR) / image_id / "boxes.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(per_image_boxes, f, indent=2)

        # Per-image masks NPZ
        masks_to_save = {
            r.replace(" ", "_"): m
            for r, m in per_image_masks.items()
            if m is not None
        }
        if masks_to_save:
            np.savez_compressed(
                str(Path(OUTPUT_DIR) / image_id / "masks.npz"),
                **masks_to_save
            )

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    results_df   = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "results_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Full results saved to {summary_path}")

    if not results_df.empty:
        region_summary = (
            results_df.groupby("region")
            .agg(
                detected  = ("detected", "sum"),
                total     = ("image_id", "count"),
                mean_dice = ("dice",     "mean"),
                mean_iou  = ("iou",      "mean"),
            )
            .reset_index()
        )
        region_summary["detection_rate"] = (
            region_summary["detected"] / region_summary["total"] * 100
        ).round(1)
        region_summary["mean_dice"] = region_summary["mean_dice"].round(3)
        region_summary["mean_iou"]  = region_summary["mean_iou"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_dice", ascending=False).to_string(index=False))

        overall_dice = results_df["dice"].dropna().mean()
        overall_iou  = results_df["iou"].dropna().mean()
        overall_det  = results_df["detected"].mean() * 100
        print(f"\nOverall mean Dice:      {overall_dice:.3f}")
        print(f"Overall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

if __name__ == "__main__":
    main()
