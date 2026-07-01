"""
grounding_dino_medsam.py
------------------------
Self-contained script: Grounding DINO + MedSAM as a formal baseline for
anatomical region grounding on the Chest ImaGenome gold subset.

Requirements: see requirements_grounding_dino.txt

Model: IDEA-Research/grounding-dino-tiny (open weights, no gated access)
       wanglab/medsam-vit-base (MedSAM, open weights)

Expected result:
  This is a NEGATIVE BASELINE. Grounding DINO has zero chest X-ray training
  and no concept of normal anatomical structures by name. Empirically (from
  sanity-check runs), almost all 15 regions produce near-identical full-image
  boxes (~full frame) with scores 0.49-0.67, giving near-zero IoU against
  gold annotations. This is a legitimate and important finding: it establishes
  the floor for zero-shot natural-image detectors on this task, and provides
  a clean contrast with domain-specific models like RadVLM.

  Key quirk fixed here: regions are queried INDIVIDUALLY (one inference call
  per region), not batched. Batching all 15 together causes token-boundary
  confusion in Grounding DINO's phrase matching, producing merged label strings.
  Individual calls give clean, correctly-labelled detections.
"""

import os

# ── CONFIG (edit these, or override via environment variables of the same name)
GOLD_CSV        = os.environ.get("GOLD_CSV", "/path/to/gold_bbox.csv")
IMAGE_DIR       = os.environ.get("IMAGE_DIR", "/path/to/mimic_cxr_images")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "./outputs/grounding_dino")
_max_images_env = os.environ.get("MAX_IMAGES")
MAX_IMAGES      = int(_max_images_env) if _max_images_env else None
BOX_THRESHOLD   = float(os.environ.get("BOX_THRESHOLD", 0.25))
TEXT_THRESHOLD  = float(os.environ.get("TEXT_THRESHOLD", 0.20))
# ──────────────────────────────────────────────────────────────────────────────

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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import SamModel, SamProcessor

from cxr_common import load_image, find_image_file, box_iou

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

print("Loading Grounding DINO...")
gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(DEVICE)
gdino_model.eval()
print("Grounding DINO loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── GROUNDING DINO INFERENCE ──────────────────────────────────────────────────

def gdino_ground_region(pil_image, region):
    """
    Run Grounding DINO for a single region phrase.
    Returns (box_xyxy, score) or (None, None) if nothing detected above threshold.

    Queried one region at a time to avoid Grounding DINO's cross-phrase token
    confusion when multiple similar phrases (e.g. 'right lung', 'left lung')
    are batched together.
    """
    inputs = gdino_processor(
        images=pil_image,
        text=[[region]],
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[pil_image.size[::-1]],
    )[0]

    if len(results["scores"]) == 0:
        return None, None

    # Take highest-scoring detection
    best_idx = results["scores"].argmax().item()
    box = [round(c, 1) for c in results["boxes"][best_idx].tolist()]
    score = round(results["scores"][best_idx].item(), 3)
    return box, score

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

def save_region_png(pil_image, region, pred_box, score, gold_box, mask,
                    iou, image_id, out_dir):
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Grounding DINO + MedSAM  |  {image_id}  |  {region}",
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
            linewidth=2, edgecolor="red", facecolor="none",
            label=f"Predicted ({score:.2f})"))
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
        "Model:    Grounding DINO + MedSAM",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Box IoU:  {iou:.3f}" if iou is not None else "Box IoU:  N/A",
        f"DINO score: {score:.3f}" if score is not None else "DINO score: N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
        "",
        "Note: NEGATIVE BASELINE.",
        "No chest X-ray training.",
        "Expected near-zero IoU.",
    ]
    axes[2].text(
        0.05, 0.95, "\n".join(stats_lines),
        transform=axes[2].transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#fff0f0", alpha=0.8)
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

        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            pred_box = None
            score    = None
            mask     = None

            try:
                pred_box, score = gdino_ground_region(pil_img, region)
            except Exception as e:
                print(f"  {region}: Grounding DINO error: {e}")

            per_image_boxes[region] = {
                "pred_box": pred_box,
                "score":    score,
            }

            if pred_box is not None:
                try:
                    mask = get_medsam_mask(pil_img, pred_box)
                except Exception as e:
                    print(f"  {region}: MedSAM error: {e}")
            per_image_masks[region] = mask

            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box  = gold_annotations[image_id].get(bbox_name)

            iou = None
            if pred_box and gold_box:
                iou = box_iou(pred_box, gold_box)

            print(
                f"  {region}: score={f'{score:.3f}' if score else 'N/A'}"
                f"  IoU={f'{iou:.3f}' if iou is not None else 'N/A'}"
                f"  pred={pred_box}"
            )

            save_region_png(
                pil_img, region, pred_box, score, gold_box,
                mask, iou, image_id, out_dir=OUTPUT_DIR
            )

            all_results.append({
                "image_id": image_id,
                "region":   region,
                "pred_box": pred_box,
                "gold_box": gold_box,
                "score":    score,
                "iou":      iou,
                "detected": pred_box is not None,
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
    print("SUMMARY  (expected: near-zero IoU across all regions)")
    print("="*60)

    results_df   = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "results_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Full results saved to {summary_path}")

    if not results_df.empty:
        region_summary = (
            results_df.groupby("region")
            .agg(
                detected = ("detected", "sum"),
                total    = ("image_id", "count"),
                mean_iou = ("iou",      "mean"),
                mean_score = ("score",  "mean"),
            )
            .reset_index()
        )
        region_summary["detection_rate"] = (
            region_summary["detected"] / region_summary["total"] * 100
        ).round(1)
        region_summary["mean_iou"]   = region_summary["mean_iou"].round(3)
        region_summary["mean_score"] = region_summary["mean_score"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_iou", ascending=False).to_string(index=False))

        overall_iou = results_df["iou"].dropna().mean()
        overall_det = results_df["detected"].mean() * 100
        print(f"\nOverall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

if __name__ == "__main__":
    main()
