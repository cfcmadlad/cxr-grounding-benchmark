"""
biovilt_medsam.py
-----------------
Self-contained script: BioViL-T + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements: see requirements_biovilt.txt
"""

import os

# ── CONFIG (edit these, or override via environment variables of the same name)
GOLD_CSV        = os.environ.get("GOLD_CSV", "/path/to/gold_bbox.csv")
IMAGE_DIR       = os.environ.get("IMAGE_DIR", "/path/to/mimic_cxr_images")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "./outputs/biovilt")
_max_images_env = os.environ.get("MAX_IMAGES")
MAX_IMAGES      = int(_max_images_env) if _max_images_env else None
PERCENTILE      = int(os.environ.get("PERCENTILE", 90))
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
from scipy import ndimage

import torch
from transformers import SamModel, SamProcessor

from cxr_common import load_image, find_image_file, box_iou, mask_dice

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

print("Loading BioViL-T...")
from health_multimodal.image import get_image_inference
from health_multimodal.image.utils import ImageModelType
from health_multimodal.text import get_bert_inference
from health_multimodal.text.utils import BertEncoderType
from health_multimodal.vlp import ImageTextInferenceEngine

image_inference = get_image_inference(ImageModelType.BIOVIL_T)
text_inference  = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
image_text_inference = ImageTextInferenceEngine(
    image_inference_engine=image_inference,
    text_inference_engine=text_inference,
)
image_text_inference.to(DEVICE)

# Compatibility patch: `batch_encode_plus` was deprecated for years and
# was finally removed from PreTrainedTokenizerBase in transformers v5.0.0
# (Jan 2026) as part of the v5 API cleanup -- NOT in 4.40 as sometimes
# assumed. requirements_biovilt.txt pins transformers<4.40 which keeps
# batch_encode_plus available, but this shim is kept as a safety net in
# case a transitive dependency ever pulls in transformers>=5 despite the
# pin. hi-ml-multimodal's BertEncoder still calls
# tokenizer.batch_encode_plus directly and was archived on GitHub
# (Nov 21, 2025), so it will never be updated to use __call__ instead.
#
# Patched at both the instance level (covers the tokenizer object already
# constructed above) and the class level (covers any tokenizer object
# hi-ml-multimodal constructs internally later, e.g. during .to(device)
# or lazy re-initialization), since which object's method is actually
# called is an internal implementation detail of hi-ml-multimodal.
def _batch_encode_plus_shim(self, batch_text_or_text_pairs, **kwargs):
    return self(batch_text_or_text_pairs, **kwargs)

if not hasattr(text_inference.tokenizer, "batch_encode_plus"):
    text_inference.tokenizer.batch_encode_plus = (
        lambda batch_text_or_text_pairs, **kwargs: text_inference.tokenizer(
            batch_text_or_text_pairs, **kwargs
        )
    )

from transformers import PreTrainedTokenizerBase
if not hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
    PreTrainedTokenizerBase.batch_encode_plus = _batch_encode_plus_shim

print("BioViL-T loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model     = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── HEATMAP TO BOX ────────────────────────────────────────────────────────────

def heatmap_to_box(sim_map, percentile=PERCENTILE):
    """
    Threshold similarity heatmap at given percentile, take the largest
    connected component, return its bounding box as [x1, y1, x2, y2].
    This is the standard approach used in the original MS-CXR benchmark.
    """
    valid = np.nan_to_num(sim_map, nan=-1e9)
    finite_vals = valid[valid > -1e8]
    if finite_vals.size == 0:
        return None, None
    thresh = np.percentile(finite_vals, percentile)
    above = valid >= thresh
    labeled, num_components = ndimage.label(above)
    if num_components == 0:
        return None, None
    sizes = ndimage.sum(above, labeled, range(1, num_components + 1))
    largest_label = int(np.argmax(sizes)) + 1
    component = labeled == largest_label
    ys, xs = np.where(component)
    # +1 on the upper bound: xs.max()/ys.max() are the last INCLUDED pixel
    # index, but every box elsewhere in this benchmark (gold boxes built
    # as [x, y, x+w, y+h], MedSAM/Grounding DINO/VLM boxes) uses an
    # EXCLUSIVE upper bound. Without +1 here, every BioViL-T box was 1
    # pixel too narrow/short compared to gold and to every other model.
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    peak_score = float(finite_vals.max())
    return box, peak_score

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

def save_region_png(pil_image, region, pred_box, gold_box, mask, iou,
                    peak_score, image_id, out_dir):
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"BioViL-T + MedSAM  |  {image_id}  |  {region}",
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
        "Model:      BioViL-T + MedSAM",
        f"Image:      {image_id}",
        f"Region:     {region}",
        "",
        f"Box IoU:    {iou:.3f}" if iou is not None else "Box IoU:    N/A",
        f"Peak sim:   {peak_score:.3f}" if peak_score is not None else "Peak sim:   N/A",
        f"Percentile: {PERCENTILE}",
        "",
        f"Pred box:   {[int(v) for v in pred_box]}" if pred_box else "Pred box:   not detected",
        f"Gold box:   {gold_box}" if gold_box else "Gold box:   not in annotations",
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
    if MAX_IMAGES is not None:
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

        # BioViL-T needs the image saved to a temp path (API takes file path)
        tmp_path = Path(OUTPUT_DIR) / f"_tmp_{image_id}.png"
        try:
            pil_img.save(tmp_path)
        except Exception as e:
            print(f"  Failed to write temp image for BioViL-T: {e}, skipping.")
            continue

        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            pred_box   = None
            peak_score = None
            mask       = None

            try:
                sim_map = image_text_inference.get_similarity_map_from_raw_data(
                    image_path=tmp_path,
                    query_text=region,
                    interpolation="bilinear",
                )
                pred_box, peak_score = heatmap_to_box(sim_map, percentile=PERCENTILE)
            except Exception as e:
                print(f"  {region}: BioViL-T error: {e}")

            per_image_boxes[region] = {
                "pred_box":   pred_box,
                "peak_score": peak_score,
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

            iou_str = f"{iou:.3f}" if iou is not None else "N/A"
            sim_str = f"{peak_score:.3f}" if peak_score is not None else "N/A"
            print(f"  {region}: IoU={iou_str}  peak_sim={sim_str}")

            save_region_png(
                pil_img, region, pred_box, gold_box, mask,
                iou, peak_score, image_id, out_dir=OUTPUT_DIR
            )

            all_results.append({
                "image_id":   image_id,
                "region":     region,
                "pred_box":   pred_box,
                "gold_box":   gold_box,
                "iou":        iou,
                "peak_score": peak_score,
                "detected":   pred_box is not None,
            })

        # Clean up temp file
        if tmp_path.exists():
            tmp_path.unlink()

        # Save per-image JSON
        json_out = Path(OUTPUT_DIR) / image_id / "boxes.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(per_image_boxes, f, indent=2)

        # Save per-image masks NPZ
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
                detected  = ("detected",   "sum"),
                total     = ("image_id",   "count"),
                mean_iou  = ("iou",        "mean"),
                mean_sim  = ("peak_score", "mean"),
            )
            .reset_index()
        )
        region_summary["detection_rate"] = (
            region_summary["detected"] / region_summary["total"] * 100
        ).round(1)
        region_summary["mean_iou"] = region_summary["mean_iou"].round(3)
        region_summary["mean_sim"] = region_summary["mean_sim"].round(3)
        print("\nPer-region results:")
        print(region_summary.sort_values("mean_iou", ascending=False).to_string(index=False))

        overall_iou = results_df["iou"].dropna().mean()
        overall_det = results_df["detected"].mean() * 100
        print(f"\nOverall mean IoU:       {overall_iou:.3f}")
        print(f"Overall detection rate: {overall_det:.1f}%")

if __name__ == "__main__":
    main()
