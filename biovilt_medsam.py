"""
biovilt_medsam.py
-----------------
Self-contained script: BioViL-T + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements: see requirements_biovilt.txt

All paths are read from config.yaml at the repo root — nothing is hardcoded.

BioViL-T produces a text-image similarity heatmap at the ORIGINAL image
resolution (get_similarity_map_from_raw_data returns a map sized to the source
image). heatmap_to_box() thresholds it at a percentile, keeps the largest
connected component, and returns [x1, y1, x2, y2] in pixel coordinates
(x = column, y = row — not transposed), which is exactly what MedSAM expects.
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

log = get_logger("biovilt")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR"])
setup_hf_home(cfg)

GOLD_CSV   = cfg["GOLD_CSV"]
IMAGE_DIR  = cfg["IMAGE_DIR"]
OUTPUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "biovilt")
MAX_IMAGES = cfg.get("MAX_IMAGES")
PERCENTILE = 90   # heatmap threshold (90 = top 10% of similarity)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import ndimage

import torch
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

# Compatibility patch for tokenizer.batch_encode_plus (hi-ml-multimodal calls it;
# newer transformers changed how it is exposed).
#
# BUG 2: the previous shim did `return text_inference.tokenizer(...)`. But
# tokenizer.__call__ itself dispatches to batch_encode_plus, so the shim called
# straight back into itself — "maximum recursion depth exceeded" on every region
# and an empty results CSV. The fix must never route through __call__.
_tok = text_inference.tokenizer
_cls_bep = getattr(type(_tok), "batch_encode_plus", None)
if _cls_bep is not None:
    # Bind the class's real implementation onto this instance. It goes straight
    # to the fast/slow _batch_encode_plus backend and never re-enters __call__.
    _tok.batch_encode_plus = _cls_bep.__get__(_tok, type(_tok))
else:
    # batch_encode_plus genuinely absent: fall back to encode_plus per item —
    # again, never through tokenizer.__call__.
    def _batch_encode_plus_shim(batch_text_or_text_pairs, **kwargs):
        if isinstance(batch_text_or_text_pairs, (list, tuple)):
            return [_tok.encode_plus(t, **kwargs) for t in batch_text_or_text_pairs]
        return _tok.encode_plus(batch_text_or_text_pairs, **kwargs)
    _tok.batch_encode_plus = _batch_encode_plus_shim
print("BioViL-T loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model     = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── HEATMAP TO BOX ────────────────────────────────────────────────────────────

def heatmap_to_box(sim_map, percentile=PERCENTILE):
    """
    Threshold similarity heatmap at ``percentile``, take the largest connected
    component, and return its bounding box as [x1, y1, x2, y2] in PIXEL coords.

    np.where returns (row_indices, col_indices) = (ys, xs); the box uses xs for
    x (columns) and ys for y (rows), i.e. x1y1x2y2 — never transposed and never
    x1y1wh. Returns (None, None) if no finite similarity / no component.
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
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    peak_score = float(finite_vals.max())
    return box, peak_score

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

        # BioViL-T's API takes an image file path; save the loaded RGB image to
        # a temp PNG so the same pixels feed BioViL-T and MedSAM.
        tmp_path = Path(OUTPUT_DIR) / f"_tmp_{image_id}.png"
        try:
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            pil_img.save(tmp_path)
        except Exception:
            log.exception("image_id=%s: failed to write temp image, skipping.", image_id)
            failed_count += 1
            return

        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)

            pred_box = None
            peak_score = None
            mask = None
            iou = None

            try:
                sim_map = image_text_inference.get_similarity_map_from_raw_data(
                    image_path=tmp_path,
                    query_text=region,
                    interpolation="bilinear",
                )
                pred_box, peak_score = heatmap_to_box(sim_map, percentile=PERCENTILE)

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
                    "pred_box": pred_box, "peak_score": peak_score
                }
                per_image_masks[region] = mask
                all_results.append({
                    "image_id": image_id, "region": region,
                    "iou": iou, "peak_score": peak_score, "detected": detected,
                })

                ps = f"{peak_score:.3f}" if peak_score is not None else "N/A"
                iou_s = f"{iou:.3f}" if iou is not None else "N/A"
                print(f"  {region}: IoU={iou_s}  peak_sim={ps}")

                try:
                    save_region_png(pil_img, region, pred_box, gold_box, mask,
                                    iou, peak_score, image_id, out_dir=OUTPUT_DIR)
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

        # Clean up temp file
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            log.exception("image_id=%s: failed to remove temp image", image_id)

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
                mean_sim=("peak_score", "mean"),
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

    print(f"\nCompleted: {images_processed} images, {regions_done} regions, "
          f"{failed_count} failures")

if __name__ == "__main__":
    main()
