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
  Paste your HF token when prompted. This only needs to be done once.

Notes on MAIRA-2's phrase grounding:
  - Designed to ground pathology findings (e.g. "pleural effusion"), not
    normal anatomy. Anatomical grounding is not its primary use case, so
    mAP will likely be lower than RadVLM's 85.3% on this task.
  - Box coordinates are output relative to MAIRA-2's internally cropped
    image (518x518). processor.adjust_box_for_original_image_size() is
    required to convert back to original image pixel coordinates.
  - transformers>=4.48.0,<4.52 is required (tested up to 4.51.3 per model card).
    This conflicts with RadVLM (==4.46.0) and BioViL-T (<4.40.0) — use a
    separate conda env (maira2).
"""

import os

# ── CONFIG (edit these, or override via environment variables of the same name)
GOLD_CSV        = os.environ.get("GOLD_CSV", "/path/to/gold_bbox.csv")
IMAGE_DIR       = os.environ.get("IMAGE_DIR", "/path/to/mimic_cxr_images")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "./outputs/maira2")
_max_images_env = os.environ.get("MAX_IMAGES")
MAX_IMAGES      = int(_max_images_env) if _max_images_env else None
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
from transformers import AutoModelForCausalLM, AutoProcessor
from transformers import SamModel, SamProcessor

from cxr_common import load_image, find_image_file, box_iou, union_box

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
    Returns pixel-coordinate box [x1, y1, x2, y2] or None if not detected.

    Box coordinates from MAIRA-2 are normalized and relative to the
    internally cropped image. processor.adjust_box_for_original_image_size
    converts them back to original image pixel coordinates.

    Edge case: for phrases describing paired/bilateral or repeated
    structures, MAIRA-2 can legitimately return MORE THAN ONE box for a
    single phrase (e.g. "left hilar structures" occasionally grounds as
    two separate components). Since the gold annotation is always a
    single box per region, we take the union (smallest enclosing box) of
    every returned box rather than silently keeping only boxes[0] and
    discarding the rest.
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

    # prediction is a list of (text, boxes_or_None) tuples
    # For phrase grounding it's typically one tuple: ('phrase text.', [(x1,y1,x2,y2), ...])
    if not prediction:
        return None, decoded_text

    _, boxes = prediction[0]
    if not boxes:
        return None, decoded_text

    # Adjust every returned box (normalized coords relative to MAIRA-2's
    # cropped view) back to original-image pixel coordinates, then take
    # the union if there's more than one.
    adjusted_boxes = []
    for raw_box in boxes:
        x1, y1, x2, y2 = maira_processor.adjust_box_for_original_image_size(
            box=raw_box,
            original_image=pil_image,
        )
        adjusted_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    final_box = adjusted_boxes[0] if len(adjusted_boxes) == 1 else union_box(adjusted_boxes)
    return final_box, decoded_text

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

        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            pred_box = None
            mask     = None
            raw_resp = ""

            try:
                pred_box, raw_resp = maira_ground_phrase(pil_img, region)
            except Exception as e:
                print(f"  {region}: MAIRA-2 error: {e}")

            per_image_boxes[region] = {
                "pred_box":     pred_box,
                "raw_response": raw_resp,
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

            print(f"  {region}: IoU={f'{iou:.3f}' if iou is not None else 'N/A'}  pred={pred_box}")

            save_region_png(
                pil_img, region, pred_box, gold_box, mask,
                iou, image_id, out_dir=OUTPUT_DIR
            )

            all_results.append({
                "image_id": image_id,
                "region":   region,
                "pred_box": pred_box,
                "gold_box": gold_box,
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
                detected = ("detected", "sum"),
                total    = ("image_id", "count"),
                mean_iou = ("iou",      "mean"),
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

if __name__ == "__main__":
    main()
