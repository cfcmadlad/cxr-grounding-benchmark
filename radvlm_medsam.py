"""
radvlm_medsam.py
----------------
Self-contained script: RadVLM + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements (install in a dedicated conda env):
    pip install torch torchvision
    pip install transformers==4.46.0
    pip install accelerate pydicom Pillow numpy scipy matplotlib pandas

MedSAM is loaded from HuggingFace (wanglab/medsam-vit-base) — no manual
checkpoint download needed.

RadVLM weights must be downloaded from PhysioNet:
    https://physionet.org/content/radvlm-model/1.0.0/
Set RADVLM_PATH below to the folder containing those weights.
"""

# ── CONFIG (edit these before running) ────────────────────────────────────────
RADVLM_PATH   = "/path/to/radvlm/weights"   # local folder with RadVLM weights
GOLD_CSV      = "/path/to/gold_bbox.csv"     # Chest ImaGenome gold annotations
IMAGE_DIR     = "/path/to/mimic_cxr_images"  # root folder with DICOM or JPG/PNG
OUTPUT_DIR    = "./outputs/radvlm"           # where results are saved
MAX_IMAGES    = None                         # set to e.g. 10 to do a quick test run
# ──────────────────────────────────────────────────────────────────────────────

import os
import re
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
# Adjust if the CSV uses different names.
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

def parse_box_from_response(response, img_w, img_h):
    """
    RadVLM outputs boxes as normalized [0,1] coordinates in its text response.
    This function tries multiple patterns to be robust to minor format variations.
    Returns [x1, y1, x2, y2] in pixel coordinates, or None if parsing fails.
    """
    # Try bracket format: [0.1, 0.2, 0.8, 0.9] or [0.1 0.2 0.8 0.9]
    patterns = [
        r"\[([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)\]",
        r"\(([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)\)",
        r"([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            vals = [float(match.group(i)) for i in range(1, 5)]
            # If values look normalized (all <= 1.0), scale to pixels
            if all(v <= 1.0 for v in vals):
                x1 = int(vals[0] * img_w)
                y1 = int(vals[1] * img_h)
                x2 = int(vals[2] * img_w)
                y2 = int(vals[3] * img_h)
            else:
                x1, y1, x2, y2 = [int(v) for v in vals]
            # Sanity check
            if x2 > x1 and y2 > y1:
                return [x1, y1, x2, y2]
    return None

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

# ── IoU / DICE ────────────────────────────────────────────────────────────────

def box_iou(pred, gold):
    """Both boxes are [x1, y1, x2, y2] in pixels."""
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

def mask_dice(pred_mask, gold_mask):
    intersection = (pred_mask & gold_mask).sum()
    if intersection == 0:
        return 0.0
    return 2 * intersection / (pred_mask.sum() + gold_mask.sum())

# ── IMAGE LOADING (DICOM + JPG/PNG) ──────────────────────────────────────────

def load_image(path):
    """Returns a PIL RGB image regardless of whether input is DICOM or JPG/PNG."""
    path = str(path)
    if path.lower().endswith((".dcm", ".dicom")):
        try:
            import pydicom
        except ImportError:
            raise ImportError("pydicom is required for DICOM files: pip install pydicom")
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(float)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    else:
        return Image.open(path).convert("RGB")

def find_image_file(image_dir, image_id):
    """Find an image file by ID, checking common extensions."""
    image_dir = Path(image_dir)
    for ext in [".jpg", ".jpeg", ".png", ".dcm", ".dicom"]:
        # flat
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
        # one level deep (MIMIC-CXR has p10/p10xxxxxx/sXXXX/image.jpg structure)
        matches = list(image_dir.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None

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
        gx1, gy1, gx2, gy2 = gold_box
        axes[0].add_patch(mpatches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor="lime", facecolor="none", label="Gold"))
    if pred_box:
        px1, py1, px2, py2 = pred_box
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
        gx1, gy1, gx2, gy2 = gold_box
        axes[1].add_patch(mpatches.Rectangle(
            (gx1, gy1), gx2 - gx1, gy2 - gy1,
            linewidth=2, edgecolor="lime", facecolor="none"))
    axes[1].axis("off")

    # Panel 3: stats
    axes[2].axis("off")
    stats_lines = [
        f"Model:   RadVLM + MedSAM",
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

# ── GOLD ANNOTATION LOADER ────────────────────────────────────────────────────

def load_gold_annotations(csv_path):
    """
    Loads Chest ImaGenome gold CSV.
    Expected columns: image_id, bbox_name, x, y, w, h
      where x,y = top-left corner, w,h = width/height in pixels.
    Returns dict: {image_id: {bbox_name: [x1, y1, x2, y2]}}
    """
    df = pd.read_csv(csv_path)
    required = {"image_id", "bbox_name", "x", "y", "w", "h"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Gold CSV is missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            "Please update REGION_TO_BBOX_NAME and the column names in load_gold_annotations()."
        )
    gold = {}
    for _, row in df.iterrows():
        iid = str(row["image_id"])
        name = str(row["bbox_name"]).lower().strip()
        box = [int(row["x"]), int(row["y"]),
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

        # Find image file
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

        # RadVLM: one call per region (fresh chat each time)
        per_image_boxes = {}
        per_image_masks = {}

        for region in REGIONS:
            prompt = (
                f"Locate the {region} in this chest X-ray. "
                f"Provide the bounding box as [x1, y1, x2, y2] "
                f"with coordinates normalized between 0 and 1."
            )
            try:
                response, _ = inference_radvlm(
                    radvlm_model, radvlm_processor, pil_img, prompt
                )
                pred_box = parse_box_from_response(response, W, H)
                if pred_box is None:
                    print(f"  {region}: could not parse box from response: {response[:80]}")
            except Exception as e:
                print(f"  {region}: inference error: {e}")
                pred_box = None
                response = ""

            per_image_boxes[region] = {
                "pred_box": pred_box,
                "raw_response": response,
            }

            # MedSAM segmentation
            mask = None
            if pred_box is not None:
                try:
                    mask = get_medsam_mask(pil_img, pred_box)
                except Exception as e:
                    print(f"  {region}: MedSAM error: {e}")
            per_image_masks[region] = mask

            # Gold box for this region
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)

            # IoU
            iou = None
            if pred_box and gold_box:
                iou = box_iou(pred_box, gold_box)

            print(f"  {region}: pred={pred_box}  gold={gold_box}  IoU={iou:.3f if iou is not None else 'N/A'}")

            # Save per-region PNG
            save_region_png(
                pil_img, region, pred_box, gold_box, mask, iou, image_id,
                out_dir=OUTPUT_DIR
            )

            all_results.append({
                "image_id": image_id,
                "region": region,
                "pred_box": pred_box,
                "gold_box": gold_box,
                "iou": iou,
                "detected": pred_box is not None,
            })

        # Save per-image JSON (boxes + scores)
        json_out = Path(OUTPUT_DIR) / image_id / "boxes.json"
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w") as f:
            json.dump(per_image_boxes, f, indent=2)

        # Save per-image masks NPZ
        npz_out = Path(OUTPUT_DIR) / image_id / "masks.npz"
        masks_to_save = {
            r.replace(" ", "_"): m
            for r, m in per_image_masks.items()
            if m is not None
        }
        if masks_to_save:
            np.savez_compressed(str(npz_out), **masks_to_save)

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    results_df = pd.DataFrame(all_results)
    summary_path = Path(OUTPUT_DIR) / "results_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Full results saved to {summary_path}")

    if not results_df.empty:
        region_summary = (
            results_df.groupby("region")
            .agg(
                detected=("detected", "sum"),
                total=("image_id", "count"),
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

if __name__ == "__main__":
    main()
