"""
chexagent_medsam.py
-------------------
Self-contained script: CheXagent-8b + MedSAM for anatomical region grounding
on the Chest ImaGenome gold subset.

Requirements: see requirements_chexagent.txt

Model: StanfordAIMI/CheXagent-8b (open weights, no gated access needed)
Paper: CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation
       https://arxiv.org/abs/2401.12208

Notes on CheXagent phrase grounding:
  - CheXagent was evaluated on phrase grounding with mIoU 0.627 / mAP 0.810
    on MS-CXR, but those results are for pathology phrases, not anatomy names.
    Expect lower scores on normal anatomical region grounding (same caveat as
    MAIRA-2).
  - Box output format is parsed from free-text response using the shared
    parse_box_from_response() in cxr_common.py. CheXagent's grounding output
    uses "<|box|> (x1,y1),(x2,y2) <|/box|>" tags with coordinates on a 0-100
    scale (confirmed against the RadVLM authors' own evaluation harness,
    which parses CheXagent output with this exact tag pattern before
    dividing by 100) -- this is checked first, with bracket/paren/bare
    4-number formats as a fallback for other checkpoint variants.
    PRINT_RAW_RESPONSES=True always prints the raw response for the first
    image so you can verify the format your specific checkpoint produces.
  - Inference format verified from the official HF model card:
    processor(images=images, text=" USER: <s>{prompt} ASSISTANT: <s>")
"""

import os

# ── CONFIG (edit these, or override via environment variables of the same name)
GOLD_CSV        = os.environ.get("GOLD_CSV", "/path/to/gold_bbox.csv")
IMAGE_DIR       = os.environ.get("IMAGE_DIR", "/path/to/mimic_cxr_images")
OUTPUT_DIR      = os.environ.get("OUTPUT_DIR", "./outputs/chexagent")
_max_images_env = os.environ.get("MAX_IMAGES")
MAX_IMAGES      = int(_max_images_env) if _max_images_env else None
PRINT_RAW_RESPONSES = os.environ.get("PRINT_RAW_RESPONSES", "1") not in ("0", "false", "False")
# ──────────────────────────────────────────────────────────────────────────────

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
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from transformers import SamModel, SamProcessor

from cxr_common import load_image, find_image_file, box_iou, parse_box_from_response

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

print("Loading CheXagent-8b...")
processor = AutoProcessor.from_pretrained(
    "StanfordAIMI/CheXagent-8b",
    trust_remote_code=True,
)
generation_config = GenerationConfig.from_pretrained(
    "StanfordAIMI/CheXagent-8b",
)
chexagent_model = AutoModelForCausalLM.from_pretrained(
    "StanfordAIMI/CheXagent-8b",
    torch_dtype=torch.float16,
    trust_remote_code=True,
).to(DEVICE)
chexagent_model.eval()
print("CheXagent-8b loaded.")

print("Loading MedSAM...")
medsam_processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
medsam_model = SamModel.from_pretrained("wanglab/medsam-vit-base").to(DEVICE)
medsam_model.eval()
print("MedSAM loaded.")

# ── CHEXAGENT PHRASE GROUNDING ────────────────────────────────────────────────

def chexagent_ground_phrase(pil_image, phrase, print_raw=False):
    """
    Run CheXagent phrase grounding for a single phrase on a single image.
    Returns pixel-coordinate box [x1, y1, x2, y2] or None if not detected.

    Inference format from official HF model card:
      processor(images=images, text=" USER: <s>{prompt} ASSISTANT: <s>")
    """
    images = [pil_image]
    prompt = (
        f'Perform phrase grounding for "{phrase}" in this chest X-ray. '
        f'Provide the bounding box as [x1, y1, x2, y2] in pixel coordinates.'
    )
    inputs = processor(
        images=images,
        text=f" USER: <s>{prompt} ASSISTANT: <s>",
        return_tensors="pt",
    ).to(device=DEVICE, dtype=torch.float16)

    prompt_length = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output = chexagent_model.generate(
            **inputs,
            generation_config=generation_config,
        )[0]

    # AutoModelForCausalLM.generate() returns the full prompt+completion
    # sequence, not just the newly generated tokens. Decoding `output` in
    # full (as opposed to `output[prompt_length:]`) means `response` is
    # dominated by the ECHOED PROMPT TEXT -- including the literal
    # instruction "Provide the bounding box as [x1, y1, x2, y2]" -- so the
    # first ~150+ characters of both the debug print and the raw_response
    # saved to JSON were the prompt being read back, not CheXagent's
    # actual answer. Slicing off the prompt tokens before decoding fixes
    # this (same pattern already used correctly in maira2_medsam.py).
    response = processor.tokenizer.decode(
        output[prompt_length:], skip_special_tokens=True
    )

    if print_raw:
        print(f"    RAW RESPONSE for '{phrase}': {response[:200]}")

    W, H = pil_image.size
    box = parse_box_from_response(response, W, H)
    return box, response

# parse_box_from_response() is imported from cxr_common. CheXagent-8b's
# real grounding output uses "<|box|> (x1,y1),(x2,y2) <|/box|>" tags with
# coordinates on a 0-100 scale, which the shared parser checks first,
# falling back to bracket/paren/bare-number formats for robustness.
# PRINT_RAW_RESPONSES=True on the first image lets you confirm the actual
# format your checkpoint produces and add a pattern to cxr_common.py if
# it differs.

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
        f"CheXagent + MedSAM  |  {image_id}  |  {region}",
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
        "Model:    CheXagent-8b + MedSAM",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Box IoU:  {iou:.3f}" if iou is not None else "Box IoU:  N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not detected",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
        "",
        "Note: CheXagent grounding is optimised",
        "for pathology, not normal anatomy.",
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
    first_image = True

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

            # Print raw responses for the first image only (for format inspection)
            print_raw = PRINT_RAW_RESPONSES and first_image

            try:
                pred_box, raw_resp = chexagent_ground_phrase(
                    pil_img, region, print_raw=print_raw
                )
            except Exception as e:
                print(f"  {region}: CheXagent error: {e}")

            per_image_boxes[region] = {
                "pred_box":     pred_box,
                "raw_response": raw_resp[:200],  # truncated for JSON
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

        first_image = False

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
