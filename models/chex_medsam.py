"""
chex_medsam.py
--------------
Self-contained thin wrapper: ChEX (native DETR-style box regression) for
anatomical region grounding on the Chest ImaGenome gold subset.

ChEX is architecturally different from every other model in this benchmark:
it is NOT a text-generation VLM that emits coordinates as text. It has a
dedicated prompt detector (DETR-style) that takes a CLIP-encoded text prompt
per anatomical region and directly regresses a bounding box for it. There is
no MedSAM step needed (unlike the phrase-grounding VLMs) -- ChEX outputs a box
directly, so "medsam" in the filename is kept only for naming consistency with
the other per-model scripts in this repo; no MedSAM model is loaded or run.

Must be run from inside the cloned ChEX repo's ``src/`` directory (its own
modules use non-package-relative imports, e.g. ``from dataset.image_transform
import ...``, exactly like the upstream ``src/evaluate.py``). The Slurm job
copies this script there and cd's into it before running, the same convention
used for biomedparse_seg.py inside the BiomedParse repo.

>>> CRITICAL COORDINATE CONVENTIONS (verified against ChEX source, not assumed) <<<

1. Boxes come out of ``ChEX.detect_prompts()`` as ``(cx, cy, w, h)``, each
   normalized to [0,1], via the sigmoid-activated DETR box head
   (model/detector/token_decoder_detector.py: predict_boxes()). This is
   NOT [x1,y1,x2,y2] and NOT pixel coordinates -- convert both aspects
   explicitly.

2. Those normalized coordinates are relative to the 224x224 SQUARE that the
   image was center-cropped-then-resized to at inference time (val_mode:
   'rect_center' -> RectangularizeCenter in dataset/image_transform.py: crop
   to a square of side ``min(H,W)`` centered on the image, THEN resize to
   224x224). They are NOT relative to the original (possibly rectangular)
   image. A box must be de-normalized against the 224 square, then have the
   crop offset added back, to land in original-image pixel space. Skipping
   the crop-offset step silently shifts every box on a non-square X-ray.

3. ChEX's detector is a *multi-region* detector even for single-anatomy
   queries (model/detector/token_decoder_detector.py: handle_multiregions()).
   The final ``output.boxes`` tensor is explicitly zeroed
   (``output.boxes_present[..., None] * boxes``) for any query the model's
   own presence head (sigmoid > 0.5) does not think is present. A box of
   literally [0,0,0,0] therefore does NOT mean "box at the top-left corner"
   -- it means "not detected". ``boxes_present`` MUST be checked per region
   BEFORE trusting a box, or every non-present region silently produces a
   plausible-looking (but fake) degenerate box.

4. Image preprocessing (dataset/datasets.py + conf/transform.yaml, verified
   against source, not the README): grayscale (mode 'L'), array / 255.0,
   center-crop-then-resize to 224x224 (RectangularizeCenter), then
   A.Normalize(mean=[0.505], std=[0.248], max_pixel_value=1.0) -- i.e.
   normalization is applied to the already-[0,1]-scaled array, not to
   0-255 values. The final tensor fed to the model has shape (N, 224, 224)
   (no explicit channel dim -- model/img_encoder/chexzero_img_encoder.py
   repeats grayscale -> 3-channel internally via
   ``einops.repeat(x, 'n h w -> n c h w', c=3)`` when x.ndim == 3).

5. Text prompts: ChEX was trained directly on Chest ImaGenome anatomy names
   (conf/dataset/anatomy_names/cig_default.yaml) with simple title-case
   prompt templates (conf/prompts/cig_anat.yaml), e.g. "right lung" ->
   "Right lung". All 15 of this benchmark's REGIONS map 1:1 onto ChEX's
   training vocabulary EXCEPT the two hilar regions, where ChEX's anatomy
   name is "right/left hilar structures" (not "... region") -- mapped
   explicitly in REGION_TO_CHEX_ANAT_NAME below, verified against
   conf/dataset/anatomy_names/cig_default.yaml on the cloned repo.

All of the above was read directly out of the ChEX source on this cluster
(not the paper, not the README) -- see the file/line references in each
comment. Verify by hand on real images before trusting any IoU number.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG + shared utils (must precede heavy imports) ────────────────────────
try:
    from cxr_common import (
        load_config, setup_hf_home, load_image, find_image_path,
        load_gold_annotations, validate_box, clamp_box, compute_iou,
        result_row, write_result_row, init_results_csv, BOX_FIELDS, get_logger,
    )
except ImportError:
    _root = os.environ.get("CXR_REPO_ROOT")
    if _root and _root not in sys.path:
        sys.path.insert(0, _root)
    from cxr_common import (
        load_config, setup_hf_home, load_image, find_image_path,
        load_gold_annotations, validate_box, clamp_box, compute_iou,
        result_row, write_result_row, init_results_csv, BOX_FIELDS, get_logger,
    )

log = get_logger("chex")
cfg = load_config(required_keys=["GOLD_CSV", "IMAGE_DIR", "OUTPUT_DIR"])

GOLD_CSV   = cfg["GOLD_CSV"]
IMAGE_DIR  = cfg["IMAGE_DIR"]
OUTPUT_DIR = os.path.join(cfg["OUTPUT_DIR"], "chex")
MAX_IMAGES = cfg.get("MAX_IMAGES")
CHEX_MODEL_NAME = os.environ.get("CHEX_MODEL_NAME", "chex_stage3")
PRINT_RAW_FIRST_N = 2   # print raw box tensors for the first N images for manual verification

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

import torch

# ── ChEX imports (only resolve when this script's own directory -- the ChEX
#    repo's src/ -- is on sys.path, exactly like upstream src/evaluate.py) ────
from util.model_utils import load_model_by_name, ModelRegistry
import model as chex_model_pkg
from model import img_encoder, txt_encoder, txt_decoder, detector
ModelRegistry.init_registries([chex_model_pkg, img_encoder, txt_encoder, txt_decoder, detector])

# ── 15 TARGET REGIONS (identical list/order to every other model script) ─────
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

# ChEX's OWN anatomy-name vocabulary (conf/dataset/anatomy_names/cig_default.yaml).
# Only the two hilar regions differ in wording from this benchmark's REGIONS.
REGION_TO_CHEX_ANAT_NAME = {
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

# Prompt templates, copied verbatim from conf/prompts/cig_anat.yaml (title case,
# no synonyms needed -- ChEX's encode_prompts() takes the first/only entry).
CHEX_ANAT_PROMPTS = {
    "right lung": ["Right lung"],
    "left lung": ["Left lung"],
    "cardiac silhouette": ["Cardiac silhouette"],
    "mediastinum": ["Mediastinum"],
    "trachea": ["Trachea"],
    "right upper lung zone": ["Right upper lung zone"],
    "right mid lung zone": ["Right mid lung zone"],
    "right lower lung zone": ["Right lower lung zone"],
    "left upper lung zone": ["Left upper lung zone"],
    "left mid lung zone": ["Left mid lung zone"],
    "left lower lung zone": ["Left lower lung zone"],
    "right hilar structures": ["Right hilar structures"],
    "left hilar structures": ["Left hilar structures"],
    "right costophrenic angle": ["Right costophrenic angle"],
    "left costophrenic angle": ["Left costophrenic angle"],
}

# Image preprocessing constants (conf/transform.yaml + conf/dataset/cig_boxclssent.yaml).
CHEX_IMG_SIZE = 224
PIXEL_MEAN = 0.505
PIXEL_STD = 0.248

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── LOAD CHEX ─────────────────────────────────────────────────────────────────

print(f"Loading ChEX model '{CHEX_MODEL_NAME}'...")
# The released checkpoint (chex_stage3/run_0/checkpoints/checkpoint_000568440.pth)
# is not named checkpoint_best.pth -- load_best=True would raise FileNotFoundError.
# load_best=False + step=-1 (default) picks the single available .pth by mtime,
# verified against the actual extracted checkpoint directory on this cluster.
chex_model, ckpt_dict = load_model_by_name(CHEX_MODEL_NAME, load_best=False, return_dict=True)
chex_model = chex_model.to(DEVICE).eval()
print(f"ChEX loaded (checkpoint step={ckpt_dict.get('step')}).")

ANAT_NAMES = [REGION_TO_CHEX_ANAT_NAME[r] for r in REGIONS]
with torch.no_grad():
    ANATOMY_TOKEN_EMB, _ = chex_model.encode_prompts(CHEX_ANAT_PROMPTS, ANAT_NAMES)
ANATOMY_TOKEN_EMB = ANATOMY_TOKEN_EMB.to(DEVICE)
print(f"Encoded {len(ANAT_NAMES)} anatomy prompts -> embedding shape {tuple(ANATOMY_TOKEN_EMB.shape)}")

# ── IMAGE PREPROCESSING (must exactly match dataset/image_transform.py val path) ──

def preprocess_image(pil_image):
    """
    Grayscale -> [0,1] float -> center-crop to square -> resize 224x224 ->
    normalize(mean=0.505, std=0.248). Returns (tensor[1,224,224], meta) where
    meta carries the info needed to map normalized boxes back to ORIGINAL
    image pixel coordinates (see cxcywh_norm_to_pixel_xyxy below).
    """
    gray = pil_image.convert("L")
    arr = np.array(gray, dtype=np.float32) / 255.0
    H, W = arr.shape
    crop_size = min(H, W)
    y0 = (H - crop_size) // 2
    x0 = (W - crop_size) // 2
    cropped = arr[y0:y0 + crop_size, x0:x0 + crop_size]
    resized = cv2.resize(cropped, (CHEX_IMG_SIZE, CHEX_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normed = (resized - PIXEL_MEAN) / PIXEL_STD
    tensor = torch.from_numpy(normed).float()  # (224, 224)
    meta = {"orig_w": W, "orig_h": H, "crop_size": crop_size, "x0": x0, "y0": y0}
    return tensor, meta


def cxcywh_norm_to_pixel_xyxy(box_cxcywh, meta):
    """Map a normalized (cx,cy,w,h) box (relative to the center-cropped square)
    back to [x1,y1,x2,y2] pixel coordinates in the ORIGINAL image."""
    cx, cy, w, h = [float(v) for v in box_cxcywh]
    cs = meta["crop_size"]
    x1c = (cx - w / 2.0) * cs
    y1c = (cy - h / 2.0) * cs
    x2c = (cx + w / 2.0) * cs
    y2c = (cy + h / 2.0) * cs
    return [x1c + meta["x0"], y1c + meta["y0"], x2c + meta["x0"], y2c + meta["y0"]]


def chex_ground_regions(pil_image, print_raw=False):
    """
    Run ChEX's prompt detector for all 15 regions on one image in a single
    batched forward pass. Returns dict region -> (pred_box_xyxy_pixel or None).
    """
    x_tensor, meta = preprocess_image(pil_image)
    x_batch = x_tensor.unsqueeze(0).to(DEVICE)  # (1, 224, 224) == (N=1, H, W)

    with torch.no_grad():
        detected = chex_model.detect_prompts(x_batch, ANATOMY_TOKEN_EMB, skip_roi_pool=True)

    boxes_cxcywh = detected.boxes[0].detach().cpu().numpy()          # (A, 4)
    boxes_present = detected.boxes_present
    if boxes_present is not None:
        boxes_present = boxes_present[0].detach().cpu().numpy().astype(bool)  # (A,)
    else:
        boxes_present = np.ones(len(REGIONS), dtype=bool)

    if print_raw:
        print(f"    RAW boxes_cxcywh (normalized, crop-square space): {boxes_cxcywh}")
        print(f"    RAW boxes_present: {boxes_present}")
        print(f"    RAW crop meta: {meta}")

    out = {}
    for i, region in enumerate(REGIONS):
        if not boxes_present[i]:
            out[region] = None
            continue
        pred_box = cxcywh_norm_to_pixel_xyxy(boxes_cxcywh[i], meta)
        out[region] = pred_box
    return out

# ── PER-REGION VISUALIZATION ──────────────────────────────────────────────────

def save_region_png(pil_image, region, pred_box, gold_box, iou, image_id, out_dir):
    img_np = np.array(pil_image)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"ChEX  |  {image_id}  |  {region}", fontsize=11, y=1.01)

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

    axes[1].axis("off")
    stats_lines = [
        "Model:    ChEX (native DETR-style box regression)",
        f"Image:    {image_id}",
        f"Region:   {region}",
        "",
        f"Box IoU:  {iou:.3f}" if iou is not None else "Box IoU:  N/A",
        "",
        f"Pred box: {pred_box}" if pred_box else "Pred box: not present (presence head < 0.5)",
        f"Gold box: {gold_box}" if gold_box else "Gold box: not in annotations",
    ]
    axes[1].text(
        0.05, 0.95, "\n".join(stats_lines),
        transform=axes[1].transAxes,
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

        try:
            region_boxes = chex_ground_regions(pil_img, print_raw=(img_idx < PRINT_RAW_FIRST_N))
        except Exception:
            failed_count += 1
            log.exception("image_id=%s: ChEX inference failed for all regions", image_id)
            for region in REGIONS:
                bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
                gold_box = gold_annotations[image_id].get(bbox_name)
                write_result_row(
                    summary_path,
                    result_row(image_id, region, None, False, gold_box, None),
                    BOX_FIELDS,
                )
            return

        for region in REGIONS:
            bbox_name = REGION_TO_BBOX_NAME.get(region, region).lower()
            gold_box = gold_annotations[image_id].get(bbox_name)
            pred_box = region_boxes.get(region)
            iou = None

            try:
                if pred_box is not None:
                    pred_box = clamp_box(pred_box, W, H)
                    if not validate_box(pred_box, W, H):
                        pred_box = None

                if pred_box is not None:
                    iou = compute_iou(pred_box, gold_box)

                detected = pred_box is not None
                write_result_row(
                    summary_path,
                    result_row(image_id, region, iou, detected, gold_box, pred_box),
                    BOX_FIELDS,
                )
                regions_done += 1
                per_image_boxes[region] = {"pred_box": pred_box}
                all_results.append({
                    "image_id": image_id, "region": region,
                    "iou": iou, "detected": detected,
                })

                print(f"  {region}: IoU={f'{iou:.3f}' if iou is not None else 'N/A'}"
                      f"  pred={pred_box}")

                try:
                    save_region_png(pil_img, region, pred_box, gold_box, iou,
                                    image_id, out_dir=OUTPUT_DIR)
                except Exception:
                    log.exception("image_id=%s region=%s: visualization failed",
                                  image_id, region)

            except Exception:
                failed_count += 1
                log.exception("image_id=%s region=%s: post-processing failed",
                              image_id, region)
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
        except Exception:
            log.exception("image_id=%s: failed to write per-image JSON", image_id)

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
