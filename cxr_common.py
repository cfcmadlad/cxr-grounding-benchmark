"""
cxr_common.py
-------------
Shared, dependency-light utilities used by every model script in this
benchmark: a unified DICOM/PNG/JPG image loader and a unified free-text
bounding-box parser.

Only depends on: numpy, Pillow, pydicom (optional, only for .dcm/.dicom).
No torch/transformers imports here so this file can be copied as-is into
the BiomedParse repo checkout (Python 3.9.19 env) without pulling in any
version-conflicting dependency.
"""

import re
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

# Some MIMIC-CXR JPEGs are known to be slightly truncated. Without this,
# Pillow raises `OSError: image file is truncated` and the whole image
# (and all 15 regions) gets skipped for no good reason.
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── IMAGE LOADING ─────────────────────────────────────────────────────────────

def _dicom_to_pil(path):
    """Load a DICOM file and return a normalized 8-bit RGB PIL image."""
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    try:
        ds = pydicom.dcmread(str(path))
    except Exception:
        # Some DICOMs in the wild are missing the preamble/File Meta group
        # (common with anonymized/re-exported MIMIC-CXR-style files).
        # force=True lets pydicom read the pixel data anyway.
        ds = pydicom.dcmread(str(path), force=True)

    arr = ds.pixel_array

    # Multi-frame DICOM: keep only the first frame.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]

    arr = arr.astype(np.float64)

    # Apply RescaleSlope/RescaleIntercept if present (raw pixel -> real units).
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    # Apply VOI LUT / windowing when available so contrast matches what a
    # radiologist would see, instead of a raw linear stretch of full bit depth.
    try:
        arr = apply_voi_lut(arr.astype(np.int16) if arr.max() < 32768 and arr.min() > -32768 else arr, ds)
        arr = arr.astype(np.float64)
    except Exception:
        pass  # fall back to plain min-max normalization below

    # MONOCHROME1 means "0 = white" (inverted grayscale) -- flip so higher
    # pixel value = brighter, matching MONOCHROME2 and standard PNG/JPG.
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr_min, arr_max = arr.min(), arr.max()
    arr = (arr - arr_min) / (arr_max - arr_min + 1e-8) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L").convert("RGB")
    return Image.fromarray(arr).convert("RGB")


def load_image(path):
    """
    Load a chest X-ray image from disk and return a 3-channel RGB PIL.Image,
    regardless of whether the input is DICOM (.dcm/.dicom) or a standard
    raster format (.png/.jpg/.jpeg/.bmp/.tif).

    Every model in this benchmark (Grounding DINO, MedSAM, BioViL-T,
    CheXagent, MAIRA-2, RadVLM, BiomedParse) expects 3-channel RGB input,
    so this is the single normalization point for all of them.
    """
    path = str(path)
    suffix = Path(path).suffix.lower()

    if suffix in (".dcm", ".dicom"):
        try:
            import pydicom  # noqa: F401
        except ImportError:
            raise ImportError(
                "pydicom is required to read DICOM files: pip install pydicom"
            )
        try:
            return _dicom_to_pil(path)
        except Exception as e:
            raise IOError(f"Failed to decode DICOM file {path}: {e}") from e

    try:
        img = Image.open(path)
        img.load()  # force full decode now so corrupt files fail here,
        # with a clear message, instead of later during model inference.
    except Exception as e:
        raise IOError(f"Failed to decode image file {path}: {e}") from e

    # Normalize every possible PIL mode (grayscale "L"/"I"/"I;16", palette
    # "P", RGBA, CMYK, ...) down to plain 3-channel RGB uint8.
    if img.mode in ("I", "I;16", "I;16B", "I;16L"):
        arr = np.array(img).astype(np.float64)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    elif img.mode == "P":
        img = img.convert("RGBA")

    return img.convert("RGB")


def find_image_file(image_dir, image_id):
    """
    Locate an image file for `image_id` under `image_dir`, checking common
    extensions both flat and nested (MIMIC-CXR uses a
    p10/p10xxxxxx/sXXXXXXXX/<dicom_id>.jpg-style nested layout).
    """
    image_dir = Path(image_dir)
    for ext in (".jpg", ".jpeg", ".png", ".dcm", ".dicom", ".bmp", ".tif", ".tiff"):
        p = image_dir / f"{image_id}{ext}"
        if p.exists():
            return p
        matches = list(image_dir.rglob(f"{image_id}{ext}"))
        if matches:
            return matches[0]
    return None


# ── FREE-TEXT BOUNDING BOX PARSING ────────────────────────────────────────────
#
# Different VLMs emit boxes in different formats:
#   - RadVLM:            "[0.12, 0.34, 0.56, 0.78]"  (normalized 0-1 floats)
#   - CheXagent-8b:       "<|box|> (12,34),(56,78) <|/box|>"  (0-100 scale)
#   - Some checkpoints:   plain "x1, y1, x2, y2" with no brackets at all
#
# parse_box_from_response() tries every known format so a single parser can
# be shared by every VLM script instead of duplicating (and subtly
# diverging) regex logic six times over.

_CHEXAGENT_BOX_TAG_RE = re.compile(
    r"<\|box\|>\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*,\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)"
)

_BRACKETED_PATTERNS = [
    # [x1, y1, x2, y2]
    r"\[\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*\]",
    # (x1, y1, x2, y2)  -- single paren group, four numbers
    r"\(\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*\)",
    # (x1,y1),(x2,y2)  -- paired-parens without the <|box|> tag wrapper
    r"\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)\s*,\s*\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)",
    # bare: x1, y1, x2, y2   (last resort, most permissive)
    r"([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)\s*[,\s]\s*([0-9.]+)",
]


def _finalize_box(vals, img_w, img_h, scale_hint=None):
    """Turn 4 raw numbers into a pixel-space [x1,y1,x2,y2] box, or None."""
    if scale_hint == "pct100":
        x1, y1, x2, y2 = (v / 100.0 for v in vals)
        x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
    elif all(0.0 <= v <= 1.0 for v in vals):
        x1, y1, x2, y2 = vals[0] * img_w, vals[1] * img_h, vals[2] * img_w, vals[3] * img_h
    elif all(0.0 <= v <= 1000.0 for v in vals) and max(vals) > img_w and max(vals) > img_h:
        # Common 0-1000 quantized-coordinate convention used by several
        # instruction-tuned VLMs (e.g. Qwen-VL-style <box>) when the raw
        # values clearly exceed the image's actual pixel dimensions.
        x1, y1, x2, y2 = (v / 1000.0 for v in vals)
        x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
    else:
        x1, y1, x2, y2 = vals

    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 > x1 and y2 > y1:
        return [x1, y1, x2, y2]
    return None


def parse_box_from_response(response, img_w, img_h):
    """
    Extract a single [x1, y1, x2, y2] pixel-space box from a free-text VLM
    response. Tries, in order:
      1. CheXagent-style <|box|> (x1,y1),(x2,y2) <|/box|> tags (0-100 scale)
      2. Bracketed/parenthesized 4-number groups (normalized 0-1 or raw pixels)
      3. Bare "x1, y1, x2, y2" text with no delimiters

    Returns None if no valid box could be parsed (caller should treat this
    as "not detected", not raise).
    """
    if not response:
        return None

    m = _CHEXAGENT_BOX_TAG_RE.search(response)
    if m:
        vals = [float(v) for v in m.groups()]
        box = _finalize_box(vals, img_w, img_h, scale_hint="pct100")
        if box is not None:
            return box

    for pattern in _BRACKETED_PATTERNS:
        match = re.search(pattern, response)
        if match:
            vals = [float(match.group(i)) for i in range(1, 5)]
            box = _finalize_box(vals, img_w, img_h)
            if box is not None:
                return box

    return None


# ── GEOMETRY ──────────────────────────────────────────────────────────────────

def box_iou(pred, gold):
    """IoU between two [x1, y1, x2, y2] boxes."""
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


def union_box(boxes):
    """Smallest box enclosing all boxes in `boxes` (list of [x1,y1,x2,y2])."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [int(x1), int(y1), int(x2), int(y2)]


def mask_dice(pred_mask, gold_mask):
    """Dice coefficient between two boolean masks."""
    pred_bool = pred_mask.astype(bool)
    gold_bool = gold_mask.astype(bool)
    intersection = (pred_bool & gold_bool).sum()
    denom = pred_bool.sum() + gold_bool.sum()
    if denom == 0:
        return 0.0
    return 2 * intersection / denom
