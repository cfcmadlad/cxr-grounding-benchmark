"""
cxr_common.py
-------------
Shared utilities for every model script in the CXR grounding benchmark.

This module is intentionally lightweight: it imports only os / sys / logging /
csv / numpy / PIL / yaml at module scope. ``pydicom`` is imported lazily inside
``load_image`` so that non-DICOM runs do not require it, and neither torch nor
transformers are imported here (that lets a script set ``HF_HOME`` from config
*before* importing the heavy ML libraries).

Everything the six model scripts share lives here so there is exactly one
implementation of each piece of logic — no local reimplementations:

  * ``load_config`` / ``setup_hf_home`` — read paths from config.yaml, fail fast.
  * ``load_image``                       — robust DICOM + PNG/JPG loader -> RGB PIL or None.
  * ``validate_box`` / ``clamp_box``     — sanity-check / clamp boxes before MedSAM.
  * ``compute_iou`` / ``compute_dice``   — the *only* IoU / Dice implementations.
  * ``result_row`` / ``write_result_row`` — incremental results_summary.csv writer.
"""

import os
import sys
import csv
import logging

import numpy as np
from PIL import Image

# PyYAML is imported lazily inside load_config() so that scripts which only use
# load_image / compute_iou do not require it at import time.


# ── LOGGING ───────────────────────────────────────────────────────────────────
# Configure the root logger once, streaming to stdout so messages land in the
# Slurm .out file. basicConfig is a no-op if logging is already configured.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def get_logger(name="cxr"):
    """Return a module logger. All failures in this codebase go through one."""
    return logging.getLogger(name)


_log = get_logger("cxr_common")


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Keys whose value must be an existing path on disk (input data / weights).
_PATH_KEYS_MUST_EXIST = {"GOLD_CSV", "IMAGE_DIR", "RADVLM_PATH", "BIOMEDPARSE_REPO"}
# Keys that are output/cache directories — created if missing rather than required.
_PATH_KEYS_CREATE = {"OUTPUT_DIR", "HF_HOME"}


def _config_path():
    """Resolve config.yaml.

    Order of precedence:
      1. ``CXR_CONFIG`` environment variable (used by the Slurm job scripts so
         that BiomedParse, which runs from inside its own repo dir, still finds
         the benchmark's config.yaml).
      2. config.yaml sitting next to this file (the repo root).
    """
    env = os.environ.get("CXR_CONFIG")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def load_config(required_keys=None):
    """Load config.yaml and validate the keys this script needs.

    ``required_keys`` is the list of keys the calling script depends on. For
    every required key: if it is missing / empty a descriptive error is printed
    and the process exits(1) immediately. Input-path keys (GOLD_CSV, IMAGE_DIR,
    RADVLM_PATH, BIOMEDPARSE_REPO) must additionally point at something that
    exists; output/cache-path keys (OUTPUT_DIR, HF_HOME) are created if absent.
    """
    try:
        import yaml
    except ImportError:
        print("ERROR: PyYAML is required (pip install pyyaml) to read config.yaml",
              file=sys.stderr)
        sys.exit(1)

    cfg_path = _config_path()
    if not os.path.exists(cfg_path):
        print(f"ERROR: config.yaml not found at: {cfg_path}", file=sys.stderr)
        print("       Create it at the repo root (see config.yaml template).",
              file=sys.stderr)
        sys.exit(1)

    try:
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"ERROR: failed to parse config.yaml ({cfg_path}): {e}",
              file=sys.stderr)
        sys.exit(1)

    if not isinstance(cfg, dict):
        print(f"ERROR: config.yaml did not parse to a mapping: {cfg_path}",
              file=sys.stderr)
        sys.exit(1)

    for key in (required_keys or []):
        if key not in cfg or cfg[key] is None or str(cfg[key]).strip() == "":
            print(f"ERROR: required config key '{key}' is missing or empty "
                  f"in {cfg_path}", file=sys.stderr)
            sys.exit(1)
        val = cfg[key]
        if key in _PATH_KEYS_MUST_EXIST:
            if not os.path.exists(str(val)):
                print(f"ERROR: config key '{key}' points to a path that does "
                      f"not exist: {val}", file=sys.stderr)
                sys.exit(1)
        elif key in _PATH_KEYS_CREATE:
            try:
                os.makedirs(str(val), exist_ok=True)
            except Exception as e:
                print(f"ERROR: could not create directory for config key "
                      f"'{key}' ({val}): {e}", file=sys.stderr)
                sys.exit(1)

    return cfg


def setup_hf_home(cfg):
    """Point HuggingFace caches at the configured location.

    Must be called BEFORE importing torch / transformers so the environment
    variables take effect. The Slurm job scripts also export these, so this is
    belt-and-suspenders for interactive runs.
    """
    hf_home = cfg.get("HF_HOME")
    if hf_home:
        os.makedirs(hf_home, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        # TRANSFORMERS_CACHE is deprecated in newer transformers but harmless
        # and still respected by older versions used by some envs here.
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home))
    return hf_home


# ── IMAGE LOADING ─────────────────────────────────────────────────────────────

def _first_value(v):
    """WindowCenter / WindowWidth may be a single DS or a MultiValue list."""
    try:
        return float(v[0])
    except (TypeError, IndexError, ValueError):
        return float(v)


def _window_to_uint8(ds, arr):
    """Map a floating-point (already rescaled) grayscale array to uint8.

    Uses the DICOM window center/width when available (proper window/level),
    otherwise falls back to min-max normalisation. Never does a raw
    ``.astype(uint8)`` on 16-bit data.
    """
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        try:
            center = _first_value(wc)
            width = _first_value(ww)
            if width > 0:
                low = center - width / 2.0
                high = center + width / 2.0
                arr = np.clip(arr, low, high)
                arr = (arr - low) / (high - low) * 255.0
                return arr.astype(np.uint8)
        except Exception:
            pass  # fall through to min-max
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    arr = (arr - amin) / (amax - amin + 1e-8) * 255.0
    return arr.astype(np.uint8)


def _color_to_uint8(arr):
    """Scale a colour (H,W,3/4) array to uint8 without a raw truncating cast."""
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float64)
    amin = float(arr.min())
    amax = float(arr.max())
    return ((arr - amin) / (amax - amin + 1e-8) * 255.0).astype(np.uint8)


def _load_dicom(path):
    """Load a DICOM file to an RGB PIL image, or None on any failure."""
    try:
        import pydicom
    except ImportError:
        _log.error("pydicom is not installed; cannot load DICOM: %s", path)
        return None

    try:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array
    except Exception as e:
        _log.error("Failed to read DICOM %s: %s", path, e)
        return None

    try:
        arr = np.asarray(arr)

        # Multi-frame DICOM -> take frame 0.
        num_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        if num_frames > 1:
            arr = arr[0]
        # A remaining >2D array that is not an (H,W,3/4) colour image is also a
        # frame stack -> take frame 0.
        while arr.ndim > 2 and arr.shape[-1] not in (3, 4):
            arr = arr[0]

        # Colour DICOM (uncommon for CXR but handle it).
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr8 = _color_to_uint8(arr)[..., :3]
            img = Image.fromarray(arr8, mode="RGB")
        else:
            # Grayscale: apply rescale slope/intercept, then window/level.
            arr = arr.astype(np.float64)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            arr = arr * slope + intercept
            arr8 = _window_to_uint8(ds, arr)
            # MONOCHROME1: high pixel value = dark -> invert to display normally.
            photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
            if photometric == "MONOCHROME1":
                arr8 = 255 - arr8
            img = Image.fromarray(arr8, mode="L").convert("RGB")

        if img.size[0] == 0 or img.size[1] == 0:
            _log.error("DICOM decoded to a degenerate 0x0 image: %s", path)
            return None
        return img
    except Exception as e:
        _log.error("Failed to process DICOM pixel data %s: %s", path, e)
        return None


def _load_pil(path, log_failure=True):
    """Load a PNG/JPG to an RGB PIL image, or None on any failure."""
    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        if log_failure:
            _log.error("Failed to open image %s: %s", path, e)
        return None
    try:
        if img.size[0] == 0 or img.size[1] == 0:
            _log.error("Image is degenerate 0x0: %s", path)
            return None
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        if log_failure:
            _log.error("Failed to convert image to RGB %s: %s", path, e)
        return None


def load_image(path):
    """Load an image (DICOM or PNG/JPG) as an RGB ``PIL.Image``, or ``None``.

    Never raises: every failure is logged with the full path and reason and
    ``None`` is returned so the caller can skip the image and keep going.

    Auto-detection by extension:
      * .dcm / .dicom            -> pydicom
      * .png / .jpg / .jpeg      -> PIL
      * anything else / no ext   -> try PIL first, then pydicom
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".dcm", ".dicom"):
        return _load_dicom(path)
    if ext in (".png", ".jpg", ".jpeg"):
        return _load_pil(path)

    # Unknown extension: try PIL quietly, then DICOM.
    img = _load_pil(path, log_failure=False)
    if img is not None:
        return img
    img = _load_dicom(path)
    if img is None:
        _log.error("Could not load %s as either an image or a DICOM.", path)
    return img


# ── BOX UTILITIES ─────────────────────────────────────────────────────────────

def clamp_box(box, img_w, img_h):
    """Clamp an [x1,y1,x2,y2] box into the image rectangle. Returns a new list.

    x coordinates are clamped to [0, img_w], y coordinates to [0, img_h]. This
    does not fix ordering (x2<=x1); use ``validate_box`` after clamping.
    """
    if box is None:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    x1 = min(max(x1, 0.0), float(img_w))
    x2 = min(max(x2, 0.0), float(img_w))
    y1 = min(max(y1, 0.0), float(img_h))
    y2 = min(max(y2, 0.0), float(img_h))
    return [x1, y1, x2, y2]


def validate_box(box, img_w, img_h):
    """Return True iff ``box`` is a usable [x1,y1,x2,y2] pixel box.

    Checks: exactly 4 numeric values, x2 > x1 and y2 > y1, and every value
    inside the image ([0,img_w] for x, [0,img_h] for y). Logs a warning and
    returns False on any violation. Called before every MedSAM inference call.
    """
    if box is None:
        _log.warning("validate_box: box is None")
        return False
    try:
        n = len(box)
    except TypeError:
        _log.warning("validate_box: box is not a sequence: %r", box)
        return False
    if n != 4:
        _log.warning("validate_box: expected 4 values, got %d: %r", n, box)
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        _log.warning("validate_box: non-numeric values: %r", box)
        return False
    if not (x2 > x1 and y2 > y1):
        _log.warning("validate_box: non-positive extent (x1=%s y1=%s x2=%s y2=%s)",
                     x1, y1, x2, y2)
        return False
    # Small floating tolerance so a box touching the border (x2 == img_w) passes.
    tol = 1e-6
    if x1 < -tol or y1 < -tol or x2 > img_w + tol or y2 > img_h + tol:
        _log.warning("validate_box: out of bounds box=%r for image %dx%d",
                     box, img_w, img_h)
        return False
    return True


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] pixel boxes.

    Returns:
      * ``None`` if either input is None or malformed.
      * ``0.0`` for non-overlapping or zero-area boxes.
      * a float in [0,1] otherwise.

    Never raises. Both boxes MUST already be in the same pixel coordinate space.
    """
    if box1 is None or box2 is None:
        return None
    try:
        if len(box1) != 4 or len(box2) != 4:
            return None
        b1 = [float(v) for v in box1]
        b2 = [float(v) for v in box2]
    except (TypeError, ValueError):
        return None

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def compute_dice(pred_mask, gold_mask):
    """Dice between two boolean masks. Returns None if either is None, else float."""
    if pred_mask is None or gold_mask is None:
        return None
    pred_bool = np.asarray(pred_mask).astype(bool)
    gold_bool = np.asarray(gold_mask).astype(bool)
    denom = int(pred_bool.sum()) + int(gold_bool.sum())
    if denom == 0:
        return 0.0
    intersection = int((pred_bool & gold_bool).sum())
    return float(2.0 * intersection / denom)


# ── RESULTS CSV (incremental) ─────────────────────────────────────────────────
# evaluate.py expects exactly these columns. dice is only written by BiomedParse.
BOX_FIELDS = ["image_id", "region", "iou", "detected", "gold_box", "pred_box"]
BOX_FIELDS_DICE = BOX_FIELDS + ["dice"]

_UNSET = object()


def box_to_str(box):
    """Serialise a box to the string form '[x1,y1,x2,y2]', or '' for None.

    Empty string is read back by pandas as NaN, so ``.notna()`` filters work.
    """
    if box is None:
        return ""
    try:
        vals = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return ""
    if len(vals) != 4:
        return ""
    return "[" + ",".join(str(v) for v in vals) + "]"


def result_row(image_id, region, iou, detected, gold_box, pred_box, dice=_UNSET):
    """Build one results_summary.csv row dict with correctly-typed values.

    * iou   -> Python float, or '' (empty) for None  (never the string 'nan').
    * detected -> Python bool True/False.
    * gold_box / pred_box -> '[x1,y1,x2,y2]' strings (or '' when absent).
    * dice  -> included only when provided (BiomedParse); float or '' for None.
    """
    row = {
        "image_id": str(image_id),
        "region": str(region),
        "iou": "" if iou is None else float(iou),
        "detected": bool(detected),
        "gold_box": box_to_str(gold_box),
        "pred_box": box_to_str(pred_box),
    }
    if dice is not _UNSET:
        row["dice"] = "" if dice is None else float(dice)
    return row


def init_results_csv(csv_path):
    """Remove any stale results_summary.csv so a fresh run starts clean.

    Rows are then appended incrementally (header written on first append), so if
    the Slurm job is killed at hour 23 the partial CSV is still valid.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if os.path.exists(csv_path):
        os.remove(csv_path)


def write_result_row(csv_path, row, fieldnames):
    """Append one row to the CSV, writing the header only if the file is new/empty."""
    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── GOLD ANNOTATIONS ──────────────────────────────────────────────────────────

def load_gold_annotations(csv_path):
    """Load the gold CSV, auto-detecting its schema.

    Two layouts are supported:

    1. **Bounding-box CSV** (Chest ImaGenome scene-graph export) with columns
       ``image_id, bbox_name, x, y, w, h`` (x,y = top-left corner, w,h in pixels).
       Returns ``{image_id: {bbox_name_lower: [x1, y1, x2, y2]}}``.

    2. **Report CSV** (the file used on the Sharanga HPC run,
       ``gold_1000k_reports.csv``) with columns ``study_id, subject_id, report``.
       There are no gold boxes in this file, so it is used only to enumerate the
       images: ``study_id`` becomes the image id and each maps to an empty box
       dict. Downstream, ``gold_box`` is therefore ``None`` and IoU stays ``None``
       while predicted boxes / detection are still recorded for every region.

    Returns an ordered ``dict`` (insertion order = CSV row order) so
    ``list(gold.keys())[:MAX_IMAGES]`` is deterministic.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    cols = set(df.columns)

    bbox_cols = {"image_id", "bbox_name", "x", "y", "w", "h"}
    if bbox_cols.issubset(cols):
        gold = {}
        for _, r in df.iterrows():
            iid = str(r["image_id"])
            name = str(r["bbox_name"]).lower().strip()
            box = [int(r["x"]), int(r["y"]),
                   int(r["x"]) + int(r["w"]),
                   int(r["y"]) + int(r["h"])]
            gold.setdefault(iid, {})[name] = box
        _log.info("Loaded bbox gold CSV: %d images with annotations.", len(gold))
        return gold

    # Report CSV: enumerate images from study_id (no gold boxes available).
    id_col = None
    for candidate in ("study_id", "image_id", "dicom_id"):
        if candidate in cols:
            id_col = candidate
            break
    if id_col is not None:
        gold = {}
        for v in df[id_col].tolist():
            iid = str(v).strip()
            if iid and iid.lower() != "nan":
                gold.setdefault(iid, {})   # empty box dict — no gold boxes
        _log.info("Loaded report gold CSV (%s): %d unique images, no gold boxes.",
                  id_col, len(gold))
        return gold

    raise ValueError(
        "Gold CSV has neither the bbox schema (image_id, bbox_name, x, y, w, h) "
        "nor an id column (study_id / image_id / dicom_id).\n"
        f"Found columns: {list(df.columns)}"
    )


def find_image_path(image_dir, image_id):
    """Resolve the on-disk path of the image for ``image_id`` under ``image_dir``.

    The images on the HPC run are ``.jpg`` files (NOT ``.dcm``), so extensions
    are tried in that order. Resolution order:
      1. flat file ``image_dir/<image_id><ext>`` for each known extension;
      2. ``image_id`` used as an already-complete filename (with extension);
      3. a recursive search under ``image_dir`` for nested layouts.
    Returns a path string, or ``None`` if nothing matches.
    """
    image_id = str(image_id)
    # 1. Flat directory, extension-by-extension (.jpg first — that is what the
    #    HPC gold_images/ directory contains).
    for ext in (".jpg", ".jpeg", ".png", ".dcm", ".dicom"):
        p = os.path.join(image_dir, image_id + ext)
        if os.path.exists(p):
            return p
    # 2. image_id may already include the extension (a full filename).
    p = os.path.join(image_dir, image_id)
    if os.path.exists(p):
        return p
    # 3. Recursive fallback for nested layouts (e.g. p10/s5xxxx/<id>.jpg).
    from pathlib import Path
    root = Path(image_dir)
    for ext in (".jpg", ".jpeg", ".png", ".dcm", ".dicom"):
        matches = list(root.rglob(f"{image_id}{ext}"))
        if matches:
            return str(matches[0])
    return None


# Backwards-compatible alias (older code imported ``find_image_file``).
find_image_file = find_image_path
