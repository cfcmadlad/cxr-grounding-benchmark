# Compatibility & Bug Audit Report

Audit scope: all 6 model scripts, `evaluate.py`, and all 6 `requirements_*.txt`
files, cross-checked against upstream documentation, GitHub source, and
model cards where accessible. Target deployment: Sharanga HPC (Rocky Linux
8.10, Slurm, A100/CUDA 12.4, no internet on compute nodes).

---

## 1. Bugs found and fixed

### 1.1 Crash-on-every-print: invalid f-string format specifiers
**Files:** `biovilt_medsam.py`, `radvlm_medsam.py`

Both scripts contained:
```python
print(f"  {region}: IoU={iou:.3f if iou is not None else 'N/A'}...")
```
This is not valid Python. Everything after the `:` in an f-string
replacement field is a *format spec*, not an inline expression -- there is
no conditional evaluation inside it. `".3f if iou is not None else 'N/A'"`
is not a legal format spec, so this raises `ValueError: Invalid format
specifier` the first time it executes (i.e. on the very first region of
the very first image). Both model scripts would have failed to produce a
single usable log line, though the underlying `results_summary.csv` and
PNGs were written *before* the print statement, so results weren't lost --
but the run would still abort partway (the crash happens inside the
per-region loop, so PNGs already saved for that region survive, but the
process dies without ever reaching per-image JSON/NPZ saving or the final
summary).

**Fix:** compute the formatted string in a separate statement before the
f-string, e.g. `iou_str = f"{iou:.3f}" if iou is not None else "N/A"`.

### 1.2 Grounding DINO requires transformers>=4.40.0, not >=4.38.0
**File:** `requirements_grounding_dino.txt`

`GroundingDinoForObjectDetection` / `AutoModelForZeroShotObjectDetection`
support and `GroundingDinoProcessor.post_process_grounded_object_detection`
were merged into `huggingface/transformers` on 2024-04-11 and first
shipped in the **4.40.0** release. The pin `transformers>=4.38.0` would
resolve a version that predates Grounding DINO support entirely, causing
`AutoModelForZeroShotObjectDetection.from_pretrained(...)` to fail with an
unrecognized-architecture error.

**Fix:** pinned to `transformers>=4.40.0`. (Also noted: transformers
4.51.0 changed `post_process_grounded_object_detection` to return integer
label ids in some code paths instead of strings -- harmless here since the
script only reads `scores`/`boxes`, but documented in the requirements
file for anyone extending it to use `labels`.)

### 1.3 CheXagent box-parsing regex doesn't match CheXagent's real output format
**File:** `chexagent_medsam.py` (parser now lives in `cxr_common.py`)

The original `parse_box_from_response()` only tried
`[x1,y1,x2,y2]`/`(x1,y1,x2,y2)`/bare-number patterns. CheXagent's actual
grounding output format is **`<|box|> (x1,y1),(x2,y2) <|/box|>`** with
coordinates on a **0-100 scale** -- confirmed against the RadVLM authors'
own evaluation harness (`radvlm/evaluation/models_loading_inference.py` in
`github.com/uzh-dqbm-cmi/RadVLM`), which parses CheXagent's output with
exactly this tag pattern before dividing by 100. None of the original
three regex patterns can match this (the two coordinate pairs are
separated by `),(` -- parentheses, not just a comma/space -- so even the
permissive bare-number fallback fails to find 4 contiguous numbers).
In practice every single CheXagent grounding call would have returned
`pred_box=None`, silently producing 0% detection rate for every region
without ever raising a visible error.

**Fix:** the shared `parse_box_from_response()` in `cxr_common.py` checks
the `<|box|>` tag format first (with 0-100 scale normalization), then
falls back to bracket/paren/bare-number parsing for other checkpoints or
prompt variations. `PRINT_RAW_RESPONSES=True` (still the default) lets you
confirm the exact format your specific checkpoint snapshot produces on
the first image of every run.

### 1.4 MAIRA-2: only the first of possibly several returned boxes was kept
**File:** `maira2_medsam.py`

`processor.convert_output_to_plaintext_or_grounded_sequence()` can return
more than one box for a single phrase (documented behavior for bilateral
or repeated structures, and observed in the RadVLM authors' own
`inference_maira2_grounding` code, which loops over multiple boxes per
phrase and adjusts each one individually). The original script did
`raw_box = boxes[0]` and dropped everything else, silently discarding a
subset of valid detections whenever MAIRA-2 grounded a phrase with
multiple regions.

**Fix:** every returned box is now individually passed through
`processor.adjust_box_for_original_image_size()`, and if more than one
box comes back, the script takes their union (smallest enclosing box) via
the new `union_box()` helper in `cxr_common.py`, since the gold annotation
is always a single box per region.

### 1.5 Six duplicated, slightly-fragile image loaders
**Files:** all 6 model scripts

Every script had its own copy-pasted `load_image()`/`find_image_file()`.
Beyond the maintenance risk of six near-identical implementations quietly
diverging, the original loader had several concrete gaps (see Section 4).

**Fix:** extracted into `cxr_common.py`, imported by all 6 scripts
(including `biomedparse_seg.py`, which needs its own copy placed inside
the cloned BiomedParse repo directory -- `setup.sh` does this
automatically).

### 1.6 hi-ml-multimodal repo archival status and `batch_encode_plus`
**Files:** `biovilt_medsam.py`, `requirements_biovilt.txt`

Confirmed via GitHub: `microsoft/hi-ml` was archived (read-only) on
**November 21, 2025**. The existing runtime patch and requirements comment
claimed `batch_encode_plus` was "removed in transformers >= 4.40" -- this
is factually wrong and was corrected. `batch_encode_plus` was deprecated
for a long time but only **actually removed in transformers v5.0.0**
(released January 2026), part of the v5 API cleanup; 4.40+ still has it
with a deprecation warning. The existing `<4.40` pin already protects
against the real removal in v5, so the risk was more theoretical than the
original comment suggested -- but since hi-ml-multimodal is now frozen
forever (no more releases possible), any future transitive dependency
resolution that pulls in transformers>=5 despite the pin would silently
break it with no upstream fix ever coming.

**Fix:** corrected the comment/documentation to state the real removal
version, and hardened the runtime shim to patch both the specific
tokenizer instance BioViL-T constructs *and* `PreTrainedTokenizerBase` at
the class level, since hi-ml-multimodal's `BertEncoder` may construct
tokenizer objects at times not visible from the top-level script.

### 1.7 evaluate.py
No functional bugs found after a full read-through. Column-by-column logic
(`compute_map_at_05`, per-region aggregation, chart generation) is
self-consistent and already handles missing model results gracefully
(the stated design goal). No changes were made to this file.

---

## 2. DICOM / PNG image pipeline hardening

All fixes below live in the new shared `cxr_common.py::load_image()`,
used by every script.

| Issue | Old behavior | Fixed behavior |
|---|---|---|
| DICOM missing preamble/File Meta (common after anonymization/re-export) | `pydicom.dcmread()` raises, whole image skipped | retried with `force=True` before giving up |
| Multi-frame DICOM (`pixel_array` returns a 3D stack) | Crashes or silently mis-shapes downstream tensors | first frame is taken explicitly when the array isn't already channel-last RGB/RGBA |
| `RescaleSlope`/`RescaleIntercept` present (common on re-processed/derived DICOMs) | Ignored -- raw pixel values normalized directly, which is wrong if slope/intercept aren't identity | applied before windowing/normalization |
| VOI LUT / windowing metadata present | Ignored -- plain full-range min-max stretch, which can wash out contrast the radiologist actually used | `pydicom.pixel_data_handlers.util.apply_voi_lut()` applied when available, falling back to plain min-max stretch if it fails for any reason |
| `MONOCHROME1` (inverted) images | Handled (this part was already correct) | unchanged, verified with a synthetic MONOCHROME1 DICOM in testing |
| Truncated/corrupt JPEGs (a handful of MIMIC-CXR files are known to be slightly truncated) | Pillow raises `OSError: image file is truncated`, image skipped entirely | `ImageFile.LOAD_TRUNCATED_IMAGES = True` set globally; corrupt files still fail loudly with a clear wrapped `IOError` rather than a silent skip further downstream |
| 16-bit grayscale PNG (`I;16` mode, sometimes produced by DICOM-to-PNG conversion tools) | `.convert("RGB")` on a 16-bit-per-channel image does NOT do a proper 16→8 bit rescale in Pillow, producing near-black or wraparound-artifact images | detected and explicitly min-max normalized to 8-bit before RGB conversion |
| Palette (`P` mode) / CMYK / RGBA PNGs | `.convert("RGB")` alone works for most but silently drops alpha-blended regions differently across Pillow versions for `P` mode | explicit `RGBA` intermediate conversion for palette images before the final RGB convert, for consistent alpha-flattening behavior |
| Channel-count consistency across all 6 models | Each script called `.convert("RGB")` independently; consistent already, but not centrally guaranteed | now guaranteed at a single choke point -- every one of the 6 scripts receives exactly the same normalized 3-channel `uint8` RGB `PIL.Image` regardless of source format, which matters because Grounding DINO, MedSAM, BioViL-T, CheXagent, MAIRA-2, RadVLM, and BiomedParse all expect 3-channel RGB input (verified against each model's processor/model card) |

Tested manually against synthetic DICOM (`MONOCHROME1`/`MONOCHROME2`,
`RescaleSlope`/`Intercept`), 8-bit grayscale PNG, 16-bit grayscale PNG,
RGBA PNG, and a deliberately corrupted file -- see commit history for the
test commands used.

---

## 3. Package versions pinned and why

| Package | Script(s) | Pin | Why |
|---|---|---|---|
| `transformers` | `grounding_dino_medsam.py` | `>=4.40.0` | Grounding DINO support shipped in 4.40.0 (see 1.2) |
| `transformers` | `biovilt_medsam.py` | `>=4.30.0,<4.40.0` | hi-ml-multimodal 0.2.2 (last release before archival) only tested against <4.40; also keeps `batch_encode_plus` available without relying solely on the runtime shim |
| `transformers` | `radvlm_medsam.py` | `==4.46.0` | Exact pin -- RadVLM's checkpoint was converted from a specific LLaVA-OneVision architecture snapshot; unrelated to the other pins, kept unchanged (already correct) |
| `transformers` | `maira2_medsam.py` | `>=4.48.0,<4.52` | Per official MAIRA-2 model card, tested up to 4.51.3; unchanged (already correct) |
| `transformers` | `chexagent_medsam.py` | `>=4.35.0,<4.47.0` | Keeps clear of the RadVLM/BioViL-T pins in case envs are ever merged; unchanged (already correct) |
| `numpy` | `biovilt_medsam.py` (via `requirements_biovilt.txt`) | `>=1.24.0,<2.0.0` | hi-ml-multimodal 0.2.2 predates NumPy 2.0 (June 2024) and, being archived, will never be updated for it -- capped to avoid ABI/dtype-promotion breakage |
| `numpy` | `biomedparse_seg.py` (via `requirements_biomedparse.txt`) | `<2.0.0` | BiomedParse's own `assets/requirements/requirements.txt` predates NumPy 2.0 under Python 3.9.19; capped defensively in case it doesn't already pin numpy itself |
| `hi-ml-multimodal` | `biovilt_medsam.py` | `==0.2.2` | Last PyPI release before the GitHub repo was archived (Nov 21, 2025); still installable from PyPI (archival doesn't remove existing releases), unchanged |
| `pydicom` | all 6 | `>=2.4.0` | `apply_voi_lut` (used in the new DICOM loader) needs a reasonably recent pydicom; `>=2.4.0` is a safe floor, unchanged from original |

---

## 4. Known remaining risks / limitations

- **hi-ml-multimodal is frozen forever.** The GitHub repo is archived
  (read-only). The PyPI package `hi-ml-multimodal==0.2.2` still installs
  fine today, but if it (or a transitive dependency like `SimpleITK`) is
  ever yanked from PyPI, or if a future pip resolver decision pulls in an
  incompatible transformers/numpy despite the pins in this repo, there
  will be no upstream fix. Recommendation: mirror the exact wheel files
  (`pip download`) to cluster-local storage once, so a working BioViL-T
  env can always be rebuilt offline even if PyPI availability changes.

- **CheXagent-8b's exact grounding prompt/output format could not be
  independently re-verified from the live HuggingFace model card during
  this audit** (huggingface.co was unreachable through this environment's
  network policy during the audit). The `<|box|>` tag format fix in
  Section 1.3 is based on the RadVLM authors' own published evaluation
  code parsing CheXagent output this exact way, which is strong secondary
  evidence, but you should still treat `PRINT_RAW_RESPONSES=True` on the
  first real run as mandatory verification, not optional -- if your
  specific checkpoint snapshot differs, add the correct pattern to
  `cxr_common.py::_BRACKETED_PATTERNS` (or a new dedicated regex ahead of
  it) before trusting a full run's numbers.

- **MedSAM checkpoint (`wanglab/medsam-vit-base`) and its `SamModel`/
  `SamProcessor` API were verified to exist and match the usage in all 5
  scripts that call it.** No changes were needed here.

- **RadVLM's exact grounding output format (normalized `[x1,y1,x2,y2]`,
  0-1 floats) was independently confirmed** against the RadVLM paper/repo
  description. No format-specific bug was found here beyond the f-string
  crash (Section 1.1); the shared parser still applies the same
  multi-format fallback chain as CheXagent for robustness against
  unexpected outputs (e.g. a stray trailing period, extra whitespace, or
  an occasional 0-1000-scale response from a differently-tuned decoding
  configuration).

- **MAIRA-2's exact multi-box return semantics could not be re-verified
  against the live HF model card** for the same network-access reason as
  CheXagent above. The union-of-boxes fix (Section 1.4) is a defensible
  and conservative way to combine multiple legitimate detections for a
  single gold region, but if you observe MAIRA-2 returning multiple boxes
  that clearly correspond to *different, unrelated* structures (rather
  than bilateral/repeated instances of the same one), the union approach
  will over-estimate the predicted box size and under-estimate IoU --
  inspect `boxes.json`'s `raw_response` field per image if MAIRA-2's
  reported IoU looks anomalously low.

- **BiomedParse's custom-class import path (`modeling.BaseModel`,
  `modeling.build_model`, `configs/biomedparse_inference.yaml`,
  `interactive_infer_image`) matches the officially documented v1
  (2D X-ray-Chest) inference pattern.** Note that the BiomedParse GitHub
  repo has since added a separate, differently-architected "v2"/3D
  pipeline (Hydra-based config, different model-loading and inference
  call signature) for volumetric modalities -- this benchmark
  intentionally continues to use the original v1 2D API, which is the one
  relevant to chest X-ray, and this script's approach is correct for that
  checkpoint. If a future BiomedParse release removes the v1 API
  entirely, this script would need to be rewritten for the v2 interface.

- **GPU/driver assumptions:** all `requirements_*.txt` header comments
  reference "CUDA 13.0, Driver 580.95.05" (the environment on which they
  were originally authored), which does not match the target Sharanga
  cluster (A100s, CUDA 12.4). This is not actually a conflict --
  `--index-url https://download.pytorch.org/whl/cu124` installs CUDA
  12.4-built PyTorch wheels regardless of the host driver's reported CUDA
  version, and NVIDIA's driver/toolkit forward compatibility covers this
  --  but the comments are misleading if read literally as cluster specs.
  Not changed (informational only, doesn't affect installed packages),
  but flagged here so nobody assumes CUDA 13.0-specific wheels are needed.

- **RadVLM/MAIRA-2/CheXagent are 16-17.5GB VRAM models** sharing a single
  A100 GPU per Slurm job (per `--gres=gpu:1`). The 32G **system RAM**
  requested in `job_radvlm.sh`/`job_maira2.sh`/`job_chexagent.sh` is
  headroom for CPU-side tensor staging, image I/O, and matplotlib
  rendering, not a proxy for VRAM -- if your cluster's A100s are the 40GB
  variant rather than 80GB, running more than one of these three jobs
  concurrently on the same node could still exceed available VRAM if
  Slurm's GPU accounting allows oversubscription; confirm your gres
  configuration prevents that.
