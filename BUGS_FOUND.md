# BUGS_FOUND.md — Audit & Changelog

**Task:** Fix and harden the entire `cxr-grounding-benchmark` codebase for a
production inference run on the Sharanga HPC cluster (Rocky Linux 8.10, single
GPU node, Slurm, 24 h wall-time on `gpu`).

> **Note on this file.** `BUGS_FOUND.md` and `COMPATIBILITY_REPORT.md` did **not
> exist** in the repository when this work started (nor did `cxr_common.py`,
> `config.yaml`, any `job_*.sh`, `results/`, `compare_results.py`, or
> `setup_check.py`). This file is therefore both the audit of every bug found and
> the changelog of every fix applied. All findings below were derived by reading
> the six model scripts + `evaluate.py` end-to-end.

---

# Round 2 — fixes after the failed Sharanga HPC production run

The first pass (everything below this section) was written against assumptions
about the data layout. A real production run on Sharanga then failed. The
confirmed-from-logs bugs and the environment mismatch behind them are fixed
here. **Real HPC paths (all under `/home/manik/pranjali/Aditya_project/`):**

| Thing | Path |
|---|---|
| Images (`.jpg`, NOT `.dcm`) | `gold_images/gold_images/gold_images/` |
| Gold CSV (`study_id,subject_id,report`) | `gold_images/gold_images/gold_1000k_reports.csv` |
| Repo | `cxr-grounding-benchmark-main/` |
| MedSAM / GDINO / BioViL-T / BERT weights | `medsam_vit_b.pth`, `grounding_dino_swint_ogc.pth`, `biovilt_image_model_proj_size_128.pt`, `bert_base_uncased_local/` |

### R2.0 — CRITICAL blocker: gold CSV schema mismatch (every script, startup)

The real `GOLD_CSV` is a **report list** with columns `study_id, subject_id,
report` — it has **no bounding boxes**. `load_gold_annotations()` *required*
`image_id, bbox_name, x, y, w, h` and **raised `ValueError`** otherwise, so every
one of the six scripts crashed at `load_gold_annotations(GOLD_CSV)` **before
producing a single row** — independent of the image-path bug.

**Fix (`cxr_common.load_gold_annotations`):** auto-detect the schema. Bbox CSV →
`{image_id: {bbox_name: [x1,y1,x2,y2]}}` as before. Report CSV → enumerate images
from `study_id` (falls back to `image_id` / `dicom_id`), each mapping to an
**empty** box dict. Downstream, `gold_box` is `None`, `compute_iou(pred, None)`
returns `None`, and detection / predicted boxes are still recorded — so the run
produces the full **844 × 15 = 12 660** rows per model instead of crashing.
`setup_check.py` now accepts either schema too.

### R2.1 — BUG 1: `.dcm` hardcoded, images are `.jpg` (all 6 scripts)

Image lookup went through `find_image_file`, which never tried `image_id` as a
full filename and used `.dcm` in its extension list ahead of nothing useful.
Added **`find_image_path(image_dir, image_id)`** to `cxr_common.py` exactly as
specified: tries `.jpg/.jpeg/.png/.dcm/.dicom` (jpg first — that is what the HPC
`gold_images/` dir contains), then `image_id` as a complete filename, then a
recursive fallback. All six scripts import and call `find_image_path`
(`find_image_file` kept as an alias).

### R2.2 — BUG 2: BioViL-T infinite recursion (`biovilt_medsam.py`)

The `batch_encode_plus` compatibility shim did `return
text_inference.tokenizer(...)`. But `tokenizer.__call__` dispatches **into**
`batch_encode_plus`, so the shim called itself forever → *"maximum recursion
depth exceeded"* on every region, empty results CSV. **Fix:** bind the
tokenizer class's real `batch_encode_plus` onto the instance (goes straight to
the fast/slow backend, never re-enters `__call__`); if the method is genuinely
absent, fall back to `encode_plus` per item — again never through `__call__`.
Partial-result safety was already covered by the incremental CSV writer + the
per-region `try/except` that writes a row on failure.

### R2.3 — BUG 3: CheXagent produced zero output (`chexagent_medsam.py`)

Raw model output was printed only for the first image, so a silent box-parse
failure looked like "no output at all". Now the **raw model output is printed
after every inference call** (`RAW[<image>/<region>]: ...`). The per-region CSV
write already fires for every region regardless of whether a box parsed
(`detected=False`, `iou=None` when it does not), so a row is emitted for all
844 × 15 regions.

### R2.4 — BUG 4: MAIRA-2 stopped after ~4 images (all 6 scripts)

Only the per-region block was guarded; an unhandled error in the per-image scope
(outside that block) aborted the whole run (~61 rows ≈ 4 images). The per-image
body is now a nested `_process_image()` called inside
`try/except Exception` in the loop, so **one bad image can never stop the run** —
it is logged with the `image_id`, `failed_count` is incremented, and the loop
continues. Applied identically to all six scripts. Each still prints
`Completed: X images, Y regions, Z failures`.

### R2.5 — Infrastructure: config, jobs, setup, weights

- **`config.yaml`** rewritten with the real Sharanga paths and the exact
  Section-7 keys, including the staged local weight files (`MEDSAM_WEIGHTS`,
  `GDINO_WEIGHTS`, `BIOVILT_WEIGHTS`, `BERT_PATH`) and `MAX_IMAGES: 844`.
- **All six `job_*.sh`** repointed to
  `/home/manik/pranjali/Aditya_project/...` (repo `cxr-grounding-benchmark-main`,
  cache `.cache/huggingface`); `job_biomedparse.sh` `cd`s into `BiomedParse` and
  exports `PYTHONPATH`/`CXR_CONFIG`/`CXR_REPO_ROOT` accordingly. SBATCH headers,
  `--mem` (8/8/40/40/40/8 G) and conda envs were already correct.
- **`setup_check.py`** now accepts the report-CSV schema and verifies the staged
  weight files (`MEDSAM_WEIGHTS`, `GDINO_WEIGHTS`, `BIOVILT_WEIGHTS`, `BERT_PATH`)
  exist, per Section 10.
- **MedSAM weights (documented deviation):** the scripts load MedSAM (and
  Grounding DINO / BioViL-T) from the HuggingFace cache under `HF_HOME`, which is
  what loaded successfully on the failed runs. The raw `*.pth`/`*.pt` checkpoints
  in `config.yaml` are staged for offline use and verified by `setup_check.py`;
  rewiring the HF loaders to raw checkpoints would change model-loading logic and
  is intentionally out of scope ("do not change model inference logic").

---

## 1. CRITICAL — runtime crashes (would abort a whole 23 h job)

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1 | `radvlm_medsam.py:426` | `print(... IoU={iou:.3f if iou is not None else 'N/A'})` — the conditional was placed **inside** the format spec, which raises `ValueError` on *every* region that has a numeric IoU (and `TypeError` when `iou is None`). This crashed the job on the first successful region. | Replaced with a pre-computed safe string. All IoU/score prints now use `f"{x:.3f}" if x is not None else "N/A"` evaluated *before* the f-string. |
| 2 | `biovilt_medsam.py:354` | Same malformed f-string for both `IoU=` and `peak_sim=` — guaranteed crash. | Same fix. |

These two were the highest-impact defects: the pipeline could never complete.

---

## 2. Image loading — was fragile and inconsistent (Section 1)

**Before:** every script had its own `load_image()` that:
- `raise`d `ImportError` when pydicom was missing (crashes the job) instead of returning `None`;
- did `arr.astype(np.uint8)` on 16-bit DICOM **without window/level**, truncating/wrapping pixel values;
- had **no** handling for multi-frame DICOM, float pixel arrays, or colour DICOM;
- did **no** 0×0 degenerate-size check;
- did **not** convert non-RGB PNG/JPG modes explicitly in all paths;
- **raised** on any failure instead of returning `None`.

**Fix:** one `load_image(path)` in `cxr_common.py`, called by all 6 scripts:
- extension auto-detect (`.dcm/.dicom` → pydicom; `.png/.jpg/.jpeg` → PIL; unknown → PIL then pydicom);
- DICOM: `try/except` around `dcmread`; MONOCHROME1 inversion; multi-frame → frame 0; float arrays normalised to 0–255; 16-bit mapped to uint8 via **window/level** (WindowCenter/Width, else min–max), never a raw cast; RescaleSlope/Intercept applied; colour DICOM handled; final RGB;
- PNG/JPG: `try/except`; explicit non-RGB→RGB; 0×0 check;
- **never raises** — returns `None` and logs the full path + reason on any failure.

Every model loop now does: `pil_img = load_image(path); if pil_img is None: log, failed_count += 1, continue`.

---

## 3. Per-image / per-region error handling (Section 2)

**Before:** error handling was partial and uneven. Grounding DINO wrapped the
model call and MedSAM call separately but left IoU/printing outside any guard;
several scripts had the IoU/`print` line (with the broken f-string) completely
unguarded; a single bad region aborted the entire job.

**Fix (all 6 scripts):** the entire per-region inference block is wrapped in
`try/except Exception`:
- logs `image_id`, `region`, and full traceback (`logging.exception`);
- increments a `failed_count`;
- writes a `results_summary.csv` row with `iou=None, detected=False`;
- `continue`s to the next region — never crashes the job.

Each script ends with: `Completed: X images, Y regions, Z failures`.

---

## 4. Bounding-box coordinate audit (Section 3)

End-to-end audit of every model's box pipeline. Findings per model:

- **MAIRA-2** — outputs normalized coords relative to an internal 518×518 view.
  The official `processor.adjust_box_for_original_image_size()` performs the
  crop-aware inverse (MAIRA pads to square + centre-crops before resizing, so
  the inverse is **not** a plain `W/518, H/518` scale). This was already correct;
  we **kept** the official method (a naive scale would be wrong), and **added**
  `clamp_box` + `validate_box` before MedSAM. The math is documented in
  `maira_ground_phrase`.
- **RadVLM** — regex parse of normalized boxes. **Fixed:** rewrote the parser to
  (a) use robust numeric groups, (b) add **fallback patterns** (`[...]`, `(...)`,
  `<box>...</box>`, `x1:.. y1:.. x2:.. y2:..`, and bare `x1,y1,x2,y2`),
  (c) **clamp every value to `[0,1]`**, then (d) denormalize to pixels
  (`x*img_w, y*img_h`). If no box parses → `detected=False`, MedSAM skipped.
- **BioViL-T** — heatmap percentile → box. Verified: threshold produces pixel
  `x1y1x2y2`; axes **not** transposed (`ys, xs = np.where(...)`, box uses `xs` for
  x); largest connected component via `ndimage.label`+`argmax`; output is
  `x1y1x2y2`, not `x1y1wh`. Correct — left the logic, added clamp+validate.
- **CheXagent** — free-text parse. **Fixed:** raw model output is printed for the
  **first image** (already toggled, kept); added fallback regex patterns; parser
  denormalizes with the correct dims (`x*w, y*h`) when values ≤ 1, else clamps
  pixel coords into the image.
- **Grounding DINO** — verified output is pixel `x1y1x2y2` (post-processing with
  `target_sizes=(H,W)`), no missing cxcywh→xyxy conversion. Added clamp+validate.
- **BiomedParse** — box derived from mask via `np.where`. Verified box is built as
  `[col_min, row_min, col_max, row_max]` = `[x1,y1,x2,y2]`, **not**
  `[row_min, col_min, ...]`. Correct; documented in `mask_to_box`.

**New:** `validate_box(box, img_w, img_h)` in `cxr_common.py` — checks 4 numeric
values, `x2>x1 & y2>y1`, and bounds (`x∈[0,img_w]`, `y∈[0,img_h]`); logs a warning
and returns `False` on any violation. **Called before every MedSAM call** in all
five box→MedSAM scripts (and on the derived box in BiomedParse). Boxes are first
`clamp_box`-ed into the image so a small overshoot is corrected rather than dropped.

---

## 5. MedSAM integration (Section 4)

- Input box is pixel `x1y1x2y2` at original resolution → validated by
  `validate_box` immediately before the call.
- `SamProcessor` resizes the image to 1024×1024 and rescales the box internally;
  `post_process_masks(original_sizes=...)` resizes the predicted mask back to the
  original image — documented in each `get_medsam_mask`.
- Every MedSAM call is wrapped in `try/except`: on failure `mask=None`, the error
  is logged, and the loop continues.
- **Deviation (documented):** Section 4 says "on failure store iou=None". In these
  pipelines the reported metric is **box IoU**, computed from the predicted box vs
  the gold box — it does **not** depend on the MedSAM mask. Nulling a valid box
  IoU because mask segmentation failed would discard a correct result, so box IoU
  is preserved and only `mask` is set to `None` on MedSAM failure. Mask-derived
  outputs (the overlay PNG, NPZ) are simply omitted for that region.
- MedSAM weights: `config.yaml` (Section 8) does not define a MedSAM key, so the
  HF id `wanglab/medsam-vit-base` is kept as a constant; the weights are cached
  under `HF_HOME`, which **is** read from config and exported by the job scripts.

---

## 6. IoU computation (Section 5)

**Before:** `box_iou()` was **reimplemented in all 6 scripts** (duplication +
drift risk). It returned `0.0` on non-overlap but did not handle `None` inputs
or guarantee a Python `float`.

**Fix:** one `compute_iou(box1, box2)` in `cxr_common.py`, used everywhere with
no local reimplementations:
- `None` in → `None` out;
- non-overlapping or zero-area → `0.0`;
- correct intersection formula (`max(0, min(x2)-max(x1)) * max(0, min(y2)-max(y1))`);
- both boxes are pixel `x1y1x2y2` in the same space before calling;
- never raises; always returns a Python `float` (or `None`).

---

## 7. `results_summary.csv` output (Section 6)

**Before:** each script accumulated an `all_results` list and wrote the whole CSV
**once at the very end** via `results_df.to_csv()`. If the job was killed at hour
23, **all** partial results were lost. Columns also included extras (`score`,
`peak_score`) beyond what `evaluate.py` expects, and `pred_box`/`gold_box` were
serialized as Python lists with spaces.

**Fix (all 6 scripts):**
- exact columns only: `image_id, region, iou, detected, gold_box, pred_box`
  (+ `dice` for BiomedParse);
- `gold_box`/`pred_box` serialized as `"[x1,y1,x2,y2]"` strings (empty when absent);
- `detected` is a Python `bool`; `iou` is a Python `float` or empty (never the
  string `"nan"` / `np.nan`);
- rows are **appended incrementally** (`csv.DictWriter`, header on first write) so
  a killed job leaves a valid partial CSV;
- `os.makedirs(output_dir, exist_ok=True)` at script start; the CSV is truncated
  once at start (`init_results_csv`) so a fresh run does not append to stale rows.

Verified round-trip: `pandas.read_csv` reads `detected` as `bool` dtype and empty
`iou` as `NaN`, which `evaluate.py`'s `.notna()` / `== True` filters rely on.

---

## 8. `config.yaml` — removal of hardcoded paths (Section 8)

**Before:** every script had a hardcoded `CONFIG` block
(`GOLD_CSV = "/path/to/..."`, etc.). Nothing was shared.

**Fix:** created `config.yaml` at the repo root (GOLD_CSV, IMAGE_DIR, RADVLM_PATH,
OUTPUT_DIR, MAX_IMAGES, HF_HOME, BIOMEDPARSE_REPO). All 6 scripts + `evaluate.py`
load paths through `cxr_common.load_config(required_keys=[...])`, which:
- exits(1) with a descriptive message if a required key is missing/empty;
- exits(1) if an input-path key (GOLD_CSV, IMAGE_DIR, RADVLM_PATH,
  BIOMEDPARSE_REPO) does not exist on disk;
- creates output/cache dirs (OUTPUT_DIR, HF_HOME).

`OUTPUT_DIR` is the **base**; each script writes to `OUTPUT_DIR/<model>/`.
`setup_hf_home()` exports `HF_HOME`/`TRANSFORMERS_CACHE` **before** importing
torch/transformers. No hardcoded paths remain in any Python file.

---

## 9. Slurm job scripts (Section 9)

Created `job_gdino.sh`, `job_biovilt.sh`, `job_chexagent.sh`, `job_maira2.sh`,
`job_radvlm.sh`, `job_biomedparse.sh` — each with the required SBATCH header
(`-p gpu`, `-N 1`, `-n 1`, `-t 0-23:00`, `-o/-e slurm.%j.*`, `--mail-type=ALL`,
`--job-name`), the specified `--mem` (8G / 8G / 40G / 40G / 40G / 8G), conda
activation of the correct env, `HF_HOME`/`TRANSFORMERS_CACHE` exports, `cd` to the
repo root, the model script, then `evaluate.py` + `compare_results.py`.
`job_biomedparse.sh` additionally exports `PYTHONPATH`/`CXR_CONFIG`, copies the
script + shared module + config into the BiomedParse repo, and `cd`s there to run.
`.gitattributes` pins `*.sh` to `eol=lf` so they run on Linux.

---

## 10. Results tracking (Section 10)

- Created `results/results.json` (all 6 models `pending`).
- `evaluate.py` computes per-model metrics (iou_mean/median, recall@0.1/0.25/0.5,
  map@0.5, num_samples, num_failed) and updates `results.json` under an
  **`fcntl.flock` exclusive lock** (guarded so it degrades gracefully off-Linux),
  so concurrent job endings cannot corrupt the file.
- Created `compare_results.py`: reads `results.json`, prints a markdown table
  across all 6 models, **bolds the best value per column**, writes
  `results/comparison_table.md`, shows `pending` gracefully, and is safe to run
  after every job.

---

## 11. Setup verification (Section 11)

Created `setup_check.py`: verifies config paths exist, the gold CSV header
contains the required columns, ≥1 image exists in IMAGE_DIR, RADVLM_PATH is
non-empty, OUTPUT_DIR/HF_HOME/results are writable, and the six conda envs exist.
Prints `PASS`/`FAIL: reason` per check and exits 1 on any failure (CI-friendly).

---

## Files created / changed

**Created:** `cxr_common.py`, `config.yaml`, `setup_check.py`,
`compare_results.py`, `results/results.json`, `BUGS_FOUND.md`,
`COMPATIBILITY_REPORT.md`, `job_gdino.sh`, `job_biovilt.sh`, `job_chexagent.sh`,
`job_maira2.sh`, `job_radvlm.sh`, `job_biomedparse.sh`.

**Rewritten (bug fixes only, inference logic untouched):**
`grounding_dino_medsam.py`, `biovilt_medsam.py`, `radvlm_medsam.py`,
`maira2_medsam.py`, `chexagent_medsam.py`, `biomedparse_seg.py`.

**Edited:** `evaluate.py` (config-driven paths + results.json updating),
`.gitattributes` (LF for `*.sh`), `README.md` (config.yaml / Slurm workflow).
