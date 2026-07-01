# Line-by-Line Bug Audit

Every `.py`, `.sh`, `.txt`, and `.md` file in this repo was read in full for
this pass (in addition to the earlier compatibility audit). Findings below
are grouped by file. All "FIXED" items were fixed directly in the code as
part of this pass. "VERIFIED — NOT A BUG" items were specifically checked
against the categories requested (off-by-one, wrong axis, division by
zero, NaN/Inf propagation, standard metric definitions, etc.) and found to
already behave correctly, with the reasoning recorded so it doesn't need
re-litigating later.

---

## cxr_common.py

### FIXED — Unreachable/incorrect 0-1000 coordinate-scale heuristic
**Was:** `_finalize_box()` had a branch treating 4 numbers as "0-1000
quantized scale" whenever `max(vals) > img_w and max(vals) > img_h`. For
this benchmark's actual images (MIMIC-CXR frames are typically
2000-3000px per side), a legitimate 0-1000-scale coordinate (max value
<=1000 by definition) can never exceed `img_w`/`img_h`, so the branch
could never fire correctly — it was dead logic that, if it ever did fire
(on a hypothetically small image), would have misinterpreted ordinary raw
pixel coordinates as a different scale.
**Fix:** removed the branch entirely. Neither model that uses the shared
free-text fallback parser (RadVLM: 0-1 floats; CheXagent: 0-100 `<|box|>`
tags, handled in a separate branch before this one) emits 0-1000-scale
boxes, so raw-pixel-coordinate is the correct sole fallback.

### FIXED — All-or-nothing normalized-scale detection (real, high-impact bug)
**Was:** `elif all(0.0 <= v <= 1.0 for v in vals):` gated the
normalized-to-pixel conversion. A VLM emitting `[0.1, 0.2, 1.02, 0.95]`
(a single coordinate slightly over 1.0 — a very plausible float-rounding
artifact at the image edge) would fail this `all(...)` check entirely,
falling through to the raw-pixel-coordinate branch, which then reinterpreted
`1.02`/`0.95` as literal pixel counts — producing a **degenerate ~1x1
pixel box** at the image's top-left corner instead of the correct
near-full-image box.
**Verified with a regression test:** before the fix,
`parse_box_from_response("[0.1, 0.2, 1.02, 0.95]", 1000, 800)` returned
`None` (the degenerate 1x1 box even failed the `x2 > x1` sanity check).
After the fix it correctly returns `[100, 160, 1000, 760]`.
**Fix:** normalized-scale detection now uses a `-0.05 <= v <= 1.05`
tolerance band (allowing for rounding slop) and clamps each individual
coordinate into `[0, 1]` before scaling, rather than an all-or-nothing
exact-range gate.

### FIXED — No coordinate clamping to image bounds
**Was:** a box straddling just past the image edge (e.g. `x2` slightly
`> img_w`) was passed through unclamped. Downstream consumers (MedSAM box
prompts, mask array indexing in `biomedparse_seg.py`) could receive
out-of-range coordinates.
**Fix:** `_finalize_box()` now clamps `x1,y1,x2,y2` into `[0, img_w]` /
`[0, img_h]` before returning.

### FIXED — `box_iou()` didn't guard against degenerate/negative-area boxes
**Was:** if either box had `x2 <= x1` or `y2 <= y1` (zero or negative
width/height — not supposed to happen given `_finalize_box()`'s own
validation, but reachable from hand-built boxes like `union_box()`
output on adversarial input), `area_pred`/`area_gold` could be zero or
negative, and the ratio `inter / (area_pred + area_gold - inter)` could
produce a nonsensical value (e.g. division producing a value outside
`[0, 1]`) instead of raising or clamping.
**Fix:** `box_iou()` now explicitly returns `0.0` if either box's area is
`<= 0`, before computing any intersection.

### FIXED — Multi-frame COLOR DICOM (4D pixel array) not handled
**Was:** the multi-frame-DICOM branch only checked
`arr.ndim == 3 and arr.shape[-1] not in (3, 4)` (grayscale multi-frame:
`(frames, H, W)`). A multi-frame **color** DICOM has shape
`(frames, H, W, samples)` — `ndim == 4` — which fell through unhandled,
and later `Image.fromarray()` on a 4D array would raise.
**Fix:** added an explicit `if arr.ndim == 4: arr = arr[0]` branch before
the grayscale-multi-frame check. (Multi-frame color CXR DICOM is rare in
practice, but this is a correctness gap worth closing defensively.)

### VERIFIED — NaN/Inf handling in `heatmap_to_box`'s heatmap thresholding
`np.nan_to_num(sim_map, nan=-1e9)` + filtering `valid[valid > -1e8]`
correctly handles an all-NaN heatmap (returns `None, None`) without
raising. Percentile edge cases (0th/100th percentile) produce valid
(if degenerate) boxes, not crashes. No changes needed.

### VERIFIED — `mask_dice()` division
`2 * intersection / denom` with `denom == 0` explicitly guarded
(`return 0.0`) before the division. No changes needed.

---

## grounding_dino_medsam.py

### FIXED — Truthy-check bug on a float score (line ~323)
**Was:** `f"score={f'{score:.3f}' if score else 'N/A'}"` — uses `if score`
(truthy) instead of `if score is not None`. A real detection score of
exactly `0.0` would incorrectly print `N/A` instead of `0.000`.
**Impact:** low in practice (`BOX_THRESHOLD=0.25` means scores are never
near zero), but it's a genuine logic bug and inconsistent with every
other `is not None` check in the same file (e.g. the IoU check two lines
below it, and the identical pattern in `save_region_png()`).
**Fix:** changed to `if score is not None`.

### FIXED — `if MAX_IMAGES:` doesn't handle `MAX_IMAGES=0` (line ~270)
**Was:** truthy check meant `MAX_IMAGES=0` (env var `"0"`) was treated the
same as `None` (process all images) instead of processing zero images.
**Fix:** changed to `if MAX_IMAGES is not None:` in `main()`.
(Same fix applied identically to all 6 scripts — see below.)

### VERIFIED — Grounding DINO prompt/input format
`gdino_processor(images=pil_image, text=[[region]], return_tensors="pt")`
matches the documented `AutoProcessor` calling convention for Grounding
DINO (list-of-list-of-phrases). `post_process_grounded_object_detection`
is called with `target_sizes=[pil_image.size[::-1]]` — correctly reversing
PIL's `(W, H)` to the `(H, W)` order the processor expects. Output boxes
are absolute pixel-space `[x1,y1,x2,y2]`, matching gold box format
(no xywh/cxcywh confusion). No changes needed.

---

## biovilt_medsam.py

### FIXED — Off-by-one in `heatmap_to_box()` (connected-component bbox)
**Was:** `box = [xs.min(), ys.min(), xs.max(), ys.max()]`. `xs.max()`/
`ys.max()` are the last **included** pixel index of the connected
component, but every other box in this benchmark (gold boxes built as
`[x, y, x+w, y+h]`, and every model's predicted boxes) uses an
**exclusive** upper bound. This made every BioViL-T-derived box exactly
1 pixel too narrow and 1 pixel too short compared to gold and to every
other model — a systematic bias in IoU computation (small in absolute
terms for typical box sizes, but a real, fixable correctness bug).
**Fix:** `box = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]`.

### FIXED — Unguarded `pil_img.save(tmp_path)` could crash the whole run
**Was:** BioViL-T's API requires the image to be written to a temp PNG
file first; this write wasn't wrapped in try/except, unlike every other
per-image/per-region operation in the same loop (load_image, inference,
MedSAM, plotting). A transient disk-write failure (e.g. quota/permission
issue on shared HPC scratch storage) would have killed the entire job
instead of skipping just that one image.
**Fix:** wrapped in try/except; logs and `continue`s to the next image on
failure, consistent with the rest of the loop's error handling.

### VERIFIED — `batch_encode_plus` shim
Confirmed both instance-level and class-level patches are applied before
any BioViL-T inference call. No changes needed beyond the compatibility
fixes already made in the prior audit pass.

### VERIFIED — percentile edge cases
`PERCENTILE=0` selects the whole image (valid, if degenerate) as a single
component; `PERCENTILE=100` selects only the peak-value pixel(s) (valid,
tiny box). Neither crashes. No changes needed.

---

## chexagent_medsam.py

### FIXED — Raw response dominated by echoed prompt, not the model's actual answer
**Was:** `response = processor.tokenizer.decode(output, skip_special_tokens=True)`
decodes `output` in full. `AutoModelForCausalLM.generate()` returns the
full prompt+completion sequence (not just newly generated tokens) for a
decoder-only model, so `response` began with the entire templated prompt
— including the literal instructional text `Provide the bounding box as
[x1, y1, x2, y2] in pixel coordinates.` — before CheXagent's actual
answer. Since:
  - the debug print (`response[:200]`) truncates to 200 characters, and
  - the JSON-saved `raw_response` field is also truncated to 200 characters,

and the echoed prompt is easily 150+ characters on its own, **the entire
purpose of `PRINT_RAW_RESPONSES=True` and the saved `raw_response`
field — letting you verify CheXagent's actual box-output format — was
defeated**: both were mostly or entirely showing the prompt being read
back, not the model's answer.
**Fix:** compute `prompt_length = inputs["input_ids"].shape[-1]` before
`generate()`, then decode only `output[prompt_length:]` — the same
correct pattern `maira2_medsam.py` already used. `parse_box_from_response()`
itself was unaffected by this bug (it operates on the full un-truncated
string via `re.search`, and the prompt's own instructional text doesn't
contain digit sequences that could false-positive-match), so past parsing
results are not invalidated by this fix — but future debugging/verification
now actually works as intended.

### FIXED — `if MAX_IMAGES:` (line ~285) — see cross-file fix below.

---

## maira2_medsam.py

No new bugs found in this pass beyond the multi-box union fix already
applied in the prior audit. Verified:
- `format_and_preprocess_phrase_grounding_input(...).to(DEVICE, torch.float16)`
  correctly chains device+dtype conversion on the returned `BatchFeature`.
- `prompt_length = processed_inputs["input_ids"].shape[-1]` + slicing
  `output_decoding[0][prompt_length:]` before decode — already correct
  (this is the pattern the CheXagent/RadVLM fixes above were modeled on).
- `union_box()` correctly combines multiple returned boxes per phrase.

### FIXED — `if MAX_IMAGES:` (line ~297) — see cross-file fix below.

---

## radvlm_medsam.py

### FIXED — Case-sensitive, substring-fragile prompt-stripping regex (real bug)
**Was:**
```python
full_response = processor.decode(output[0], skip_special_tokens=True)
response = re.split(r"(user|assistant)", full_response)[-1].strip()
```
Two real problems:
1. **Case sensitivity:** the pattern only matches lowercase `user`/
   `assistant`. RadVLM is built on
   `LlavaOnevisionForConditionalGeneration`, and LLaVA/Vicuna-family chat
   templates conventionally render roles as `USER`/`ASSISTANT`
   (uppercase) — exactly the convention this repo's own
   `chexagent_medsam.py` uses in its prompt string
   (`" USER: <s>{prompt} ASSISTANT: <s>"`). If RadVLM's template follows
   the same convention, the lowercase-only regex **never matches**, and
   `re.split` returns the original string unchanged as a single-element
   list — meaning `response` would silently be the **entire**
   prompt+completion text, not just the model's answer.
2. **Substring fragility:** even when it does match, splitting on bare
   `user`/`assistant` substrings anywhere in the decoded text (rather than
   anchoring to actual chat-template role boundaries) has no protection
   against those words appearing elsewhere in the text.
**Fix:** replaced with the same token-length-slicing approach used to fix
CheXagent above: capture `prompt_length = inputs["input_ids"].shape[-1]`
before `generate()`, then decode only `output[0][prompt_length:]`. This
sidesteps case-sensitivity and substring-collision risk entirely by
operating in token space instead of string space. The now-unused
`import re` was also removed (`re` was only used at this one call site;
`parse_box_from_response()`'s own regex work lives in `cxr_common.py` and
imports `re` separately there).

### FIXED — `if MAX_IMAGES:` (line ~288) — see cross-file fix below.

### VERIFIED — box-parsing coverage
RadVLM's confirmed real output format (normalized `[x1,y1,x2,y2]` floats,
per the RadVLM paper/repo) is handled by the shared parser's primary
bracket pattern. The shared parser's fallback chain (paren-pairs, bare
numbers) provides defense-in-depth against minor format deviations
without needing per-model-specific regex duplicated in this file.

---

## biomedparse_seg.py

### FIXED — Off-by-one in `mask_to_box()` (same class of bug as BioViL-T)
**Was:** `[xs.min(), ys.min(), xs.max(), ys.max()]` — inclusive upper
bound, inconsistent with gold boxes and every other model's exclusive
upper-bound convention.
**Fix:** `[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]`.

### FIXED — `gold_box_to_mask()` vulnerable to negative-index wraparound
**Was:** `mask[y1:y2, x1:x2] = True` with no bounds checking. If `x1` or
`y1` were ever negative (a plausible data-quality issue in a gold CSV, not
purely hypothetical), Python's slicing semantics treat a negative start
index as "from the end of the array" — silently selecting a **completely
wrong region** of the mask array instead of raising an error or safely
clipping.
**Fix:** clamp `x1,x2` to `[0, img_w]` and `y1,y2` to `[0, img_h]` before
slicing.

### FIXED — Silent misalignment if `interactive_infer_image()` returns the wrong number of masks
**Was:** `for region, pred_raw in zip(REGIONS, pred_masks_raw):` — if
BiomedParse's API ever returned fewer (or more) masks than the 15 region
prompts submitted (e.g. it silently drops an unrecognized prompt), `zip()`
truncates to the shorter list with **no error, warning, or indication**
that some regions were skipped for that image.
**Fix:** added an explicit length check right after the inference call;
mismatches now print a warning and skip the image (rather than silently
producing partial/misaligned results that would look like clean data to
`evaluate.py`).

### VERIFIED — mask-resize logic
When `pred_mask.shape != (H, W)` (BiomedParse's native inference
resolution differs from the original image), the resize path correctly
re-derives the mask from `pred_raw` (the original float array) rather
than from the already-wrong-shaped boolean `pred_mask`, and `.resize((W, H), ...)`
uses PIL's `(width, height)` order correctly, matching numpy's resulting
`(H, W)` array shape. No changes needed.

### FIXED — `if MAX_IMAGES:` (line ~263) — see cross-file fix below.

---

## Cross-file fix: `if MAX_IMAGES:` → `if MAX_IMAGES is not None:`

**Files:** all 6 model scripts, one occurrence each in `main()`.
**Was:** `if MAX_IMAGES: image_ids = image_ids[:MAX_IMAGES]`. Since
`MAX_IMAGES` is set from an environment variable
(`int(_max_images_env) if _max_images_env else None`), setting
`MAX_IMAGES=0` (e.g. to sanity-check the pipeline runs zero images without
touching real data) produces `MAX_IMAGES = 0`, and `if 0:` is falsy in
Python — silently falling through to "process every image" instead of
"process zero images." Low real-world likelihood (nobody intentionally
sets `MAX_IMAGES=0` for a real run) but a genuine logic bug matching the
"off-by-one / wrong truthiness" category explicitly called out for this
audit.
**Fix:** changed to `if MAX_IMAGES is not None:` in all 6 scripts.

---

## evaluate.py

Re-verified against the specific standard-definition questions raised for
this pass:

### VERIFIED — "mAP@0.5" is a detection-rate proxy, not classic COCO mAP — by design, not a bug
`compute_map_at_05()` computes `(iou >= 0.5).mean() * 100` over
**detected-only** rows (`detected == True` and `gold_box` present) — this
is not the same as textbook mAP (precision-recall curve integration across
confidence thresholds, with missed detections counted as false negatives
in the denominator). However: this benchmark's task (exactly one gold box
per anatomical region per image, not multi-instance detection) is a
phrase-grounding-style task, and the docstring explicitly states this
metric is chosen to match "the RadVLM paper's Table 4" convention, which
uses the same detection-accuracy-at-IoU-0.5 definition common in
phrase-grounding literature (as opposed to COCO-style multi-instance AP).
The separately-reported `Detection Rate` column in `overall_summary.csv`
lets a reader see when a model's mAP@0.5 is inflated by a low detection
rate (few attempts, but the ones attempted happened to be accurate). Not
changed — changing this would silently break comparability with the
85.3% reference line the charts already draw from the RadVLM paper.

### VERIFIED — Dice/IoU column aggregation with missing data
`.dropna().mean()` on an empty `Series` returns `NaN` without raising
(verified against the currently installed pandas/numpy in this
environment); the `if not np.isnan(...)` guards correctly convert that to
`None` for CSV/table output. `warnings.filterwarnings("ignore")` at the
top of the file suppresses the benign "Mean of empty slice" RuntimeWarning
numpy would otherwise print. No changes needed.

### VERIFIED — empty/missing model handling
A model with a `results_summary.csv` containing only a header row (0 data
rows — e.g. every image in a `MAX_IMAGES=10` test run failed to load)
produces an empty-but-correctly-columned DataFrame; every downstream
aggregation (`groupby`, `.dropna().mean()`, `compute_map_at_05`) already
handles this without raising, per the function-level checks reviewed
above. No changes needed.

No code changes were made to `evaluate.py` in this pass.

---

## Inter-script consistency (per-file schema check)

Confirmed by direct inspection of every `all_results.append({...})` /
`per_image_boxes[...] = {...}` block across all 6 scripts:

| Column | gdino | biovilt | radvlm | maira2 | chexagent | biomedparse |
|---|---|---|---|---|---|---|
| `image_id` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `region` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pred_box` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `gold_box` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `iou` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `detected` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dice` | — | — | — | — | — | ✓ (only model with this column, handled conditionally by `evaluate.py`) |
| model-specific extra (`score`, `peak_score`) | `score` | `peak_score` | — | — | — | — |

All 6 scripts write `results_summary.csv` to `Path(OUTPUT_DIR) /
"results_summary.csv"`, and every per-image `boxes.json`/`masks.npz` under
`Path(OUTPUT_DIR) / image_id / ...` — identical structure, matching what
`evaluate.py::load_model_results()` expects
(`Path(output_dir) / "results_summary.csv"`) and what the per-region PNG
viewers assume. No inconsistencies found.

---

## DICOM/PNG loader trace (per model)

Traced `load_image()`'s output through every model's actual input
requirement:

| Model | Expected input | What it receives |
|---|---|---|
| Grounding DINO (`AutoProcessor`/`AutoModelForZeroShotObjectDetection`) | RGB PIL image | `load_image()` output, `.convert("RGB")` guaranteed |
| MedSAM (`SamProcessor`/`SamModel`, used by 5 of 6 scripts) | RGB PIL image | same |
| BioViL-T (`ImageTextInferenceEngine.get_similarity_map_from_raw_data`) | file path (reads its own file from disk with its own preprocessing) | `load_image()` output is saved to a temp PNG first (`pil_img.save(tmp_path)`), so BioViL-T's own internal loader — not `cxr_common.load_image()` — actually decodes the temp file. This is an intentional exception, documented in the script; `cxr_common.load_image()` still does the DICOM decode/normalization up front so BioViL-T's internal loader only ever sees a clean, already-normalized PNG regardless of whether the source was DICOM |
| CheXagent-8b (`AutoProcessor`, `trust_remote_code=True`) | RGB PIL image(s) list | `load_image()` output |
| MAIRA-2 (`format_and_preprocess_phrase_grounding_input(frontal_image=...)`) | RGB PIL image | `load_image()` output |
| RadVLM (`LlavaOnevisionForConditionalGeneration`/`AutoProcessor`) | RGB PIL image | `load_image()` output |
| BiomedParse (`interactive_infer_image(model, image, prompts)`) | RGB PIL image, batch size 1 | `load_image()` output |

No model is silently receiving a wrong channel count or the wrong image
object — every one of the 6 scripts calls `load_image()` (or, for
BioViL-T, saves its already-normalized output before its own
model-specific file-based API takes over) and gets a 3-channel RGB
`uint8` `PIL.Image` regardless of source format.
