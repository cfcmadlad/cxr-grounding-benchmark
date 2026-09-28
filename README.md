# CXR Grounding Benchmark

Benchmarking visual grounding models for localizing 15 anatomical regions in
chest X-rays against radiologist-validated gold standard annotations (Chest
ImaGenome). **Eight models are evaluated** under one standardized protocol: the
same 959 images, the same 15 region definitions, the same IoU metric. For each
model and image, the pipeline produces 15 per-region PNGs showing predicted vs
gold bounding boxes (plus segmentation mask overlays where applicable),
per-image JSON exports, and a final IoU/detection-rate comparison table. Images
and model weights are not included — access requires a PhysioNet credentialed
account.

---

## Results Summary

Mean IoU is computed two ways and both are reported, because they answer
different questions and disagree sharply for models with low detection rates:

- **Conditional Mean IoU** — averaged only over the region queries a model
  actually returned a usable box for. Answers "how accurate is the model when
  it answers."
- **Unconditional Mean IoU** — averaged over all 14,385 region queries,
  counting every undetected region as IoU 0. Answers "how much correct,
  localized area does the model deliver overall," including the queries it
  misses. **Use this one for ranking models against each other** — conditional
  IoU alone rewards abstention and makes a low-detection-rate model look
  better than it is.

| Model | Conditional Mean IoU | Unconditional Mean IoU | Detection Rate | Images |
|---|---|---|---|---|
| **RadVLM** | 0.764 | 0.763 | 99.9% | 959 |
| MAIRA-2 | 0.433 | 0.313 | 72.1% | 959 |
| BiomedParse | 0.268 | 0.254 | 94.7% | 959 |
| ChEX | 0.220 | 0.214 | 97.4% | 959 |
| BioViL-T | 0.152 | 0.152 | 100.0% | 959 |
| MedGemma 1.5 (4B) | 0.144 | 0.144 | 100.0% | 959 |
| CheXagent-8b | 0.095 | 0.011 | 11.3% | 959 |
| Grounding DINO | 0.083 | 0.083 | 100.0% | 959 |

14,385 region-instances scored per model (959 images × 15 regions); one image
is missing its gold annotation for "right lower lung zone" and is excluded
from that single region's IoU for every model equally (see Known Issues).

RadVLM's training data explicitly includes Chest ImaGenome box-grounding
supervision using this exact coordinate convention — it is the only model in
this benchmark trained end-to-end on the task being measured, which is the
primary reason for the gap between it and every other model. See Known Issues
below before citing CheXagent's or MAIRA-2's numbers as final.

---

## What Prompt Was Used for Each Model

**The prompt text is NOT identical across models.** These are eight
architecturally incompatible systems — chat-based text generators, a native
object detector, a CLIP-style text-conditioned box regressor, and a
segmentation model — and there is no single prompt string that all eight
accept. What IS held constant across every model: the same 15 anatomical
region names, and the same task (produce a box for this named region, from
the image alone). **No model was given a pre-existing radiology report as
input** — every model grounds directly from the image plus the region
name/prompt.

| Model | Input mechanism | Exact prompt / query |
|---|---|---|
| Grounding DINO | Native phrase-grounding detector call | Region name passed directly as the detection query (no full-sentence prompt) |
| BioViL-T | Native phrase-grounding / similarity-heatmap API | Region name as the text phrase; box derived by thresholding the similarity heatmap |
| CheXagent-8b | Chat prompt (`USER: ... ASSISTANT:`) | `Perform phrase grounding for "{phrase}" in this chest X-ray. Provide the bounding box as [x1, y1, x2, y2] in pixel coordinates.` |
| MedGemma 1.5 | Chat prompt | `Locate the {region} in this chest X-ray. Provide the bounding box as [x1, y1, x2, y2] with coordinates normalized between 0 and 1.` |
| RadVLM | LLaVA-OneVision chat template | `Locate the {region} in this chest X-ray. Provide the bounding box as [x1, y1, x2, y2] with coordinates normalized between 0 and 1.` |
| MAIRA-2 | MAIRA-2's own grounded-reporting generation call | Region name passed as the grounding target; MAIRA-2 generates its own report-style text and links a box to it when the phrasing reads as a "finding" (see Known Issues) |
| BiomedParse | Text-prompted segmentation API | Region name as the segmentation prompt; box = bounding rectangle of the predicted mask |
| ChEX | CLIP-style text encoder, not a generative prompt | Title-case anatomy name from ChEX's own training vocabulary (`conf/dataset/anatomy_names/cig_default.yaml`), e.g. `"Right lung."` — all 15 regions encoded once per image in a single batched call |

---

## Known Issues and Findings (read before citing per-model numbers)

These were found via a systematic audit (per-region left/right cross-checks,
raw-response inspection, and source-level verification for ChEX) done after
the initial full runs. They materially change how CheXagent's and ChEX's
scores should be interpreted, and are documented here rather than silently
fixed, so anyone using this benchmark can see exactly what was checked.

### ChEX: genuine left/right confusion (verified against source, not a harness bug)
ChEX's predictions for left-labeled regions systematically land near the
correct *right*-side structure instead of the correct left-side one (e.g.
"left lung" prediction matches "right lung" gold at IoU 0.41 vs only 0.16
against its own "left lung" gold). This was traced through ChEX's actual
`encode_prompts()`/`detect_prompts()` source
(`chex/src/model/chex.py:448-515`): prompt order is correctly preserved
end-to-end, box regression is conditioned purely on each prompt's own text
embedding, and there is no fixed-vocabulary index lookup that could explain a
harness-side swap. "Right lung" and "left lung" produce genuinely different
(not duplicated) box coordinates on the same image, ruling out a simple
aliasing bug. The most likely explanation is that ChEX's text encoder does not
sufficiently separate "left" from "right" anatomical prompts to drive its box
head to the correct hemisphere — a real property of the trained model, not
something fixable in this evaluation harness. Reported as a finding, not
patched.

### CheXagent: likely prompt-format mismatch (not fully resolved — do not treat 0.095/0.011 as final)
Inspecting raw model responses shows CheXagent frequently returns the literal
placeholder text `"[x1, y1, x2, y2]"` from our prompt instruction instead of
generating real coordinates. Critically, on the queries where it *does*
ground successfully, it spontaneously uses its own native output format
(`<ref>ClassName</ref><box>(x1,y1),(x2,y2)</box>`) rather than the
`[x1,y1,x2,y2]` format our prompt asks for. This strongly suggests our current
prompt does not match how CheXagent was trained to respond, and that its
11.3% detection rate understates its real capability. **A corrected prompt
that elicits CheXagent's native `<ref><box>` format should be tested before
these numbers are used in any final comparison or publication.**

### MAIRA-2: task-framing mismatch on whole-organ queries (not fully resolved)
MAIRA-2's detection rate varies enormously by region — 7.3% for "right lung"
vs 94.1% for "left mid lung zone" — and the pattern maps precisely onto
whether the region name reads like a radiology-report *finding*. For "right
lung"/"left lung" MAIRA-2 returns generic negative-findings report boilerplate
("No confluent pulmonary infiltrates are seen...") with no box, since MAIRA-2
is a *grounded reporting* model trained to link report findings to boxes, and
a whole organ is rarely phrased as a finding in real reports. For
"mediastinum," "trachea," and the lung zones, which do read like plausible
report-sentence subjects, it correctly returns `<obj>findings in the
X.<box>...</box></obj>`. **Reframing whole-organ queries to read like a
report finding should be tested before treating 0.433/0.313 as final.**

### MedGemma 1.5: mild, likely genuine left/right confusion
A milder version of ChEX's pattern shows up on 3 of 15 regions (right upper
lung zone, right lower lung zone, right hilar region). Since MedGemma's
prompt is a plain-English sentence with no coordinate-mapping code involved
(`f"Locate the {region}..."`), this is most likely a genuine zero-shot
vision-language model limitation rather than a code bug, but it has not been
independently confirmed the way ChEX's issue was.

### One missing gold annotation
Image `acb299f2-449ffbaf-848f8dc9-07d91ecc-73d7bc8d` has no gold box for
"right lower lung zone." All 8 models still produced a prediction for it; the
instance is excluded from Conditional Mean IoU and counted as 0 in
Unconditional Mean IoU for every model equally, so it does not change any
model's ranking relative to the others.

### Costophrenic angles are not uniformly "hard" — check recall@0.1, not just IoU
Raw IoU on the costophrenic angles is low for most models, but RadVLM's
recall@0.1 (fraction of predictions with IoU ≥ 0.1) is 95% and its
gold-center-inside-predicted-box rate is ~90% for both angles — meaning it
does find the correct location almost every time; the low raw IoU reflects
that a small anatomical target is punished disproportionately by IoU geometry
for any small offset, not a real localization failure. Grounding DINO shows
the opposite pattern (93-95% center-in-box despite 0% recall@0.1), which
indicates a degenerate near-whole-image predicted box, not real localization.
**Don't cite "costophrenic angle is inherently hard for every model" without
this context** — it's true for most models but specifically false for RadVLM.

---

## Configuration & Running on Sharanga HPC

All paths live in **`config.yaml`** at the repo root — the model scripts have
no hardcoded paths. Edit `config.yaml` once (GOLD_CSV, IMAGE_DIR,
RADVLM_PATH, OUTPUT_DIR, MAX_IMAGES, HF_HOME, BIOMEDPARSE_REPO), then:

```bash
# 1. Verify the environment before submitting any job (CI-friendly, exits 1 on failure)
python setup_check.py

# 2. Submit one Slurm job per model (each activates its own conda env,
#    runs the model, then writes results_summary.csv):
sbatch job_gdino.sh
sbatch job_biovilt.sh
sbatch job_chexagent.sh
sbatch job_maira2.sh
sbatch job_medgemma15.sh
sbatch job_chex.sh
sbatch job_biomedparse.sh
sbatch job_radvlm.sh
```

Each job writes its `results_summary.csv` **incrementally** (one row per
region) to `OUTPUT_DIR/<model>/`, so partial results survive if a job is
killed near the wall-time limit.

Before running full jobs, always run the smaller `*_smoketest` variant first
(a handful of images via `config_smoketest.yaml`) and read the raw model
responses printed to the Slurm `.out` log by hand — every real bug found in
this project (MedGemma's y-before-x coordinate order, CheXagent's prompt
placeholder echo, ChEX's laterality issue) was caught this way, never by
trusting a plausible-looking IoU number.

### A note on HPC queue time and script efficiency
This cluster's queue is heavily contended and wait times are unpredictable
(anywhere from minutes to 47+ hours were observed for the same tiny job
resubmitted at different times). Before submitting a full run:
- Submit the same job to 2-3 partitions at once (e.g. `gpu_v100_2`,
  `gpu_a100_8`, `gpu_h100_4`) and cancel the losers once one starts, rather
  than waiting in a single queue.
- Check whether the model script re-encodes the image once per region or
  once per image — RadVLM's first version re-ran the vision encoder on every
  one of the 15 per-region prompts; encoding the image once and reusing it
  across regions (or batching all 15 prompts into one multi-turn call) cuts
  wall-clock time substantially and should be done before submitting a
  15-region-per-image model for the first time.
- Bump wall-time past what the smoke test's per-image rate implies you need —
  MedGemma's first full run hit the 23h cap at 954/959 images and had to be
  resubmitted with a 30h cap.

---

## Target Regions (15)

Right Lung, Left Lung, Cardiac Silhouette, Mediastinum, Trachea,
Right/Left Upper Lung Zone, Right/Left Mid Lung Zone, Right/Left Lower Lung
Zone, Right/Left Hilar Region, Right/Left Costophrenic Angle.

---

## Models

### 1. Grounding DINO + MedSAM — `grounding_dino_medsam.py`
**Papers:** [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

Open-vocabulary object detector trained on natural images, zero chest X-ray
training. Included as a negative baseline. MedSAM applied on top of each
detected box to produce segmentation masks.

**Conda env:** `gdino` · **VRAM:** ~0.6GB · **Access:** Open weights

```bash
conda create -n gdino python=3.10 -y && conda activate gdino
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_grounding_dino.txt
python grounding_dino_medsam.py
```

---

### 2. BioViL-T + MedSAM — `biovilt_medsam.py`
**Papers:** [BioViL-T (Bannur et al., CVPR 2023)](https://arxiv.org/abs/2301.04558) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

Microsoft model pretrained on MIMIC-CXR image-report pairs via contrastive
learning. Produces a similarity heatmap per text prompt; box derived by
thresholding at a configurable percentile and taking the largest connected
component.

**Conda env:** `biovilt` · **VRAM:** ~2GB · **Access:** Open weights

```bash
conda create -n biovilt python=3.10 -y && conda activate biovilt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_biovilt.txt
python biovilt_medsam.py
```

---

### 3. RadVLM + MedSAM — `radvlm_medsam.py`
**Papers:** [RadVLM (Deperrois et al., 2025)](https://arxiv.org/abs/2502.03333) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

7B multitask conversational VLM built on LLaVA-OneVision, instruction-tuned on
1M+ chest X-ray image-instruction pairs **including explicit anatomical
grounding on Chest ImaGenome using this exact coordinate convention**. The
strongest model in this benchmark by a wide margin (0.764 unconditional mean
IoU) — see Results Summary for why.

⚠️ **Weights require separate PhysioNet access:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0).

**Conda env:** `radvlm` · **VRAM:** ~16GB · **Access:** PhysioNet (credentialed)

```bash
conda create -n radvlm python=3.10 -y && conda activate radvlm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_radvlm.txt
python radvlm_medsam.py
```

---

### 4. MAIRA-2 + MedSAM — `maira2_medsam.py`
**Papers:** [MAIRA-2 (Bannur et al., 2024)](https://arxiv.org/abs/2406.04449) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

Microsoft grounded radiology report generation model. **See Known Issues** —
its low detection rate on whole-organ regions is likely a task-framing
mismatch rather than a pure capability gap.

⚠️ **Gated HF access required:** [huggingface.co/microsoft/maira-2](https://huggingface.co/microsoft/maira-2).

**Conda env:** `maira2` · **VRAM:** ~16GB · **Access:** HuggingFace checkbox

```bash
conda create -n maira2 python=3.10 -y && conda activate maira2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_maira2.txt
huggingface-cli login
python maira2_medsam.py
```

---

### 5. CheXagent-8b + MedSAM — `chexagent_medsam.py`
**Paper:** [CheXagent (Chen et al., 2024)](https://arxiv.org/abs/2401.12208)

Stanford AIMI foundation model instruction-tuned on 28 chest X-ray datasets.
**See Known Issues** — its low detection rate is likely a prompt-format
mismatch rather than a pure capability gap.

**Conda env:** `chexagent` · **VRAM:** ~17.5GB · **Access:** Open weights

```bash
conda create -n chexagent python=3.10 -y && conda activate chexagent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_chexagent.txt
python chexagent_medsam.py
```

---

### 6. MedGemma 1.5 (4B) — `medgemma15_medsam.py`
**Model:** [MedGemma (Google)](https://huggingface.co/google/medgemma-4b-it)

General-purpose medical vision-language model, used zero-shot with no
grounding-specific fine-tuning. Outputs a box on a 0-1000 normalized grid as
free text, parsed with a multi-pattern regex.

**Conda env:** `radvlm` (same transformers version works) · **VRAM:** ~10GB · **Access:** Open weights (gated checkbox on HF)

```bash
python medgemma15_medsam.py
```

---

### 7. ChEX — `chex_medsam.py`
**Paper:** ChEX (native DETR-style box regression)

Purpose-built anatomical grounding model: a CLIP-style text encoder feeds a
DETR-style box-regression head with an explicit presence gate, trained
directly on Chest ImaGenome anatomy names. No MedSAM step — ChEX outputs a
box directly. **See Known Issues** for its verified left/right confusion.
Must be run from inside the cloned ChEX repo's `src/` directory (its modules
use non-package-relative imports).

**Conda env:** `chex` (built via `mamba`, not classic `conda` — see setup notes below) · **VRAM:** ~4GB

```bash
# classic `conda env create` OOMs/times out on this environment.yaml on this
# cluster; bootstrap mamba into its own env first, then:
mamba env create -f environment.yaml
conda activate chex
python chex_medsam.py   # run from inside the cloned ChEX repo's src/ directory
```

---

### 8. BiomedParse — `biomedparse_seg.py`
**Paper:** [BiomedParse (Zhao et al., Nature Methods 2025)](https://arxiv.org/abs/2405.12971)

Microsoft foundation model for joint segmentation/detection/recognition
across 9 imaging modalities. Outputs masks directly from text prompts; a
bounding box is derived from each predicted mask for cross-model comparison.

⚠️ **Special setup:** uses custom model classes not in any pip package. Script
must be placed and run from inside the cloned repo.

**Conda env:** `biomedparse` · **Python:** 3.9.19 · **VRAM:** ~1.5GB · **Access:** Open weights

```bash
git clone https://github.com/microsoft/BiomedParse.git && cd BiomedParse
conda create -n biomedparse python=3.9.19 -y && conda activate biomedparse
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r assets/requirements/requirements.txt
pip install -r requirements_biomedparse.txt
python biomedparse_seg.py
```

---

## Environment Summary

| Script | Env | Python | transformers |
|---|---|---|---|
| `grounding_dino_medsam.py` | `gdino` | 3.10 | >=4.38 |
| `biovilt_medsam.py` | `biovilt` | 3.10 | >=4.30,<4.40 |
| `radvlm_medsam.py` | `radvlm` | 3.10 | ==4.46.0 |
| `medgemma15_medsam.py` | `radvlm` | 3.10 | ==4.46.0 |
| `maira2_medsam.py` | `maira2` | 3.10 | >=4.48,<4.52 |
| `chexagent_medsam.py` | `chexagent` | 3.10 | >=4.35,<4.47 |
| `chex_medsam.py` | `chex` | 3.10 | (in repo, via mamba) |
| `biomedparse_seg.py` | `biomedparse` | 3.9.19 | (in repo) |

Each model needs its own conda environment — do not mix them.

---

## Data Access

- **Images + gold annotations:** [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) — PhysioNet credentialed access required
- **RadVLM weights:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0) — separate PhysioNet access request required

Neither images, annotations, nor model weights may be redistributed publicly
per PhysioNet data use agreements. This repository contains only code —
scripts, prompts, and parsers — so that results are reproducible by anyone
with their own PhysioNet access.
