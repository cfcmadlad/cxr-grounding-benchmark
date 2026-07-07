# COMPATIBILITY_REPORT.md

Environment / dependency compatibility notes for running the six models on the
**Sharanga HPC cluster** (Rocky Linux 8.10, single GPU node, Slurm, 24 h
wall-time on the `gpu` partition).

> This file did not exist previously; it documents the compatibility landscape
> that shaped the fixes recorded in `BUGS_FOUND.md`.

---

## 1. transformers version conflicts require separate conda envs

The models pin **mutually incompatible** `transformers` versions, so they cannot
share one environment. Each has its own conda env (see the table) and its own
requirements file.

| Model | conda env | Python | transformers | Notes |
|-------|-----------|--------|--------------|-------|
| Grounding DINO | `gdino` | 3.10 | `>=4.38` | Open weights. Lightest env. |
| BioViL-T | `biovilt` | 3.10 | `>=4.30,<4.40` | `hi-ml-multimodal==0.2.2`; needs `batch_encode_plus` (removed ≥4.40). |
| RadVLM | `radvlm` | 3.10 | `==4.46.0` | LLaVA-OneVision; exact pin. PhysioNet weights. |
| MAIRA-2 | `maira2` | 3.10 | `>=4.48,<4.52` | Gated HF; `trust_remote_code=True`. |
| CheXagent | `chexagent` | 3.10 | `>=4.35,<4.47` | Open weights; `trust_remote_code=True`. |
| BiomedParse | `biomedparse` | 3.9.19 | (repo-pinned) | Custom classes; run from inside the cloned repo. |

**Implication for the shared code:** `cxr_common.py` must import cleanly under
**all** of these transformers versions. It therefore imports **only** `os`, `sys`,
`csv`, `logging`, `numpy`, `PIL`, and (lazily) `yaml` / `pydicom` / `pandas` — it
does **not** import torch or transformers. This also lets each script set
`HF_HOME` from config *before* the heavy ML imports.

---

## 2. BioViL-T `batch_encode_plus` shim

`hi-ml-multimodal==0.2.2` calls `tokenizer.batch_encode_plus`, removed in
transformers ≥ 4.40. The `biovilt` env pins `<4.40`, and the script *also*
installs a runtime shim (`text_inference.tokenizer.batch_encode_plus = ...`) as a
safety net. Both are kept. `hi-ml-multimodal` is from an archived repo, so the
pinned version must not be upgraded.

BioViL-T's `get_similarity_map_from_raw_data(...)` returns the similarity map at
the **original image resolution**, so the heatmap-derived box is already in
original pixel coordinates — the same space MedSAM and the gold boxes use. No
extra rescale is needed (verified during the coordinate audit).

---

## 3. Gated / credentialed access

- **MAIRA-2** — gated on HuggingFace: accept the disclaimer at
  `huggingface.co/microsoft/maira-2`, then `huggingface-cli login` once. Weights
  cache under `HF_HOME`.
- **RadVLM** — weights are **not** on HuggingFace; download from PhysioNet
  (`physionet.org/content/radvlm-model/1.0.0`) and point `RADVLM_PATH` at the
  folder. `load_config` fails fast if that path is missing.
- **Images + gold annotations** — Chest ImaGenome / MIMIC-CXR, PhysioNet
  credentialed. Not redistributable.

---

## 4. BiomedParse runs from inside its own repo

BiomedParse uses custom classes (`modeling`, `utilities`, `inference_utils`, …)
that are only importable when the **script directory is the cloned repo**. The
Slurm job therefore copies `biomedparse_seg.py` (+ `cxr_common.py`, `config.yaml`)
into `BIOMEDPARSE_REPO`, exports `PYTHONPATH`, `CXR_CONFIG`, `CXR_REPO_ROOT`, and
`cd`s into the repo to run. `biomedparse_seg.py` has an import fallback that adds
`CXR_REPO_ROOT` to `sys.path` if `cxr_common` is not immediately importable.

BiomedParse is the only model with **no MedSAM step** — it emits masks directly;
a box is derived from the mask for cross-model IoU comparison, but Dice is its
primary metric (`dice` column present only in its `results_summary.csv`).

---

## 5. MedSAM is shared by 5 of the 6 models

All models except BiomedParse load `wanglab/medsam-vit-base` from HuggingFace.
Because `HF_HOME` is set per config and exported by every job, the MedSAM weights
are downloaded once and reused across envs from the same cache directory.

---

## 6. CUDA / driver

Requirements files target NVIDIA RTX A5000 (24 GB), CUDA 13.0 driver, with
PyTorch installed from the CUDA 12.4 wheels (forward-compatible with the 13.0
driver). VRAM footprints: gdino ~0.6 GB, biovilt ~2 GB, biomedparse ~1.5 GB,
radvlm/maira2 ~16 GB, chexagent ~17.5 GB — all fit a single 24 GB GPU, which is
why job memory (`--mem`) is set to 8 G for the light models and 40 G (host RAM,
not VRAM) for the large VLMs.

---

## 7. Slurm / wall-time considerations

The `gpu` partition enforces a 24 h wall-time. With `MAX_IMAGES: 500` × 15
regions, the large VLMs (one generate call per region) are the tightest fit. The
**incremental CSV writing** (Section 7 of `BUGS_FOUND.md`) is the key mitigation:
if a job is killed near hour 23, its `results_summary.csv` is already valid and
`evaluate.py` / `compare_results.py` still produce partial results. `results.json`
is written under an `fcntl.flock` lock so multiple jobs finishing together do not
corrupt it.
