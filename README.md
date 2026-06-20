# CXR Grounding Benchmark

Benchmarking visual grounding models for localizing 15 anatomical regions in chest X-rays against radiologist-validated gold standard annotations (Chest ImaGenome, 500 patients). Six models are evaluated. For each model and image, the pipeline produces 15 per-region PNGs showing predicted vs gold bounding boxes and segmentation mask overlays, along with per-image JSON/NPZ exports and a final IoU/Dice/mAP@0.5 comparison table. Images and model weights are not included — access requires a PhysioNet credentialed account.

---

## Target Regions (15)

Right Lung, Left Lung, Cardiac Silhouette, Mediastinum, Trachea,
Right/Left Upper Lung Zone, Right/Left Mid Lung Zone, Right/Left Lower Lung Zone,
Right/Left Hilar Region, Right/Left Costophrenic Angle.

---

## Models

### 1. Grounding DINO + MedSAM — `grounding_dino_medsam.py`
**Papers:** [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

Grounding DINO is an open-vocabulary object detector trained on natural images. Included as a **negative baseline** — zero chest X-ray training, expected near-zero IoU across all regions. MedSAM (SAM fine-tuned on 1.5M medical image-mask pairs) is applied on top of each detected box to produce segmentation masks.

**Conda env:** `gdino` · **VRAM:** ~0.6GB · **Access:** Open weights

```bash
conda create -n gdino python=3.10 -y && conda activate gdino
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_grounding_dino.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
python grounding_dino_medsam.py
```

---

### 2. BioViL-T + MedSAM — `biovilt_medsam.py`
**Papers:** [BioViL-T (Bannur et al., CVPR 2023)](https://arxiv.org/abs/2301.04558) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

BioViL-T is a Microsoft model pretrained on MIMIC-CXR image-report pairs using contrastive learning. Produces a similarity heatmap per text prompt; a bounding box is derived by thresholding at a configurable percentile and taking the largest connected component. Fixes Grounding DINO's laterality-marker confusion but struggles with fine-grained zone differentiation. Note: the `hi-ml-multimodal` package is from an archived repo (Nov 2025); a `batch_encode_plus` compatibility patch is included in the script.

**Conda env:** `biovilt` · **VRAM:** ~2GB · **Access:** Open weights

```bash
conda create -n biovilt python=3.10 -y && conda activate biovilt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_biovilt.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
# tune PERCENTILE (default 90) if boxes are too large/small
python biovilt_medsam.py
```

---

### 3. RadVLM + MedSAM — `radvlm_medsam.py`
**Papers:** [RadVLM (Deperrois et al., 2025)](https://arxiv.org/abs/2502.03333) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

RadVLM is a 7B multitask conversational VLM built on LLaVA-OneVision, instruction-tuned on 1M+ chest X-ray image-instruction pairs including explicit anatomical grounding on Chest ImaGenome. The primary model of this benchmark — the only one trained with direct box supervision on the exact 15-region anatomical grounding task. Reports mAP@0.5 of 85.3% on anatomical grounding. Outputs bounding boxes as normalized coordinates in free text, parsed with a multi-pattern regex.

⚠️ **Weights require separate PhysioNet access:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0). Download weights and set `RADVLM_PATH` in the script.

**Conda env:** `radvlm` · **VRAM:** ~16GB · **Access:** PhysioNet (credentialed)

```bash
conda create -n radvlm python=3.10 -y && conda activate radvlm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_radvlm.txt
# edit RADVLM_PATH, GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script
# set MAX_IMAGES=10 for first run
python radvlm_medsam.py
```

---

### 4. MAIRA-2 + MedSAM — `maira2_medsam.py`
**Papers:** [MAIRA-2 (Bannur et al., 2024)](https://arxiv.org/abs/2406.04449) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

MAIRA-2 is a Microsoft model combining a radiology-specific image encoder (Rad-DINO) with a Vicuna-7B LLM, trained for grounded radiology report generation. Phrase grounding was primarily trained on pathology findings rather than normal anatomy names, so mAP on the 15 anatomical regions is expected to be lower than RadVLM. Box coordinates are output relative to MAIRA-2's internally cropped 518×518 view; `adjust_box_for_original_image_size` converts them back to pixel coordinates.

⚠️ **Gated HF access required:** visit [huggingface.co/microsoft/maira-2](https://huggingface.co/microsoft/maira-2), click "Agree and access repository", then run `huggingface-cli login`.

**Conda env:** `maira2` · **VRAM:** ~16GB · **Access:** HuggingFace checkbox

```bash
conda create -n maira2 python=3.10 -y && conda activate maira2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_maira2.txt
huggingface-cli login   # paste HF token; only needed once
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
python maira2_medsam.py
```

---

### 5. CheXagent + MedSAM — `chexagent_medsam.py`
**Paper:** [CheXagent (Chen et al., 2024)](https://arxiv.org/abs/2401.12208)

CheXagent-8b is a Stanford AIMI foundation model instruction-tuned on 28 chest X-ray datasets. Grounding was evaluated primarily on pathology findings so normal anatomy mAP is expected to be weak. Uses the official HF model card inference format with a phrase grounding prompt. Box output is parsed from free text; raw model responses are printed for the first image to allow format verification.

**Conda env:** `chexagent` · **VRAM:** ~17.5GB · **Access:** Open weights

```bash
conda create -n chexagent python=3.10 -y && conda activate chexagent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_chexagent.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
# check printed raw responses on first run to verify box parsing format
python chexagent_medsam.py
```

---

### 6. BiomedParse — `biomedparse_seg.py`
**Paper:** [BiomedParse (Zhao et al., Nature Methods 2025)](https://arxiv.org/abs/2405.12971)

BiomedParse is a Microsoft foundation model for joint segmentation, detection, and recognition across 9 imaging modalities including chest X-ray. Outputs masks directly from text prompts — no intermediate bounding box step and no MedSAM needed. All 15 regions are processed in a single batched inference call. A bounding box is derived from each predicted mask for cross-model comparison, but Dice is the primary metric.

⚠️ **Special setup:** uses custom model classes not in any pip package. Script must be placed and run from inside the cloned repo.

**Conda env:** `biomedparse` · **Python: 3.9.19** · **VRAM:** ~1.5GB · **Access:** Open weights

```bash
git clone https://github.com/microsoft/BiomedParse.git && cd BiomedParse
conda create -n biomedparse python=3.9.19 -y && conda activate biomedparse
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r assets/requirements/requirements.txt
pip install -r requirements_biomedparse.txt   # place this file inside BiomedParse/
# place biomedparse_seg.py inside BiomedParse/, edit config at top, then:
python biomedparse_seg.py
```

---

## Evaluation

Run after any or all model scripts complete:

```bash
conda activate radvlm
python evaluate.py
```

Results saved to `outputs/evaluation/`:

| File | Contents |
|---|---|
| `per_region_iou_table.csv` | Main results table — mean IoU per region per model |
| `overall_summary.csv` | Per-model detection rate, mean IoU, mAP@0.5 |
| `map_by_model.png` | mAP@0.5 bar chart with RadVLM 85.3% reference line |
| `per_region_iou_by_model.png` | Grouped bar chart per region, all models |
| `iou_heatmap.png` | Color-coded IoU grid — green = high, red = low |
| `per_region_dice_table.csv` | Dice scores (BiomedParse only) |

`evaluate.py` handles missing model results gracefully — run it incrementally as each model finishes.

---

## Environment Summary

| Script | Env | Python | transformers |
|---|---|---|---|
| `grounding_dino_medsam.py` | `gdino` | 3.10 | >=4.38 |
| `biovilt_medsam.py` | `biovilt` | 3.10 | >=4.30,<4.40 |
| `radvlm_medsam.py` | `radvlm` | 3.10 | ==4.46.0 |
| `maira2_medsam.py` | `maira2` | 3.10 | >=4.48,<4.52 |
| `chexagent_medsam.py` | `chexagent` | 3.10 | >=4.35,<4.47 |
| `biomedparse_seg.py` | `biomedparse` | 3.9.19 | (in repo) |

Each model needs its own conda environment — do not mix them.

---

## Data Access

- **Images + gold annotations:** [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) — PhysioNet credentialed access required
- **RadVLM weights:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0) — separate PhysioNet access request required

Neither images, annotations, nor model weights may be redistributed publicly per PhysioNet data use agreements.