# CXR Grounding Benchmark

A benchmark for anatomical region localization in chest X-rays, evaluating six visual grounding models against the [Chest ImaGenome](https://physionet.org/content/chest-imagenome/1.0.0/) gold standard annotations (500 radiologist-validated MIMIC-CXR patients). Part of a clinical knowledge graph pipeline project at BITS Pilani under Prof. Manik Gupta. For each model and image, the pipeline produces 15 per-region PNGs showing predicted vs gold bounding boxes and segmentation mask overlays, along with per-image JSON/NPZ exports and a final IoU/Dice/mAP@0.5 comparison table. MIMIC-CXR images and model weights are not included; access requires a PhysioNet credentialed account and the relevant data use agreements.

---

## Target Regions (15)

Right Lung, Left Lung, Cardiac Silhouette, Mediastinum, Trachea,
Right/Left Upper Lung Zone, Right/Left Mid Lung Zone, Right/Left Lower Lung Zone,
Right/Left Hilar Region, Right/Left Costophrenic Angle.

---

## Models

### 1. Grounding DINO + MedSAM — `grounding_dino_medsam.py`
**Papers:** [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

Grounding DINO is an open-vocabulary object detector trained on natural images (COCO, Objects365). It is included as a **negative baseline** — it has zero chest X-ray training and no concept of normal anatomy. Empirically, all 15 regions produce near-identical full-image boxes, giving near-zero IoU. This establishes the floor for zero-shot natural-image detectors and provides a clean contrast with domain-specific models. MedSAM (SAM fine-tuned on 1.5M medical image-mask pairs) is applied on top of each detected box to produce segmentation masks.

**Conda env:** `gdino` · **VRAM:** ~0.6GB · **Access:** Open weights

```bash
conda create -n gdino python=3.10 -y
conda activate gdino
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_grounding_dino.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
python grounding_dino_medsam.py
```

---

### 2. BioViL-T + MedSAM — `biovilt_medsam.py`
**Papers:** [BioViL-T (Bannur et al., CVPR 2023)](https://arxiv.org/abs/2301.04558) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

BioViL-T (Biomedical Vision-Language with Temporal) is a Microsoft model pretrained on MIMIC-CXR image-report pairs using contrastive learning. Unlike a detector, it produces a **similarity heatmap** between each image patch and a text prompt. A bounding box is derived from this heatmap by thresholding at a configurable percentile and taking the largest connected component — the standard approach used in the MS-CXR benchmark. Fixes Grounding DINO's laterality-marker confusion (mediastinum no longer locks onto the corner marker), but struggles with fine-grained zone differentiation. Note: the `hi-ml-multimodal` package is from a GitHub repo archived in Nov 2025; a compatibility patch for `batch_encode_plus` is included in the script.

**Conda env:** `biovilt` · **VRAM:** ~2GB · **Access:** Open weights

```bash
conda create -n biovilt python=3.10 -y
conda activate biovilt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_biovilt.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
# tune PERCENTILE (default 90) if boxes are too large/small
python biovilt_medsam.py
```

---

### 3. RadVLM + MedSAM — `radvlm_medsam.py`
**Papers:** [RadVLM (Deperrois et al., 2025)](https://arxiv.org/abs/2502.03333) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

RadVLM is a 7B multitask conversational VLM built on LLaVA-OneVision, instruction-tuned on 1M+ chest X-ray image-instruction pairs including explicit anatomical grounding on Chest ImaGenome. It is the **primary model** of this benchmark — the only one trained with direct box supervision on the exact 15-region anatomical grounding task. Reports mAP@0.5 of 85.3% on anatomical grounding in its paper. Outputs bounding boxes as normalized coordinates in free text; the script parses these with a multi-pattern regex and scales to pixel coordinates.

⚠️ **Weights require separate PhysioNet access:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0/). Download weights, set `RADVLM_PATH` in the script.

**Conda env:** `radvlm` · **VRAM:** ~16GB · **Access:** PhysioNet (credentialed)

```bash
conda create -n radvlm python=3.10 -y
conda activate radvlm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_radvlm.txt
# edit RADVLM_PATH, GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script
# set MAX_IMAGES=10 for first run
python radvlm_medsam.py
```

---

### 4. MAIRA-2 + MedSAM — `maira2_medsam.py`
**Papers:** [MAIRA-2 (Bannur et al., 2024)](https://arxiv.org/abs/2406.04449) · [MedSAM (Ma et al., 2024)](https://arxiv.org/abs/2304.12306)

MAIRA-2 is a Microsoft model combining a radiology-specific image encoder (Rad-DINO) with a Vicuna-7B LLM, trained for grounded radiology report generation. Its phrase grounding was primarily trained on **pathology findings** (e.g. "pleural effusion", "cardiomegaly") rather than normal anatomy names, so mAP on this benchmark's 15 anatomical regions is expected to be lower than RadVLM. Box coordinates are output relative to MAIRA-2's internally cropped 518×518 view; `adjust_box_for_original_image_size` is applied to convert to original pixel coordinates.

⚠️ **Gated HF access required:** visit [huggingface.co/microsoft/maira-2](https://huggingface.co/microsoft/maira-2), click "Agree and access repository", then run `huggingface-cli login`.

**Conda env:** `maira2` · **VRAM:** ~16GB · **Access:** HuggingFace checkbox

```bash
conda create -n maira2 python=3.10 -y
conda activate maira2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_maira2.txt
huggingface-cli login   # paste HF token; only needed once
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
python maira2_medsam.py
```

---

### 5. CheXagent + MedSAM — `chexagent_medsam.py`
**Paper:** [CheXagent (Chen et al., 2024)](https://arxiv.org/abs/2401.12208)

CheXagent-8b is a Stanford AIMI foundation model instruction-tuned on 28 chest X-ray datasets via their CheXinstruct dataset. Like MAIRA-2, its grounding was evaluated primarily on pathology findings, so normal anatomy mAP is expected to be weak. The script uses the official HF model card inference format (`USER: <s>... ASSISTANT: <s>`) with a phrase grounding prompt. Box output is parsed from free text with a multi-pattern regex. Raw model responses are printed for the first image so the output format can be verified and the parser adjusted if needed.

**Conda env:** `chexagent` · **VRAM:** ~17.5GB · **Access:** Open weights

```bash
conda create -n chexagent python=3.10 -y
conda activate chexagent
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements_chexagent.txt
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
# check printed raw responses on first run to verify box parsing format
python chexagent_medsam.py
```

---

### 6. BiomedParse — `biomedparse_seg.py`
**Paper:** [BiomedParse (Zhao et al., Nature Methods 2025)](https://arxiv.org/abs/2405.12971)

BiomedParse is a Microsoft foundation model for joint segmentation, detection, and recognition across 9 imaging modalities including chest X-ray. It is structurally different from the other models: it **outputs masks directly from text prompts** with no intermediate bounding box step and no MedSAM needed. All 15 regions are processed in a single batched inference call. A bounding box is derived from each predicted mask for cross-model comparison, but Dice is the primary metric. Published in Nature Methods 2025.

⚠️ **Special setup:** BiomedParse uses custom classes not in any pip package. The script must be placed and run from inside the cloned repo. Follow the setup order exactly.

**Conda env:** `biomedparse` · **Python: 3.9.19** · **VRAM:** ~1.5GB · **Access:** Open weights

```bash
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse
conda create -n biomedparse python=3.9.19 -y
conda activate biomedparse
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install -r assets/requirements/requirements.txt
pip install -r requirements_biomedparse.txt   # place this file inside BiomedParse/
# place biomedparse_seg.py inside BiomedParse/, then:
# edit GOLD_CSV, IMAGE_DIR, OUTPUT_DIR at top of script, set MAX_IMAGES=10 first
python biomedparse_seg.py
```

---

## Evaluation

After all (or some) model scripts have completed, run:

```bash
conda activate radvlm   # any env with pandas + matplotlib works
python evaluate.py
```

Outputs saved to `outputs/evaluation/`:

| File | Contents |
|---|---|
| `per_region_iou_table.csv` | Main results table — mean IoU per region per model |
| `overall_summary.csv` | Per-model detection rate, mean IoU, mAP@0.5 |
| `map_by_model.png` | mAP@0.5 bar chart with RadVLM 85.3% reference line |
| `per_region_iou_by_model.png` | Grouped bar chart per region, all models |
| `iou_heatmap.png` | Color-coded IoU grid — green = high, red = low |
| `per_region_dice_table.csv` | Dice scores (BiomedParse only) |

`evaluate.py` handles missing model results gracefully — run it incrementally as each model finishes rather than waiting for all six.

---

## Conda Environment Summary

| Script | Env | Python | transformers |
|---|---|---|---|
| `grounding_dino_medsam.py` | `gdino` | 3.10 | >=4.38 |
| `biovilt_medsam.py` | `biovilt` | 3.10 | >=4.30,<4.40 |
| `radvlm_medsam.py` | `radvlm` | 3.10 | ==4.46.0 |
| `maira2_medsam.py` | `maira2` | 3.10 | >=4.48,<4.52 |
| `chexagent_medsam.py` | `chexagent` | 3.10 | >=4.35,<4.47 |
| `biomedparse_seg.py` | `biomedparse` | **3.9.19** | (in repo) |

> **Note:** Each model needs its own conda environment because they require conflicting `transformers` versions. Do not mix them.

---

## Data Access

All scripts require:
- **MIMIC-CXR images:** [physionet.org/content/mimic-cxr](https://physionet.org/content/mimic-cxr/2.0.0/) — requires PhysioNet credentialed access
- **Chest ImaGenome gold annotations:** [physionet.org/content/chest-imagenome](https://physionet.org/content/chest-imagenome/1.0.0/) — requires PhysioNet credentialed access
- **RadVLM weights:** [physionet.org/content/radvlm-model/1.0.0](https://physionet.org/content/radvlm-model/1.0.0/) — requires separate PhysioNet access request

Neither images, annotations, nor model weights may be redistributed publicly per PhysioNet data use agreements.