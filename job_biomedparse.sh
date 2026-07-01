#!/bin/bash
#SBATCH --job-name=cxr_biomedparse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

# ── EDIT THESE BEFORE SUBMITTING ─────────────────────────────────────────────
GOLD_CSV="/path/to/gold_bbox.csv"
IMAGE_DIR="/path/to/mimic_cxr_images"
MAX_IMAGES=10   # set to "" for a full run once this test run looks correct
BIOMEDPARSE_REPO_DIR="$HOME/BiomedParse"   # path to the cloned BiomedParse repo
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# OUTPUT_DIR is deliberately an ABSOLUTE path back into this repo's
# outputs/ folder (not "./outputs/biomedparse"), because the python
# script itself must be run from inside $BIOMEDPARSE_REPO_DIR (see below),
# and evaluate.py later expects every model's results_summary.csv under
# THIS repo's ./outputs/<model>/ regardless of where each script executed.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$REPO_DIR/outputs/biomedparse"
export GOLD_CSV IMAGE_DIR OUTPUT_DIR MAX_IMAGES

echo "Job started on $(hostname) at $(date), SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate biomedparse

# BiomedParse's custom model classes (BaseModel, build_model, ...) only
# import correctly when run from inside the cloned BiomedParse repo, so
# unlike the other 5 job scripts this one must cd there instead of into
# this repo. biomedparse_seg.py and cxr_common.py must already have been
# copied into $BIOMEDPARSE_REPO_DIR by setup.sh.
cd "$BIOMEDPARSE_REPO_DIR"
mkdir -p "$OUTPUT_DIR"

python biomedparse_seg.py

echo "Job finished at $(date)"
