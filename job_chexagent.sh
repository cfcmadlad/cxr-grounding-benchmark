#!/bin/bash
#SBATCH --job-name=cxr_chexagent
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

# ── EDIT THESE BEFORE SUBMITTING ─────────────────────────────────────────────
GOLD_CSV="/path/to/gold_bbox.csv"
IMAGE_DIR="/path/to/mimic_cxr_images"
OUTPUT_DIR="./outputs/chexagent"
MAX_IMAGES=10   # set to "" for a full run once this test run looks correct
PRINT_RAW_RESPONSES=1   # keep on for the first run to verify box-parsing format
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export GOLD_CSV IMAGE_DIR OUTPUT_DIR MAX_IMAGES PRINT_RAW_RESPONSES

echo "Job started on $(hostname) at $(date), SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chexagent

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUTPUT_DIR"

python chexagent_medsam.py

echo "Job finished at $(date)"
