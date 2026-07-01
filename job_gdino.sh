#!/bin/bash
#SBATCH --job-name=cxr_gdino
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

# ── EDIT THESE BEFORE SUBMITTING ─────────────────────────────────────────────
GOLD_CSV="/path/to/gold_bbox.csv"
IMAGE_DIR="/path/to/mimic_cxr_images"
OUTPUT_DIR="./outputs/grounding_dino"
MAX_IMAGES=10   # set to "" for a full run once this test run looks correct
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export GOLD_CSV IMAGE_DIR OUTPUT_DIR MAX_IMAGES

echo "Job started on $(hostname) at $(date), SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gdino

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUTPUT_DIR"

python grounding_dino_medsam.py

echo "Job finished at $(date)"
