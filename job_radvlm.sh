#!/bin/bash
#SBATCH --job-name=cxr_radvlm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

# ── EDIT THESE BEFORE SUBMITTING ─────────────────────────────────────────────
RADVLM_PATH="/path/to/radvlm/weights"   # downloaded from PhysioNet, see README
GOLD_CSV="/path/to/gold_bbox.csv"
IMAGE_DIR="/path/to/mimic_cxr_images"
OUTPUT_DIR="./outputs/radvlm"
MAX_IMAGES=10   # set to "" for a full run once this test run looks correct
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export RADVLM_PATH GOLD_CSV IMAGE_DIR OUTPUT_DIR MAX_IMAGES

echo "Job started on $(hostname) at $(date), SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate radvlm

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUTPUT_DIR"

python radvlm_medsam.py

echo "Job finished at $(date)"
