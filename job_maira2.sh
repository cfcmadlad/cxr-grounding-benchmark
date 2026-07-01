#!/bin/bash
#SBATCH --job-name=cxr_maira2
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
OUTPUT_DIR="./outputs/maira2"
MAX_IMAGES=10   # set to "" for a full run once this test run looks correct
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export GOLD_CSV IMAGE_DIR OUTPUT_DIR MAX_IMAGES

echo "Job started on $(hostname) at $(date), SLURM_JOB_ID=${SLURM_JOB_ID:-N/A}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# MAIRA-2 needs a gated-repository HF token available in this shell.
# huggingface-cli login (done once in setup.sh) stores it under
# ~/.cache/huggingface/token, which is picked up automatically -- no
# extra export needed here unless you use a non-default HF_HOME.

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate maira2

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUTPUT_DIR"

python maira2_medsam.py

echo "Job finished at $(date)"
