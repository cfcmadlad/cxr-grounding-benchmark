#!/bin/bash
# RadVLM on all 29 Chest ImaGenome regions (optimised: shared image-prefix KV cache,
# batched region decoding, no MedSAM/PNGs). See models/radvlm_29regions.py.
#
# The partition is chosen at submit time, e.g.
#     sbatch -p gpu_a100_8 jobs/job_radvlm29.sh
#     sbatch -p gpu_h100_4 jobs/job_radvlm29.sh
# It is safe to submit to several partitions at once: a lock in the output dir makes
# every job after the first exit immediately, and a finished run is never repeated.
# Needs an 80 GB GPU for the fastest setting (A100-80GB / H100); smaller GPUs fall
# back to smaller batches automatically.
#
# Smoke test (8 images, separate output dir):
#     CXR_CONFIG=$PWD/config_smoketest.yaml sbatch -p gpu_h100_4 --export=ALL jobs/job_radvlm29.sh
#SBATCH -p gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-04:00
#SBATCH -o slurm.radvlm29.%j.out
#SBATCH -e slurm.radvlm29.%j.err
#SBATCH --job-name="radvlm29"
#SBATCH --mem 48G

# NOTE: must stay BELOW the #SBATCH block -- sbatch stops parsing directives at the
# first executable line.
set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate radvlm

# ~/.local must not shadow the env's pinned torch/numpy.
export PYTHONNOUSERSITE=1

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# All weights are local; fail loudly instead of hanging if something tries the network
# (compute nodes have no reliable internet).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

REPO=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
cd "$REPO"

echo "host: $(hostname)   job: ${SLURM_JOB_ID:-none}   config: ${CXR_CONFIG:-$REPO/config.yaml}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python models/radvlm_29regions.py
