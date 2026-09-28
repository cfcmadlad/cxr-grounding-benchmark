#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-01:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="radvlm_smoke"
#SBATCH --mem 40G

set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate radvlm

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
export CXR_CONFIG=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/config_smoketest.yaml

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

export PYTHONPATH="/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed:${PYTHONPATH:-}"
python models/radvlm_medsam.py
