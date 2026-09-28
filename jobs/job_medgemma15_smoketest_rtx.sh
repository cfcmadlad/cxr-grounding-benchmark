#!/bin/bash
#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-01:00
#SBATCH -o slurm_mg15_full_smoke.%j.out
#SBATCH -e slurm_mg15_full_smoke.%j.err
#SBATCH --job-name="mg15fsmoke"
#SBATCH --mem 40G

set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate gdino
export PYTHONNOUSERSITE=1

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# HF_TOKEN must be set in your own shell environment before submitting this job
# (never hardcode a real token in a script that goes into version control).
# export HF_TOKEN=<your-token-here>   # or: huggingface-cli login
export CXR_CONFIG=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/config_smoketest.yaml

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
export PYTHONPATH="/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed:${PYTHONPATH:-}"
python models/medgemma15_medsam.py
