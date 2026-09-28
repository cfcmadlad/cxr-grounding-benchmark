#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 1-06:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="medgemma15"
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

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
export PYTHONPATH="/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed:${PYTHONPATH:-}"
python models/medgemma15_medsam.py

# Aggregation deliberately does NOT run here -- see job_chexagent.sh for why.
