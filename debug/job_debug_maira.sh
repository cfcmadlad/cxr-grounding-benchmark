#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:20
#SBATCH -o slurm_debug.%j.out
#SBATCH -e slurm_debug.%j.err
#SBATCH --job-name="debugmaira"
#SBATCH --mem 40G
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate maira2
export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# HF_TOKEN must be set in your own shell environment before submitting this job
# (never hardcode a real token in a script that goes into version control).
# export HF_TOKEN=<your-token-here>   # or: huggingface-cli login
cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
python debug/debug_maira_ground.py
