#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:25
#SBATCH -o slurm_debug.%j.out
#SBATCH -e slurm_debug.%j.err
#SBATCH --job-name="dbgmaira2"
#SBATCH --mem 40G

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate maira2

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# Weights are already in the cache; stay offline so no token is needed here.
export HF_HUB_OFFLINE=1

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
python debug_maira_v2.py
echo "DEBUG EXIT CODE: $?"
