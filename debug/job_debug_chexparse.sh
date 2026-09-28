#!/bin/bash
#SBATCH -p gpu_v100_2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-00:15
#SBATCH -o slurm_debug.%j.out
#SBATCH -e slurm_debug.%j.err
#SBATCH --job-name="dbgchexparse"
#SBATCH --mem 8G

set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate gdino

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
python debug/debug_chexagent_parse.py
