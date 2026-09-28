#!/bin/bash
#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-12:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="chex"
#SBATCH --mem 24G

set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate chex
export PYTHONNOUSERSITE=1

export CXR_CONFIG=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/config.yaml
export CXR_REPO_ROOT=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

# ChEX's own modules use non-package-relative imports (from dataset.image_transform
# import ...), same as upstream src/evaluate.py -- must run with cwd = chex/src.
cp /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/cxr_common.py \
   /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/models/chex_medsam.py \
   /home/manik/pranjali/Aditya_project/chex/src/

cd /home/manik/pranjali/Aditya_project/chex/src
python chex_medsam.py

# Aggregation deliberately does NOT run here -- see job_chexagent.sh for why.
