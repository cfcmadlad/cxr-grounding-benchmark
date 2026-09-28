#!/bin/bash
#SBATCH -p gpu_rtx_pro_6000_6_csis_hyd
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-01:00
#SBATCH -o slurm_chex_smoke.%j.out
#SBATCH -e slurm_chex_smoke.%j.err
#SBATCH --job-name="chexsmoke"
#SBATCH --mem 24G

set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate chex
export PYTHONNOUSERSITE=1

export CXR_CONFIG=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/config_smoketest.yaml
export CXR_REPO_ROOT=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

echo "PYTHONNOUSERSITE check:"
PYTHONNOUSERSITE=1 python -c "import torch, numpy; print('torch', torch.__file__); print('numpy', numpy.__version__); print('cuda available:', torch.cuda.is_available())"

# ChEX's own modules use non-package-relative imports (from dataset.image_transform
# import ...), same as upstream src/evaluate.py -- must run with cwd = chex/src.
cp /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/cxr_common.py \
   /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/chex_medsam.py \
   /home/manik/pranjali/Aditya_project/chex/src/

cd /home/manik/pranjali/Aditya_project/chex/src
python chex_medsam.py
