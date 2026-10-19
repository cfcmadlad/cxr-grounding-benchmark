#!/bin/bash
# CPU-only: draw gold (green) + RadVLM (red) boxes on all 959 images x 29 regions.
# Output: outputs/radvlm29/overlays/<region>/<image_id>.jpg   (re-runnable, skips existing)
#SBATCH -p big_compute
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=128
#SBATCH -t 0-01:00
#SBATCH --mem 128G
#SBATCH -o slurm.radvlm29_overlays.%j.out
#SBATCH --job-name="radvlm29_ov"

set -euo pipefail
source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate radvlm
export PYTHONNOUSERSITE=1
cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
python -u models/radvlm29_overlays.py
