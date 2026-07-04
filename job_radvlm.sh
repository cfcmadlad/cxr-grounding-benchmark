#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-23:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="radvlm"
#SBATCH --mem 40G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate radvlm

export HF_HOME=/home/manik/cxr-grounding-benchmark/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/cxr-grounding-benchmark/.cache/huggingface

cd /home/manik/cxr-grounding-benchmark

python radvlm_medsam.py

cd /home/manik/cxr-grounding-benchmark
python evaluate.py
python compare_results.py
