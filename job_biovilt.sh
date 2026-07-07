#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-23:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="biovilt"
#SBATCH --mem 8G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate biovilt

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-main

python biovilt_medsam.py

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-main
python evaluate.py
python compare_results.py
