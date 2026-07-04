#!/bin/bash
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-23:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="biomedparse"
#SBATCH --mem 8G

source ~/anaconda3/etc/profile.d/conda.sh
conda activate biomedparse

export HF_HOME=/home/manik/cxr-grounding-benchmark/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/cxr-grounding-benchmark/.cache/huggingface
# BiomedParse runs from inside its own repo, so make the benchmark's shared
# module and config discoverable from there.
export PYTHONPATH=/home/manik/cxr-grounding-benchmark:$PYTHONPATH
export CXR_CONFIG=/home/manik/cxr-grounding-benchmark/config.yaml
export CXR_REPO_ROOT=/home/manik/cxr-grounding-benchmark

cd /home/manik/cxr-grounding-benchmark

# The BiomedParse custom classes are only importable with the repo dir as the
# script dir, so copy the script (and shared module/config as a fallback) in.
cp cxr_common.py config.yaml biomedparse_seg.py \
   /home/manik/cxr-grounding-benchmark/BiomedParse/

cd /home/manik/cxr-grounding-benchmark/BiomedParse
python biomedparse_seg.py

cd /home/manik/cxr-grounding-benchmark
python evaluate.py
python compare_results.py
