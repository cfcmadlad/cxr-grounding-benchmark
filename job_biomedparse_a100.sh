#!/bin/bash
#SBATCH -p gpu_a100_8
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 0-23:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="biomedparse"
#SBATCH --mem 8G

# Fail the job on the first error, on an unset variable, and on any failure
# inside a pipeline. Without this Slurm only reports the LAST command's exit
# code, so a crashed model script still shows up as COMPLETED 0:0.
# NOTE: this must stay BELOW the #SBATCH block -- sbatch stops parsing #SBATCH
# directives at the first executable line.
set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate biomedparse2

# The biomedparse2 env is python 3.9, and ~/.local/lib/python3.9/site-packages
# sits AHEAD of the env on sys.path. Without this it would shadow the env's
# pinned torch 2.4.0/cu124 and numpy 1.26.4 with torch 2.8.0/cu128 + numpy 2.0.2.
export PYTHONNOUSERSITE=1

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# HF_TOKEN must be set in your own shell environment before submitting this job
# (never hardcode a real token in a script that goes into version control).
# export HF_TOKEN=<your-token-here>   # or: huggingface-cli login
# BiomedParse runs from inside its own repo, so make the benchmark's shared
# module and config discoverable from there.
export PYTHONPATH=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed:${PYTHONPATH:-}
export CXR_CONFIG=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed/config.yaml
export CXR_REPO_ROOT=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

# The BiomedParse custom classes are only importable with the repo dir as the
# script dir, so copy the script (and shared module/config as a fallback) in.
cp cxr_common.py config.yaml biomedparse_seg.py \
   /home/manik/pranjali/Aditya_project/BiomedParse/

cd /home/manik/pranjali/Aditya_project/BiomedParse
echo "DEBUG: HF_TOKEN is set to: ${HF_TOKEN:0:10}... (length ${#HF_TOKEN})"
python biomedparse_seg.py

# Aggregation deliberately does NOT run here any more. Having every job run
# evaluate.py at the end raced against models that were still writing their
# results_summary.csv. Run ./run_aggregation.sh manually, once, after all
# models are confirmed complete.
