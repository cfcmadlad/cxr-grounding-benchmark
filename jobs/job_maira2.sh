#!/bin/bash
#SBATCH -p gpu_h100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -t 0-23:00
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --mail-type=ALL
#SBATCH --job-name="maira2"
#SBATCH --mem 40G

# Fail the job on the first error, on an unset variable, and on any failure
# inside a pipeline. Without this Slurm only reports the LAST command's exit
# code, so a crashed model script still shows up as COMPLETED 0:0.
# NOTE: this must stay BELOW the #SBATCH block -- sbatch stops parsing #SBATCH
# directives at the first executable line.
set -euo pipefail

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate maira2

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface
# HF_TOKEN must be set in your own shell environment before submitting this job
# (never hardcode a real token in a script that goes into version control).
# export HF_TOKEN=<your-token-here>   # or: huggingface-cli login

cd /home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed

export PYTHONPATH="/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed:${PYTHONPATH:-}"
python models/maira2_medsam.py

# Aggregation deliberately does NOT run here any more. Having every job run
# evaluate.py at the end raced against models that were still writing their
# results_summary.csv. Run ./run_aggregation.sh manually, once, after all
# models are confirmed complete.
