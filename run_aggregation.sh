#!/bin/bash
# run_aggregation.sh
# -----------------------------------------------------------------------------
# Builds the aggregate results (results/results.json, results/comparison_table.md
# and outputs/evaluation/*) from whatever per-model results_summary.csv files are
# on disk.
#
# Run this MANUALLY, exactly once, after every model job has finished. It used to
# run at the tail of every job_*.sh, which meant a job finishing early would read
# another model's half-written CSV and record those partial numbers as "complete"
# (that is how CheXagent was once recorded at N=5384 mid-run).
#
# Usage (login node is fine -- this is CPU-only pandas/matplotlib work):
#     ./run_aggregation.sh
#
# It refuses to run if any model job is still queued or running.
# -----------------------------------------------------------------------------

set -euo pipefail

REPO=/home/manik/pranjali/Aditya_project/cxr-grounding-benchmark-fixed
cd "$REPO"

# --- refuse to aggregate while benchmark jobs are still in flight ------------
LIVE=$(squeue -u "$USER" -h -o '%j' \
       | grep -E '^(gdino|biovilt|chexagent|maira2|biomedparse|radvlm)$' || true)
if [ -n "$LIVE" ]; then
    echo "REFUSING TO RUN: these benchmark jobs are still queued/running:" >&2
    echo "$LIVE" | sed 's/^/  /' >&2
    echo "Wait for them to finish, then re-run." >&2
    exit 1
fi

# --- show what is about to be aggregated ------------------------------------
echo "Per-model results_summary.csv files found:"
for m in grounding_dino biovilt maira2 chexagent biomedparse radvlm; do
    f="outputs/$m/results_summary.csv"
    if [ -f "$f" ]; then
        printf '  %-16s %8d data rows   (mtime %s)\n' \
            "$m" "$(($(wc -l < "$f") - 1))" "$(stat -c %y "$f" | cut -d. -f1)"
    else
        printf '  %-16s MISSING\n' "$m"
    fi
done
echo

source /apps/spack/opt/spack/linux-rocky8-zen2/gcc-11.2.0/anaconda3-2022.05-od5lltp3ijbed4uvsrut4fifckrgsbbf/etc/profile.d/conda.sh
conda activate gdino

export HF_HOME=/home/manik/pranjali/Aditya_project/.cache/huggingface
export TRANSFORMERS_CACHE=/home/manik/pranjali/Aditya_project/.cache/huggingface

python evaluate.py
python compare_results.py

echo
echo "Aggregation complete."
echo "  results/results.json"
echo "  results/comparison_table.md"
echo "  outputs/evaluation/"
