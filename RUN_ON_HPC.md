# Running the CXR Grounding Benchmark on Sharanga HPC (BITS Pilani Hyderabad)

Target environment: Rocky Linux 8.10, Slurm, A100 GPUs, CUDA 12.4, conda
available on login. Compute nodes have **no internet access** -- all
package installation must happen where the internet-accessible node is
(consult your cluster's docs/admins for which node that is; typically a
login or data-transfer node, never inside an `sbatch` job on a compute node).

---

## 1. Clone and initial setup

```bash
git clone https://github.com/cfcmadlad/cxr-grounding-benchmark.git
cd cxr-grounding-benchmark
```

Get PhysioNet-credentialed access first (this benchmark cannot run without
it) -- see the "Data Access" section of `README.md`:
- Chest ImaGenome + MIMIC-CXR images and gold annotations
- RadVLM weights (separate PhysioNet request)
- MAIRA-2 gated HuggingFace checkbox (`huggingface.co/microsoft/maira-2`)

Transfer the MIMIC-CXR/Chest ImaGenome images and gold annotation CSV onto
cluster storage now (e.g. via `rsync`/`scp` from a machine with PhysioNet
access), since compute nodes can't download them later.

---

## 2. Build the conda environments (interactive Slurm session)

`setup.sh` must run somewhere with internet access AND with a GPU visible
(so `pip install torch --index-url .../cu124` resolves correctly), which
usually means the login node needs no GPU only if it can still resolve
CUDA wheels correctly. If unsure, request a short interactive GPU session
on a node that also has outbound internet:

```bash
srun -p gpu --gres=gpu:1 --time=02:00:00 --pty bash

bash setup.sh                 # builds all 6 envs (~30-60 min total)
# or build one at a time, e.g. while iterating:
bash setup.sh gdino
bash setup.sh biovilt
bash setup.sh biomedparse
```

`setup.sh` is safe to re-run -- it skips `conda create` for envs that
already exist and simply re-runs `pip install` (idempotent) for the rest.

**Two manual steps setup.sh cannot do for you** (each needs a browser,
which compute/login nodes won't have):

```bash
# RadVLM weights (after PhysioNet access is approved):
#   download from https://physionet.org/content/radvlm-model/1.0.0/
#   to a path visible from compute nodes, then set RADVLM_PATH in job_radvlm.sh

# MAIRA-2 gated access (one-time per HPC user account):
#   1. From a browser: visit https://huggingface.co/microsoft/maira-2
#      and click "Agree and access repository"
#   2. From the cluster: conda run -n maira2 huggingface-cli login
#      and paste your HF token
```

---

## 3. Edit each job script's config block

Every `job_<model>.sh` has an editable block near the top:

```bash
GOLD_CSV="/path/to/gold_bbox.csv"
IMAGE_DIR="/path/to/mimic_cxr_images"
OUTPUT_DIR="./outputs/<model>"
MAX_IMAGES=10
```

`job_radvlm.sh` additionally has `RADVLM_PATH`. `job_biomedparse.sh`
additionally has `BIOMEDPARSE_REPO_DIR` (defaults to `$HOME/BiomedParse`,
matching where `setup.sh` clones it).

**Leave `MAX_IMAGES=10` for the first submission of every model** -- this
is a smoke test to confirm the environment, paths, and (for the VLM
scripts) the box-parsing regex actually work on your specific checkpoint
before committing GPU-hours to a full run. Once a test run's
`results_summary.csv` and per-region PNGs look sane, edit `MAX_IMAGES=""`
(empty -> processes every image) and resubmit.

---

## 4. Submit jobs

```bash
sbatch job_gdino.sh
sbatch job_biovilt.sh
sbatch job_biomedparse.sh
sbatch job_radvlm.sh        # needs RadVLM weights + 32G mem, ~16GB VRAM
sbatch job_maira2.sh        # needs HF gated access + 32G mem, ~16GB VRAM
sbatch job_chexagent.sh     # 32G mem, ~17.5GB VRAM
```

All 6 are independent and can run concurrently (each has its own conda
env and output directory) as long as your account has enough concurrent
GPU allocation.

---

## 5. Monitor jobs

```bash
squeue -u $USER                          # see queued/running jobs
squeue -u $USER --start                  # estimated start times if queued

# follow a specific job's live output (job ID from squeue or sbatch's return)
tail -f slurm.<jobid>.out
tail -f slurm.<jobid>.err

# for CheXagent specifically: PRINT_RAW_RESPONSES=1 prints the model's raw
# text output for the first image, so check slurm.<jobid>.out early to
# confirm box_from_response() is actually matching the output format:
grep -A2 "RAW RESPONSE" slurm.<jobid>.out
```

If a job fails, `slurm.<jobid>.err` almost always has the Python
traceback. Common early failures and fixes are in `COMPATIBILITY_REPORT.md`.

---

## 6. Run evaluation

After any subset of the 6 model jobs finishes (evaluate.py handles missing
models gracefully -- you don't need to wait for all 6):

```bash
srun -p gpu --gres=gpu:1 --pty bash    # or run on a CPU-only node/partition;
                                        # evaluate.py itself needs no GPU
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate radvlm   # any env with pandas/matplotlib/numpy works
cd cxr-grounding-benchmark
python evaluate.py
```

Outputs land in `outputs/evaluation/`:
- `per_region_iou_table.csv` -- main results table
- `overall_summary.csv` -- per-model detection rate / mean IoU / mAP@0.5
- `map_by_model.png`, `per_region_iou_by_model.png`, `iou_heatmap.png`
- `per_region_dice_table.csv` (BiomedParse)

Re-run `python evaluate.py` any time after more models finish -- it
recomputes everything from each model's `results_summary.csv` on disk.
