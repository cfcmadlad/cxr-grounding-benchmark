#!/bin/bash
# setup.sh
# --------
# Builds all 6 conda environments for the CXR grounding benchmark on
# Sharanga (Rocky Linux 8.10, Slurm, A100/CUDA 12.4).
#
# MUST be run inside an interactive GPU Slurm session, NOT on the login
# node, because:
#   - compute nodes have no internet access, so this needs to run somewhere
#     that does (check with your cluster admins which nodes have outbound
#     access -- typically a designated data-transfer/login node, not a
#     compute node) OR all wheels/packages must already be mirrored locally
#   - `pip install` for the *-8b/-7b models briefly needs GPU-visible CUDA
#     libraries to resolve the correct torch/xformers wheel variants
#
#   srun -p gpu --gres=gpu:1 --pty bash
#   bash setup.sh                      # build every env
#   bash setup.sh gdino biovilt        # build only specific envs
#
# Safe to re-run: each `conda create` is skipped if the env already exists.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

ALL_ENVS=(gdino biovilt radvlm maira2 chexagent biomedparse)
ENVS_TO_BUILD=("$@")
if [ ${#ENVS_TO_BUILD[@]} -eq 0 ]; then
    ENVS_TO_BUILD=("${ALL_ENVS[@]}")
fi

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

echo "=================================================================="
echo "CXR Grounding Benchmark -- environment setup"
echo "Envs to build: ${ENVS_TO_BUILD[*]}"
echo "=================================================================="

conda_env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

contains() {
    local needle="$1"; shift
    for x in "$@"; do [ "$x" = "$needle" ] && return 0; done
    return 1
}

# ── 1. Grounding DINO + MedSAM ────────────────────────────────────────────────
if contains gdino "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [1/6] gdino ──"
    conda_env_exists gdino || conda create -n gdino python=3.10 -y
    conda run -n gdino pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    conda run -n gdino pip install -r requirements_grounding_dino.txt
    echo "gdino env ready."
fi

# ── 2. BioViL-T + MedSAM ──────────────────────────────────────────────────────
if contains biovilt "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [2/6] biovilt ──"
    conda_env_exists biovilt || conda create -n biovilt python=3.10 -y
    conda run -n biovilt pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    conda run -n biovilt pip install -r requirements_biovilt.txt
    echo "biovilt env ready."
fi

# ── 3. RadVLM + MedSAM ────────────────────────────────────────────────────────
if contains radvlm "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [3/6] radvlm ──"
    conda_env_exists radvlm || conda create -n radvlm python=3.10 -y
    conda run -n radvlm pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    conda run -n radvlm pip install -r requirements_radvlm.txt
    echo "radvlm env ready."
    echo "NOTE: RadVLM weights require separate PhysioNet credentialed access:"
    echo "      https://physionet.org/content/radvlm-model/1.0.0/"
    echo "      Download them to a compute-node-visible path and set RADVLM_PATH"
    echo "      in config.yaml before submitting."
fi

# ── 4. MAIRA-2 + MedSAM ───────────────────────────────────────────────────────
if contains maira2 "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [4/6] maira2 ──"
    conda_env_exists maira2 || conda create -n maira2 python=3.10 -y
    conda run -n maira2 pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    conda run -n maira2 pip install -r requirements_maira2.txt
    echo "maira2 env ready."
    echo "NOTE: MAIRA-2 requires one-time gated HF access:"
    echo "        1. Visit https://huggingface.co/microsoft/maira-2 and click"
    echo "           'Agree and access repository' (needs internet -- do this"
    echo "           from a browser, not the cluster)."
    echo "        2. Run: conda run -n maira2 huggingface-cli login"
    echo "           and paste your HF token. Only needed once per user account."
fi

# ── 5. CheXagent + MedSAM ─────────────────────────────────────────────────────
if contains chexagent "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [5/6] chexagent ──"
    conda_env_exists chexagent || conda create -n chexagent python=3.10 -y
    conda run -n chexagent pip install torch torchvision --index-url "$TORCH_INDEX_URL"
    conda run -n chexagent pip install -r requirements_chexagent.txt
    echo "chexagent env ready."
fi

# ── 6. BiomedParse ────────────────────────────────────────────────────────────
if contains biomedparse "${ENVS_TO_BUILD[@]}"; then
    echo; echo "── [6/6] biomedparse ──"
    BIOMEDPARSE_REPO_DIR="${BIOMEDPARSE_REPO_DIR:-$HOME/BiomedParse}"

    if [ ! -d "$BIOMEDPARSE_REPO_DIR" ]; then
        git clone https://github.com/microsoft/BiomedParse.git "$BIOMEDPARSE_REPO_DIR"
    else
        echo "BiomedParse repo already present at $BIOMEDPARSE_REPO_DIR, skipping clone."
    fi

    conda_env_exists biomedparse || conda create -n biomedparse python=3.9.19 -y
    conda run -n biomedparse conda install -y -c pytorch -c nvidia \
        pytorch torchvision torchaudio pytorch-cuda=12.4

    ( cd "$BIOMEDPARSE_REPO_DIR" && conda run -n biomedparse pip install -r assets/requirements/requirements.txt )
    conda run -n biomedparse pip install -r requirements_biomedparse.txt

    # biomedparse_seg.py and cxr_common.py must live inside the BiomedParse
    # repo checkout for its custom `modeling`/`utilities` imports to resolve.
    cp biomedparse_seg.py cxr_common.py "$BIOMEDPARSE_REPO_DIR/"
    echo "Copied biomedparse_seg.py and cxr_common.py into $BIOMEDPARSE_REPO_DIR"
    echo "biomedparse env ready."
fi

echo
echo "=================================================================="
echo "Setup complete for: ${ENVS_TO_BUILD[*]}"
echo "Next: edit paths in config.yaml (GOLD_CSV / IMAGE_DIR / OUTPUT_DIR /"
echo "MAX_IMAGES / weight paths), verify with 'python setup_check.py', then"
echo "submit with: sbatch job_<model>.sh"
echo "=================================================================="
