#!/usr/bin/env bash
# setup_katana_env.sh
# Baseline - Reflexion (Shinn et al., 2023)
# Author : Gurdiraj Bal (z5386590)
#
# One-time setup on a UNSW Katana LOGIN node (has internet). Two stages:
#   Stage A (venv):     builds a Python 3.11 venv on scratch with vLLM + the
#                       project's runtime deps. ~12GB, fits current scratch.
#   Stage B (weights):  downloads the gated Llama-3.3-70B-Instruct (~140GB bf16)
#                       to scratch. Needs (1) HF access to the gated repo,
#                       (2) an HF token, (3) enough free scratch (raise quota
#                       first - default 128GB is not enough for bf16).
#
# Usage:
#   bash setup_katana_env.sh venv       # Stage A only (safe to run now)
#   HF_TOKEN=hf_xxx bash setup_katana_env.sh weights   # Stage B (after access+quota)
#   HF_TOKEN=hf_xxx bash setup_katana_env.sh all        # both

set -euo pipefail

SCRATCH="/srv/scratch/${USER}"
VENV_PATH="${VENV_PATH:-${SCRATCH}/reflexion_venv}"
MODEL_PATH="${MODEL_PATH:-${SCRATCH}/models/Llama-3.3-70B-Instruct}"
MODEL_REPO="${MODEL_REPO:-meta-llama/Llama-3.3-70B-Instruct}"
export HF_HOME="${HF_HOME:-${SCRATCH}/hf_cache}"

module load python/3.11.3 2>/dev/null || true
module load cuda/12.5.0 2>/dev/null || true

stage_venv() {
    echo "=== Stage A: venv at ${VENV_PATH} ==="
    if [[ ! -d "${VENV_PATH}" ]]; then
        python3 -m venv "${VENV_PATH}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    pip install --upgrade pip
    # vLLM pulls a matching torch; openai=client for the katana backend;
    # evalplus=get_mbpp_plus/get_human_eval_plus + the sandbox test data.
    pip install vllm openai "evalplus"
    echo "venv ready:"; python -c "import vllm, openai, evalplus; print('vllm', vllm.__version__)"
}

stage_weights() {
    echo "=== Stage B: download ${MODEL_REPO} -> ${MODEL_PATH} ==="
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "ERROR: HF_TOKEN not set. Get a token (read scope) from a HF account" >&2
        echo "       that has been granted access to ${MODEL_REPO}, then re-run:" >&2
        echo "       HF_TOKEN=hf_xxx bash setup_katana_env.sh weights" >&2
        exit 1
    fi
    # Guard against the known-too-small default scratch: bf16 needs ~140GB.
    avail_gb=$(df -BG --output=avail "${SCRATCH}" 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "scratch free: ${avail_gb}GB"
    if [[ -n "${avail_gb}" && "${avail_gb}" -lt 160 ]]; then
        echo "WARNING: <160GB free on scratch - bf16 weights (~140GB) plus venv" >&2
        echo "         may not fit. Raise your scratch quota before continuing." >&2
        read -r -p "Continue anyway? [y/N] " ans
        [[ "${ans}" == "y" || "${ans}" == "Y" ]] || exit 1
    fi
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    pip install -q "huggingface_hub[cli]"
    mkdir -p "${MODEL_PATH}"
    HF_TOKEN="${HF_TOKEN}" huggingface-cli download "${MODEL_REPO}" \
        --local-dir "${MODEL_PATH}" \
        --exclude "original/*" "*.pth"   # skip the redundant PyTorch-consolidated copy
    echo "weights staged at ${MODEL_PATH}"
    du -sh "${MODEL_PATH}"
}

case "${1:-all}" in
    venv)    stage_venv ;;
    weights) stage_weights ;;
    all)     stage_venv; stage_weights ;;
    *) echo "usage: $0 {venv|weights|all}" >&2; exit 2 ;;
esac
