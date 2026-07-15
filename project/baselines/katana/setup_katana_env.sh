#!/usr/bin/env bash
# setup_katana_env.sh
# Shared Katana infrastructure for the external baselines (Reflexion, Prochemy, ...)
# Author : Gurdiraj Bal (z5386590)
#
# One-time setup on a UNSW Katana LOGIN node (has internet). Builds a single
# shared vLLM venv + weights used by every baseline (serve_and_run.sh points at
# them). Two stages:
#   Stage A (venv):     builds a Python 3.11 venv on scratch with vLLM + the
#                       project's runtime deps. ~12GB.
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
VENV_PATH="${VENV_PATH:-${SCRATCH}/vllm_venv}"
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
    # Remove torchcodec: vLLM pulls it for multimodal *video* loading (irrelevant
    # to text code-gen), but its .so needs system FFmpeg libs absent on the compute
    # nodes. A *broken* install raises OSError from `vllm.multimodal.video`'s
    # `import torchcodec`, which slips past vLLM's `except ImportError` guard and
    # crashes `vllm serve` at startup. Uninstalling it lets the guard catch a clean
    # ModuleNotFoundError instead. (Fixing FFmpeg via modules hits a GLIBCXX conflict.)
    pip uninstall -y torchcodec 2>/dev/null || true
    echo "venv ready:"; python -c "import vllm, openai, evalplus; print('vllm', vllm.__version__)"
}

stage_weights() {
    echo "=== Stage B: download ${MODEL_REPO} -> ${MODEL_PATH} ==="
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    # NB: `huggingface-cli` is deprecated/removed in huggingface_hub >=1.0; the
    # command is now `hf`. Use the venv's hf (python 3.11, has `packaging`), NOT
    # any stale ~/.local/bin/hf (python 3.10) that may shadow it.
    HF="${VENV_PATH}/bin/hf"

    # Auth: prefer a cached login (`hf auth login`, token stored mode-600 under
    # $HF_HOME) so the raw token never needs to be passed on a command line or
    # through env. Fall back to an explicit HF_TOKEN if set. HF_HOME must match
    # between login and here (set at the top of this script) so the token is found.
    if [[ -z "${HF_TOKEN:-}" ]] && ! "${HF}" auth whoami >/dev/null 2>&1; then
        echo "ERROR: not authenticated to Hugging Face (HF_HOME=${HF_HOME})." >&2
        echo "  Recommended (token stays out of shell history / env) - run:" >&2
        echo "    source ${VENV_PATH}/bin/activate" >&2
        echo "    export HF_HOME=${HF_HOME}" >&2
        echo "    hf auth login          # paste your token" >&2
        echo "  then re-run:  bash setup_katana_env.sh weights" >&2
        echo "  (Or, less preferred: HF_TOKEN=hf_xxx bash setup_katana_env.sh weights)" >&2
        echo "  The HF account must have been granted access to ${MODEL_REPO}." >&2
        exit 1
    fi

    # Guard against too-small scratch: bf16 needs ~140GB.
    avail_gb=$(df -BG --output=avail "${SCRATCH}" 2>/dev/null | tail -1 | tr -dc '0-9')
    echo "scratch free: ${avail_gb}GB"
    if [[ -n "${avail_gb}" && "${avail_gb}" -lt 160 ]]; then
        echo "WARNING: <160GB free on scratch - bf16 weights (~140GB) plus venv" >&2
        echo "         may not fit. Raise your scratch quota before continuing." >&2
        read -r -p "Continue anyway? [y/N] " ans
        [[ "${ans}" == "y" || "${ans}" == "Y" ]] || exit 1
    fi

    mkdir -p "${MODEL_PATH}"
    # hf download picks up HF_TOKEN from env if set, else the cached login above.
    # One --exclude per pattern: the new `hf download` treats a bare trailing
    # pattern as a positional filename (and then ignores --exclude entirely).
    # Excludes skip the redundant PyTorch-consolidated copy (~140GB) so only the
    # safetensors vLLM needs are pulled - essential to fit in scratch.
    "${HF}" download "${MODEL_REPO}" \
        --local-dir "${MODEL_PATH}" \
        --exclude "original/*" --exclude "*.pth"
    echo "weights staged at ${MODEL_PATH}"
    du -sh "${MODEL_PATH}"
}

case "${1:-all}" in
    venv)    stage_venv ;;
    weights) stage_weights ;;
    all)     stage_venv; stage_weights ;;
    *) echo "usage: $0 {venv|weights|all}" >&2; exit 2 ;;
esac
