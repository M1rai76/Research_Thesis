#!/usr/bin/env bash
# serve_and_run.sh
# Shared Katana infrastructure for the external baselines (Reflexion, Prochemy, ...)
# Author : Gurdiraj Bal (z5386590)
#
# Scheduler-agnostic inner script for running any baseline batch on a UNSW Katana
# GPU node. Does the three things a job needs, in order:
#   1. Launch a vLLM OpenAI-compatible server for Llama-3.3-70B-Instruct on the
#      allocated node, served under the name `llama-3.3-70b-versatile` so the
#      model slug (and therefore every output filename) matches the existing
#      Groq runs exactly - keeping comparisons apples-to-apples.
#   2. Block until the server answers /v1/models.
#   3. Run the baseline command passed as arguments ("$@"), from $RUN_DIR.
# Always tears the server down on exit (success, failure, or signal).
#
# Baseline-agnostic: everything specific to a baseline (which batch script to run,
# from which directory, with which flags) is supplied by the caller:
#     RUN_DIR=/path/to/baseline/dir \
#       bash serve_and_run.sh python run_prochemy.py --dataset humaneval ...
# The per-baseline PBS/SLURM wrappers (run_*.pbs in this folder) set RUN_DIR and
# pass the command. Everything Katana-site-specific (weights path, GPU count,
# venv/module setup) is read from environment variables with sensible defaults.

set -euo pipefail

# --- Serving config (override via the job wrapper's env) ----------------------
# Path to the model weights. On Katana, pre-stage the gated Llama-3.3-70B-Instruct
# weights to scratch and point here (compute nodes typically have no outbound
# internet). A HF repo id also works if the node can reach the Hub.
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
# Must stay llama-3.3-70b-versatile so output filenames match the existing runs.
SERVED_NAME="${SERVED_NAME:-llama-3.3-70b-versatile}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"   # = number of GPUs in the allocation
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"    # prompts + optimised system prompts stay well under this
VLLM_PORT="${VLLM_PORT:-8000}"

# What to run once the server is up. RUN_DIR is where the batch script lives;
# the command itself is this script's positional arguments ("$@").
RUN_DIR="${RUN_DIR:?set RUN_DIR to the baseline directory (e.g. .../baselines/prochemy)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The batch talks to the local vLLM server via the shared `katana` backend
# (see generate_samples.get_client). These are what get_client reads.
export KATANA_BASE_URL="${KATANA_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
export KATANA_API_KEY="${KATANA_API_KEY:-EMPTY}"

# --- Activate the project environment ----------------------------------------
# The PBS/SLURM wrapper does the `module load` lines; here we just activate the
# vLLM venv (built by setup_katana_env.sh) if VENV_PATH is set.
if [[ -n "${VENV_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
fi

echo "=== Katana baseline run ==="
echo "Model weights : ${MODEL_PATH}"
echo "Served as     : ${SERVED_NAME}"
echo "Tensor //     : ${TENSOR_PARALLEL}"
echo "Endpoint      : ${KATANA_BASE_URL}"
echo "Run dir       : ${RUN_DIR}"
echo "Command       : $*"
echo

# --- 1. Launch vLLM in the background ----------------------------------------
echo "[serve] starting vLLM ..."
vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --port "${VLLM_PORT}" \
    > "${SCRIPT_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "[serve] stopping vLLM (pid ${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- 2. Wait until the server is ready ----------------------------------------
echo "[serve] waiting for ${KATANA_BASE_URL}/models ..."
for i in $(seq 1 120); do   # up to ~20 min; 70B load from scratch can be slow
    if curl -sf "${KATANA_BASE_URL}/models" >/dev/null 2>&1; then
        echo "[serve] ready after ~$((i * 10))s"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[serve] vLLM died during startup - see vllm_server.log" >&2
        exit 1
    fi
    sleep 10
done
if ! curl -sf "${KATANA_BASE_URL}/models" >/dev/null 2>&1; then
    echo "[serve] server never became ready - see vllm_server.log" >&2
    exit 1
fi

# --- 3. Run the baseline command ---------------------------------------------
echo "[batch] running: $*"
cd "${RUN_DIR}"
"$@"

echo "[batch] done. Outputs written under project/{results,samples}/."
