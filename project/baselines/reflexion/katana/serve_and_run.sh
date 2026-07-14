#!/usr/bin/env bash
# serve_and_run.sh
# Baseline - Reflexion (Shinn et al., 2023)
# Author : Gurdiraj Bal (z5386590)
#
# Scheduler-agnostic inner script for running the MBPP+ Reflexion baseline on a
# UNSW Katana GPU node. Does the three things a job needs to do, in order:
#   1. Launch a vLLM OpenAI-compatible server for Llama-3.3-70B-Instruct on the
#      allocated node, served under the name `llama-3.3-70b-versatile` so the
#      model slug (and therefore every output filename) matches the existing
#      Groq cot/minimal runs exactly - keeping the MBPP comparison apples-to-apples.
#   2. Block until the server answers /v1/models.
#   3. Run run_reflexion_batch.py against that server over localhost.
# Always tears the server down on exit (success, failure, or signal).
#
# This script is deliberately scheduler-agnostic: call it from either a PBS
# (`qsub`) or SLURM (`sbatch`) wrapper - see run_reflexion_mbpp.job in this
# folder. Everything Katana-site-specific (weights path, GPU count, HF token,
# venv/module setup) is read from environment variables with sensible defaults,
# so the wrapper only has to export the handful that differ on your allocation.

set -euo pipefail

# --- Site-configurable (override via the job wrapper's env) -------------------
# Path to the model weights. On Katana, pre-stage the gated Llama-3.3-70B-Instruct
# weights to scratch and point here (compute nodes typically have no outbound
# internet). A HF repo id also works if the node can reach the Hub.
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.3-70B-Instruct}"
# Must stay llama-3.3-70b-versatile so output filenames match the existing runs.
SERVED_NAME="${SERVED_NAME:-llama-3.3-70b-versatile}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"   # = number of GPUs in the allocation
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"    # prompts are small; 512 output tokens
VLLM_PORT="${VLLM_PORT:-8000}"
DATASET="${DATASET:-mbpp}"
MAX_REPAIR_ROUNDS="${MAX_REPAIR_ROUNDS:-2}"
REFLEXION_MEMORY_SIZE="${REFLEXION_MEMORY_SIZE:-1}"

# Repo layout: this script lives in project/baselines/reflexion/katana/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REFLEXION_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$(dirname "$REFLEXION_DIR")")"   # .../project

# The batch talks to the local vLLM server via the shared `katana` backend
# (see generate_samples.get_client). These are what get_client reads.
export KATANA_BASE_URL="${KATANA_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
export KATANA_API_KEY="${KATANA_API_KEY:-EMPTY}"

# --- Activate the project environment ----------------------------------------
# Adjust to your Katana setup: `module load` lines and/or venv activation.
# Example (uncomment / edit for your allocation):
#   module load python/3.11 cuda/12.4
if [[ -n "${VENV_PATH:-}" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
elif [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/.venv/bin/activate"
fi

echo "=== Reflexion MBPP+ on Katana ==="
echo "Model weights : ${MODEL_PATH}"
echo "Served as     : ${SERVED_NAME}"
echo "Tensor //     : ${TENSOR_PARALLEL}"
echo "Endpoint      : ${KATANA_BASE_URL}"
echo "Dataset       : ${DATASET}"
echo "Project dir   : ${PROJECT_DIR}"
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

# --- 3. Run the Reflexion batch ----------------------------------------------
echo "[batch] running run_reflexion_batch.py ..."
cd "${REFLEXION_DIR}"
python run_reflexion_batch.py \
    --dataset "${DATASET}" \
    --model "${SERVED_NAME}" \
    --backend katana \
    --max_repair_rounds "${MAX_REPAIR_ROUNDS}" \
    --reflexion_memory_size "${REFLEXION_MEMORY_SIZE}" \
    --resume-missing

echo "[batch] done. Trajectories + round JSONLs written under project/{results,samples}/."
echo "Next: run the Docker EvalPlus step on the round1/round2 JSONLs (see katana/README.md)."
