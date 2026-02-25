#!/bin/bash
# -*- coding: utf-8 -*-
#
# Bootstrap-only pipeline: same as run_pipeline_tabu.sh but WITHOUT CMA/Tabu loop.
#
# This script:
# 1. Starts vLLM server (Devstral-Small-2507) on GPUs 0,1,2,3
# 2. Starts CSC Tabu Search server on port 8000 (bootstrap still PUTs task to CSC)
# 3. Waits for both servers to be ready
# 4. Runs ONLY the bootstrap phase (N agent attempts → eval → extract intentions → PUT to CSC)
# 5. Does NOT run the Tabu/CMA loop after bootstrap
#
# Use for experiments: e.g. NUM_ATTEMPTS=50 --only_instance_ids=id1,id2 to run
# 50 bootstrap attempts per instance on selected instances only.
#
# To stop: kill <PID> (SIGTERM). Avoid kill -9 or trap cannot run.
#

set -euo pipefail

# Configuration (same as run_pipeline_tabu.sh)
VLLM_PORT=8080
CSC_PORT=8000
VLLM_MODEL="mistralai/Devstral-Small-2507"
VLLM_CONTEXT=64000
VLLM_GPU_UTIL=0.7
VLLM_GPUS="0,1,2,3"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
CSC_DTYPE="${CSC_DTYPE:-float16}"
CSC_QUANTIZATION="${CSC_QUANTIZATION:-none}"

SUBSET="${SUBSET:-bash_only}"
if [ "$SUBSET" = "bash_only" ]; then
  DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
else
  DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Lite}"
fi
INSTANCE_ID="${INSTANCE_ID:-django__django-11265}"
MINI_MODEL="${MINI_MODEL:-openai/mistralai/Devstral-Small-2507}"
SPLIT="${SPLIT:-dev}"
MAX_STEPS_BOOTSTRAP="${MAX_STEPS_BOOTSTRAP:-5}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"
START_AT="${START_AT:-0}"
RUN_ID="${RUN_ID:-}"
# Number of bootstrap attempts per instance (e.g. 50 for bootstrap-only experiments)
NUM_ATTEMPTS="${NUM_ATTEMPTS:-3}"

# Tabu params (bootstrap still PUTs to CSC with these for consistency)
TABU_TENURE="${TABU_TENURE:-50}"
SIGMA_LOCAL="${SIGMA_LOCAL:-300}"
SIGMA_KICK="${SIGMA_KICK:-1000}"
STAGNATION_THRESHOLD="${STAGNATION_THRESHOLD:-5}"
KICK_PROBABILITY="${KICK_PROBABILITY:-0.25}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
MINI_CONFIG="${MINI_CONFIG:-${SCRIPT_DIR}/csc_swe_loop/swebench_minimal.yaml}"
if [ -n "${TMPDIR}" ]; then
  RUN_BASE="${TMPDIR}"
elif [ -n "${WORK_DIR}" ]; then
  RUN_BASE="${WORK_DIR}"
else
  RUN_BASE="${SCRIPT_DIR}"
fi
LOG_DIR="${SCRIPT_DIR}/logs"
OUT_SUFFIX=""
[ -n "${RUN_ID:-}" ] && OUT_SUFFIX="_run_${RUN_ID}"
# Bootstrap-only output dir (separate from full pipeline)
DATASET_OUT_BASE="${RUN_BASE}/dataset_runs_bootstrap${OUT_SUFFIX}"
mkdir -p "$LOG_DIR"
mkdir -p "$DATASET_OUT_BASE"
export SWE_RUN_EVAL_LOG_DIR="${LOG_DIR}/run_evaluation"
mkdir -p "$SWE_RUN_EVAL_LOG_DIR"

if [ "$RUN_BASE" != "$SCRIPT_DIR" ]; then
  export HF_HOME="${RUN_BASE}/hf_cache"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  mkdir -p "$HF_DATASETS_CACHE"
  if [ -d "${HOME}/.cache/huggingface" ]; then
    echo "Pre-populating HF cache from PD -> TMP..."
    rsync -a "${HOME}/.cache/huggingface/" "${HF_HOME}/" 2>/dev/null || true
  fi
  echo "HF cache on TMP: HF_HOME=$HF_HOME"
fi

VLLM_LOG="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
CSC_LOG="${LOG_DIR}/csc_tabu_$(date +%Y%m%d_%H%M%S).log"
BOOTSTRAP_LOG="${LOG_DIR}/bootstrap_only_$(date +%Y%m%d_%H%M%S).log"

VLLM_PID=""
CSC_PID=""
PODMAN_SERVICE_PID=""
STARTED_VLLM=0
STARTED_CSC=0
STARTED_PODMAN_SERVICE=0
SKIP_CLEANUP=0
PODMAN_SOCKET="/run/user/$(id -u)/podman/podman.sock"

setup_docker_socket() {
    echo "=== Setting up Docker/Podman socket ==="
    if [ -n "${DOCKER_HOST:-}" ]; then
        echo "Using existing DOCKER_HOST: $DOCKER_HOST"
        return 0
    fi
    if [ -S "/var/run/docker.sock" ]; then
        export DOCKER_HOST="unix:///var/run/docker.sock"
        echo "Using Docker socket: /var/run/docker.sock"
        return 0
    fi
    if [ -S "$PODMAN_SOCKET" ]; then
        export DOCKER_HOST="unix://$PODMAN_SOCKET"
        echo "Using existing Podman socket: $PODMAN_SOCKET"
        return 0
    fi
    if command -v podman >/dev/null 2>&1; then
        echo "Starting Podman socket service..."
        mkdir -p "$(dirname "$PODMAN_SOCKET")"
        podman system service --time=0 "unix://$PODMAN_SOCKET" &
        PODMAN_SERVICE_PID=$!
        STARTED_PODMAN_SERVICE=1
        local max_wait=10 waited=0
        while [ ! -S "$PODMAN_SOCKET" ] && [ $waited -lt $max_wait ]; do
            sleep 1
            waited=$((waited + 1))
            echo "Waiting for Podman socket... ($waited/$max_wait)"
        done
        if [ -S "$PODMAN_SOCKET" ]; then
            export DOCKER_HOST="unix://$PODMAN_SOCKET"
            echo "Podman socket ready: $PODMAN_SOCKET"
            return 0
        fi
    fi
    echo "ERROR: No Docker or Podman available."
    return 1
}

cleanup() {
    if [ "$SKIP_CLEANUP" -eq 1 ]; then
        return 0
    fi
    echo ""
    echo "=== Cleaning up ==="
    if pkill -TERM -P $$ 2>/dev/null; then
        sleep 2
        pkill -9 -P $$ 2>/dev/null || true
    fi
    if [ "$STARTED_VLLM" -eq 1 ] && [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "Killing vLLM (PID: $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$VLLM_PID" 2>/dev/null || true
    fi
    if [ "$STARTED_CSC" -eq 1 ] && [ -n "$CSC_PID" ] && kill -0 "$CSC_PID" 2>/dev/null; then
        echo "Killing CSC Tabu server (PID: $CSC_PID)..."
        kill "$CSC_PID" 2>/dev/null || true
        sleep 2
        kill -9 "$CSC_PID" 2>/dev/null || true
    fi
    if [ "$STARTED_PODMAN_SERVICE" -eq 1 ] && [ -n "$PODMAN_SERVICE_PID" ] && kill -0 "$PODMAN_SERVICE_PID" 2>/dev/null; then
        kill "$PODMAN_SERVICE_PID" 2>/dev/null || true
    fi
    if [ "$RUN_BASE" != "$SCRIPT_DIR" ] && [ -d "$DATASET_OUT_BASE" ]; then
        echo "Final sync: dataset_runs_bootstrap TMP -> PD..."
        mkdir -p "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}"
        rsync -a "$DATASET_OUT_BASE/" "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/" 2>/dev/null || \
            cp -r "$DATASET_OUT_BASE/"* "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/" 2>/dev/null || true
    fi
    if [ "$RUN_BASE" != "$SCRIPT_DIR" ] && [ -n "${TMPDIR:-}" ]; then
        echo "Cleaning TMPDIR contents..."
        rm -rf "${TMPDIR:?}/project" "${TMPDIR:?}/hf_cache" "${TMPDIR:?}/logs" "$DATASET_OUT_BASE" 2>/dev/null || true
    fi
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

check_server() {
    local url=$1 name=$2
    if curl -s "$url" >/dev/null 2>&1; then
        echo "✓ $name is already running at $url"
        return 0
    else
        return 1
    fi
}

wait_for_server() {
    local url=$1 name=$2 max_wait=${3:-300} elapsed=0
    echo "Waiting for $name to be ready at $url..."
    while [ $elapsed -lt $max_wait ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo "✓ $name is ready!"
            return 0
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  ... still waiting (${elapsed}s/${max_wait}s)"
    done
    echo "✗ $name failed to start within ${max_wait}s"
    return 1
}

mirror_to_tmp() {
    if [ "$RUN_BASE" = "$SCRIPT_DIR" ]; then
        return 0
    fi
    echo ""
    echo "=== Mirroring project to local storage (TMP) ==="
    local LOCAL_PROJECT="${RUN_BASE}/project"
    mkdir -p "$LOCAL_PROJECT"
    rsync -a --delete \
        --exclude='dataset_runs*' \
        --exclude='logs' \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        "$SCRIPT_DIR/" "$LOCAL_PROJECT/"
    if [ -f "$LOCAL_PROJECT/.venv/bin/activate" ]; then
        sed -i "s|VIRTUAL_ENV=.*|VIRTUAL_ENV=\"${LOCAL_PROJECT}/.venv\"|" \
            "$LOCAL_PROJECT/.venv/bin/activate"
    fi
    cd "$LOCAL_PROJECT"
    export SWE_RUN_WORKING_DIR="$LOCAL_PROJECT"
    if [[ "$MINI_CONFIG" == "${SCRIPT_DIR}"* ]]; then
        MINI_CONFIG="${LOCAL_PROJECT}${MINI_CONFIG#${SCRIPT_DIR}}"
    fi
    echo "  Working dir: $(pwd)"
}

start_vllm() {
    VLLM_PORT=8080
    if check_server "http://localhost:${VLLM_PORT}/v1/models" "vLLM"; then
        STARTED_VLLM=0
        VLLM_PID=""
        return 0
    fi
    echo "=== Starting vLLM server ==="
    echo "Model: $VLLM_MODEL Port: $VLLM_PORT"
    source .venv/bin/activate
    export CUDA_VISIBLE_DEVICES="$VLLM_GPUS"
    export VLLM_USE_CPU=0
    export HF_HUB_DOWNLOAD_TIMEOUT=300
    export HF_HUB_DOWNLOAD_TIMEOUT_STREAM=300
    if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        echo "ERROR: CUDA is not available."
        return 1
    fi
    VLLM_CMD="CUDA_VISIBLE_DEVICES=$VLLM_GPUS .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model $VLLM_MODEL --port $VLLM_PORT --tokenizer_mode mistral --config_format mistral --load_format mistral \
        --tensor-parallel-size 4 --max-model-len $VLLM_CONTEXT --gpu-memory-utilization $VLLM_GPU_UTIL --dtype $VLLM_DTYPE --enforce-eager"
    [ -n "$VLLM_QUANTIZATION" ] && VLLM_CMD="$VLLM_CMD --quantization $VLLM_QUANTIZATION"
    eval "$VLLM_CMD" > "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    STARTED_VLLM=1
    echo "vLLM started with PID: $VLLM_PID"
    if wait_for_server "http://localhost:${VLLM_PORT}/v1/models" "vLLM" 300; then
        return 0
    else
        echo "vLLM failed to start. Check: $VLLM_LOG"
        tail -50 "$VLLM_LOG"
        return 1
    fi
}

start_csc() {
    if check_server "http://localhost:${CSC_PORT}/health" "CSC Tabu"; then
        STARTED_CSC=0
        CSC_PID=""
        return 0
    fi
    echo "=== Starting CSC Tabu Search server ==="
    touch "$CSC_LOG"
    source .venv/bin/activate
    CSC_DTYPE="$CSC_DTYPE" CSC_QUANTIZATION="$CSC_QUANTIZATION" \
    .venv/bin/python -m uvicorn csc_server_tabu:app --host 0.0.0.0 --port "$CSC_PORT" > "$CSC_LOG" 2>&1 &
    CSC_PID=$!
    STARTED_CSC=1
    sleep 2
    if ! kill -0 "$CSC_PID" 2>/dev/null; then
        echo "ERROR: CSC server died. Check: $CSC_LOG"
        cat "$CSC_LOG"
        return 1
    fi
    if wait_for_server "http://localhost:${CSC_PORT}/health" "CSC Tabu" 300 || \
       wait_for_server "http://localhost:${CSC_PORT}/" "CSC Tabu" 10; then
        echo "CSC Tabu server is ready."
        return 0
    else
        echo "CSC failed to start. Check: $CSC_LOG"
        tail -50 "$CSC_LOG"
        return 1
    fi
}

run_bootstrap_only() {
    echo ""
    echo "=== Running BOOTSTRAP ONLY (no Tabu/CMA loop) ==="
    echo "Subset: $SUBSET Split: $SPLIT Max instances: $MAX_INSTANCES Start at: $START_AT"
    echo "Num attempts per instance: $NUM_ATTEMPTS"
    echo "Out: $DATASET_OUT_BASE"
    echo "Log: $BOOTSTRAP_LOG"
    [ -n "${ONLY_INSTANCE_IDS:-}" ] && echo "Only instance IDs: $ONLY_INSTANCE_IDS"
    [ -n "${RUN_ID:-}" ] && echo "Run ID: $RUN_ID"

    if [ "$RUN_BASE" != "$SCRIPT_DIR" ] && [ -d "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}" ]; then
        echo "Syncing PD dataset_runs_bootstrap -> TMP for resume..."
        mkdir -p "$DATASET_OUT_BASE"
        rsync -a "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/" "$DATASET_OUT_BASE/" 2>/dev/null || cp -r "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/"* "$DATASET_OUT_BASE/" 2>/dev/null || true
    fi

    source .venv/bin/activate
    export MSWEA_COST_TRACKING="ignore_errors"
    export MSWEA_SILENT_STARTUP=1
    export OPENAI_API_BASE="http://localhost:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    export DOCKER_HOST="${DOCKER_HOST:-unix://$PODMAN_SOCKET}"

    if [ -z "${MSWEA_DOCKER_EXECUTABLE:-}" ]; then
        if command -v docker >/dev/null 2>&1; then
            export MSWEA_DOCKER_EXECUTABLE="$(command -v docker)"
        else
            local _docker_api
            _docker_api="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/docker_api.py"
            [ "$RUN_BASE" != "$SCRIPT_DIR" ] && [ -f "${RUN_BASE}/project/docker_api.py" ] && _docker_api="${RUN_BASE}/project/docker_api.py"
            chmod +x "$_docker_api" 2>/dev/null || true
            export MSWEA_DOCKER_EXECUTABLE="$_docker_api"
            echo "Using Python Docker API CLI: $_docker_api"
        fi
    fi

    local EFFECTIVE_ENV_CLASS="${ENVIRONMENT_CLASS:-docker}"
    if [[ "$EFFECTIVE_ENV_CLASS" == "apptainer" || "$EFFECTIVE_ENV_CLASS" == "singularity" ]]; then
        if [[ "${DOCKER_HOST}" == tcp://* || "${DOCKER_HOST}" == ssh://* ]]; then
            echo "Switching to docker (remote DOCKER_HOST)."
            EFFECTIVE_ENV_CLASS="docker"
        else
            export APPTAINER_CACHEDIR="${RUN_BASE}/apptainer_cache"
            export SINGULARITY_CACHEDIR="${RUN_BASE}/apptainer_cache"
            mkdir -p "${RUN_BASE}/apptainer_cache"
        fi
    fi

    local PD_DIR_ARG=""
    [ "$RUN_BASE" != "$SCRIPT_DIR" ] && PD_DIR_ARG="--pd_dir ${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}" && export SWE_RUN_WORKING_DIR="${RUN_BASE}/project"

    local ONLY_IDS_ARG=""
    [ -n "${ONLY_INSTANCE_IDS:-}" ] && ONLY_IDS_ARG="--only_instance_ids ${ONLY_INSTANCE_IDS}"

    .venv/bin/python run_dataset.py \
        --csc "http://localhost:${CSC_PORT}" \
        --subset "$SUBSET" \
        --split "$SPLIT" \
        --dataset_name "$DATASET_NAME" \
        --mini_config "$MINI_CONFIG" \
        --mini_model "$MINI_MODEL" \
        --environment_class "${EFFECTIVE_ENV_CLASS}" \
        --out "$DATASET_OUT_BASE" \
        --max_instances "$MAX_INSTANCES" \
        --start_at "$START_AT" \
        --num_attempts "$NUM_ATTEMPTS" \
        --max_steps "$MAX_STEPS_BOOTSTRAP" \
        --bootstrap-only \
        --resume \
        --algorithm tabu \
        --tabu_tenure "$TABU_TENURE" \
        --sigma_local "$SIGMA_LOCAL" \
        --sigma_kick "$SIGMA_KICK" \
        --stagnation_threshold "$STAGNATION_THRESHOLD" \
        --kick_probability "$KICK_PROBABILITY" \
        --eval_backend "${EVAL_BACKEND:-docker}" \
        ${PD_DIR_ARG:-} \
        ${ONLY_IDS_ARG:-} \
        2>&1 | tee "$BOOTSTRAP_LOG"

    local exit_code=${PIPESTATUS[0]}

    if [ "$RUN_BASE" != "$SCRIPT_DIR" ] && [ -d "$DATASET_OUT_BASE" ]; then
        echo "Syncing dataset_runs_bootstrap TMP -> PD..."
        mkdir -p "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}"
        rsync -a "$DATASET_OUT_BASE/" "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/" 2>/dev/null || cp -r "$DATASET_OUT_BASE/"* "${SCRIPT_DIR}/dataset_runs_bootstrap${OUT_SUFFIX}/" 2>/dev/null || true
    fi

    if [ $exit_code -ne 0 ]; then
        echo "✗ Bootstrap-only run failed (exit code: $exit_code)"
        return 1
    fi
    echo "✓ Bootstrap-only run completed"
    return 0
}

show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Bootstrap-only pipeline: same as run_pipeline_tabu.sh but runs ONLY the bootstrap
phase (N agent attempts, eval, extract intentions, PUT to CSC). No Tabu/CMA loop.

Environment variables:
  INSTANCE_ID          - Instance ID (single-instance mode)
  MINI_MODEL           - Model name (default: openai/mistralai/Devstral-Small-2507)
  SUBSET, SPLIT        - Dataset subset/split (default: bash_only, dev)
  NUM_ATTEMPTS         - Bootstrap attempts per instance (default: 3; use 50 for experiments)
  MAX_STEPS_BOOTSTRAP  - Max steps per attempt (default: 5)
  MAX_INSTANCES        - Instances to run (default: 1)
  START_AT             - Start at dataset index (default: 0)
  ONLY_INSTANCE_IDS   - Comma-separated instance IDs (e.g. id1,id2,id3)
  RUN_ID               - Output to dataset_runs_bootstrap_run_<RUN_ID>
  EVAL_BACKEND         - docker (default) or modal

Examples:
  ./run_pipeline_bootstrap.sh

  # 50 bootstrap attempts on selected instances
  NUM_ATTEMPTS=50 ONLY_INSTANCE_IDS="django__django-11265,sqlfluff__sqlfluff-1625" ./run_pipeline_bootstrap.sh

  # 10 instances, 50 attempts each
  MAX_INSTANCES=10 NUM_ATTEMPTS=50 ./run_pipeline_bootstrap.sh
EOF
}

main() {
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        SKIP_CLEANUP=1
        show_usage
        exit 0
    fi
    echo "=========================================="
    echo "  Bootstrap-only pipeline (no Tabu loop)"
    echo "=========================================="
    echo "  Num attempts: $NUM_ATTEMPTS  Max instances: $MAX_INSTANCES  Start at: $START_AT"
    echo "  Out: $DATASET_OUT_BASE"
    echo ""

    if [ "${EVAL_BACKEND:-docker}" = "docker" ]; then
        if ! setup_docker_socket; then
            echo "Failed to setup Docker/Podman. Exiting."
            exit 1
        fi
    else
        echo "Eval backend: ${EVAL_BACKEND}"
    fi

    mirror_to_tmp

    if ! start_vllm; then
        echo "Failed to start vLLM. Exiting."
        exit 1
    fi
    if ! start_csc; then
        echo "Failed to start CSC. Exiting."
        exit 1
    fi

    if ! run_bootstrap_only; then
        echo "Bootstrap-only run failed. Exiting."
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo "  Bootstrap-only run completed"
    echo "=========================================="
    echo "  Logs: vLLM=$VLLM_LOG  CSC=$CSC_LOG  Bootstrap=$BOOTSTRAP_LOG"
    echo "  Output: $DATASET_OUT_BASE (or dataset_runs_bootstrap${OUT_SUFFIX} on PD)"
    echo ""
}

main "$@"
