#!/bin/bash
# -*- coding: utf-8 -*-
#
# Automated pipeline runner for CSC × mini-swe-agent × SWE-bench
# TABU SEARCH VERSION
#
# This script:
# 1. Starts vLLM server (Devstral-Small-2507) on GPUs 0,1,2,3 with 40k context
# 2. Starts CSC Tabu Search server on GPU (port 8000)
# 3. Waits for both servers to be ready
# 4. Runs bootstrap phase (3 agent attempts → select best as initial solution)
# 5. Runs Tabu Search loop
# 6. Handles cleanup on exit
#
# Key difference from GA version:
# - Uses Tabu Search instead of Genetic Algorithm
# - Bootstrap selects best solution (not anchors) as starting point
# - Tabu parameters instead of GA parameters
#
# To stop: use kill <PID> (SIGTERM) so cleanup runs and children are killed.
# Do NOT use kill -9: trap cannot run, vLLM/python children keep running.
#

set -euo pipefail

# Configuration
# Note: VLLM_PORT is explicitly set to 8080 in start_vllm() to prevent env var override
VLLM_PORT=8080
CSC_PORT=8000
VLLM_MODEL="mistralai/Devstral-Small-2507"
VLLM_CONTEXT=64000
VLLM_GPU_UTIL=0.7  # Reduced to 0.25 (from 0.70) to share GPU with CSC server
VLLM_GPUS="0,1,2,3"
# Quantization options for vLLM (empty = no quantization, or: awq, gptq, squeezellm, fp8, marlin, etc.)
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}"
# Dtype for vLLM (auto, half, float16, bfloat16, float, float32)
VLLM_DTYPE="${VLLM_DTYPE:-auto}"
# CSC server dtype (float16, bfloat16, float32, auto)
CSC_DTYPE="${CSC_DTYPE:-float16}"
# CSC server quantization (none, 4bit, 8bit)
CSC_QUANTIZATION="${CSC_QUANTIZATION:-none}"

# Default arguments (can be overridden via env vars)
SUBSET="${SUBSET:-bash_only}"
# Dataset depends on subset: Verified (bash_only) vs Lite (other subsets)
if [ "$SUBSET" = "bash_only" ]; then
  DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-Bench_Verified}"
else
  DATASET_NAME="${DATASET_NAME:-princeton-nlp/SWE-bench_Lite}"
fi
# Instance ID only used in single-instance mode (MAX_INSTANCES=1). Override with INSTANCE_ID=...
INSTANCE_ID="${INSTANCE_ID:-django__django-11265}"

MINI_MODEL="${MINI_MODEL:-openai/mistralai/Devstral-Small-2507}"
SPLIT="${SPLIT:-dev}"
ROUNDS="${ROUNDS:-80}"
K="${K:-1}"
MAX_STEPS_BOOTSTRAP="${MAX_STEPS_BOOTSTRAP:-5}"
MAX_STEPS_TABU="${MAX_STEPS_TABU:-5}"
# Multi-instance mode: set MAX_INSTANCES > 1 to run on dataset instead of single instance
MAX_INSTANCES="${MAX_INSTANCES:-1}"
START_AT="${START_AT:-6}"

# Tabu Search specific parameters
TABU_TENURE="${TABU_TENURE:-50}"
SIGMA_LOCAL="${SIGMA_LOCAL:-300}"
SIGMA_KICK="${SIGMA_KICK:-1000}"
STAGNATION_THRESHOLD="${STAGNATION_THRESHOLD:-5}"
KICK_PROBABILITY="${KICK_PROBABILITY:-0.25}"

# Directories (must be before MINI_CONFIG which uses SCRIPT_DIR)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
# Agent config: swebench_minimal.yaml forbids creating new files; override with MINI_CONFIG=path
MINI_CONFIG="${MINI_CONFIG:-${SCRIPT_DIR}/csc_swe_loop/swebench_minimal.yaml}"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Log files
VLLM_LOG="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
CSC_LOG="${LOG_DIR}/csc_tabu_$(date +%Y%m%d_%H%M%S).log"
BOOTSTRAP_LOG="${LOG_DIR}/bootstrap_tabu_$(date +%Y%m%d_%H%M%S).log"
TABU_LOG="${LOG_DIR}/tabu_$(date +%Y%m%d_%H%M%S).log"

# PIDs
VLLM_PID=""
CSC_PID=""
PODMAN_SERVICE_PID=""
STARTED_VLLM=0
STARTED_CSC=0
STARTED_PODMAN_SERVICE=0
SKIP_CLEANUP=0

# Podman socket path
PODMAN_SOCKET="/run/user/$(id -u)/podman/podman.sock"

# Setup Docker/Podman socket
setup_docker_socket() {
    echo "=== Setting up Docker/Podman socket ==="
    
    # If DOCKER_HOST already set (e.g. remote Docker via SSH tunnel: tcp://127.0.0.1:2375), use it
    if [ -n "${DOCKER_HOST:-}" ]; then
        echo "Using existing DOCKER_HOST: $DOCKER_HOST"
        return 0
    fi
    
    # Check if docker socket exists
    if [ -S "/var/run/docker.sock" ]; then
        export DOCKER_HOST="unix:///var/run/docker.sock"
        echo "Using Docker socket: /var/run/docker.sock"
        return 0
    fi
    
    # Check if podman socket already exists
    if [ -S "$PODMAN_SOCKET" ]; then
        export DOCKER_HOST="unix://$PODMAN_SOCKET"
        echo "Using existing Podman socket: $PODMAN_SOCKET"
        return 0
    fi
    
    # Try to start podman socket service
    echo "Starting Podman socket service..."
    mkdir -p "$(dirname "$PODMAN_SOCKET")"
    
    # Start podman system service in background
    podman system service --time=0 "unix://$PODMAN_SOCKET" &
    PODMAN_SERVICE_PID=$!
    STARTED_PODMAN_SERVICE=1
    
    # Wait for socket to be created
    local max_wait=10
    local waited=0
    while [ ! -S "$PODMAN_SOCKET" ] && [ $waited -lt $max_wait ]; do
        sleep 1
        waited=$((waited + 1))
        echo "Waiting for Podman socket... ($waited/$max_wait)"
    done
    
    if [ -S "$PODMAN_SOCKET" ]; then
        export DOCKER_HOST="unix://$PODMAN_SOCKET"
        echo "Podman socket ready: $PODMAN_SOCKET"
        echo "DOCKER_HOST set to: $DOCKER_HOST"
        return 0
    else
        echo "ERROR: Failed to start Podman socket service"
        return 1
    fi
}

# Cleanup function (runs on EXIT, INT, TERM — NOT on kill -9)
# To stop pipeline: kill <PID> (SIGTERM). Avoid kill -9 or children keep running.
cleanup() {
    if [ "$SKIP_CLEANUP" -eq 1 ]; then
        return 0
    fi
    echo ""
    echo "=== Cleaning up ==="
    
    # Kill child processes first (bootstrap/tabu python) so they stop calling LLM
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
        echo "Killing Podman service (PID: $PODMAN_SERVICE_PID)..."
        kill "$PODMAN_SERVICE_PID" 2>/dev/null || true
    fi
    
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

# Check if servers are already running
check_server() {
    local url=$1
    local name=$2
    if curl -s "$url" >/dev/null 2>&1; then
        echo "✓ $name is already running at $url"
        return 0
    else
        return 1
    fi
}

# Wait for server to be ready
wait_for_server() {
    local url=$1
    local name=$2
    local max_wait=${3:-300}
    local elapsed=0
    
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

# Start vLLM server
start_vllm() {
    # Ensure VLLM_PORT is set correctly (override any env var)
    VLLM_PORT=8080
    
    if check_server "http://localhost:${VLLM_PORT}/v1/models" "vLLM"; then
        STARTED_VLLM=0
        VLLM_PID=""
        return 0
    fi
    
    echo "=== Starting vLLM server ==="
    echo "Model: $VLLM_MODEL"
    echo "Context: $VLLM_CONTEXT"
    echo "GPUs: $VLLM_GPUS"
    echo "Port: $VLLM_PORT (DEBUG: using port ${VLLM_PORT} for vLLM)"
    echo "Dtype: $VLLM_DTYPE"
    echo "Quantization: ${VLLM_QUANTIZATION:-none}"
    echo "Log: $VLLM_LOG"
    
    # Activate venv and start vLLM
    source .venv/bin/activate
    VLLM_PORT=8080
    # Explicitly set CUDA devices and ensure CUDA is available
    export CUDA_VISIBLE_DEVICES="$VLLM_GPUS"
    export VLLM_USE_CPU=0
    # Increase HuggingFace timeout for model download (default is 10s, increase to 300s)
    export HF_HUB_DOWNLOAD_TIMEOUT=300
    export HF_HUB_DOWNLOAD_TIMEOUT_STREAM=300
    
    # Verify CUDA is available before starting
    if ! .venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        echo "ERROR: CUDA is not available. Cannot start vLLM."
        return 1
    fi
    
    # Build vLLM command with optional quantization
    VLLM_CMD="CUDA_VISIBLE_DEVICES=$VLLM_GPUS .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model $VLLM_MODEL \
        --port $VLLM_PORT \
        --tokenizer_mode mistral \
        --config_format mistral \
        --load_format mistral \
        --tensor-parallel-size 4 \
        --max-model-len $VLLM_CONTEXT \
        --gpu-memory-utilization $VLLM_GPU_UTIL \
        --dtype $VLLM_DTYPE"
    
    # Add quantization if specified
    if [ -n "$VLLM_QUANTIZATION" ]; then
        VLLM_CMD="$VLLM_CMD --quantization $VLLM_QUANTIZATION"
    fi
    
    eval "$VLLM_CMD" > "$VLLM_LOG" 2>&1 &
    
    VLLM_PID=$!
    STARTED_VLLM=1
    echo "vLLM started with PID: $VLLM_PID"
    
    # Wait for vLLM to be ready (can take 2-3 minutes)
    if wait_for_server "http://localhost:${VLLM_PORT}/v1/models" "vLLM" 300; then
        echo "vLLM is ready!"
        return 0
    else
        echo "vLLM failed to start. Check log: $VLLM_LOG"
        tail -50 "$VLLM_LOG"
        return 1
    fi
}

# Start CSC Tabu Search server
start_csc() {
    if check_server "http://localhost:${CSC_PORT}/health" "CSC Tabu"; then
        STARTED_CSC=0
        CSC_PID=""
        return 0
    fi
    
    echo "=== Starting CSC Tabu Search server ==="
    echo "Port: $CSC_PORT"
    echo "Dtype: $CSC_DTYPE"
    echo "Quantization: $CSC_QUANTIZATION"
    echo "Tabu tenure: $TABU_TENURE"
    echo "Sigma local: $SIGMA_LOCAL"
    echo "Sigma kick: $SIGMA_KICK"
    echo "Log: $CSC_LOG"
    
    # Create log file immediately to ensure it exists
    touch "$CSC_LOG"
    
    # Activate venv and start CSC Tabu Search server with dtype/quantization env vars
    source .venv/bin/activate
    
    CSC_DTYPE="$CSC_DTYPE" CSC_QUANTIZATION="$CSC_QUANTIZATION" \
    .venv/bin/python -m uvicorn csc_server_tabu:app \
        --host 0.0.0.0 \
        --port "$CSC_PORT" \
        > "$CSC_LOG" 2>&1 &
    
    CSC_PID=$!
    STARTED_CSC=1
    echo "CSC Tabu server started with PID: $CSC_PID"
    echo "CSC log file: $CSC_LOG"
    
    # Wait a moment for process to start
    sleep 2
    
    # Check if process is still running
    if ! kill -0 "$CSC_PID" 2>/dev/null; then
        echo "ERROR: CSC Tabu server process died immediately. Check log: $CSC_LOG"
        if [ -f "$CSC_LOG" ]; then
            cat "$CSC_LOG"
        else
            echo "Log file does not exist"
        fi
        return 1
    fi
    
    # Wait for CSC to be ready (can take 2-3 minutes for model loading)
    if wait_for_server "http://localhost:${CSC_PORT}/health" "CSC Tabu" 300 || \
       wait_for_server "http://localhost:${CSC_PORT}/" "CSC Tabu" 10 || \
       wait_for_server "http://localhost:${CSC_PORT}/openapi.json" "CSC Tabu" 10; then
        # Verify models are loaded and algorithm is Tabu Search
        if curl -s "http://localhost:${CSC_PORT}/health" 2>/dev/null | grep -q '"algorithm":"TabuSearch"'; then
            echo "CSC Tabu Search server is ready with models loaded!"
            return 0
        elif curl -s "http://localhost:${CSC_PORT}/" 2>/dev/null | grep -q '"algorithm":"Tabu Search"'; then
            echo "CSC Tabu Search server is responding. Verifying /tasks endpoint..."
            if curl -s -X PUT "http://localhost:${CSC_PORT}/tasks/test" -H "Content-Type: application/json" \
               -d '{"task":{"task_context":"test"},"tabu":{"tabu_tenure":50}}' >/dev/null 2>&1; then
                echo "CSC Tabu Search server is ready (endpoint test passed)"
                return 0
            else
                echo "WARNING: CSC Tabu server is responding but /tasks endpoint may not work. Check log: $CSC_LOG"
                tail -50 "$CSC_LOG"
                return 1
            fi
        else
            echo "CSC server is running but may not be Tabu Search version. Check log: $CSC_LOG"
            curl -s "http://localhost:${CSC_PORT}/" || true
            if [ -f "$CSC_LOG" ]; then
                tail -50 "$CSC_LOG"
            fi
            return 1
        fi
    else
        echo "CSC Tabu server failed to start. Check log: $CSC_LOG"
        if [ -f "$CSC_LOG" ]; then
            tail -50 "$CSC_LOG"
        else
            echo "ERROR: CSC log file does not exist. Process may have crashed immediately."
        fi
        return 1
    fi
}

# Run on multiple instances (dataset mode) with Tabu Search
run_dataset() {
    local DATASET_LOG="${LOG_DIR}/dataset_tabu_$(date +%Y%m%d_%H%M%S).log"
    echo ""
    echo "=== Running Dataset Mode (Tabu Search) ==="
    echo "Subset: $SUBSET"
    echo "Split: $SPLIT"
    echo "Max instances: $MAX_INSTANCES"
    echo "Start at: $START_AT"
    echo "Config: $MINI_CONFIG"
    echo "Rounds: $ROUNDS, K: $K"
    echo "Tabu tenure: $TABU_TENURE"
    echo "Sigma local/kick: $SIGMA_LOCAL / $SIGMA_KICK"
    echo "Stagnation threshold: $STAGNATION_THRESHOLD"
    echo "Kick probability: $KICK_PROBABILITY"
    echo "Log: $DATASET_LOG"
    
    source .venv/bin/activate
    
    export MSWEA_COST_TRACKING="ignore_errors"
    export OPENAI_API_BASE="http://localhost:${VLLM_PORT}/v1"
    export OPENAI_API_KEY="dummy"
    export PATH="/home/user/.local/bin:$PATH"
    export DOCKER_HOST="${DOCKER_HOST:-unix://$PODMAN_SOCKET}"
    echo "DOCKER_HOST: $DOCKER_HOST"
    
    # Pass Tabu-specific parameters via environment or args
    .venv/bin/python run_dataset.py \
        --csc "http://127.0.0.1:${CSC_PORT}" \
        --subset "$SUBSET" \
        --split "$SPLIT" \
        --dataset_name "$DATASET_NAME" \
        --mini_config "$MINI_CONFIG" \
        --mini_model "$MINI_MODEL" \
        --environment_class "${ENVIRONMENT_CLASS:-docker}" \
        --rounds "$ROUNDS" \
        --k "$K" \
        --k_after "$K" \
        --max_instances "$MAX_INSTANCES" \
        --start_at "$START_AT" \
        --out "dataset_runs/${SUBSET}_${SPLIT}_tabu" \
        --max_steps "$MAX_STEPS_TABU" \
        --cache \
        --resume \
        --algorithm tabu \
        --tabu_tenure "$TABU_TENURE" \
        --sigma_local "$SIGMA_LOCAL" \
        --sigma_kick "$SIGMA_KICK" \
        --stagnation_threshold "$STAGNATION_THRESHOLD" \
        --kick_probability "$KICK_PROBABILITY" \
        --eval_backend "${EVAL_BACKEND:-docker}" \
        2>&1 | tee "$DATASET_LOG"
    
    local dataset_exit=${PIPESTATUS[0]}
    if [ $dataset_exit -ne 0 ]; then
        echo "✗ Dataset run (Tabu Search) failed (exit code: $dataset_exit)"
        return 1
    fi
    
    echo "✓ Dataset run (Tabu Search) completed"
    return 0
}

# Show usage
show_usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Automated pipeline runner for CSC × mini-swe-agent × SWE-bench
TABU SEARCH VERSION

Key differences from GA version:
- Uses Tabu Search algorithm instead of Genetic Algorithm
- Bootstrap generates 3 solutions, selects best as starting point
- Tabu list prevents revisiting same solutions
- Accepts moves with score >= current (plateau navigation)
- Kick moves for diversification when stuck

Environment variables (can be set before running):
  INSTANCE_ID          - SWE-bench instance ID (default: sqlfluff__sqlfluff-1625)
  MINI_MODEL           - Model name for mini-swe-agent (default: openai/mistralai/Devstral-Small-2507)
  DATASET_NAME         - SWE-bench harness dataset (default: from SUBSET; Verified for bash_only)
  SUBSET               - Dataset subset (default: bash_only; use lite for Lite)
  SPLIT                - Dataset split (default: dev)
  ROUNDS               - Number of Tabu Search rounds (default: 50)
  K                    - Number of candidates per round (default: 1)
  MAX_STEPS_BOOTSTRAP  - Max steps per bootstrap attempt (default: 5)
  MAX_STEPS_TABU       - Max steps per Tabu candidate (default: 5)
  MAX_INSTANCES        - Number of instances to run (default: 1)
  START_AT             - Index of first instance in dataset (default: 6)

  Tabu Search specific:
  TABU_TENURE          - How long solutions stay tabu (default: 50)
  SIGMA_LOCAL          - Std for local neighborhood moves (default: 0.5)
  SIGMA_KICK           - Std for kick/diversification moves (default: 3.0)
  STAGNATION_THRESHOLD - Iterations without improvement before kick (default: 10)
  KICK_PROBABILITY     - Base probability of kick move (default: 0.15)

  Environment:
  ENVIRONMENT_CLASS    - Agent execution: docker (default), singularity, or apptainer (HPC; set MSWEA_SINGULARITY_EXECUTABLE=apptainer if needed)
  EVAL_BACKEND         - Patch evaluation: docker (default, local Docker/Podman) or modal (Modal cloud, no local Docker needed; run 'modal token new' first)

Examples:
  # Run single instance with defaults
  ./run_pipeline_tabu.sh

  # Run on 10 instances from the dataset
  MAX_INSTANCES=10 ./run_pipeline_tabu.sh

  # Run from instance index 20 (single instance or first of N)
  START_AT=20 MAX_INSTANCES=1 ./run_pipeline_tabu.sh

  # Aggressive Tabu Search (longer tenure, more kicks)
  TABU_TENURE=100 KICK_PROBABILITY=0.25 ./run_pipeline_tabu.sh

  # Conservative Tabu Search (shorter tenure, smaller moves)
  TABU_TENURE=20 SIGMA_LOCAL=0.3 ./run_pipeline_tabu.sh

  # Use Apptainer/Singularity instead of Docker (e.g. on HPC)
  ENVIRONMENT_CLASS=apptainer EVAL_BACKEND=modal ./run_pipeline_tabu.sh

  # Use Modal for evaluation only (agent still uses Docker)
  EVAL_BACKEND=modal ./run_pipeline_tabu.sh

EOF
}

# Main execution
main() {
    # Check for help flag
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        SKIP_CLEANUP=1
        show_usage
        exit 0
    fi
    echo "=========================================="
    echo "  CSC × mini-swe-agent Pipeline Runner"
    echo "  (TABU SEARCH VERSION)"
    echo "=========================================="
    echo ""
    echo "Configuration:"
    if [ "$MAX_INSTANCES" -gt 1 ]; then
        echo "  Mode: DATASET (multi-instance)"
        echo "  Max instances: $MAX_INSTANCES"
    else
        echo "  Mode: SINGLE INSTANCE"
    fi
    echo "  Start at index: $START_AT"
    echo "  Model: $MINI_MODEL"
    echo "  Dataset: $DATASET_NAME"
    echo "  Subset: $SUBSET, Split: $SPLIT"
    echo "  Tabu Search rounds: $ROUNDS, k: $K"
    echo "  Max steps (bootstrap): $MAX_STEPS_BOOTSTRAP"
    echo "  Max steps (Tabu): $MAX_STEPS_TABU"
    echo ""
    echo "  Tabu Search parameters:"
    echo "    Tabu tenure: $TABU_TENURE"
    echo "    Sigma local: $SIGMA_LOCAL"
    echo "    Sigma kick: $SIGMA_KICK"
    echo "    Stagnation threshold: $STAGNATION_THRESHOLD"
    echo "    Kick probability: $KICK_PROBABILITY"
    echo ""
    echo "  vLLM port: $VLLM_PORT"
    echo "  vLLM dtype: $VLLM_DTYPE"
    echo "  vLLM quantization: ${VLLM_QUANTIZATION:-none}"
    echo "  CSC port: $CSC_PORT"
    echo "  CSC dtype: $CSC_DTYPE"
    echo "  CSC quantization: $CSC_QUANTIZATION"
    echo ""
    
    # Setup Docker/Podman socket (needed for SWE-bench harness when using docker eval backend)
    if [ "${EVAL_BACKEND:-docker}" = "docker" ]; then
        if ! setup_docker_socket; then
            echo "Failed to setup Docker/Podman socket. Exiting."
            exit 1
        fi
    else
        echo "Eval backend: ${EVAL_BACKEND} (skipping Docker/Podman socket setup)"
    fi
    
    # Start servers
    if ! start_vllm; then
        echo "Failed to start vLLM. Exiting."
        exit 1
    fi
    
    if ! start_csc; then
        echo "Failed to start CSC Tabu Search server. Exiting."
        exit 1
    fi
    
    # Run pipeline - always use run_dataset so START_AT and MAX_INSTANCES control instances
    if [ "$MAX_INSTANCES" -gt 1 ]; then
        echo "Running in DATASET mode with Tabu Search (${MAX_INSTANCES} instances, start_at=$START_AT)"
    else
        echo "Running in SINGLE INSTANCE mode with Tabu Search (instance at index START_AT=$START_AT)"
    fi
    if ! run_dataset; then
        echo "Dataset run (Tabu Search) failed. Exiting."
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo "  Run completed! (Tabu Search)"
    echo "=========================================="
    echo ""
    echo "Logs:"
    echo "  vLLM: $VLLM_LOG"
    echo "  CSC:  $CSC_LOG"
    echo "  Dataset: (see LOG_DIR/dataset_tabu_*.log)"
    echo ""
}

# Run main function
main "$@"
