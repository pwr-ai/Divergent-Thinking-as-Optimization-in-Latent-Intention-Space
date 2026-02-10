# CSC × mini-swe-agent × SWE-bench loop (bootstrap + CMA-ES)

This package provides:
1) **bootstrap_then_cma.py**: reads a SWE-bench task, generates 3 anchor intentions, evaluates them (mini -> SWE-bench harness),
   then **PUTs** the task into your CSC Intention Server with anchor weights derived from the scores.
2) **cma_loop.py**: runs the iterative loop: CSC suggest -> mini patch -> SWE-bench eval -> score -> CSC feedback, until solved or budget exhausted.

## Requirements
- Your CSC Intention Server running (FastAPI) e.g. http://127.0.0.1:8000
- Docker working (SWE-bench harness runs containers)
- Python packages:
  - httpx pyyaml datasets swebench mini-swe-agent

## Quickstart

### Automated Pipeline (Recommended)

The easiest way to run the full pipeline is using the automated script:

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with default settings (Bash Only = Verified, default instance django__django-11265)
./run_pipeline.sh

# Or customize (for Lite subset default instance is sqlfluff__sqlfluff-1625):
INSTANCE_ID="django__django-14007" \
SUBSET="bash_only" \
MINI_MODEL="openai/mistralai/Devstral-Small-2507" \
ROUNDS=3 \
K=4 \
./run_pipeline.sh

# Use Lite subset instead of Bash Only (Verified):
SUBSET="lite" ./run_pipeline.sh
```

The script automatically:
1. Checks if vLLM/CSC servers are already running (reuses them if found)
2. Starts vLLM server (Devstral-Small-2507) on GPUs 0,1,2,3 with 48k context if needed
3. Starts CSC server on CPU (port 8000) if needed
4. Waits for both servers to be ready
5. Runs bootstrap phase (3 agent attempts → extract intentions)
6. Runs CMA loop
7. Handles cleanup on exit (kills only processes started by this script)

All logs are saved to `logs/` directory with timestamps.

### Manual Setup

If you prefer to run components manually:

#### 0) Start your CSC server
```bash
uvicorn csc_server:app --host 0.0.0.0 --port 8000
```

#### 1) Bootstrap anchors + initialize CSC task
```bash
# Bash Only subset (Verified dataset, 500 instances); --dataset_name is derived from subset
python bootstrap_then_cma.py \
  --csc http://127.0.0.1:8000 \
  --subset bash_only --split dev \
  --instance_id <instance_id> \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --environment_class docker

# Or Lite subset (300 instances)
python bootstrap_then_cma.py \
  --csc http://127.0.0.1:8000 \
  --subset lite --split dev \
  --instance_id sympy__sympy-15599 \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --environment_class docker
```

#### 2) Run CMA loop
```bash
python cma_loop.py \
  --csc http://127.0.0.1:8000 \
  --subset bash_only --split dev \
  --instance_id <instance_id> \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --environment_class docker \
  --rounds 6 --k 8 --cache
```

## Subsets (SWE-bench)
- **bash_only** (default): SWE-bench Bash Only setting — [Verified](https://www.swebench.com/bash-only.html) dataset (500 instances), mini-SWE-agent minimal bash environment. Default instance: `django__django-11265`. Verified has NO sqlfluff instances.
- **lite**: SWE-bench Lite (300 instances). Default instance: `sqlfluff__sqlfluff-1625`. Use `--subset lite` or `SUBSET=lite`.
- **verified**: Same dataset as bash_only; alias `bash_only` is for clarity. Instance IDs differ between Lite and Verified.

## Environment (Docker vs Apptainer/Singularity)
- **Default**: `--environment_class docker` (or set `ENVIRONMENT_CLASS=docker` in the pipeline scripts).
- **Apptainer/Singularity** (e.g. on HPC): use `--environment_class apptainer` or `--environment_class singularity`. The code maps `apptainer` to mini-swe-agent’s `singularity` backend. If your binary is `apptainer`, set `MSWEA_SINGULARITY_EXECUTABLE=apptainer` before running. The **evaluation** of patches (SWE-bench harness) still uses Docker/Podman in this repo; only the **agent execution** runs in Singularity/Apptainer.

## Notes
- The mini runner mirrors the `mini-extra swebench-single` pattern: it loads the SWE-bench instance, creates the sandbox environment,
  and runs the InteractiveAgent on an injected task that prepends a "HIGH-PRIORITY FIX INTENTION". The returned string from `agent.run(...)`
  is treated as the patch (unified diff) to evaluate.
- SWE-bench harness result schemas can differ by version. The scorer tries common fields; if your harness writes richer JSON files, adapt
  `score_from_instance_result()` in `csc_swe_loop/scoring.py`.
