# Architecture: Latent Intention Search

## Overview

This system implements **intention-guided patch generation** via Tabu Search in a latent embedding space. It treats the coding agent as a black box and optimizes over high-level solution strategies (intentions) rather than code directly.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PHASE 1: BOOTSTRAP (3 attempts -> extract intentions)                   │
│  ═════════════════════════════════════════════════════                    │
│                                                                          │
│  ┌──────────────┐     ┌────────────────────┐     ┌───────────────────┐  │
│  │  SWE-bench   │────>│   Coding Agent     │────>│  SWE-bench        │  │
│  │  Instance    │     │  (3 attempts,      │     │  Harness          │  │
│  │              │     │   no intention)     │     │  (eval patches)   │  │
│  └──────────────┘     └────────────────────┘     └───────────────────┘  │
│                               │                            │            │
│                               v                            v            │
│                       ┌───────────────┐           ┌──────────────┐      │
│                       │  3 patches +  │           │  3 scores    │      │
│                       │  agent traces │           │  (f2p x p2p) │      │
│                       └───────┬───────┘           └──────┬───────┘      │
│                               │                          │              │
│                               v                          v              │
│                       ┌────────────────────────────────────────┐        │
│                       │  LLM: Extract intentions from traces   │        │
│                       │  "What was the fix strategy here?"     │        │
│                       └─────────────────┬──────────────────────┘        │
│                                         │                               │
│                                         v                               │
│                               ┌───────────────────┐                     │
│                               │  Anchor           │                     │
│                               │  Intentions       │                     │
│                               │  (embedded into   │                     │
│                               │   Z via SFR)      │                     │
│                               └─────────┬─────────┘                     │
│                                         │                               │
│  PHASE 2: TABU SEARCH IN LATENT SPACE   │                               │
│  ═══════════════════════════════════════│═════════════════════════       │
│                                         v                               │
│                       ┌──────────────────────────────┐                  │
│                       │       CSC Server             │                  │
│                       │                              │                  │
│                       │   Current solution: z_t      │                  │
│                       │          │                   │                  │
│                       │          v                   │                  │
│                       │   Neighborhood operator:     │                  │
│                       │   z' = z_t + epsilon         │                  │
│                       │   (local or kick move)       │                  │
│                       │          │                   │                  │
│                       │          v                   │                  │
│                       │   Tabu check:                │                  │
│                       │   z' not in T_t?             │<── feedback      │
│                       │          │                   │    (scores)      │
│                       │          v                   │                  │
│                       │   xRAG decode:               │                  │
│                       │   z' --> intention text      │                  │
│                       │                              │                  │
│                       └──────────────┬───────────────┘                  │
│                                      │                                  │
│                                      v                                  │
│                       ┌────────────────────────┐                        │
│                       │    Coding Agent        │                        │
│                       │   intention --> patch  │                        │
│                       └────────────┬───────────┘                        │
│                                    │                                    │
│                                    v                                    │
│                       ┌───────────────────────┐                         │
│                       │   SWE-bench Harness   │                         │
│                       │   patch --> score      │── loop until solved --> │
│                       └───────────────────────┘                         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. CSC Server (`csc_server.py`)

**CSC = Continuous Semantic Conditioning**

FastAPI server that manages the Tabu Search in a 4096-dimensional embedding space.

**Key responsibilities:**
- Embeds anchor intentions from bootstrap via SFR-Embedding-Mistral
- Generates neighborhood candidates (local perturbation or kick moves)
- Maintains a tabu list of recently visited intentions to prevent cycling
- Decodes embedding vectors to text intentions via xRAG
- Implements aspiration criteria (accept tabu move if best-ever)
- Hard-bans failed intentions (score=0, empty patch)

**Endpoints:**
```
PUT  /tasks/{task_id}          # Initialize task with anchors + tabu config
POST /tasks/{task_id}/suggest  # Generate next intention from neighborhood
POST /tasks/{task_id}/feedback # Report score, update current solution
GET  /tasks/{task_id}/state    # Debug: inspect tabu state
GET  /health                   # Server health check
```

**Tabu Search parameters (paper defaults):**
- `sigma_local = 300` — Local perturbation scale
- `sigma_kick = 1000` — Kick scale (for escaping local optima)
- `tabu_tenure = 50` — Rounds before a tabu entry expires
- `stagnation_threshold = 5` — Rounds without improvement before kick
- `kick_probability = 0.25` — Probability of random kick vs. directed move

**Models:**
- `Hannibal046/xrag-7b` — LLM decoder (embedding -> text)
- `Salesforce/SFR-Embedding-Mistral` — Encoder (text -> 4096-dim embedding)

### 2. Dataset Runner (`run_dataset.py`)

Orchestrates the full pipeline across a dataset split:

1. Load SWE-bench instances
2. For each instance:
   - Run N bootstrap attempts (no intention)
   - Evaluate patches via SWE-bench harness
   - Extract intentions from traces using LLM
   - PUT task to CSC server with anchor intentions
   - Run Tabu Search loop: suggest -> execute -> score -> feedback
3. Save per-instance results to `dataset_runs/<subset>_<split>/`

Supports `--resume` for fault-tolerant execution and `--bootstrap-only` for ablation experiments.

### 3. Mini Runner (`csc_swe_loop/mini_runner.py`)

Wrapper around `mini-swe-agent` that:
- Injects the decoded intention as a "HIGH-PRIORITY FIX INTENTION" prefix
- Prepends it to the problem statement
- Runs the agent in a containerized environment (Docker/Singularity)
- Extracts the generated patch (unified diff)

### 4. Evaluation (`csc_swe_loop/swebench_eval.py`)

Interfaces with the SWE-bench harness to evaluate patches in Docker containers.

### 5. Scoring (`csc_swe_loop/scoring.py`)

```python
score = fail_to_pass_ratio * pass_to_pass_ratio
```

- `fail_to_pass (f2p)`: Fraction of originally failing tests that now pass
- `pass_to_pass (p2p)`: Fraction of originally passing tests that still pass
- An instance is **resolved** when f2p = 1.0 and p2p = 1.0

### 6. Intent Client (`csc_swe_loop/intent_client.py`)

HTTP client for the CSC server. Handles PUT (initialize), suggest, and feedback requests.

## Data Flow

```
run_dataset.py
    |
    |-- bootstrap phase (N=3):
    |       run_mini_on_instance(intention=None) --> patch
    |       evaluate_patch(patch) --> score
    |       extract_intention_from_solution(trace) --> anchor_intention
    |
    |-- CSCClient.put_task(anchors, tabu_config)
    |
    |-- tabu search loop (R=50 rounds):
            CSCClient.suggest() --> intention text
            run_mini_on_instance(intention=intention) --> patch
            evaluate_patch(patch) --> score
            CSCClient.feedback(score)
            if resolved: break
```

## Environment

### Required Services

1. **vLLM server** (port 8080): Serves Devstral-Small-2507 for the coding agent
2. **CSC server** (port 8000): Manages Tabu Search + xRAG decoding
3. **Docker daemon**: Required for SWE-bench containerized evaluation

### Resource Requirements

- ~80 GB VRAM total across GPUs
  - vLLM: ~40-50 GB (Devstral 24B, fp8, 64k context)
  - CSC: ~10-15 GB (xRAG-7B + SFR-Embedding, 4-bit quantized)
- Docker/Podman for running SWE-bench test containers
