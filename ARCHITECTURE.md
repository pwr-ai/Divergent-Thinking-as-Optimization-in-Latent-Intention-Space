# CSC-SWE: Intention-Driven Coding Agent with CMA-ES Optimization

## Overview

This repository implements an **intention-driven coding agent** architecture that uses evolutionary optimization (CMA-ES) in embedding space to search for the best "fix intention" for software bugs.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PHASE 1: BOOTSTRAP (3 attempts → extract intentions)                       │
│  ═══════════════════════════════════════════════════                        │
│                                                                              │
│  ┌──────────────┐     ┌────────────────────┐     ┌───────────────────────┐  │
│  │  SWE-bench   │────▶│   Coding Agent     │────▶│   SWE-bench Harness   │  │
│  │  Instance    │     │  (3 attempts)      │     │   (evaluate patches)  │  │
│  │              │     │                    │     │                       │  │
│  │ problem_stmt │     │ problem → patch    │     │ patch → score         │  │
│  └──────────────┘     └────────────────────┘     └───────────────────────┘  │
│                               │                            │                │
│                               ▼                            ▼                │
│                       ┌───────────────┐           ┌──────────────┐          │
│                       │  3 patches +  │           │  3 scores    │          │
│                       │  agent traces │           │  (f2p × p2p) │          │
│                       └───────┬───────┘           └──────┬───────┘          │
│                               │                          │                  │
│                               ▼                          ▼                  │
│                       ┌─────────────────────────────────────────┐           │
│                       │  LLM: Extract intentions from solutions │           │
│                       │  "What was the fix strategy here?"      │           │
│                       └─────────────────┬───────────────────────┘           │
│                                         │                                   │
│                                         ▼                                   │
│                               ┌───────────────────┐                         │
│                               │  3 Anchor         │                         │
│                               │  Intentions       │                         │
│                               │  (weighted by     │                         │
│                               │   scores)         │                         │
│                               └─────────┬─────────┘                         │
│                                         │                                   │
│  PHASE 2: CMA-ES OPTIMIZATION LOOP      │                                   │
│  ═══════════════════════════════════════│═══════════════════════════════    │
│                                         ▼                                   │
│                       ┌──────────────────────────────┐                      │
│                       │       CSC Server             │                      │
│                       │   ┌────────────────────┐     │                      │
│                       │   │ Embed anchors →    │     │                      │
│                       │   │ CMA-ES centroid    │     │                      │
│                       │   └────────────────────┘     │                      │
│                       │   ┌────────────────────┐     │                      │
│                       │   │ CMA-ES: sample k   │◀────┼──── feedback(scores) │
│                       │   │ new embeddings     │     │                      │
│                       │   └────────────────────┘     │                      │
│                       │   ┌────────────────────┐     │                      │
│                       │   │ xRAG: decode to    │     │                      │
│                       │   │ text intentions    │     │                      │
│                       │   └────────────────────┘     │                      │
│                       └──────────────┬───────────────┘                      │
│                                      │                                      │
│                                      ▼                                      │
│                       ┌────────────────────────┐                            │
│                       │    Coding Agent        │                            │
│                       │   intention → patch    │                            │
│                       └────────────┬───────────┘                            │
│                                    │                                        │
│                                    ▼                                        │
│                       ┌───────────────────────┐                             │
│                       │   SWE-bench Harness   │                             │
│                       │   patch → score       │──── loop until solved ────▶ │
│                       └───────────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Idea

The system works in two phases:

### Phase 1: Bootstrap (Extract Intentions from Attempts)
1. **Run coding agent 3 times** directly on the problem (no intention)
2. **Evaluate patches** using SWE-bench harness → get scores
3. **Extract intentions** from patches/traces using LLM
   - "What fix strategy did this patch implement?"
   - Reverse-engineer the intention from the solution
4. **Weight intentions** by their scores

### Phase 2: CMA-ES Optimization Loop
5. **Initialize CMA-ES** at centroid of anchor embeddings (weighted)
6. **Sample k new embeddings** from CMA-ES
7. **Decode embeddings → intentions** using xRAG
8. **Execute intentions** via coding agent → patches
9. **Score patches** using SWE-bench harness
10. **Feedback to CMA-ES** and repeat

## Components

### 1. CSC Server (`csc_server.py`)

**CSC = CMA-ES + Semantic + Conditioning**

FastAPI server that:
- Manages tasks with anchor intentions
- Runs CMA-ES optimization in embedding space (4096-dim)
- Uses **xRAG** (retrieval-augmented generation) to decode embeddings → text intentions
- Stores state per task for iterative optimization

**Key endpoints:**
```
PUT  /tasks/{task_id}          # Initialize task with anchors
POST /tasks/{task_id}/suggest  # Get k new intention candidates from CMA-ES
POST /tasks/{task_id}/feedback # Report scores back to update CMA-ES
GET  /tasks/{task_id}/state    # Debug current CMA state
```

**Models used:**
- `Hannibal046/xrag-7b` - LLM for decoding embeddings to text
- `Salesforce/SFR-Embedding-Mistral` - Retriever for text → embedding

### 2. Bootstrap Phase (`bootstrap_then_cma.py`)

**Step 1: Run coding agent 3 times (no intention)**

```python
for attempt in range(3):
    patch = run_mini_on_instance(
        instance=instance,
        intention=None,  # No intention - just problem statement
        model_name=model,
    )
    score = evaluate_patch(patch)  # SWE-bench harness
    attempts.append((patch, score, agent_trace))
```

**Step 2: Extract intentions from solutions using LLM**

```python
for patch, score, trace in attempts:
    intention = llm.extract_intention(
        problem_statement=problem,
        patch=patch,
        agent_trace=trace,
        prompt="""
        Analyze this patch and agent trace. Extract the fix intention:
        - What was the root cause identified?
        - What file/function was modified?
        - What was the fix strategy?
        
        Format as detailed intention with code snippet and Pros/Cons.
        """
    )
    anchors.append((intention, score))
```

**Step 3: Extracted intention format**

```
After collecting from_expression_elements, check if there's more than one table.

Changes to src/sqlfluff/rules/L031.py:

def _lint_aliases_in_join(self, base_table, from_expression_elements, ...):
    # If only one from_expression_element exists, there's no join
    if len(from_expression_elements) <= 1:
        return None
    # ... rest of existing logic ...

Pros:
- Very minimal change (2 lines)
- Uses existing data structures

Cons:
- Assumes from_expression_elements count correlates to join presence
```

**Step 4: Weight anchors by scores and initialize CMA-ES**

### 3. CMA Loop (`cma_loop.py`)

Iterative optimization:

```
for round in range(rounds):
    candidates = csc.suggest(n=k)          # Get k intentions from CMA-ES
    for candidate in candidates:
        patch = mini_agent.run(intention)  # Execute intention
        score = harness.evaluate(patch)    # f2p × p2p score
        evaluations.append(score)
    csc.feedback(evaluations)              # Update CMA-ES
```

### 4. Mini Runner (`csc_swe_loop/mini_runner.py`)

Wrapper around `mini-swe-agent` that:
- Injects the intention as "HIGH-PRIORITY FIX INTENTION"
- Prepends it to the problem statement
- Runs agent in Docker container
- Extracts the generated patch

### 5. Evaluation (`csc_swe_loop/swebench_eval.py`)

Uses SWE-bench harness to evaluate patches:
- `fail_to_pass (f2p)` - ratio of failing tests that now pass
- `pass_to_pass (p2p)` - ratio of passing tests that still pass
- `score = f2p × p2p` - penalizes regressions

### 6. Dataset Runner (`run_dataset.py`)

Runs the full pipeline across a dataset:
- Bootstrap + CMA loop for each instance
- Resume support (`--resume`)
- Caching to avoid re-evaluation
- Results saved to `dataset_runs/<subset>_<split>/results.jsonl`

## File Structure

```
.
├── csc_server.py              # CSC Intention Server (FastAPI + CMA-ES + xRAG)
├── bootstrap_then_cma.py      # Single instance: bootstrap anchors + init CMA
├── cma_loop.py                # Single instance: run CMA optimization loop
├── run_dataset.py             # Dataset runner: iterate over instances
├── requirements.txt           # Python dependencies
├── csc_swe_loop/
│   ├── intent_client.py       # HTTP client for CSC server
│   ├── mini_runner.py         # Wrapper for mini-swe-agent
│   ├── swebench_eval.py       # SWE-bench harness integration
│   ├── swebench_utils.py      # Dataset loading utilities
│   └── scoring.py             # f2p × p2p scoring logic
└── dataset_runs/              # Output directory for results
```

## How to Run

### 1. Start vLLM Server (for coding agent)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Devstral-Small-2507 \
  --port 8080 \
  --tensor-parallel-size 4 \
  --max-model-len 32768
```

### 2. Start CSC Server

```bash
python -m uvicorn csc_server:app --host 0.0.0.0 --port 8000
```

### 3. Run on Single Instance

```bash
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="dummy"
export MSWEA_COST_TRACKING="ignore_errors"

# Bootstrap (3 agent attempts → extract intentions → init CMA)
python bootstrap_then_cma.py \
  --csc http://127.0.0.1:8000 \
  --instance_id sqlfluff__sqlfluff-1625 \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --environment_class docker

# CMA Loop (optimize intentions)
python cma_loop.py \
  --csc http://127.0.0.1:8000 \
  --instance_id sqlfluff__sqlfluff-1625 \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --rounds 5 --k 4
```

### 4. Run on Dataset

```bash
python run_dataset.py \
  --csc http://127.0.0.1:8000 \
  --subset lite --split dev \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --rounds 6 --k 8 \
  --cache --resume
```

## Scoring

The score combines two metrics from SWE-bench:

```python
score = fail_to_pass_ratio × pass_to_pass_ratio
```

- `fail_to_pass (f2p)`: What fraction of originally failing tests now pass?
- `pass_to_pass (p2p)`: What fraction of originally passing tests still pass?
- `resolved`: True if f2p=1.0 and p2p=1.0 (all tests pass, no regressions)

## CMA-ES in Embedding Space

The key innovation is optimizing **intentions** not patches:

1. **Embed anchors** → 4096-dim vectors via SFR-Embedding-Mistral
2. **CMA-ES** explores the embedding neighborhood
3. **xRAG decodes** embedding vectors → text intentions
4. **Agent executes** text intention → code patch
5. **Harness scores** patch → scalar feedback
6. **CMA-ES updates** based on scores

This allows "semantic search" for good fix strategies rather than random prompting.

## Requirements

- Python 3.11+
- CUDA GPUs (for vLLM and xRAG models)
- Docker/Podman (for SWE-bench harness)
- ~80GB VRAM total (for Devstral + xRAG models)

## Known Limitations

1. **Weak executor models loop** - Small models (Qwen-3B, etc.) struggle with `sed` commands and get stuck
2. **Context overflow** - Long conversations exceed model context limits
3. **GPU memory** - Running CSC + vLLM requires significant VRAM

## Future Work

- [ ] Better file editing tools for agent (avoid `sed`)
- [ ] Stronger executor models (Claude, GPT-4)
- [ ] Parallel evaluation of candidates
- [ ] Multi-objective optimization (speed, correctness, code quality)
