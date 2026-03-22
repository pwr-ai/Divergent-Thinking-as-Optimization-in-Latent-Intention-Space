# Beyond Discrete Search: Divergent Thinking as Optimization in Latent Intention Space

This repository contains the source code for the paper *"Beyond Discrete Search: Divergent Thinking as Optimization in Latent Intention Space"*.

We introduce a framework that recasts LLM-based coding as optimization over a latent space of **intentions** — natural-language descriptions of high-level solution strategies. A Tabu Search explores this space with feedback guidance, steering a coding agent toward correct solutions without modifying model weights.

On **SWE-Bench Verified**, the method raises the resolution rate of a quantized 24B open-weight model from 45.2% to **70.2%**, reaching parity with frontier models 25x its size.

## Method Overview

The system operates in two phases at inference time, treating the coding agent as a black box:

1. **Bootstrap Phase**: The agent executes 3 times without guidance. Anchor intentions are reverse-engineered from reasoning traces and embedded into a latent space Z.

2. **Tabu Search Phase**: A neighborhood operator proposes candidate vectors near the current solution. Each candidate is decoded into a natural-language intention, the agent is conditioned on it, and the evaluation score feeds back into the search loop. A tabu list prevents cycling and encourages exploration.

```
Problem Statement
       |
       v
  [Bootstrap: 3 unguided agent runs]
       |
       v
  Extract anchor intentions --> Embed into Z (4096-dim)
       |
       v
  [Tabu Search Loop]
       |
       +---> Neighborhood operator: propose candidate z'
       |          |
       |          v
       |     Decode z' --> intention text (via CSC/xRAG)
       |          |
       |          v
       |     Coding agent conditioned on intention --> patch
       |          |
       |          v
       |     SWE-bench harness --> score
       |          |
       |          v
       +---- Update current solution, tabu list
       |
       v
  Resolved patch (or budget exhausted)
```

## Results

| Model | Method | Params | Res. (%) |
|---|---|---|---|
| Devstral-Small | mini-SWE-agent (official) | 24B | 56.4 |
| Devstral-Small | mini-SWE-agent (limited) | 24B | 45.2 |
| **Devstral-Small** | **Intention search + mini-SWE-agent** | **24B + 14B** | **70.2** |
| Gemini 3 Pro | mini-SWE-agent | --- | 69.6 |
| DeepSeek V3.2 | mini-SWE-agent | 685B | 70.0 |
| Claude Sonnet 4.5 | mini-SWE-agent | --- | 71.4 |
| GPT-5.2 | mini-SWE-agent | --- | 72.8 |

## Reproducing Paper Results

### Hardware Requirements

- **GPUs**: 4x GPUs with ~80 GB VRAM total (e.g., 4x A100 40GB or 2x A100 80GB)
  - vLLM serves Devstral-Small-2507 (24B, fp8) across 4 GPUs
  - CSC server loads xRAG-7B + SFR-Embedding-Mistral (~14B total, 4-bit quantized)
- **Docker/Podman**: Required for SWE-bench containerized evaluation
- **Python 3.11+**

### Required Models

Download these HuggingFace models (downloaded automatically on first run):
- `mistralai/Devstral-Small-2507` — Base coding agent (24B)
- `Hannibal046/xrag-7b` — Decoder: embedding to text intention
- `Salesforce/SFR-Embedding-Mistral` — Encoder: text to embedding

### Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit environment variables
cp .env.example .env
source .env

# 3. Run the full pipeline on a single instance
./run_pipeline_tabu.sh

# 4. Reproduce full paper results (500 instances, R=50 rounds)
./reproduce.sh
```

### Step-by-Step Manual Setup

```bash
# 1. Start vLLM server (Devstral-Small, fp8 quantization)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Devstral-Small-2507 \
  --port 8080 \
  --tokenizer_mode mistral --config_format mistral --load_format mistral \
  --tensor-parallel-size 4 \
  --max-model-len 64000 \
  --quantization fp8

# 2. Start CSC Intention Server (Tabu Search + xRAG)
CSC_QUANTIZATION=4bit python -m uvicorn csc_server:app --host 0.0.0.0 --port 8000

# 3. Run on a single instance
export OPENAI_API_BASE="http://localhost:8080/v1"
export OPENAI_API_KEY="dummy"
export MSWEA_COST_TRACKING="ignore_errors"

python run_dataset.py \
  --csc http://127.0.0.1:8000 \
  --subset bash_only --split dev \
  --mini_model openai/mistralai/Devstral-Small-2507 \
  --environment_class docker \
  --rounds 50 --k 1 \
  --algorithm tabu \
  --tabu_tenure 50 \
  --sigma_local 300 \
  --sigma_kick 1000 \
  --stagnation_threshold 5 \
  --kick_probability 0.25 \
  --max_instances 1 \
  --resume
```

## Project Structure

```
.
├── csc_server.py                # CSC Intention Server (FastAPI + Tabu Search + xRAG)
├── run_dataset.py               # Dataset runner: bootstrap + search for each instance
├── run_pipeline_tabu.sh         # Automated pipeline (start servers + run search)
├── run_pipeline_bootstrap.sh    # Bootstrap-only pipeline (for ablation experiments)
├── reproduce.sh                 # Reproduce paper results (Table 1)
├── docker_api.py                # Docker API wrapper for environments without docker CLI
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
│
├── csc_swe_loop/                # Core package
│   ├── intent_client.py         #   HTTP client for CSC server
│   ├── mini_runner.py           #   Wrapper for mini-swe-agent
│   ├── swebench_eval.py         #   SWE-bench harness integration
│   ├── swebench_utils.py        #   Dataset loading utilities
│   ├── scoring.py               #   fail-to-pass x pass-to-pass scoring
│   └── prompts/
│       └── mini_task.md         #   Agent task prompt template
│
├── scripts/                     # Utility scripts
│   ├── check_intention_repeatability.py
│   └── max_solved_index.py
│
├── analysis/                    # Analysis and evaluation scripts
│   ├── analyze_runs.py
│   ├── analyze_tabu_intentions.py
│   ├── analyze_ablation.py
│   ├── sample_instances.py
│   └── test_scoring.py
│
└── future_work/                 # Alternative search variants (not in paper)
    ├── csc_server_cmaes.py      #   CMA-ES search server
    ├── csc_server_ga.py         #   Genetic algorithm search server
    ├── bootstrap_then_cma.py    #   CMA-ES bootstrap script
    ├── cma_loop.py              #   CMA-ES optimization loop
    └── run_pipeline_cmaes.sh    #   CMA-ES pipeline script
```

## CSC Module (Continuous Semantic Conditioning)

The CSC module provides the encoder (phi) and decoder (delta) for the latent intention space:

- **Encoder** (`Salesforce/SFR-Embedding-Mistral`): Maps text intentions to 4096-dimensional embeddings
- **Decoder** (`Hannibal046/xrag-7b`): Decodes embedding vectors back to natural-language intentions via the xRAG framework

This implements the continuous semantic conditioning framework referenced in the paper. The CSC server runs as a FastAPI service that the search loop queries for intention generation and embedding.

## Scoring

Patches are scored by the SWE-bench harness:

```
score = fail_to_pass_ratio x pass_to_pass_ratio
```

- `fail_to_pass (f2p)`: Fraction of originally failing tests that now pass
- `pass_to_pass (p2p)`: Fraction of originally passing tests that still pass
- `resolved`: True iff f2p = 1.0 and p2p = 1.0

## Other Search Variants

The `future_work/` directory contains alternative search strategies explored during development:
- **CMA-ES**: Covariance Matrix Adaptation Evolution Strategy in embedding space
- **Genetic Algorithm**: Population-based search with crossover and mutation

These are not reported in the paper but may be useful for future research.

## Citation

```bibtex
@inproceedings{bystronski2026latent-intention-search,
  title     = {Beyond Discrete Search: Divergent Thinking as Optimization in Latent Intention Space},
  author    = {Bystro\'{n}ski, Mateusz},
  year      = {2026},
  note      = {Under review}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
