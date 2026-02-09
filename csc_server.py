#!/usr/bin/env python
"""
CSC Intention Server (FastAPI)

Stateful server that:
- accepts a "task" with anchors (seed intentions) and optional context
- runs CMA-ES in embedding space
- decodes CMA points into intentions via xRAG (retrieval-embedding conditioned generation)
- returns batches of candidate intentions
- accepts feedback scores and updates CMA-ES state (tell)

Endpoints:
- PUT  /tasks/{task_id}         -> set/reset task, anchors, CMA config
- POST /tasks/{task_id}/suggest -> ask CMA-ES and decode to intentions (returns batch_id + candidates)
- POST /tasks/{task_id}/feedback-> tell CMA-ES with (candidate_id, score)
- GET  /tasks/{task_id}/state   -> debug state

Run:
  pip install fastapi uvicorn httpx pydantic cma torch transformers
  uvicorn csc_server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import cma
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# --- xRAG / SFR imports ---
from xRAG.src.model import SFR, XMistralForCausalLM
from xRAG.src.language_modeling.utils import get_retrieval_embeds, XRAG_TOKEN


# =============================================================================
# Globals: models (loaded on startup)
# =============================================================================
device = None
retriever_device = None
_dtype_llm = None
_dtype_retr = None

llm = None
llm_tokenizer = None
retriever = None
retriever_tokenizer = None


def initialize_models() -> None:
    """Initialize device, dtypes, and models with automatic device mapping."""
    global device, retriever_device, _dtype_llm, _dtype_retr
    global llm, llm_tokenizer, retriever, retriever_tokenizer
    
    import sys
    print("[CSC] initialize_models() called", flush=True)
    sys.stdout.flush()

    if torch.cuda.is_available():
        _dtype_llm = torch.float16
        _dtype_retr = torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        _dtype_llm = torch.float16
        _dtype_retr = torch.float16
    else:
        _dtype_llm = torch.float32
        _dtype_retr = torch.float32

    llm_name_or_path = os.environ.get("CSC_XRAG_LLM", "Hannibal046/xrag-7b")
    retriever_name_or_path = os.environ.get("CSC_XRAG_RETR", "Salesforce/SFR-Embedding-Mistral")

    print(f"[CSC] Loading LLM: {llm_name_or_path} (dtype={_dtype_llm}) device_map=auto")
    _llm = XMistralForCausalLM.from_pretrained(
        llm_name_or_path,
        torch_dtype=_dtype_llm,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()

    _llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name_or_path,
        add_eos_token=False,
        use_fast=False,
        padding_side="left",
    )
    _llm.set_xrag_token_id(_llm_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

    # Patch prepare_inputs_embeds (as in your script)
    def patched_prepare_inputs_embeds(input_ids, retrieval_embeds):
        inputs_embeds = _llm.model.embed_tokens(input_ids)
        retrieval_embeds = retrieval_embeds.view(-1, _llm.retriever_hidden_size)

        num_xrag_tokens = torch.sum(input_ids == _llm.xrag_token_id).item()
        num_retrieval_embeds = retrieval_embeds.shape[0]
        assert num_xrag_tokens == num_retrieval_embeds, (num_xrag_tokens, num_retrieval_embeds)

        retrieval_embeds = _llm.projector(retrieval_embeds.to(inputs_embeds.dtype))
        retrieval_embeds = retrieval_embeds.to(inputs_embeds.device)
        inputs_embeds[input_ids == _llm.xrag_token_id] = retrieval_embeds
        return inputs_embeds

    _llm.prepare_inputs_embeds = patched_prepare_inputs_embeds

    # Infer device for input tensors (embedding layer)
    if hasattr(_llm, "hf_device_map") and _llm.hf_device_map:
        embed_layer_names = ["model.embed_tokens", "embed_tokens"]
        device_candidate = None
        for name in embed_layer_names:
            if name in _llm.hf_device_map:
                device_candidate = torch.device(_llm.hf_device_map[name])
                break
        if device_candidate is None:
            try:
                device_candidate = next(_llm.model.embed_tokens.parameters()).device
            except Exception:
                device_candidate = torch.device(next(iter(_llm.hf_device_map.values())))
        _device = device_candidate
    else:
        try:
            _device = next(_llm.model.embed_tokens.parameters()).device
        except Exception:
            _device = _llm.device if hasattr(_llm, "device") else torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )

    print(f"[CSC] Using device for LLM inputs: {_device}")

    print(f"[CSC] Loading retriever: {retriever_name_or_path} (dtype={_dtype_retr}) device_map=auto")
    _retriever = SFR.from_pretrained(
        retriever_name_or_path,
        torch_dtype=_dtype_retr,
        device_map="auto",
    ).eval()
    _retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)

    if hasattr(_retriever, "hf_device_map") and _retriever.hf_device_map:
        _retriever_device = torch.device(next(iter(_retriever.hf_device_map.values())))
    else:
        _retriever_device = _retriever.device if hasattr(_retriever, "device") else _device

    print(f"[CSC] Using device for retriever inputs: {_retriever_device}")
    print("[CSC] Models loaded.")

    llm = _llm
    llm_tokenizer = _llm_tokenizer
    retriever = _retriever
    retriever_tokenizer = _retriever_tokenizer
    device = _device
    retriever_device = _retriever_device


# =============================================================================
# xRAG helpers
# =============================================================================
rag_template = """[INST] Background: {document}

Question: {prompt} [/INST] Propose fix intention based on the background. The answer is:"""


def embed_text(documents: List[str]) -> torch.Tensor:
    """Return retrieval embeddings tensor shape [B, D] for the given documents."""
    with torch.no_grad():
        toks = retriever_tokenizer(
            documents,
            max_length=1024,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        embs = get_retrieval_embeds(
            retriever,
            input_ids=toks["input_ids"].to(retriever_device),
            attention_mask=toks["attention_mask"].to(retriever_device),
        )
    return embs


def generate_from_embedding(
    embedding: torch.Tensor,
    prompt: str,
    max_new_tokens: int = 120,
    do_sample: bool = False,
) -> str:
    """Use xRAG-7B with a single retrieval embedding to generate text."""
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)

    embedding = embedding.to(device)

    formatted_prompt = rag_template.format_map(dict(document=XRAG_TOKEN, prompt=prompt))
    encoded = llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)

    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=llm_tokenizer.pad_token_id,
            retrieval_embeds=embedding,
        )
    text = llm_tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    return text.split("The answer is:", 1)[-1].strip()


def refine_intention_for_task(
    intention: str,
    task_context: str,
    max_new_tokens: int = 300,
    do_sample: bool = False,
) -> str:
    """Refine/adjust an intention to fit a given task context. Include code details."""
    # Extract only bug description, not instructions
    ctx = extract_bug_description(task_context)
    # Limit to 3000 chars for prompt
    ctx = ctx[:3000] if len(ctx) > 3000 else ctx

    sys_msg = """Adapt the fix intention to this specific bug context.

Your output MUST include:
1. What causes the bug (1-2 sentences)
2. The fix approach with specific file/function names
3. Code snippet showing the key change

Be detailed and specific. Include actual code."""

    user_msg = f"""Bug Context:
{ctx}

Generic Intention:
{intention}

Specific detailed fix plan:"""

    formatted_prompt = f"[INST] {sys_msg}\n\n{user_msg} [/INST]"

    encoded = llm_tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)
    attention_mask = encoded.attention_mask.to(device)
    input_length = input_ids.shape[1]

    with torch.no_grad():
        out = llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            pad_token_id=llm_tokenizer.pad_token_id,
        )

    generated_ids = out[0][input_length:]
    s = llm_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return s.strip()


def extract_bug_description(task_context: str) -> str:
    """
    Extract only the bug description from task_context, removing instructions.
    Looks for <pr_description>...</pr_description> or extracts content before <instructions>.
    """
    if not task_context:
        return "Bug context not provided."
    
    # Try to extract <pr_description> section
    if "<pr_description>" in task_context and "</pr_description>" in task_context:
        start = task_context.find("<pr_description>") + len("<pr_description>")
        end = task_context.find("</pr_description>")
        if start > len("<pr_description>") and end > start:
            return task_context[start:end].strip()
    
    # If no pr_description tags, extract everything before <instructions>
    if "<instructions>" in task_context:
        end = task_context.find("<instructions>")
        return task_context[:end].strip()
    
    # Fallback: return first 2000 chars (should contain the bug description)
    return task_context[:2000].strip()


def looks_like_junk(s: str) -> bool:
    """Check if intention is empty or gibberish. Code is ALLOWED now."""
    s = (s or "").strip()
    if not s:
        return True
    if len(s) < 10:
        return True
    # Only reject if it's just repeated characters or obvious garbage
    if len(set(s)) < 5:
        return True
    return False


# =============================================================================
# API models
# =============================================================================
class Anchor(BaseModel):
    id: str
    text: str
    weight: float = 1.0


class CMAConfig(BaseModel):
    """
    CMA in *embedding space*.
    dim can be omitted; if omitted we infer it from the retriever embedding size.
    """
    dim: Optional[int] = None
    sigma0: float = 10.0  # Very large sigma for maximum diversity in embedding space
    popsize: int = 32
    seed: int = 123
    # Bounds for CMA-ES in pycma are typically [lower, upper] scalars or per-dim arrays.
    bounds: Optional[List[float]] = None  # e.g. [-3.0, 3.0]
    maxiter: Optional[int] = None


class TaskSpec(BaseModel):
    """
    task_context: the bug/issue context used in prompts (your BENCHMARK_TASK or agent-provided).
    intention_prompt: optional override prompt to generate raw intentions from embeddings.
    """
    task_context: str = ""
    anchors: List[Anchor] = Field(default_factory=list)
    use_refinement: bool = False
    intention_prompt: Optional[str] = None
    # Novelty bonus: adds diversity component to score
    novelty_lambda: float = 0.0  # weight for novelty component: score_total = score_task + λ * novelty
    novelty_mode: str = "batch_distance"  # "anchor_distance" or "batch_distance"
    novelty_decay_tau: float = 0.0  # if >0, novelty weight decays: λ_eff = λ * 0.5^(iter_updates/tau). 0 = no decay.


class SetTaskReq(BaseModel):
    task: TaskSpec
    cma: CMAConfig


class Candidate(BaseModel):
    candidate_id: str
    z: List[float]
    intention_raw: str
    intention: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class SuggestReq(BaseModel):
    n: int = 16
    # generation params
    max_new_tokens: int = 120
    do_sample: bool = False


class SuggestResp(BaseModel):
    task_id: str
    batch_id: str
    candidates: List[Candidate]


class Evaluation(BaseModel):
    candidate_id: str
    score: float
    aux: Optional[Dict[str, Any]] = None


class FeedbackReq(BaseModel):
    batch_id: str
    evaluations: List[Evaluation]
    maximize: bool = True  # if True, CMA cost = -score


class FeedbackResp(BaseModel):
    task_id: str
    updated: int
    best: Optional[Dict[str, Any]] = None


# =============================================================================
# Server state
# =============================================================================
@dataclass
class PendingCandidate:
    x: np.ndarray
    z_hash: str
    intention_raw: str
    intention: str
    meta: Dict[str, Any]


class TaskState:
    def __init__(self, spec: TaskSpec, cfg: CMAConfig):
        self.spec = spec
        self.cfg = cfg
        self.lock = asyncio.Lock()

        # infer dim from retriever embeddings if not provided
        inferred_dim = None
        if spec.anchors:
            with torch.no_grad():
                e = embed_text([spec.anchors[0].text])
            inferred_dim = int(e.shape[-1])

        dim = cfg.dim or inferred_dim
        if dim is None:
            raise ValueError("Cannot infer dim: provide cma.dim or provide at least 1 anchor.")

        self.dim = dim

        # CMA init at mean=anchor centroid (in embedding space) if anchors provided; else zeros
        # Also store anchor embeddings for novelty calculation
        if spec.anchors:
            texts = [a.text for a in spec.anchors]
            weights = np.array([a.weight for a in spec.anchors], dtype=np.float64)
            weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)

            with torch.no_grad():
                E = embed_text(texts).detach().to("cpu").numpy().astype(np.float64)  # [A,D]
            x0 = (E * weights[:, None]).sum(axis=0)
            # Store anchor embeddings for novelty calculation
            self.anchor_embeddings = E  # [A, D]
        else:
            x0 = np.zeros(dim, dtype=np.float64)
            self.anchor_embeddings = None

        opts: Dict[str, Any] = {"popsize": cfg.popsize, "seed": cfg.seed}
        if cfg.maxiter is not None:
            opts["maxiter"] = cfg.maxiter
        if cfg.bounds is not None:
            opts["bounds"] = cfg.bounds

        self.es = cma.CMAEvolutionStrategy(x0, cfg.sigma0, opts)

        # pending batches: batch_id -> candidate_id -> PendingCandidate
        self.pending: Dict[str, Dict[str, PendingCandidate]] = {}

        # basic cache to avoid regenerating for same z (hash)
        self.cache: Dict[str, Tuple[str, str]] = {}  # z_hash -> (raw, refined)

        # stats
        self.iter_updates = 0
        self.best_score = None
        self.best_candidate = None
        self.last_update_ts = time.time()

    def _update_popsize(self, new_popsize: int) -> None:
        """
        Update CMA-ES popsize while preserving optimization state.
        Reinitializes CMA-ES with new popsize, preserving mean and sigma.
        """
        if self.es.popsize == new_popsize:
            return
        
        # Save current state
        x0 = self.es.mean
        sigma0 = self.es.sigma
        
        # Reinitialize with new popsize
        opts: Dict[str, Any] = {"popsize": new_popsize, "seed": self.cfg.seed}
        if self.cfg.maxiter is not None:
            opts["maxiter"] = self.cfg.maxiter
        if self.cfg.bounds is not None:
            opts["bounds"] = self.cfg.bounds
        
        self.es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        # Restore iteration count (approximate, since we reinitialized)
        # Note: This is a simplification - full state restoration would be more complex

    def _compute_novelty(
        self, 
        z: np.ndarray, 
        other_z_in_batch: List[np.ndarray],
        mode: str
    ) -> float:
        """
        Compute novelty score for a candidate z.
        
        Args:
            z: candidate embedding vector [D]
            other_z_in_batch: list of other candidate embeddings in the same batch
            mode: "anchor_distance" or "batch_distance"
        
        Returns:
            novelty score (higher = more novel/diverse)
        """
        if mode == "anchor_distance":
            # Novelty = min cosine distance from anchors (higher distance = more novel)
            if self.anchor_embeddings is None or len(self.anchor_embeddings) == 0:
                return 0.0
            
            # Normalize z and anchor embeddings for cosine distance
            z_norm = z / (np.linalg.norm(z) + 1e-10)
            anchor_norms = self.anchor_embeddings / (
                np.linalg.norm(self.anchor_embeddings, axis=1, keepdims=True) + 1e-10
            )
            
            # Cosine similarity (1 - cosine_distance)
            cosines = np.dot(anchor_norms, z_norm)  # [A]
            # Cosine distance = 1 - cosine similarity
            distances = 1.0 - cosines
            # Novelty = minimum distance (how far from closest anchor)
            return float(np.min(distances))
        
        elif mode == "batch_distance":
            # Novelty = minimum distance from other candidates in batch
            if len(other_z_in_batch) == 0:
                return 1.0  # If alone in batch, maximum novelty
            
            # Normalize for cosine distance
            z_norm = z / (np.linalg.norm(z) + 1e-10)
            other_norms = np.array([
                z_other / (np.linalg.norm(z_other) + 1e-10) 
                for z_other in other_z_in_batch
            ])  # [N, D]
            
            # Cosine similarities
            cosines = np.dot(other_norms, z_norm)  # [N]
            # Cosine distance = 1 - cosine similarity
            distances = 1.0 - cosines
            # Novelty = minimum distance (how far from closest other candidate)
            return float(np.min(distances))
        
        else:
            raise ValueError(f"Unknown novelty_mode: {mode}")

    def _hash_z(self, x: np.ndarray) -> str:
        # float64 -> stable bytes. You can quantize if needed.
        b = np.asarray(x, dtype=np.float32).tobytes()
        return hashlib.sha256(b).hexdigest()

    def _default_intention_prompt(self) -> str:
        # Extract only bug description (pr_description), not instructions
        ctx = extract_bug_description(self.spec.task_context)
        # Keep it simple: this prompt is what xRAG answers to, conditioned on embedding.
        return f"""Describe in ONE sentence the core fix approach for this bug.

Rules:
- State WHAT to change conceptually (not HOW or code)
- Be concise: max 20 words
- No code blocks

Bug Context:
{ctx}

Core fix approach (one sentence):"""

    def _decode_candidate(self, x: np.ndarray, max_new_tokens: int, do_sample: bool) -> Tuple[str, str]:
        """
        Convert CMA point x (embedding space) -> raw intention (xRAG) -> refined intention (optional).
        Returns (raw, refined)
        """
        z_hash = self._hash_z(x)
        if z_hash in self.cache:
            return self.cache[z_hash]

        emb = torch.tensor(x, device=device, dtype=_dtype_retr)

        prompt = self.spec.intention_prompt or self._default_intention_prompt()
        raw = generate_from_embedding(
            emb,
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        ).strip()

        # Keep more context to preserve diversity - don't truncate to first sentence
        # This allows different embeddings to generate different intentions
        raw = raw.strip()

        if looks_like_junk(raw):
            # hard fallback: short generic (still deterministic)
            raw = "Propose a minimal, targeted logic change to prevent the false positive rule trigger."

        refined = raw
        if self.spec.use_refinement and self.spec.task_context.strip():
            refined = refine_intention_for_task(raw, self.spec.task_context).strip()
            if ". " in refined:
                refined = refined.split(". ")[0].strip() + "."
            refined = refined.strip()

        self.cache[z_hash] = (raw, refined)
        return raw, refined


TASKS: Dict[str, TaskState] = {}


# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="CSC Intention Server", version="0.1")


@app.on_event("startup")
def _startup():
    import sys
    try:
        print("[CSC] Starting model initialization...", flush=True)
        sys.stdout.flush()
        initialize_models()
        print("[CSC] Model initialization completed successfully.", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[CSC] ERROR during model initialization: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        # Don't raise - let server start but endpoints will return 503


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "models_loaded": llm is not None and retriever is not None,
        "tasks_count": len(TASKS),
    }


@app.get("/")
async def root():
    """Root endpoint for testing."""
    return {
        "service": "CSC Intention Server",
        "version": "0.1",
        "endpoints": [
            "GET /health",
            "PUT /tasks/{task_id}",
            "POST /tasks/{task_id}/suggest",
            "POST /tasks/{task_id}/feedback",
            "GET /tasks/{task_id}/state",
        ],
        "models_loaded": llm is not None and retriever is not None,
    }


@app.put("/tasks/{task_id}")
async def set_task(task_id: str, req: SetTaskReq):
    """
    Set or reset a task.
    This resets CMA-ES state for that task_id.
    """
    # Check if models are initialized
    if llm is None or retriever is None:
        raise HTTPException(
            status_code=503, 
            detail="CSC server models not initialized. Check server logs."
        )
    
    try:
        st = TaskState(req.task, req.cma)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    TASKS[task_id] = st
    print(f"[CSC] Task {task_id} created: novelty_lambda={st.spec.novelty_lambda}, novelty_mode={st.spec.novelty_mode}")
    return {
        "task_id": task_id,
        "status": "ready",
        "dim": st.dim,
        "popsize": st.cfg.popsize,
        "use_refinement": st.spec.use_refinement,
        "novelty_lambda": st.spec.novelty_lambda,
        "novelty_mode": st.spec.novelty_mode,
    }


@app.post("/tasks/{task_id}/suggest", response_model=SuggestResp)
async def suggest(task_id: str, req: SuggestReq):
    """
    Ask CMA-ES for candidate points and decode them into intentions.
    Returns batch_id for later feedback().
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]

    if req.n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    async with st.lock:
        # Set popsize to match req.n for consistency between ask() and tell()
        # This ensures CMA-ES gets feedback for all points it generates
        st._update_popsize(req.n)
        
        X = st.es.ask()
        # X should now have exactly req.n points (or we take first req.n if there's a mismatch)
        if len(X) > req.n:
            X = X[:req.n]

        batch_id = uuid.uuid4().hex
        st.pending[batch_id] = {}

        candidates: List[Candidate] = []
        for x in X:
            x = np.asarray(x, dtype=np.float64)
            cid = uuid.uuid4().hex
            z_hash = st._hash_z(x)

            raw, refined = st._decode_candidate(x, max_new_tokens=req.max_new_tokens, do_sample=req.do_sample)

            meta = {
                "z_hash": z_hash,
                "refined": st.spec.use_refinement,
                "ts": time.time(),
            }

            st.pending[batch_id][cid] = PendingCandidate(
                x=x,
                z_hash=z_hash,
                intention_raw=raw,
                intention=refined,
                meta=meta,
            )

            candidates.append(
                Candidate(
                    candidate_id=cid,
                    z=list(map(float, x.tolist())),
                    intention_raw=raw,
                    intention=refined,
                    meta=meta,
                )
            )

        return SuggestResp(task_id=task_id, batch_id=batch_id, candidates=candidates)


@app.post("/tasks/{task_id}/feedback", response_model=FeedbackResp)
async def feedback(task_id: str, req: FeedbackReq):
    """
    Provide evaluation scores for candidates from a batch.
    Performs CMA-ES tell().

    maximize=True => CMA cost = -score (so higher score is better)
    """
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]

    async with st.lock:
        if req.batch_id not in st.pending:
            raise HTTPException(status_code=400, detail="unknown batch_id")
        batch = st.pending.pop(req.batch_id)

        # Collect all candidate embeddings for batch_distance novelty calculation
        all_z_in_batch = [pc.x for pc in batch.values()]

        # align X and costs with novelty bonus
        X: List[np.ndarray] = []
        costs: List[float] = []
        best_local = None

        for ev in req.evaluations:
            if ev.candidate_id not in batch:
                continue
            pc = batch[ev.candidate_id]
            X.append(pc.x)
            
            # Compute novelty bonus if enabled (with optional decay over time)
            score_task = float(ev.score)
            novelty_bonus = 0.0
            
            if st.spec.novelty_lambda > 0.0:
                # Decay: early iterations reward diversity, later focus on task score
                if st.spec.novelty_decay_tau > 0:
                    decay = 0.5 ** (st.iter_updates / st.spec.novelty_decay_tau)
                    lambda_eff = st.spec.novelty_lambda * decay
                else:
                    lambda_eff = st.spec.novelty_lambda
                if lambda_eff > 0:
                    other_z = [z for z in all_z_in_batch if not np.array_equal(z, pc.x)]
                    novelty = st._compute_novelty(
                        pc.x, 
                        other_z, 
                        st.spec.novelty_mode
                    )
                    novelty_bonus = lambda_eff * novelty
            
            # Total score = task score + novelty bonus
            score_total = score_task + novelty_bonus
            
            # Convert to cost for CMA-ES (CMA minimizes, so cost = -score if maximizing)
            cost = -score_total if req.maximize else score_total
            costs.append(cost)

            # Track best candidate using total score (including novelty)
            if best_local is None or score_total > best_local["score"]:
                best_local = {
                    "candidate_id": ev.candidate_id,
                    "score": score_total,
                    "score_task": score_task,
                    "novelty": novelty_bonus / st.spec.novelty_lambda if st.spec.novelty_lambda > 0 else 0.0,
                    "intention": pc.intention,
                    "intention_raw": pc.intention_raw,
                    "aux": ev.aux,
                }

        if not X:
            raise HTTPException(status_code=400, detail="no matching evaluations in batch")

        st.es.tell(X, costs)
        st.iter_updates += 1
        st.last_update_ts = time.time()
        
        # Log novelty info
        if st.spec.novelty_lambda > 0:
            print(f"[CSC] feedback: {len(X)} evals, novelty_lambda={st.spec.novelty_lambda}, costs={[f'{c:.4f}' for c in costs]}")

        # track best overall (in "score" coordinates, not cost)
        if best_local is not None:
            if st.best_score is None or best_local["score"] > st.best_score:
                st.best_score = best_local["score"]
                st.best_candidate = best_local

        return FeedbackResp(
            task_id=task_id,
            updated=len(X),
            best=st.best_candidate,
        )


@app.get("/tasks/{task_id}/state")
async def state(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]
    async with st.lock:
        return {
            "task_id": task_id,
            "dim": st.dim,
            "popsize": st.cfg.popsize,
            "sigma0": st.cfg.sigma0,
            "iter_updates": st.iter_updates,
            "best_score": st.best_score,
            "best_candidate": st.best_candidate,
            "pending_batches": len(st.pending),
            "last_update_ts": st.last_update_ts,
        }
