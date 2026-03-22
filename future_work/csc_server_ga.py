#!/usr/bin/env python
"""
CSC Intention Server (FastAPI) – Genetic Algorithm Version

Replaces CMA-ES with a steady-state Genetic Algorithm in embedding space:
- Individuals are retriever-embedding vectors x (float64 numpy arrays)
- Crossover: affine mixing / interpolation-extrapolation with lambda (λ)
      child = (1-λ)*a + λ*b
  where λ can be sampled from unions of intervals
- Mutation: gaussian noise (sigma)
- Population update: keep bounded, replace worst
- Hard ban: bad intentions (score=0, empty patch) are banned and never proposed again

Endpoints:
- PUT  /tasks/{task_id}         -> set/reset task, anchors, GA config
- POST /tasks/{task_id}/suggest -> generate intentions from GA
- POST /tasks/{task_id}/feedback-> update population with scores
- GET  /tasks/{task_id}/state   -> debug state
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
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
    global device, retriever_device, _dtype_llm, _dtype_retr
    global llm, llm_tokenizer, retriever, retriever_tokenizer

    import sys
    print("[CSC-GA] initialize_models() called", flush=True)
    sys.stdout.flush()

    # Get dtype from environment variable (float16, bfloat16, float32, auto)
    csc_dtype = os.environ.get("CSC_DTYPE", "auto").lower()
    # Get quantization from environment variable (none, 4bit, 8bit)
    csc_quantization = os.environ.get("CSC_QUANTIZATION", "none").lower()
    
    # Determine dtype
    if csc_dtype == "bfloat16":
        _dtype_llm = torch.bfloat16
        _dtype_retr = torch.bfloat16
    elif csc_dtype == "float32":
        _dtype_llm = torch.float32
        _dtype_retr = torch.float32
    elif csc_dtype == "float16":
        _dtype_llm = torch.float16
        _dtype_retr = torch.float16
    else:  # auto
        if torch.cuda.is_available():
            _dtype_llm = torch.float16
            _dtype_retr = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _dtype_llm = torch.float16
            _dtype_retr = torch.float16
        else:
            _dtype_llm = torch.float32
            _dtype_retr = torch.float32
    
    print(f"[CSC-GA] Configured dtype: {csc_dtype} -> torch dtype: {_dtype_llm}")
    print(f"[CSC-GA] Configured quantization: {csc_quantization}")

    llm_name_or_path = os.environ.get("CSC_XRAG_LLM", "Hannibal046/xrag-7b")
    retriever_name_or_path = os.environ.get("CSC_XRAG_RETR", "Salesforce/SFR-Embedding-Mistral")

    # Build loading kwargs based on quantization
    llm_load_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    
    if csc_quantization == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            llm_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_dtype_llm,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("[CSC-GA] Using 4-bit quantization (bitsandbytes NF4)")
        except ImportError:
            print("[CSC-GA] WARNING: bitsandbytes not available, falling back to no quantization")
            llm_load_kwargs["torch_dtype"] = _dtype_llm
    elif csc_quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            llm_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("[CSC-GA] Using 8-bit quantization (bitsandbytes)")
        except ImportError:
            print("[CSC-GA] WARNING: bitsandbytes not available, falling back to no quantization")
            llm_load_kwargs["torch_dtype"] = _dtype_llm
    else:
        # No quantization - use specified dtype
        llm_load_kwargs["torch_dtype"] = _dtype_llm

    print(f"[CSC-GA] Loading LLM: {llm_name_or_path} (dtype={_dtype_llm}, quant={csc_quantization}) device_map=auto")
    _llm = XMistralForCausalLM.from_pretrained(
        llm_name_or_path,
        **llm_load_kwargs,
    ).eval()

    _llm_tokenizer = AutoTokenizer.from_pretrained(
        llm_name_or_path,
        add_eos_token=False,
        use_fast=False,
        padding_side="left",
    )
    _llm.set_xrag_token_id(_llm_tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

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

    if hasattr(_llm, "hf_device_map") and _llm.hf_device_map:
        device_candidate = None
        for name in ["model.embed_tokens", "embed_tokens"]:
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

    print(f"[CSC-GA] Using device for LLM inputs: {_device}")

    print(f"[CSC-GA] Loading retriever: {retriever_name_or_path} (dtype={_dtype_retr}) device_map=auto")
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

    print(f"[CSC-GA] Using device for retriever inputs: {_retriever_device}")
    print("[CSC-GA] Models loaded.")

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
    with torch.no_grad():
        toks = retriever_tokenizer(
            documents,
            max_length=512,
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


def extract_bug_description(task_context: str) -> str:
    """Extract only the bug description from task_context, removing instructions."""
    if not task_context:
        return "Bug context not provided."
    
    if "<pr_description>" in task_context and "</pr_description>" in task_context:
        start = task_context.find("<pr_description>") + len("<pr_description>")
        end = task_context.find("</pr_description>")
        if start > len("<pr_description>") and end > start:
            return task_context[start:end].strip()
    
    if "<instructions>" in task_context:
        end = task_context.find("<instructions>")
        return task_context[:end].strip()
    
    return task_context[:2000].strip()


def looks_like_junk(s: str) -> bool:
    s = (s or "").strip()
    if not s:
        return True
    if len(s) < 10:
        return True
    if len(set(s)) < 5:
        return True
    return False


def _normalize_intention(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


# =============================================================================
# API models
# =============================================================================
class Anchor(BaseModel):
    id: str
    text: str
    weight: float = 1.0


class GAConfig(BaseModel):
    dim: Optional[int] = None
    seed: int = 123

    # population
    pop_size: int = 15
    elite_frac: float = 0.4
    min_pop_to_crossover: int = 3

    # mutation
    sigma: float = 1.0  # gaussian noise std in embedding space

    # crossover lambda sampling - three modes:
    # 1) Interpolation: local mixing / refinement (λ close to 0.5)
    interp_lo: float = 0.4
    interp_hi: float = 0.6
    # 2) Extrapolation: aggressive exploration (λ far from [0,1])
    extrap_lo: float = 3.0
    extrap_hi: float = 6.0
    # 3) Super-extrapolation: even more aggressive (paper peak ~10)
    extrap2_lo: float = 6.0
    extrap2_hi: float = 10.0
    # 4) Moderate extrapolation: buffer zone (λ slightly outside [0,1])
    mod_extrap_lo: float = 1.2
    mod_extrap_hi: float = 2.0

    # Schedule: probability of extrapolation decreases over iterations
    # p_extrap starts at extrap_prob_start, decays to extrap_prob_end over extrap_decay_iters
    extrap_prob_start: float = 0.95   # initial p(extrapolation)
    extrap_prob_end: float = 0.2     # final p(extrapolation)
    extrap_decay_iters: int = 100    # iterations to decay from start to end

    # Within extrapolation, split between normal extrap vs super-extrap
    super_extrap_frac: float = 0.3   # fraction of extrap that uses [6,10] instead of [3,6]

    # Legacy (kept for backwards compat, but ignored if new params are set)
    lam_lo: float = 3
    lam_hi: float = 6
    extrap_prob: float = 0.3

    # ban controls
    max_banned: int = 10000
    max_seen_intentions: int = 10000
    ban_threshold: float = 0.001  # score below this triggers soft-ban

    # pair memory
    pair_memory: int = 50000


class TaskSpec(BaseModel):
    task_context: str = ""
    anchors: List[Anchor] = Field(default_factory=list)
    use_refinement: bool = False
    intention_prompt: Optional[str] = None


class SetTaskReq(BaseModel):
    task: TaskSpec
    ga: GAConfig


class Candidate(BaseModel):
    candidate_id: str
    z: List[float]
    intention_raw: str
    intention: str
    meta: Dict[str, Any] = Field(default_factory=dict)


class SuggestReq(BaseModel):
    n: int = 8
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
    maximize: bool = True


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
    i_hash: str
    meta: Dict[str, Any]


@dataclass
class Individual:
    x: np.ndarray
    score: float
    z_hash: str
    i_hash: str
    intention: str
    intention_raw: str
    meta: Dict[str, Any]


class TaskState:
    def __init__(self, spec: TaskSpec, cfg: GAConfig):
        self.spec = spec
        self.cfg = cfg
        self.lock = asyncio.Lock()

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Population and tracking
        self.pop: List[Individual] = []
        self.pending: Dict[str, Dict[str, PendingCandidate]] = {}
        self.decode_cache: Dict[str, Tuple[str, str, str]] = {}

        # Banned embeddings and seen intentions
        self.banned_z: Dict[str, float] = {}
        self.banned_z_fifo: Deque[str] = deque()
        self.seen_i: Dict[str, float] = {}
        self.seen_i_fifo: Deque[str] = deque()

        # Pair memory to avoid repeating crossovers
        self.seen_pairs: Dict[str, float] = {}
        self.seen_pairs_fifo: Deque[str] = deque()

        self.iter_updates = 0  # rounds (one per feedback batch)
        self.total_evals = 0   # individual evaluations (for schedule: per-iteration)
        self.best_score: Optional[float] = None
        self.best_candidate: Optional[Dict[str, Any]] = None
        self.last_update_ts = time.time()

        # Infer dim
        inferred_dim = None
        if spec.anchors:
            with torch.no_grad():
                e = embed_text([spec.anchors[0].text])
            inferred_dim = int(e.shape[-1])

        dim = cfg.dim or inferred_dim
        if dim is None:
            raise ValueError("Cannot infer dim: provide ga.dim or provide at least 1 anchor.")
        self.dim = int(dim)

        # Init x0 as anchor centroid
        if spec.anchors:
            texts = [a.text for a in spec.anchors]
            weights = np.array([a.weight for a in spec.anchors], dtype=np.float64)
            weights = weights / (weights.sum() if weights.sum() > 0 else 1.0)
            with torch.no_grad():
                E = embed_text(texts).detach().float().to("cpu").numpy().astype(np.float64)
            x0 = (E * weights[:, None]).sum(axis=0)
            self.anchor_embeddings = E
        else:
            x0 = np.zeros(self.dim, dtype=np.float64)
            self.anchor_embeddings = None
        self.x0 = x0.astype(np.float64)

        # Seed population with anchors
        if spec.anchors:
            texts = [a.text for a in spec.anchors]
            with torch.no_grad():
                E = embed_text(texts).detach().float().to("cpu").numpy().astype(np.float64)

            for a, x in zip(spec.anchors, E):
                intention = (a.text or "").strip()
                if looks_like_junk(intention):
                    continue

                z_hash = self._hash_z(x)
                i_hash = self._hash_intention(intention)

                if z_hash in self.banned_z:
                    continue
                if i_hash in self.seen_i:
                    continue

                self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)

                seed_score = float(a.weight)
                self.pop.append(
                    Individual(
                        x=x.astype(np.float64),
                        score=seed_score,
                        z_hash=z_hash,
                        i_hash=i_hash,
                        intention=intention,
                        intention_raw=intention,
                        meta={"op": "anchor_seed", "anchor_id": a.id, "anchor_weight": float(a.weight)},
                    )
                )

            if len(self.pop) > int(self.cfg.pop_size):
                self.pop.sort(key=lambda p: p.score, reverse=True)
                self.pop = self.pop[:int(self.cfg.pop_size)]

    # ---------------- hashing & bookkeeping ----------------
    def _hash_z(self, x: np.ndarray) -> str:
        b = np.asarray(x, dtype=np.float32).tobytes()
        return hashlib.sha256(b).hexdigest()

    def _hash_intention(self, intention: str) -> str:
        return hashlib.sha256(_normalize_intention(intention).encode("utf-8")).hexdigest()

    def _remember_fifo(self, d: Dict[str, float], fifo: Deque[str], key: str, maxlen: int) -> None:
        if key in d:
            return
        d[key] = time.time()
        fifo.append(key)
        while len(fifo) > maxlen:
            old = fifo.popleft()
            d.pop(old, None)

    def _ban(self, z_hash: str, i_hash: Optional[str] = None) -> None:
        self._remember_fifo(self.banned_z, self.banned_z_fifo, z_hash, self.cfg.max_banned)
        if i_hash:
            self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)

    # ---------------- prompt / decode ----------------
    def _default_intention_prompt(self) -> str:
        ctx = extract_bug_description(self.spec.task_context)
        return f"""Describe in ONE sentence the core fix approach for this bug.

Rules:
- State WHAT to change conceptually (not HOW or code)
- Be concise: max 20 words
- No code blocks

Bug Context:
{ctx}

Core fix approach (one sentence):"""

    def _decode_candidate(self, x: np.ndarray, max_new_tokens: int, do_sample: bool) -> Tuple[str, str, str, str]:
        z_hash = self._hash_z(x)
        if z_hash in self.decode_cache:
            raw, refined, i_hash = self.decode_cache[z_hash]
            return z_hash, raw, refined, i_hash

        emb = torch.tensor(x, device=device, dtype=_dtype_retr)
        prompt = self.spec.intention_prompt or self._default_intention_prompt()
        raw = generate_from_embedding(emb, prompt, max_new_tokens=max_new_tokens, do_sample=True).strip()

        if looks_like_junk(raw):
            raw = "Propose a minimal, targeted logic change to prevent the false positive rule trigger."

        refined = raw
        i_hash = self._hash_intention(refined)
        self.decode_cache[z_hash] = (raw, refined, i_hash)
        return z_hash, raw, refined, i_hash

    # ---------------- GA mechanics ----------------
    def _compute_extrap_prob(self) -> float:
        """
        Compute current p(extrapolation) based on schedule.
        Uses total_evals (per-iteration) so each evaluated child counts.
        Starts at extrap_prob_start, decays to extrap_prob_end over extrap_decay_iters.
        Always clamped to [0.05, 0.95] to ensure some of each mode.
        """
        start = float(self.cfg.extrap_prob_start)
        end = float(self.cfg.extrap_prob_end)
        decay_iters = max(1, int(self.cfg.extrap_decay_iters))
        
        # Linear decay by total evaluations (each child = one step)
        progress = min(1.0, self.total_evals / decay_iters)
        p_ex = start + (end - start) * progress
        
        # Clamp to ensure we always have some of each mode
        return max(0.05, min(0.95, p_ex))

    def _sample_lambda(self) -> float:
        """
        Sample λ for crossover: child = (1-λ)*a + λ*b
        
        Three modes with schedule:
        1) Interpolation [0.4, 0.6]: local refinement, λ close to 0.5
        2) Extrapolation [3, 6]: aggressive exploration
        3) Super-extrapolation [6, 10]: even more aggressive (paper peak ~10)
        
        Schedule: early iterations favor extrapolation (exploration),
        later iterations favor interpolation (exploitation/refinement).
        """
        p_ex = self._compute_extrap_prob()
        
        # Decide: extrapolation vs interpolation
        if random.random() < p_ex:
            # Extrapolation mode
            super_frac = float(self.cfg.super_extrap_frac)
            
            # Decide: normal extrap [3,6] vs super-extrap [6,10] vs moderate [1.2,2.0]
            r = random.random()
            if r < super_frac:
                # Super-extrapolation [6, 10]
                lo = float(self.cfg.extrap2_lo)
                hi = float(self.cfg.extrap2_hi)
            elif r < super_frac + 0.2:
                # Moderate extrapolation [1.2, 2.0] - buffer zone (20% of extrap)
                lo = float(self.cfg.mod_extrap_lo)
                hi = float(self.cfg.mod_extrap_hi)
            else:
                # Normal extrapolation [3, 6]
                lo = float(self.cfg.extrap_lo)
                hi = float(self.cfg.extrap_hi)
            
            # Randomly go negative or positive
            lam_magnitude = lo + (hi - lo) * random.random()
            if random.random() < 0.5:
                return -lam_magnitude  # negative extrapolation
            else:
                return 1.0 + lam_magnitude  # positive extrapolation
        else:
            # Interpolation mode [0.4, 0.6] - local mixing
            interp_lo = float(self.cfg.interp_lo)
            interp_hi = float(self.cfg.interp_hi)
            return interp_lo + (interp_hi - interp_lo) * random.random()

    def _lambda_bucket(self, lam: float) -> str:
        """
        Bucket λ values for pair memory (avoid repeating similar crossovers).
        Updated for new ranges:
        - Interpolation [0.4, 0.6] → "interp"
        - Moderate extrap [1.2, 2.0] or [-2.0, -1.2] → "mod_pos" / "mod_neg"
        - Normal extrap [3, 6] → "ext_pos_N" / "ext_neg_N"
        - Super extrap [6, 10] → "sup_pos_N" / "sup_neg_N"
        """
        if lam < -6:
            # Super-extrapolation negative
            return f"sup_neg_{int(min(9, max(0, math.floor((abs(lam) - 6) * 2))))}"
        elif lam < -2:
            # Normal extrapolation negative [3, 6]
            return f"ext_neg_{int(min(9, max(0, math.floor((abs(lam) - 2) * 2))))}"
        elif lam < -1:
            # Moderate extrapolation negative [1.2, 2.0]
            return "mod_neg"
        elif lam < 0.3:
            # Unusual interpolation (shouldn't happen often with new config)
            return "edge_lo"
        elif lam <= 0.7:
            # Interpolation [0.4, 0.6] - bucket as single group
            return "interp"
        elif lam < 1.0:
            # Unusual interpolation
            return "edge_hi"
        elif lam < 2.0:
            # Moderate extrapolation positive [1.2, 2.0]
            return "mod_pos"
        elif lam < 7.0:
            # Normal extrapolation positive [3, 6] (starts at 1+3=4)
            return f"ext_pos_{int(min(9, max(0, math.floor((lam - 1) * 2))))}"
        else:
            # Super-extrapolation positive [6, 10] (starts at 1+6=7)
            return f"sup_pos_{int(min(9, max(0, math.floor((lam - 7) * 2))))}"

    def _pick_parent_index(self) -> int:
        n = len(self.pop)
        if n == 0:
            return -1
        idx = list(range(n))
        idx.sort(key=lambda i: self.pop[i].score, reverse=True)
        elite_n = max(1, int(self.cfg.elite_frac * n))
        elite = idx[:elite_n]
        if random.random() < 0.80:
            return random.choice(elite)
        return random.choice(idx)

    def _make_child_x(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        meta: Dict[str, Any] = {}

        if len(self.pop) < self.cfg.min_pop_to_crossover:
            # Warmup: noise around x0
            sigma = float(self.cfg.sigma)
            x = self.x0 + np.random.normal(0.0, sigma, size=(self.dim,)).astype(np.float64)
            meta.update({"op": "warmup_noise", "sigma": sigma})
            return x, meta

        i = self._pick_parent_index()
        j = self._pick_parent_index()
        if i < 0 or j < 0:
            sigma = float(self.cfg.sigma)
            x = self.x0 + np.random.normal(0.0, sigma, size=(self.dim,)).astype(np.float64)
            meta.update({"op": "fallback_noise", "sigma": sigma})
            return x, meta

        if i == j and len(self.pop) >= 2:
            j = (j + 1) % len(self.pop)

        a = self.pop[i]
        b = self.pop[j]

        lam = self._sample_lambda()
        lam_bucket = self._lambda_bucket(lam)

        ida = a.z_hash[:16]
        idb = b.z_hash[:16]
        pkey = "|".join(sorted([ida, idb])) + f"|{lam_bucket}"

        tries = 0
        while pkey in self.seen_pairs and tries < 6:
            lam = self._sample_lambda()
            lam_bucket = self._lambda_bucket(lam)
            pkey = "|".join(sorted([ida, idb])) + f"|{lam_bucket}"
            tries += 1

        self._remember_fifo(self.seen_pairs, self.seen_pairs_fifo, pkey, self.cfg.pair_memory)

        # Crossover: child = (1-λ)*a + λ*b
        x = (1.0 - lam) * a.x + lam * b.x
        print(f"Delta: {a.x - b.x}")
        # Mutation
        sigma = float(self.cfg.sigma)
        x = x + np.random.normal(0.0, sigma, size=x.shape).astype(np.float64)

        # Determine mode from lam value
        if -0.1 < lam < 1.1:
            lam_mode = "interp"
        elif abs(lam) < 3 or (lam > 1 and lam < 4):
            lam_mode = "mod_extrap"
        elif abs(lam) < 7 or (lam > 1 and lam < 8):
            lam_mode = "extrap"
        else:
            lam_mode = "super_extrap"

        meta.update({
            "op": "crossover",
            "parent_a": ida,
            "parent_b": idb,
            "parent_a_score": float(a.score),
            "parent_b_score": float(b.score),
            "lam": float(lam),
            "lam_bucket": lam_bucket,
            "lam_mode": lam_mode,
            "p_extrap": round(self._compute_extrap_prob(), 3),
            "iter_updates": self.iter_updates,
            "total_evals": self.total_evals,
            "sigma": sigma,
        })
        return x.astype(np.float64), meta

    def _try_generate_candidate(
        self,
        max_new_tokens: int,
        do_sample: bool,
        attempt_limit: int = 12,
    ) -> Tuple[np.ndarray, str, str, str, str, Dict[str, Any]]:
        for _ in range(attempt_limit):
            x, meta = self._make_child_x()
            z_hash, raw, refined, i_hash = self._decode_candidate(x, max_new_tokens=max_new_tokens, do_sample=do_sample)

            if z_hash in self.banned_z:
                continue
            if i_hash in self.seen_i:
                continue
            if looks_like_junk(refined):
                continue

            self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)

            meta = dict(meta)
            meta.update({"z_hash": z_hash, "i_hash": i_hash, "ts": time.time()})
            return x, z_hash, raw, refined, i_hash, meta

        # Fallback
        x = (self.x0 + np.random.normal(0.0, float(self.cfg.sigma), size=(self.dim,))).astype(np.float64)
        z_hash, raw, refined, i_hash = self._decode_candidate(x, max_new_tokens=max_new_tokens, do_sample=False)
        self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)
        meta = {"op": "fallback", "z_hash": z_hash, "i_hash": i_hash, "ts": time.time()}
        return x, z_hash, raw, refined, i_hash, meta

    def _insert_individual(self, ind: Individual) -> None:
        if ind.z_hash in self.banned_z:
            return

        # Check for duplicate by i_hash in population
        for existing in self.pop:
            if existing.i_hash == ind.i_hash:
                # Update score if better
                if ind.score > existing.score:
                    existing.score = ind.score
                    existing.meta = ind.meta
                return

        self.pop.append(ind)
        
        # Keep bounded
        if len(self.pop) > int(self.cfg.pop_size):
            worst_i = min(range(len(self.pop)), key=lambda i: self.pop[i].score)
            self.pop.pop(worst_i)

        # Update best
        if self.best_score is None or ind.score > self.best_score:
            self.best_score = ind.score
            self.best_candidate = {
                "score": float(ind.score),
                "intention": ind.intention,
                "intention_raw": ind.intention_raw,
                "z_hash": ind.z_hash,
                "i_hash": ind.i_hash,
                "meta": ind.meta,
            }

    def summary(self) -> Dict[str, Any]:
        pop_scores = [p.score for p in self.pop]
        return {
            "dim": self.dim,
            "pop_size": len(self.pop),
            "pop_capacity": int(self.cfg.pop_size),
            "score_min": float(min(pop_scores)) if pop_scores else None,
            "score_max": float(max(pop_scores)) if pop_scores else None,
            "score_mean": float(sum(pop_scores) / len(pop_scores)) if pop_scores else None,
            "banned_z": len(self.banned_z),
            "seen_intentions": len(self.seen_i),
            "seen_pairs": len(self.seen_pairs),
            "iter_updates": self.iter_updates,
            "total_evals": self.total_evals,
            "best_score": self.best_score,
            "best_candidate": self.best_candidate,
            "last_update_ts": self.last_update_ts,
        }


TASKS: Dict[str, TaskState] = {}

# =============================================================================
# FastAPI app
# =============================================================================
app = FastAPI(title="CSC Intention Server (GA)", version="0.2")


@app.on_event("startup")
def _startup():
    import sys
    try:
        print("[CSC-GA] Starting model initialization...", flush=True)
        sys.stdout.flush()
        initialize_models()
        print("[CSC-GA] Model initialization completed successfully.", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[CSC-GA] ERROR during model initialization: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": llm is not None and retriever is not None,
        "tasks_count": len(TASKS),
        "algorithm": "GA",
    }


@app.get("/")
async def root():
    return {
        "service": "CSC Intention Server (GA)",
        "version": "0.2",
        "algorithm": "Genetic Algorithm",
        "models_loaded": llm is not None and retriever is not None,
    }


@app.put("/tasks/{task_id}")
async def set_task(task_id: str, req: SetTaskReq):
    if llm is None or retriever is None:
        raise HTTPException(status_code=503, detail="Models not initialized.")
    try:
        st = TaskState(req.task, req.ga)
    except Exception as e:
        import traceback
        print(f"[CSC-GA] set_task failed for {task_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    TASKS[task_id] = st
    print(f"[CSC-GA] Task {task_id} created: pop_size={st.cfg.pop_size}, sigma={st.cfg.sigma}")
    return {
        "task_id": task_id,
        "status": "ready",
        "dim": st.dim,
        "pop_size": int(st.cfg.pop_size),
        "sigma": float(st.cfg.sigma),
        "initial_pop": len(st.pop),
    }


@app.post("/tasks/{task_id}/suggest", response_model=SuggestResp)
async def suggest(task_id: str, req: SuggestReq):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    if req.n <= 0:
        raise HTTPException(status_code=400, detail="n must be > 0")

    st = TASKS[task_id]
    async with st.lock:
        batch_id = uuid.uuid4().hex
        st.pending[batch_id] = {}

        candidates: List[Candidate] = []
        for _ in range(req.n):
            x, z_hash, raw, refined, i_hash, meta = st._try_generate_candidate(
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                attempt_limit=12,
            )
            cid = uuid.uuid4().hex
            st.pending[batch_id][cid] = PendingCandidate(
                x=x,
                z_hash=z_hash,
                intention_raw=raw,
                intention=refined,
                i_hash=i_hash,
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
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]

    async with st.lock:
        if req.batch_id not in st.pending:
            raise HTTPException(status_code=400, detail="unknown batch_id")
        batch = st.pending.pop(req.batch_id)

        updated = 0
        best_local: Optional[Dict[str, Any]] = None

        for ev in req.evaluations:
            if ev.candidate_id not in batch:
                continue
            pc = batch[ev.candidate_id]
            aux = ev.aux or {}
            score = float(ev.score)
            st.total_evals += 1  # each evaluation counts for schedule (per-iteration)

            # Ban if score too low or empty patch
            if score < st.cfg.ban_threshold or aux.get("note") == "empty patch":
                st._ban(pc.z_hash, pc.i_hash)
                updated += 1
                continue

            if not math.isfinite(score):
                st._ban(pc.z_hash, pc.i_hash)
                updated += 1
                continue

            if looks_like_junk(pc.intention):
                st._ban(pc.z_hash, pc.i_hash)
                updated += 1
                continue

            # Insert into population
            ind = Individual(
                x=pc.x,
                score=score if req.maximize else -score,
                z_hash=pc.z_hash,
                i_hash=pc.i_hash,
                intention=pc.intention,
                intention_raw=pc.intention_raw,
                meta={**pc.meta, "feedback_aux": aux, "feedback_ts": time.time()},
            )
            st._insert_individual(ind)
            updated += 1

            if best_local is None or ind.score > float(best_local["score"]):
                best_local = {
                    "candidate_id": ev.candidate_id,
                    "score": float(ind.score),
                    "intention": ind.intention,
                    "intention_raw": ind.intention_raw,
                    "aux": aux,
                }

        st.iter_updates += 1
        st.last_update_ts = time.time()
        
        print(f"[CSC-GA] feedback: {updated} evals, pop_size={len(st.pop)}, best_score={st.best_score}")

        return FeedbackResp(task_id=task_id, updated=updated, best=st.best_candidate or best_local)


@app.get("/tasks/{task_id}/state")
async def state(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]
    async with st.lock:
        return {"task_id": task_id, **st.summary(), "pending_batches": len(st.pending)}
