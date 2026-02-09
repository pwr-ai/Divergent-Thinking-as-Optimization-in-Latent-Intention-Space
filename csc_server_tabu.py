#!/usr/bin/env python
"""
CSC Intention Server (FastAPI) – Tabu Search Version

Replaces GA with Tabu Search in embedding space:
- Maintains a tabu list of solution fingerprints (embeddings, intentions, AST diff hashes)
- Uses neighborhood moves instead of crossover:
  - Local moves: x' = x + ε (directional noise towards anchors or PCA direction)
  - Kick moves: larger jumps when stuck
- Accepts new candidate if score >= current (not just >), crucial for plateau navigation
- Aspiration criteria: accept tabu move if it's the best ever seen
- Hard ban: bad intentions (score=0, empty patch) are banned forever

Endpoints:
- PUT  /tasks/{task_id}         -> set/reset task, anchors, Tabu config
- POST /tasks/{task_id}/suggest -> generate intentions from Tabu Search
- POST /tasks/{task_id}/feedback-> update current solution with scores
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
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

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
    print("[CSC-TABU] initialize_models() called", flush=True)
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
    
    print(f"[CSC-TABU] Configured dtype: {csc_dtype} -> torch dtype: {_dtype_llm}")
    print(f"[CSC-TABU] Configured quantization: {csc_quantization}")

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
            print("[CSC-TABU] Using 4-bit quantization (bitsandbytes NF4)")
        except ImportError:
            print("[CSC-TABU] WARNING: bitsandbytes not available, falling back to no quantization")
            llm_load_kwargs["torch_dtype"] = _dtype_llm
    elif csc_quantization == "8bit":
        try:
            from transformers import BitsAndBytesConfig
            llm_load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            print("[CSC-TABU] Using 8-bit quantization (bitsandbytes)")
        except ImportError:
            print("[CSC-TABU] WARNING: bitsandbytes not available, falling back to no quantization")
            llm_load_kwargs["torch_dtype"] = _dtype_llm
    else:
        # No quantization - use specified dtype
        llm_load_kwargs["torch_dtype"] = _dtype_llm

    print(f"[CSC-TABU] Loading LLM: {llm_name_or_path} (dtype={_dtype_llm}, quant={csc_quantization}) device_map=auto")
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

    print(f"[CSC-TABU] Using device for LLM inputs: {_device}")

    print(f"[CSC-TABU] Loading retriever: {retriever_name_or_path} (dtype={_dtype_retr}) device_map=auto")
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

    print(f"[CSC-TABU] Using device for retriever inputs: {_retriever_device}")
    print("[CSC-TABU] Models loaded.")

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
# Solution fingerprinting for tabu list
# =============================================================================
@dataclass
class SolutionFingerprint:
    """
    Comprehensive fingerprint of a solution for tabu list.
    Includes multiple levels of identification to catch "paraphrases".
    """
    z_hash: str  # embedding hash
    i_hash: str  # intention text hash
    # Additional fingerprint components (populated from feedback aux data)
    fix_mechanism: Optional[str] = None  # e.g., "__eq__/__hash__/__lt__"
    touched_files: Optional[Tuple[str, ...]] = None  # sorted tuple of file paths
    key_symbols: Optional[Tuple[str, ...]] = None  # key class/function names
    ast_diff_hash: Optional[str] = None  # hash of AST diff if available
    
    def matches(self, other: 'SolutionFingerprint', strict: bool = False) -> bool:
        """Check if two fingerprints represent the same solution."""
        # Always match on embedding hash
        if self.z_hash == other.z_hash:
            return True
        # Always match on intention hash
        if self.i_hash == other.i_hash:
            return True
        
        if strict:
            # In strict mode, also check semantic fingerprints
            if self.fix_mechanism and other.fix_mechanism:
                if self.fix_mechanism == other.fix_mechanism:
                    return True
            if self.touched_files and other.touched_files:
                if self.touched_files == other.touched_files:
                    return True
            if self.ast_diff_hash and other.ast_diff_hash:
                if self.ast_diff_hash == other.ast_diff_hash:
                    return True
        
        return False
    
    def to_key(self) -> str:
        """Generate a compact key for dict storage."""
        # Primary: use intention hash (more stable than embedding)
        return self.i_hash


# =============================================================================
# API models
# =============================================================================
class Anchor(BaseModel):
    id: str
    text: str
    weight: float = 1.0


class TabuConfig(BaseModel):
    dim: Optional[int] = None
    seed: int = 123

    # Tabu list parameters
    tabu_tenure: int = 50  # how long a solution stays tabu
    max_tabu_size: int = 1000  # max size of tabu list
    use_strict_fingerprint: bool = False  # use semantic fingerprints (files, symbols)
    
    # Neighborhood parameters
    sigma_local: float = 0.5  # std for local moves
    sigma_kick: float = 3.0  # std for kick moves
    
    # Direction bias (towards anchors or along PCA)
    anchor_attraction: float = 0.3  # probability of biasing towards an anchor
    pca_bias: float = 0.2  # probability of biasing along population PCA
    
    # Kick parameters (diversification)
    stagnation_threshold: int = 10  # iterations without improvement before kick
    kick_probability: float = 0.15  # base probability of kick move
    kick_intensity: float = 2.0  # multiplier for kick magnitude
    
    # Aspiration criteria
    use_aspiration: bool = True  # allow tabu moves if they're best ever
    
    # Acceptance criteria (key difference from classic tabu)
    accept_equal: bool = True  # accept if score >= current (not just >)
    
    # Memory and banning
    max_banned: int = 10000
    max_seen_intentions: int = 10000
    ban_threshold: float = 0.001


class TaskSpec(BaseModel):
    task_context: str = ""
    anchors: List[Anchor] = Field(default_factory=list)
    use_refinement: bool = False
    intention_prompt: Optional[str] = None
    # Initial solution from bootstrap (optional)
    initial_solution: Optional[Dict[str, Any]] = None


class SetTaskReq(BaseModel):
    task: TaskSpec
    tabu: TabuConfig


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
class TabuEntry:
    """Entry in the tabu list with expiration."""
    fingerprint: SolutionFingerprint
    added_at: int  # iteration when added
    expires_at: int  # iteration when it expires
    score: float  # score when it was tabu'd


@dataclass
class PendingCandidate:
    x: np.ndarray
    z_hash: str
    intention_raw: str
    intention: str
    i_hash: str
    meta: Dict[str, Any]
    fingerprint: SolutionFingerprint


@dataclass
class CurrentSolution:
    """The current solution in Tabu Search."""
    x: np.ndarray
    score: float
    z_hash: str
    i_hash: str
    intention: str
    intention_raw: str
    meta: Dict[str, Any]
    fingerprint: SolutionFingerprint


class TaskState:
    def __init__(self, spec: TaskSpec, cfg: TabuConfig):
        self.spec = spec
        self.cfg = cfg
        self.lock = asyncio.Lock()

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)

        # Current solution (single solution focus for Tabu Search)
        self.current: Optional[CurrentSolution] = None
        
        # Best solution ever seen (for aspiration)
        self.best_ever: Optional[CurrentSolution] = None
        self.best_score: Optional[float] = None
        self.best_candidate: Optional[Dict[str, Any]] = None
        
        # Tabu list
        self.tabu_list: Dict[str, TabuEntry] = {}  # key -> TabuEntry
        self.tabu_fifo: Deque[str] = deque()
        
        # History for PCA computation
        self.solution_history: List[np.ndarray] = []
        self.max_history: int = 100
        
        # Stagnation tracking
        self.iterations_without_improvement: int = 0
        
        # Pending candidates and tracking
        self.pending: Dict[str, Dict[str, PendingCandidate]] = {}
        self.decode_cache: Dict[str, Tuple[str, str, str]] = {}

        # Banned embeddings and seen intentions (hard bans)
        self.banned_z: Dict[str, float] = {}
        self.banned_z_fifo: Deque[str] = deque()
        self.seen_i: Dict[str, float] = {}
        self.seen_i_fifo: Deque[str] = deque()

        self.iter_updates = 0
        self.total_evals = 0
        self.last_update_ts = time.time()

        # Infer dim
        inferred_dim = None
        if spec.anchors:
            with torch.no_grad():
                e = embed_text([spec.anchors[0].text])
            inferred_dim = int(e.shape[-1])

        dim = cfg.dim or inferred_dim
        if dim is None:
            raise ValueError("Cannot infer dim: provide tabu.dim or provide at least 1 anchor.")
        self.dim = int(dim)

        # Compute anchor embeddings
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
        
        # Initialize from bootstrap solution if provided
        if spec.initial_solution:
            self._init_from_bootstrap(spec.initial_solution)
        elif spec.anchors:
            # Initialize current solution from best anchor
            best_anchor = max(spec.anchors, key=lambda a: a.weight)
            idx = [a.id for a in spec.anchors].index(best_anchor.id)
            init_x = self.anchor_embeddings[idx].copy()
            intention = best_anchor.text.strip()
            z_hash = self._hash_z(init_x)
            i_hash = self._hash_intention(intention)
            fingerprint = SolutionFingerprint(z_hash=z_hash, i_hash=i_hash)
            
            self.current = CurrentSolution(
                x=init_x,
                score=float(best_anchor.weight),
                z_hash=z_hash,
                i_hash=i_hash,
                intention=intention,
                intention_raw=intention,
                meta={"op": "init_anchor", "anchor_id": best_anchor.id},
                fingerprint=fingerprint,
            )
            self.best_ever = self.current
            self.best_score = self.current.score
            self.best_candidate = self._solution_to_dict(self.current)
            self.solution_history.append(init_x.copy())

    def _init_from_bootstrap(self, init_sol: Dict[str, Any]) -> None:
        """Initialize from a bootstrap solution."""
        # Extract embedding if provided, otherwise use x0
        if "z" in init_sol and init_sol["z"]:
            init_x = np.array(init_sol["z"], dtype=np.float64)
        else:
            init_x = self.x0.copy()
        
        intention = init_sol.get("intention", "").strip()
        intention_raw = init_sol.get("intention_raw", intention)
        score = float(init_sol.get("score", 0.0))
        
        z_hash = self._hash_z(init_x)
        i_hash = self._hash_intention(intention)
        fingerprint = SolutionFingerprint(z_hash=z_hash, i_hash=i_hash)
        
        # Add semantic fingerprint if available
        aux = init_sol.get("aux", {})
        if aux:
            if "fix_mechanism" in aux:
                fingerprint.fix_mechanism = aux["fix_mechanism"]
            if "touched_files" in aux:
                fingerprint.touched_files = tuple(sorted(aux["touched_files"]))
            if "key_symbols" in aux:
                fingerprint.key_symbols = tuple(sorted(aux["key_symbols"]))
            if "ast_diff_hash" in aux:
                fingerprint.ast_diff_hash = aux["ast_diff_hash"]
        
        self.current = CurrentSolution(
            x=init_x,
            score=score,
            z_hash=z_hash,
            i_hash=i_hash,
            intention=intention,
            intention_raw=intention_raw,
            meta={"op": "init_bootstrap", **init_sol.get("meta", {})},
            fingerprint=fingerprint,
        )
        self.best_ever = self.current
        self.best_score = score
        self.best_candidate = self._solution_to_dict(self.current)
        self.solution_history.append(init_x.copy())
        print(f"[CSC-TABU] Initialized from bootstrap: score={score}, intention={intention[:50]}...")

    def _solution_to_dict(self, sol: CurrentSolution) -> Dict[str, Any]:
        return {
            "score": float(sol.score),
            "intention": sol.intention,
            "intention_raw": sol.intention_raw,
            "z_hash": sol.z_hash,
            "i_hash": sol.i_hash,
            "meta": sol.meta,
        }

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

    # ---------------- Tabu list management ----------------
    def _add_to_tabu(self, fingerprint: SolutionFingerprint, score: float) -> None:
        """Add a solution to the tabu list."""
        key = fingerprint.to_key()
        entry = TabuEntry(
            fingerprint=fingerprint,
            added_at=self.iter_updates,
            expires_at=self.iter_updates + self.cfg.tabu_tenure,
            score=score,
        )
        self.tabu_list[key] = entry
        self.tabu_fifo.append(key)
        
        # Enforce max size
        while len(self.tabu_fifo) > self.cfg.max_tabu_size:
            old_key = self.tabu_fifo.popleft()
            self.tabu_list.pop(old_key, None)
    
    def _is_tabu(self, fingerprint: SolutionFingerprint) -> bool:
        """Check if a solution is tabu."""
        key = fingerprint.to_key()
        if key not in self.tabu_list:
            return False
        
        entry = self.tabu_list[key]
        # Check if expired
        if entry.expires_at <= self.iter_updates:
            del self.tabu_list[key]
            return False
        
        # Check strict fingerprint matching if enabled
        if self.cfg.use_strict_fingerprint:
            return entry.fingerprint.matches(fingerprint, strict=True)
        
        return True
    
    def _cleanup_expired_tabu(self) -> None:
        """Remove expired entries from tabu list."""
        expired = [k for k, v in self.tabu_list.items() if v.expires_at <= self.iter_updates]
        for k in expired:
            del self.tabu_list[k]

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

    # ---------------- Neighborhood operators ----------------
    def _compute_pca_direction(self) -> Optional[np.ndarray]:
        """Compute principal direction from solution history."""
        if len(self.solution_history) < 3:
            return None
        
        history = np.array(self.solution_history[-self.max_history:])
        mean = history.mean(axis=0)
        centered = history - mean
        
        try:
            # SVD to get principal direction
            U, S, Vh = np.linalg.svd(centered, full_matrices=False)
            return Vh[0]  # First principal component
        except Exception:
            return None
    
    def _get_anchor_direction(self, current_x: np.ndarray) -> Optional[np.ndarray]:
        """Get direction towards a random anchor."""
        if self.anchor_embeddings is None or len(self.anchor_embeddings) == 0:
            return None
        
        # Pick random anchor
        idx = random.randint(0, len(self.anchor_embeddings) - 1)
        anchor = self.anchor_embeddings[idx]
        
        direction = anchor - current_x
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            return direction / norm
        return None
    
    def _make_neighbor(self, current_x: np.ndarray, is_kick: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate a neighbor solution using various strategies."""
        meta: Dict[str, Any] = {}
        
        sigma = self.cfg.sigma_kick if is_kick else self.cfg.sigma_local
        if is_kick:
            sigma *= self.cfg.kick_intensity
        
        # Base noise
        noise = np.random.normal(0.0, sigma, size=(self.dim,)).astype(np.float64)
        
        # Apply directional bias
        direction_type = "random"
        
        r = random.random()
        if r < self.cfg.anchor_attraction:
            # Bias towards anchor
            anchor_dir = self._get_anchor_direction(current_x)
            if anchor_dir is not None:
                # Mix noise with anchor direction
                anchor_strength = random.uniform(0.3, 0.7) * sigma
                noise = noise * 0.5 + anchor_dir * anchor_strength
                direction_type = "anchor"
        elif r < self.cfg.anchor_attraction + self.cfg.pca_bias:
            # Bias along PCA direction
            pca_dir = self._compute_pca_direction()
            if pca_dir is not None:
                # Move along or against PCA
                pca_strength = random.uniform(0.3, 0.7) * sigma
                if random.random() < 0.5:
                    pca_strength = -pca_strength
                noise = noise * 0.5 + pca_dir * pca_strength
                direction_type = "pca"
        
        new_x = current_x + noise
        
        meta.update({
            "op": "kick" if is_kick else "local_move",
            "sigma": float(sigma),
            "direction_type": direction_type,
            "stagnation": self.iterations_without_improvement,
        })
        
        return new_x.astype(np.float64), meta

    def _should_kick(self) -> bool:
        """Determine if we should do a kick move."""
        # Always kick if stagnating
        if self.iterations_without_improvement >= self.cfg.stagnation_threshold:
            return True
        
        # Random kick with increasing probability based on stagnation
        base_prob = self.cfg.kick_probability
        stag_bonus = 0.02 * self.iterations_without_improvement
        return random.random() < (base_prob + stag_bonus)

    def _try_generate_candidate(
        self,
        max_new_tokens: int,
        do_sample: bool,
        attempt_limit: int = 15,
    ) -> Tuple[np.ndarray, str, str, str, str, Dict[str, Any], SolutionFingerprint]:
        """Generate a candidate solution using Tabu Search neighborhood."""
        
        # Get current position (or x0 if no current solution)
        if self.current is not None:
            current_x = self.current.x
        else:
            current_x = self.x0
        
        is_kick = self._should_kick()
        
        for attempt in range(attempt_limit):
            x, meta = self._make_neighbor(current_x, is_kick=is_kick)
            z_hash, raw, refined, i_hash = self._decode_candidate(x, max_new_tokens=max_new_tokens, do_sample=do_sample)
            
            fingerprint = SolutionFingerprint(z_hash=z_hash, i_hash=i_hash)
            
            # Check hard bans
            if z_hash in self.banned_z:
                continue
            if i_hash in self.seen_i:
                continue
            if looks_like_junk(refined):
                continue
            
            # Check tabu status
            is_tabu_candidate = self._is_tabu(fingerprint)
            
            if is_tabu_candidate:
                # Check aspiration: allow if would be best ever
                if self.cfg.use_aspiration and self.best_score is not None:
                    # We don't know the score yet, so we can't apply aspiration here
                    # We'll handle aspiration in feedback when we know the score
                    meta["tabu_status"] = "tabu_pending_aspiration"
                else:
                    # Skip this candidate
                    continue
            else:
                meta["tabu_status"] = "not_tabu"

            self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)
            
            meta.update({
                "z_hash": z_hash,
                "i_hash": i_hash,
                "ts": time.time(),
                "attempt": attempt,
                "is_kick": is_kick,
            })
            return x, z_hash, raw, refined, i_hash, meta, fingerprint

        # Fallback: random jump
        x = (self.x0 + np.random.normal(0.0, self.cfg.sigma_kick, size=(self.dim,))).astype(np.float64)
        z_hash, raw, refined, i_hash = self._decode_candidate(x, max_new_tokens=max_new_tokens, do_sample=False)
        fingerprint = SolutionFingerprint(z_hash=z_hash, i_hash=i_hash)
        self._remember_fifo(self.seen_i, self.seen_i_fifo, i_hash, self.cfg.max_seen_intentions)
        meta = {
            "op": "fallback_random",
            "z_hash": z_hash,
            "i_hash": i_hash,
            "ts": time.time(),
            "tabu_status": "fallback",
        }
        return x, z_hash, raw, refined, i_hash, meta, fingerprint

    def _update_current_solution(
        self,
        x: np.ndarray,
        score: float,
        z_hash: str,
        i_hash: str,
        intention: str,
        intention_raw: str,
        meta: Dict[str, Any],
        fingerprint: SolutionFingerprint,
    ) -> bool:
        """
        Update current solution based on Tabu Search acceptance criteria.
        Key difference: accept if score >= current (not just >).
        Returns True if accepted.
        """
        # Create the new solution
        new_sol = CurrentSolution(
            x=x,
            score=score,
            z_hash=z_hash,
            i_hash=i_hash,
            intention=intention,
            intention_raw=intention_raw,
            meta=meta,
            fingerprint=fingerprint,
        )
        
        # Check if this is tabu
        is_tabu_candidate = self._is_tabu(fingerprint)
        
        # Check aspiration criteria (best ever overrides tabu)
        aspiration_override = False
        if is_tabu_candidate and self.cfg.use_aspiration:
            if self.best_score is None or score > self.best_score:
                aspiration_override = True
                meta["aspiration_override"] = True
        
        if is_tabu_candidate and not aspiration_override:
            # Reject tabu move
            return False
        
        # Acceptance criteria: score >= current (crucial for plateau navigation)
        current_score = self.current.score if self.current else float('-inf')
        
        if self.cfg.accept_equal:
            accepted = score >= current_score
        else:
            accepted = score > current_score
        
        if accepted:
            # Add old solution to tabu list (if exists)
            if self.current is not None:
                self._add_to_tabu(self.current.fingerprint, self.current.score)
            
            # Update current
            self.current = new_sol
            
            # Update history
            self.solution_history.append(x.copy())
            if len(self.solution_history) > self.max_history:
                self.solution_history.pop(0)
            
            # Check for improvement
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_ever = new_sol
                self.best_candidate = self._solution_to_dict(new_sol)
                self.iterations_without_improvement = 0
            else:
                self.iterations_without_improvement += 1
            
            return True
        
        # Not accepted - still add to tabu to avoid revisiting
        self._add_to_tabu(fingerprint, score)
        self.iterations_without_improvement += 1
        return False

    def summary(self) -> Dict[str, Any]:
        current_info = None
        if self.current:
            current_info = {
                "score": float(self.current.score),
                "intention": self.current.intention[:100],
                "z_hash": self.current.z_hash[:16],
            }
        
        return {
            "dim": self.dim,
            "current_solution": current_info,
            "tabu_list_size": len(self.tabu_list),
            "tabu_tenure": self.cfg.tabu_tenure,
            "banned_z": len(self.banned_z),
            "seen_intentions": len(self.seen_i),
            "solution_history_size": len(self.solution_history),
            "iterations_without_improvement": self.iterations_without_improvement,
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
app = FastAPI(title="CSC Intention Server (Tabu Search)", version="0.3")


@app.on_event("startup")
def _startup():
    import sys
    try:
        print("[CSC-TABU] Starting model initialization...", flush=True)
        sys.stdout.flush()
        initialize_models()
        print("[CSC-TABU] Model initialization completed successfully.", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"[CSC-TABU] ERROR during model initialization: {e}", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": llm is not None and retriever is not None,
        "tasks_count": len(TASKS),
        "algorithm": "TabuSearch",
    }


@app.get("/")
async def root():
    return {
        "service": "CSC Intention Server (Tabu Search)",
        "version": "0.3",
        "algorithm": "Tabu Search",
        "models_loaded": llm is not None and retriever is not None,
    }


@app.put("/tasks/{task_id}")
async def set_task(task_id: str, req: SetTaskReq):
    if llm is None or retriever is None:
        raise HTTPException(status_code=503, detail="Models not initialized.")
    try:
        st = TaskState(req.task, req.tabu)
    except Exception as e:
        import traceback
        print(f"[CSC-TABU] set_task failed for {task_id}: {e}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
    TASKS[task_id] = st
    print(f"[CSC-TABU] Task {task_id} created: tabu_tenure={st.cfg.tabu_tenure}, sigma_local={st.cfg.sigma_local}")
    return {
        "task_id": task_id,
        "status": "ready",
        "dim": st.dim,
        "tabu_tenure": int(st.cfg.tabu_tenure),
        "sigma_local": float(st.cfg.sigma_local),
        "sigma_kick": float(st.cfg.sigma_kick),
        "has_initial_solution": st.current is not None,
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
            x, z_hash, raw, refined, i_hash, meta, fingerprint = st._try_generate_candidate(
                max_new_tokens=req.max_new_tokens,
                do_sample=req.do_sample,
                attempt_limit=15,
            )
            cid = uuid.uuid4().hex
            st.pending[batch_id][cid] = PendingCandidate(
                x=x,
                z_hash=z_hash,
                intention_raw=raw,
                intention=refined,
                i_hash=i_hash,
                meta=meta,
                fingerprint=fingerprint,
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

        # Cleanup expired tabu entries
        st._cleanup_expired_tabu()

        updated = 0
        best_local: Optional[Dict[str, Any]] = None
        accepted_count = 0

        # Sort evaluations by score (descending) to process best first
        sorted_evals = sorted(req.evaluations, key=lambda e: e.score, reverse=True)

        for ev in sorted_evals:
            if ev.candidate_id not in batch:
                continue
            pc = batch[ev.candidate_id]
            aux = ev.aux or {}
            score = float(ev.score)
            st.total_evals += 1

            # Enrich fingerprint with semantic data from aux
            if aux:
                if "fix_mechanism" in aux:
                    pc.fingerprint.fix_mechanism = aux["fix_mechanism"]
                if "touched_files" in aux:
                    pc.fingerprint.touched_files = tuple(sorted(aux["touched_files"]))
                if "key_symbols" in aux:
                    pc.fingerprint.key_symbols = tuple(sorted(aux["key_symbols"]))
                if "ast_diff_hash" in aux:
                    pc.fingerprint.ast_diff_hash = aux["ast_diff_hash"]

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

            # Try to update current solution
            effective_score = score if req.maximize else -score
            meta = {**pc.meta, "feedback_aux": aux, "feedback_ts": time.time()}
            
            accepted = st._update_current_solution(
                x=pc.x,
                score=effective_score,
                z_hash=pc.z_hash,
                i_hash=pc.i_hash,
                intention=pc.intention,
                intention_raw=pc.intention_raw,
                meta=meta,
                fingerprint=pc.fingerprint,
            )
            
            if accepted:
                accepted_count += 1
            
            updated += 1

            if best_local is None or effective_score > float(best_local.get("score", float('-inf'))):
                best_local = {
                    "candidate_id": ev.candidate_id,
                    "score": float(effective_score),
                    "intention": pc.intention,
                    "intention_raw": pc.intention_raw,
                    "aux": aux,
                    "accepted": accepted,
                }

        st.iter_updates += 1
        st.last_update_ts = time.time()
        
        current_score = st.current.score if st.current else None
        print(f"[CSC-TABU] feedback: {updated} evals, {accepted_count} accepted, "
              f"current_score={current_score}, best_score={st.best_score}, "
              f"tabu_size={len(st.tabu_list)}, stagnation={st.iterations_without_improvement}")

        return FeedbackResp(task_id=task_id, updated=updated, best=st.best_candidate or best_local)


@app.get("/tasks/{task_id}/state")
async def state(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(status_code=404, detail="unknown task_id")
    st = TASKS[task_id]
    async with st.lock:
        return {"task_id": task_id, **st.summary(), "pending_batches": len(st.pending)}
