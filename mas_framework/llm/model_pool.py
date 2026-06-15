"""
Base-model pool for multi-model MAS generation.

A model pool maps each base-model name (as understood by LLMRegistry, e.g.
an Ollama short name or a HuggingFace Hub ID) to a short natural-language
description of its capabilities and cost profile.  The descriptions are
sentence-embedded (same encoder as the role embeddings) and used by the
ARGDesigner model-selection head to score candidate models by similarity,
mirroring how roles are selected.  Because scoring is similarity-based,
models can be added to the pool at inference time without retraining the
head's output dimension.

Pool specification accepted by load_model_pool():
  None / ""            → DEFAULT_MODEL_POOL
  path to a JSON file  → {"model_name": "description", ...}
  comma-separated str  → names only; descriptions fall back to the default
                         pool entry when known, else the bare name
"""

import json
import os
import pickle

import torch

# Default pool: local open models served via an OpenAI-compatible endpoint
# (vLLM / Ollama).  Descriptions characterise capability and cost so the
# semantic embeddings are informative for routing decisions.
DEFAULT_MODEL_POOL = {
    "Qwen/Qwen3-4B": (
        "Qwen3 4B: a mid-sized general-purpose model with strong mathematical "
        "reasoning, step-by-step problem solving and solid code generation. "
        "Moderate inference cost."
    ),
    "llama3.2": (
        "Llama 3.2 3B: a small, fast instruction-following model. Good at "
        "summarising, reviewing and aggregating other agents' answers. "
        "Low inference cost, weaker at multi-step math."
    ),
    "gemma3": (
        "Gemma 3 4B: a compact general model with good language understanding "
        "and commonsense reasoning. Low inference cost, decent arithmetic, "
        "weaker at long-form code."
    ),
}


def load_model_pool(spec=None) -> dict:
    """Resolve a pool specification into an ordered {name: description} dict."""
    if not spec:
        return dict(DEFAULT_MODEL_POOL)

    if os.path.exists(spec):
        with open(spec, "r", encoding="utf-8") as f:
            pool = json.load(f)
        if not isinstance(pool, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in pool.items()
        ):
            raise ValueError(
                f"Model pool file {spec} must be a JSON object of "
                "{model_name: description} strings"
            )
        return pool

    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise ValueError(f"Empty model pool specification: {spec!r}")
    return {n: DEFAULT_MODEL_POOL.get(n, n) for n in names}


def precompute_model_embeddings(model_pool: dict, save_path: str = None) -> dict:
    """Sentence-embed "name: description" for every model in the pool.

    Returns {model_name: torch.Tensor[384]} and optionally caches it as a
    pickle, mirroring precompute_role_embeddings.
    """
    from mas_framework.llm.profile_embedding import get_sentence_model

    encoder = get_sentence_model()
    embeddings = {
        name: torch.tensor(encoder.encode(f"{name}: {description.strip()}"))
        for name, description in model_pool.items()
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Precomputed {len(embeddings)} model embeddings, saved to {save_path}")

    return embeddings


def get_model_embeddings(model_pool: dict, cache_path: str = None) -> dict:
    """Load cached model embeddings, recomputing if the cache is missing or
    does not cover every model in the pool."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            embeddings = pickle.load(f)
        if all(name in embeddings for name in model_pool):
            return embeddings
        print("Model embedding cache does not cover the pool, recomputing")
    return precompute_model_embeddings(model_pool, cache_path)
