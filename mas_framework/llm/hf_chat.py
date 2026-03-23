"""
HuggingFace Transformers direct-inference backend.

Use this for HPC / Slurm environments where no external server (Ollama) can
run alongside the job.  The model is loaded once per process and reused for
all subsequent requests.

Model naming convention
-----------------------
Pass a HuggingFace Hub model ID (must contain a '/') as the model name, e.g.
    "Qwen/Qwen3-8B"
    "meta-llama/Llama-3.2-3B-Instruct"

The model is loaded in bfloat16 on CUDA when available, else float32 on CPU.

GPU utilisation
---------------
A background batcher thread collects pending agen() calls and groups them into
a single model.generate() call (batch_size > 1).  This amortises the
weight-loading cost across multiple sequences and significantly improves GPU
utilisation compared to batch_size=1 inference.

Set the environment variable HF_INFERENCE_BATCH_SIZE (default: 4) to control
the maximum batch size.  Larger values increase GPU utilisation but also
increase peak VRAM usage.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from typing import Dict, List, Optional, Tuple, Union

from mas_framework.llm.format import Message
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry

_model_lock = threading.Lock()
_loaded_models: Dict[str, Tuple] = {}  # model_id -> (tokenizer, model)

# ---------------------------------------------------------------------------
# Per-model batcher — one instance per model_id
# ---------------------------------------------------------------------------

class _Batcher:
    """
    Collects agen() requests and dispatches them in batches to model.generate().

    Each request is a (input_text, max_tokens, temperature, result_future) tuple.
    The background thread waits up to `wait_ms` milliseconds for the batch to
    fill before flushing, so latency stays bounded even at low throughput.
    """

    def __init__(self, model_id: str, max_batch: int = 4, wait_ms: int = 50):
        self.model_id = model_id
        self.max_batch = max_batch
        self.wait_ms = wait_ms
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, input_text: str, max_tokens: int, temperature: float) -> asyncio.Future:
        """Submit a request and return an asyncio.Future for the result."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._q.put((input_text, max_tokens, temperature, loop, fut))
        return fut

    def _loop(self):
        import torch

        while True:
            # Block until at least one request arrives
            first = self._q.get()
            batch = [first]

            # Collect more requests up to max_batch, waiting at most wait_ms
            deadline = self.wait_ms / 1000.0
            while len(batch) < self.max_batch:
                try:
                    item = self._q.get(timeout=deadline)
                    batch.append(item)
                    deadline = 0.0  # drain immediately once we have work
                except queue.Empty:
                    break

            self._dispatch(batch)

    def _dispatch(self, batch):
        import torch

        tokenizer, model = _load_model(self.model_id)
        device = next(model.parameters()).device

        input_texts   = [item[0] for item in batch]
        max_tokens    = max(item[1] for item in batch)
        temperature   = batch[0][2]   # use first request's temperature

        # Tokenize with left-padding (required for decoder-only batch generation)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        inputs = tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else 1.0,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_len = inputs["input_ids"].shape[1]
            results = [
                tokenizer.decode(out[input_len:], skip_special_tokens=True)
                for out in outputs
            ]
            for i, (_, _, _, loop, fut) in enumerate(batch):
                loop.call_soon_threadsafe(fut.set_result, results[i])

        except Exception as exc:
            for _, _, _, loop, fut in batch:
                loop.call_soon_threadsafe(fut.set_exception, exc)


_batchers: Dict[str, _Batcher] = {}
_batcher_lock = threading.Lock()


def _get_batcher(model_id: str) -> _Batcher:
    if model_id not in _batchers:
        with _batcher_lock:
            if model_id not in _batchers:
                max_batch = int(os.getenv("HF_INFERENCE_BATCH_SIZE", "4"))
                _batchers[model_id] = _Batcher(model_id, max_batch=max_batch)
    return _batchers[model_id]


# ---------------------------------------------------------------------------
# Model loader (unchanged)
# ---------------------------------------------------------------------------

def _load_model(model_id: str):
    """Load tokenizer + model once; cache globally for the process lifetime."""
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    with _model_lock:
        if model_id in _loaded_models:
            return _loaded_models[model_id]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = os.getenv("HF_MODEL_CACHE", None)
        print(f"[HFChat] Loading model '{model_id}' (this may take a moment) ...")

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        model.eval()
        print(f"[HFChat] Model '{model_id}' loaded.")
        _loaded_models[model_id] = (tokenizer, model)

    return _loaded_models[model_id]


# ---------------------------------------------------------------------------
# Prompt formatting helper
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, messages: List[Dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<{role}>\n{content}\n</{role}>")
    parts.append("<assistant>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# LLM class
# ---------------------------------------------------------------------------

@LLMRegistry.register("HFChat")
class HFChat(LLM):
    """Direct HuggingFace inference with automatic request batching."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        msg_dicts = (
            messages
            if isinstance(messages, list) and messages and isinstance(messages[0], dict)
            else [{"role": m["role"], "content": m["content"]} for m in messages]
        )

        tokenizer, _ = _load_model(self.model_name)
        input_text = _format_prompt(tokenizer, msg_dicts)

        batcher = _get_batcher(self.model_name)
        return await batcher.submit(input_text, max_tokens, temperature)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation — runs a single-item batch directly."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE

        msg_dicts = (
            messages
            if isinstance(messages, list) and messages and isinstance(messages[0], dict)
            else [{"role": m["role"], "content": m["content"]} for m in messages]
        )

        import torch
        tokenizer, model = _load_model(self.model_name)
        input_text = _format_prompt(tokenizer, msg_dicts)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = next(model.parameters()).device
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)
