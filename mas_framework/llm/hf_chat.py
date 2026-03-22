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
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Dict, List, Optional, Tuple, Union

from mas_framework.llm.format import Message
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry

_model_lock = threading.Lock()
_loaded_models: Dict[str, Tuple] = {}  # model_id -> (tokenizer, model)


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


def _generate_sync(
    model_id: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
) -> str:
    import torch

    tokenizer, model = _load_model(model_id)

    # Use apply_chat_template if the tokenizer supports it
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<{role}>\n{content}\n</{role}>")
        parts.append("<assistant>")
        input_text = "\n".join(parts)

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

    # Decode only newly generated tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


@LLMRegistry.register("HFChat")
class HFChat(LLM):
    """Direct HuggingFace inference — no server required."""

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

        # Convert Message objects to plain dicts expected by transformers
        msg_dicts = (
            messages
            if isinstance(messages, list) and messages and isinstance(messages[0], dict)
            else [{"role": m["role"], "content": m["content"]} for m in messages]
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            _generate_sync,
            self.model_name,
            msg_dicts,
            max_tokens,
            temperature,
        )
        return response

    def gen(
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

        return _generate_sync(self.model_name, msg_dicts, max_tokens, temperature)
