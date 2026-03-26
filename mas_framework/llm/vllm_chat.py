"""
vLLM inference backend.

vLLM uses PagedAttention + continuous batching, giving 3-10x higher
throughput than HuggingFace transformers for the same model.

Activation
----------
Set the environment variable USE_VLLM=1 before running any script.
The --llm_name argument stays the same (e.g. "Qwen/Qwen3-8B").

    USE_VLLM=1 uv run rlhf --phase collect ...
    USE_VLLM=1 uv run baseline --method cot ...

Tuning knobs (env vars)
-----------------------
VLLM_TENSOR_PARALLEL_SIZE   GPUs to shard the model across.
                             Defaults to all visible CUDA devices.
VLLM_GPU_MEMORY_UTILIZATION Fraction of GPU VRAM to allocate for
                             the KV cache (default: 0.90).
VLLM_MAX_MODEL_LEN          Maximum sequence length (default: model max).
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Dict, List, Optional, Union

import vllm  # noqa: F401 — raises ImportError early if vllm is not installed

from mas_framework.llm.format import Message
from mas_framework.llm.llm import LLM
from mas_framework.llm.llm_registry import LLMRegistry

# ---------------------------------------------------------------------------
# Globals — one engine + tokenizer per model ID
# ---------------------------------------------------------------------------

_engines: Dict[str, object] = {}
_tokenizers: Dict[str, object] = {}
_init_lock = threading.Lock()


def _get_engine_and_tokenizer(model_id: str):
    """Load the AsyncLLMEngine and tokenizer once; cache for the process life."""
    if model_id in _engines:
        return _engines[model_id], _tokenizers[model_id]

    with _init_lock:
        if model_id in _engines:
            return _engines[model_id], _tokenizers[model_id]

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from transformers import AutoTokenizer

        try:
            import torch
            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 1

        tp = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", str(max(n_gpus, 1))))
        gpu_mem = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        max_model_len = os.getenv("VLLM_MAX_MODEL_LEN")

        print(
            f"[VLLMChat] Loading '{model_id}' "
            f"(tensor_parallel={tp}, gpu_mem={gpu_mem:.2f}) ..."
        )

        engine_args_kwargs = dict(
            model=model_id,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        if max_model_len:
            engine_args_kwargs["max_model_len"] = int(max_model_len)

        engine_args = AsyncEngineArgs(**engine_args_kwargs)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Load tokenizer on CPU (small, no CUDA needed)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        _engines[model_id] = engine
        _tokenizers[model_id] = tokenizer
        print(f"[VLLMChat] '{model_id}' ready.")

    return _engines[model_id], _tokenizers[model_id]


# ---------------------------------------------------------------------------
# Prompt formatting (same as hf_chat.py)
# ---------------------------------------------------------------------------

def _format_prompt(tokenizer, messages: List[Dict]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs: Dict = dict(tokenize=False, add_generation_prompt=True)
        if os.getenv("DISABLE_THINKING", "").lower() in ("1", "true", "yes"):
            try:
                return tokenizer.apply_chat_template(
                    messages, enable_thinking=False, **kwargs
                )
            except TypeError:
                pass  # non-Qwen3 tokenizer — ignore gracefully
        return tokenizer.apply_chat_template(messages, **kwargs)
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

@LLMRegistry.register("VLLMChat")
class VLLMChat(LLM):
    """
    vLLM async inference.

    Concurrent agen() calls are automatically batched by vLLM's continuous
    batching scheduler — no manual batcher thread needed.  GPU utilization
    is typically 60-90% compared to 20-40% with HuggingFace transformers.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        from vllm import SamplingParams

        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            from mas_framework.llm.llm import qwen3_sampling
            sp = qwen3_sampling()
            temperature = sp["temperature"]
        else:
            from mas_framework.llm.llm import qwen3_sampling
            sp = qwen3_sampling()

        msg_dicts = (
            messages
            if isinstance(messages, list) and messages and isinstance(messages[0], dict)
            else [{"role": m["role"], "content": m["content"]} for m in messages]
        )

        engine, tokenizer = _get_engine_and_tokenizer(self.model_name)
        prompt = _format_prompt(tokenizer, msg_dicts)

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=sp["top_p"] if temperature > 0 else 1.0,
            top_k=sp["top_k"] if temperature > 0 else -1,
            min_p=0.0,
        )

        request_id = str(uuid.uuid4())
        final_output = None
        async for output in engine.generate(prompt, sampling_params, request_id):
            final_output = output

        if final_output is None or not final_output.outputs:
            return ""

        # Accumulate token usage into global singletons
        from mas_framework.utils.globals import PromptTokens, CompletionTokens
        PromptTokens.instance().value += len(final_output.prompt_token_ids)
        CompletionTokens.instance().value += len(final_output.outputs[0].token_ids)

        from mas_framework.llm.hf_chat import _strip_thinking
        return _strip_thinking(final_output.outputs[0].text)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Synchronous wrapper — runs a single agen() call in a new event loop."""
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.agen(messages, max_tokens, temperature, num_comps)
        )
