"""
OpenAI-compatible chat backend.

Used for two purposes:
  1. Ollama local models (short names like "gemma3", "llama3.2")
  2. vLLM OpenAI-compatible server (USE_VLLM_SERVER=1)

Connection is controlled by env vars read at import time:
  LOCAL_BASE_URL   (default: http://localhost:11434/v1)
  LOCAL_API_KEY    (default: ollama)
"""

import asyncio
import os
from typing import List, Union, Optional, Any, Dict

from tenacity import retry, wait_random_exponential, stop_after_attempt
import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

from mas_framework.llm.format import Message
from mas_framework.llm.llm import LLM, qwen3_sampling
from mas_framework.llm.llm_registry import LLMRegistry

load_dotenv()
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL", "http://localhost:11434/v1")
LOCAL_API_KEY  = os.getenv("LOCAL_API_KEY",  "ollama")


# One client per (base_url, event loop): a new client per call leaked connections.
_CLIENTS: Dict[Any, AsyncOpenAI] = {}

REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
MAX_AGENT_TOKENS = os.getenv("MAX_AGENT_TOKENS", "").strip()   # reply cap; unset = none


def _get_client(base_url: str) -> AsyncOpenAI:
    try:
        loop_key = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_key = None
    key = (base_url, loop_key)
    client = _CLIENTS.get(key)
    if client is None:
        # No keep-alive: idle connections dropped by vLLM otherwise block the pool.
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=0, max_connections=64),
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=30.0, pool=60.0),
        )
        client = AsyncOpenAI(base_url=base_url, api_key=LOCAL_API_KEY,
                             http_client=http_client, max_retries=0)
        _CLIENTS[key] = client
    return client


@LLMRegistry.register('GPTChat')
class GPTChat(LLM):

    def __init__(self, model_name: str):
        self.model_name = model_name

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        sp = qwen3_sampling()
        if temperature is None:
            temperature = sp["temperature"]

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        no_think = os.getenv("DISABLE_THINKING", "1").lower() in ("1", "true", "yes")
        extra_body = {"top_k": sp.get("top_k", -1)}
        if no_think:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}

        # Don't pass max_tokens when unset — the server will use all remaining
        # context after the input, avoiding "max_tokens too large" errors when
        # the default equals the full context window (e.g. --max-model-len 8192).
        create_kwargs: Dict[str, Any] = dict(
            messages=messages,
            model=self.model_name,
            temperature=temperature,
            top_p=sp.get("top_p", 1.0),
            extra_body=extra_body,
        )
        if max_tokens is None and MAX_AGENT_TOKENS.isdigit():
            max_tokens = int(MAX_AGENT_TOKENS)
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        client = _get_client(LOCAL_BASE_URL)
        completion = await client.chat.completions.create(**create_kwargs)

        if completion.usage:
            from mas_framework.utils.globals import (
                PromptTokens, CompletionTokens,
                task_prompt_tokens, task_completion_tokens, task_tokens,
            )
            pt = completion.usage.prompt_tokens
            ct = completion.usage.completion_tokens
            PromptTokens.instance().value    += pt
            CompletionTokens.instance().value += ct
            task_prompt_tokens.set(task_prompt_tokens.get() + pt)
            task_completion_tokens.set(task_completion_tokens.get() + ct)
            counter = task_tokens.get()
            if counter is not None:
                counter['prompt'] += pt
                counter['completion'] += ct

        return completion.choices[0].message.content or ""

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        import asyncio
        return asyncio.get_event_loop().run_until_complete(
            self.agen(messages, max_tokens, temperature, num_comps)
        )
