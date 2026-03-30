import os
from abc import ABC, abstractmethod
from typing import List, Union, Optional

from mas_framework.llm.format import Message


# ---------------------------------------------------------------------------
# Qwen3 recommended sampling parameters (https://huggingface.co/Qwen/Qwen3-4B)
#
#   Non-thinking  DISABLE_THINKING=1  temp=0.7  top_p=0.8  top_k=20  min_p=0
#   Thinking      DISABLE_THINKING=0  temp=0.6  top_p=0.95 top_k=20  min_p=0
#
# min_p=0 disables min-p filtering (the default), so it is not passed
# explicitly to avoid compatibility issues with older transformers versions.
# ---------------------------------------------------------------------------
_SAMPLING_NO_THINK = dict(temperature=0.7, top_p=0.8,  top_k=20)
_SAMPLING_THINK    = dict(temperature=0.6, top_p=0.95, top_k=20)


def qwen3_sampling() -> dict:
    """Return the Qwen3-recommended sampling params for the current thinking mode."""
    no_think = os.getenv("DISABLE_THINKING", "1").lower() in ("1", "true", "yes")
    return _SAMPLING_NO_THINK.copy() if no_think else _SAMPLING_THINK.copy()


class LLM(ABC):
    # 1000 is too small for reasoning models (e.g. Qwen3) whose <think> chain
    # alone can exceed 1000 tokens, causing </think> to never be emitted and
    # leaving the answer parser with truncated internal reasoning.
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TEMPERATURE = 0.2      # fallback for non-Qwen3 models
    DEFUALT_NUM_COMPLETIONS = 1

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
        ) -> Union[List[str], str]:

        pass
