from abc import ABC, abstractmethod
from typing import List, Union, Optional

from mas_framework.llm.format import Message


class LLM(ABC):
    # 1000 is too small for reasoning models (e.g. Qwen3) whose <think> chain
    # alone can exceed 1000 tokens, causing </think> to never be emitted and
    # leaving the answer parser with truncated internal reasoning.
    DEFAULT_MAX_TOKENS = 8192
    DEFAULT_TEMPERATURE = 0.2
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
