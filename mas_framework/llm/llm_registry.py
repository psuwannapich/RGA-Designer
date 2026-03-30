import os
from typing import Optional
from class_registry import ClassRegistry

from mas_framework.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    DEFAULT_LOCAL_MODEL = "llama3.2"

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name == "":
            model_name = cls.DEFAULT_LOCAL_MODEL

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif '/' in model_name:
            # HuggingFace Hub model ID (e.g. "Qwen/Qwen3-8B").
            #
            # USE_VLLM_SERVER=1  → GPTChat calling an external vllm serve process
            #                      (set LOCAL_BASE_URL / LOCAL_API_KEY accordingly)
            # USE_VLLM=1         → VLLMChat loading the model in-process
            # (default)          → HFChat via HuggingFace transformers
            if os.getenv('USE_VLLM_SERVER', '').lower() in ('1', 'true', 'yes'):
                import mas_framework.llm.gpt_chat  # noqa: F401
                model = cls.registry.get('GPTChat', model_name)
            elif os.getenv('USE_VLLM', '').lower() in ('1', 'true', 'yes'):
                try:
                    import mas_framework.llm.vllm_chat  # noqa: F401
                    model = cls.registry.get('VLLMChat', model_name)
                except ImportError:
                    print(
                        "[LLMRegistry] vLLM not installed — falling back to "
                        "HuggingFace backend. Install with: uv pip install vllm"
                    )
                    import mas_framework.llm.hf_chat  # noqa: F401
                    model = cls.registry.get('HFChat', model_name)
            else:
                import mas_framework.llm.hf_chat  # noqa: F401
                model = cls.registry.get('HFChat', model_name)
        else:
            # Ollama-style short name (e.g. "gemma3", "llama3.2")
            model = cls.registry.get('GPTChat', model_name)

        return model
