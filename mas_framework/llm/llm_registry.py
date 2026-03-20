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
        else:
            model = cls.registry.get('GPTChat', model_name)

        return model
