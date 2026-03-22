from mas_framework.prompt.prompt_set_registry import PromptSetRegistry
from mas_framework.prompt.mmlu_prompt_set import MMLUPromptSet
from mas_framework.prompt.humaneval_prompt_set import HumanEvalPromptSet
from mas_framework.prompt.gsm8k_prompt_set import GSM8KPromptSet
from mas_framework.prompt.AQuA_prompt_set import AQUAPromptSet

# Register multiarith and svamp as aliases for the gsm8k prompt set
# (same math domain — same agent roles and prompt templates)
PromptSetRegistry.register('multiarith')(GSM8KPromptSet)
PromptSetRegistry.register('svamp')(GSM8KPromptSet)

__all__ = ['MMLUPromptSet',
           'HumanEvalPromptSet',
           'GSM8KPromptSet',
           'AQUAPromptSet',
           'PromptSetRegistry',]