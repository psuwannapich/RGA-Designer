import sys
import random
from contextvars import ContextVar
from typing import Union, Literal, List

# Task-local token counters.  Each asyncio Task gets an isolated copy when it is
# created, so LLM calls that happen inside one task only increment that task's
# counter.  Reset to 0 at the start of each request to get accurate per-request
# counts regardless of eval_batch_size.
task_prompt_tokens: ContextVar[int] = ContextVar('task_prompt_tokens', default=0)
task_completion_tokens: ContextVar[int] = ContextVar('task_completion_tokens', default=0)

# A mutable counter, so LLM calls made in per-agent child tasks are counted too
# (a child task's .set() on the ContextVars above never reaches the parent).
task_tokens: ContextVar = ContextVar('task_tokens', default=None)


def new_task_tokens() -> dict:
    counter = {'prompt': 0, 'completion': 0}
    task_tokens.set(counter)
    return counter

class Singleton:
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def reset(self):
        self.value = 0.0

class Cost(Singleton):
    def __init__(self):
        self.value = 0.0

class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""
