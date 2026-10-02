import asyncio
import os
import re
from typing import List, Any, Dict, Optional

from mas_framework.graph.node import Node
from mas_framework.agents.agent_registry import AgentRegistry
from mas_framework.llm.llm_registry import LLMRegistry
from mas_framework.prompt.prompt_set_registry import PromptSetRegistry
from mas_framework.tools.coding.python_executor import execute_code_get_return
from benchmark_datasets.gsm8k_dataset import gsm_get_predict


_QA = re.compile(r"^Q:\s*(.*?)(?=^A:)^A:\s*(.*?)(?=^Q:|\Z)", re.S | re.M)


def split_few_shot_pairs(text: str):
    """Split a demo block into (question, answer) pairs."""
    if not text:
        return []
    return [(q.strip(), a.strip()) for q, a in _QA.findall(text) if q.strip() and a.strip()]


# "inline": the original prompt (demo glued onto the question; used to pretrain
# ARG-Designer). "turns": the demo is replayed as completed turns (used by RGA-Designer).
SOLVER_FEWSHOT_MODE = os.getenv("RGA_SOLVER_FEWSHOT", "inline")


@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    def __init__(self, id: Optional[str] =None, role:str = None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """             
        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        if kwargs.get("omit_few_shot"):
            user_prompt = f'Q:{raw_inputs["task"]}'
        else:
            user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        if self.role == "Math Solver":
            user_prompt += "(Hint: The answer is near to"
            for id, info in spatial_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            for id, info in temporal_info.items():
                user_prompt += " "+gsm_get_predict(info["output"])
            user_prompt += ")."
        else:
            for id, info in spatial_info.items():
                spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
            for id, info in temporal_info.items():
                temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
            user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
            user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return system_prompt, user_prompt
    
    def _build_messages(self, input:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any]):
        turns = SOLVER_FEWSHOT_MODE == "turns"
        system_prompt, user_prompt = self._process_inputs(
            input, spatial_info, temporal_info, omit_few_shot=turns)
        messages = [{'role':'system','content':system_prompt}]
        if turns:
            getter = getattr(self.prompt_set, "get_answer_few_shot", None)
            demo = getter(self.role) if getter else ""
            for q, a in split_few_shot_pairs(demo):
                messages.append({'role':'user','content':f"Q:{q}"})
                messages.append({'role':'assistant','content':a})
        messages.append({'role':'user','content':user_prompt})
        return messages

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        response = self.llm.gen(self._build_messages(input, spatial_info, temporal_info))
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
        response = await self.llm.agen(self._build_messages(input, spatial_info, temporal_info))
        if self.role == "Programming Expert":
            # Blocking subprocess call; keep it off the event loop.
            answer = await asyncio.to_thread(
                execute_code_get_return,
                response.lstrip("```python\n").rstrip("\n```"),
            )
            response += f"\nthe answer is {answer}"
        # print(f"#################system_prompt:{system_prompt}")
        # print(f"#################user_prompt:{user_prompt}")
        # print(f"#################response:{response}")
        return response