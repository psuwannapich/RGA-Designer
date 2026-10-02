import asyncio
import re
from typing import List, Any, Dict, Optional

from mas_framework.graph.node import Node
from mas_framework.agents.agent_registry import AgentRegistry
from mas_framework.llm.llm_registry import LLMRegistry
from mas_framework.prompt.prompt_set_registry import PromptSetRegistry
from mas_framework.tools.coding.python_executor import PyExecutor

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    def __init__(self, id: Optional[str] =None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results
    
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        spatial_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```"):  # is python code
                self.internal_tests = self.extract_example(raw_inputs)
                output = info['output'].lstrip("```python\n").rstrip("\n```")
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}.\n At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str}\n\n"
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        # _process_inputs runs the candidate code; keep it off the event loop.
        system_prompt, user_prompt = await asyncio.to_thread(
            self._process_inputs, input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        return response


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    def __init__(self, id: Optional[str] =None, domain: str = "", llm_name: str = "", ):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        self.role = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()          
        system_prompt = f"{self.role}.\n {self.constraint}"
        
        spatial_str = ""
        for id, info in spatial_info.items():
            spatial_str += id + ": " + info['output'] + "\n\n"
        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = f"{decision_few_shot} The task is:\n\n {raw_inputs['task']}.\n At the same time, the output of other agents is as follows:\n\n{spatial_str}"
        return system_prompt, user_prompt
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
  
        system_prompt, user_prompt = self._process_inputs(input, spatial_info, temporal_info)
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = await self.llm.agen(message)
        # print(f"################system prompt:{system_prompt}")
        # print(f"################user prompt:{user_prompt}")
        # print(f"################response:{response}")
        return response

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: Optional[str] =None, domain: str = "", llm_name: str = "", ):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
                
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: Optional[str] =None, domain: str = "", llm_name: str = "", ):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)
        
    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None
    
    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output
    
    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            # print(processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output


# FinalReferTurns: the stock FinalRefer glues its few-shot demo onto the live task, and
# small models copy the demo's answer. Here the demo is its own user/assistant turn.
def _split_few_shot(text: str):
    """Split a "Q:... A:..." demo into (question, answer); (None, None) if malformed."""
    if not text:
        return None, None
    m = re.search(r"^\s*Q\s*:(.*?)^\s*A\s*:(.*)$", text, re.S | re.M)
    if not m:
        return None, None
    q, a = m.group(1).strip(), m.group(2).strip()
    return (q, a) if q and a else (None, None)


@AgentRegistry.register('FinalReferTurns')
class FinalReferTurns(Node):
    def __init__(self, id: Optional[str] = None, domain: str = "", llm_name: str = "", agent_name: str = ""):
        super().__init__(id, agent_name or type(self).__name__, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)

    def _spatial_str(self, spatial_info: Dict[str, Any]) -> str:
        out = ""
        for nid, info in spatial_info.items():
            out += nid + ": " + info['output'] + "\n\n"
        return out

    def _process_inputs(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any],
                        temporal_info: Dict[str, Any], **kwargs) -> List[Any]:
        msgs = self._build_messages(raw_inputs, spatial_info)
        system = next((m['content'] for m in msgs if m['role'] == 'system'), "")
        user = msgs[-1]['content'] if msgs else ""
        return system, user

    def _build_messages(self, raw_inputs: Dict[str, str], spatial_info: Dict[str, Any]) -> List[Dict[str, str]]:
        role = self.prompt_set.get_decision_role()
        constraint = self.prompt_set.get_decision_constraint()
        messages = [{'role': 'system', 'content': f"{role}.\n {constraint}"}]

        q, a = _split_few_shot(self.prompt_set.get_decision_few_shot())
        if q and a:
            messages.append({'role': 'user', 'content': q})
            messages.append({'role': 'assistant', 'content': a})

        user = (f"The task is:\n\n {raw_inputs['task']}.\n"
                f" At the same time, the output of other agents is as follows:\n\n"
                f"{self._spatial_str(spatial_info)}")
        messages.append({'role': 'user', 'content': user})
        return messages

    def _execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        return self.llm.gen(self._build_messages(input, spatial_info))

    async def _async_execute(self, input: Dict[str, str], spatial_info: Dict[str, Any], temporal_info: Dict[str, Any], **kwargs):
        return await self.llm.agen(self._build_messages(input, spatial_info))
