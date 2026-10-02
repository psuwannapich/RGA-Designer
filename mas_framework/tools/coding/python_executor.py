#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ast
import astunparse
from typing import List

from mas_framework.tools.coding.executor_utils import function_with_timeout
from mas_framework.tools.coding.subprocess_runner import run_snippet, run_snippet_capture
from mas_framework.tools.coding.executor_types import ExecuteResult, Executor


def get_call_str(assert_statement: str) -> str:
    ast_parsed = ast.parse(assert_statement)
    try:
        call_str = ast_parsed.body[0].test.left # type: ignore
    except:
        call_str = ast_parsed.body[0].test # type: ignore

    return astunparse.unparse(call_str).strip()

def get_output(func: str, assert_statement: str, timeout: int = 5) -> str:
    try:
        exec(f"from typing import *\n{func}", globals())
        func_call = get_call_str(assert_statement)
        output = function_with_timeout(eval, (func_call, globals()), timeout)
        return output
    except TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return str(e)
    
def execute_code_get_return(code: str, timeout: int = 5):
    # Runs in a killable subprocess: an in-process exec of a generated infinite loop hangs the job.
    status, payload = run_snippet_capture(code, timeout=timeout)
    if status == "OK":
        return payload
    if status == "TIMEOUT":
        return f"Error occurred: timed out after {timeout}s"
    if status == "ERR":
        return f"Error occurred: {payload}"
    return None

class PyExecutor(Executor):
    def execute(self, func: str, tests: List[str], timeout: int = 5, verbose: bool = True) -> ExecuteResult:
        # Combine function code and assert statement
        imports = 'from typing import *'
        func_test_list = [f'{imports}\n{func}\n{test}' for test in tests]

        # Run the tests and collect the results
        success_tests = []
        failed_tests = []
        is_passing = True
        num_tests = len(func_test_list)
        for i in range(num_tests):
            ok, out = run_snippet(func_test_list[i], timeout=timeout)
            if ok:
                success_tests.append(tests[i])
            else:
                failed_tests.append(f"{tests[i]} # output: {out}")
                is_passing = False

        state = [test in success_tests for test in tests]

        feedback = "Tests passed:\n" + "\n".join(success_tests) + "\n\nTests failed:"
        feedback += "\n" + "\n".join(failed_tests)
        return is_passing, feedback, tuple(state)

    def evaluate(self, name: str, func: str, test: str, timeout: int = 5) -> bool:
        """
        Evaluates the implementation on Human-Eval Python.

        probably should be written in a dataset-agnostic way but not now
        """
        
        code = f"""{func}

{test}

check({name})
    """
        ok, _ = run_snippet(code, timeout=timeout)
        return ok
        