import re


def humaneval_data_process(dataset):
    """Convert raw HumanEval JSONL records to the standard {task, answer, ...} format."""
    list_data_dict = []
    for data in dataset:
        item = {
            "task": data["prompt"],
            "answer": data["test"],          # test suite (used by the executor)
            "entry_point": data["entry_point"],
            "name": data.get("name", ""),
        }
        list_data_dict.append(item)
    return list_data_dict


def humaneval_get_predict(pred_str: str) -> str:
    """Extract the Python code block from an LLM response."""
    # Try fenced code block first
    match = re.search(r"```(?:python)?\s*\n(.*?)```", pred_str, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back: return the whole string (executor will handle syntax errors)
    return pred_str.strip()
