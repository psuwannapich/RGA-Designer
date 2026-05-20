import csv
import glob
import os
import re

import numpy as np
import pandas as pd
from abc import ABC
from typing import Union, List, Literal, Any, Dict


def mmlu_data_process(data_dir: str, split: str = "test") -> list:
    """
    Load MMLU CSV files and return a flat list of {task, answer} dicts
    compatible with cold_start.py.

    Parameters
    ----------
    data_dir : path to the MMLU data root (contains test/, val/, dev/ sub-dirs)
    split    : which split to load ('test', 'val', or 'dev')
    """
    split_path = os.path.join(data_dir, split)
    csv_files = sorted(glob.glob(os.path.join(split_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{split_path}'. "
            "Run: python datasets/MMLU/download.py"
        )

    records = []
    for path in csv_files:
        # Derive subject name from filename: "abstract_algebra_test.csv" → "abstract_algebra"
        basename = os.path.basename(path)
        subject = "_".join(basename.split("_")[:-1])  # strip trailing split tag
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                question, a, b, c, d, answer = (
                    row[0], row[1], row[2], row[3], row[4], row[5]
                )
                task = (
                    f"{question}\n"
                    f"Option A: {a}\n"
                    f"Option B: {b}\n"
                    f"Option C: {c}\n"
                    f"Option D: {d}"
                )
                records.append({"task": task, "answer": answer.strip().upper(), "subject": subject})
    return records


def mmlu_get_predict(pred_str: str) -> str:
    """Extract a single capital letter answer (A/B/C/D) from an LLM response."""
    # Look for explicit "answer is X" pattern first
    match = re.search(r"answer\s+is\s*:?\s*(?:Option\s+)?([A-D])", pred_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fall back to last capital letter found
    letters = re.findall(r"\b([A-D])\b", pred_str)
    return letters[-1].upper() if letters else ""


class MMLUDataset(ABC):
    def __init__(self,
                 split: Union[Literal['dev'], Literal['val'], Literal['test']],
                 data_dir: str = None,
                 ) -> None:

        self._split = split

        if data_dir is None:
            # Default: data lives next to this file in MMLU/data/
            data_dir = os.path.join(os.path.dirname(__file__), "MMLU", "data")
        data_path = os.path.join(data_dir, self._split) + os.sep
        self._total_df: pd.DataFrame = self._load_data(data_path)

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'

    @staticmethod
    def _load_data(
            data_path: str,
    ) -> pd.DataFrame:

        rng = np.random.default_rng(888)

        csv_paths = glob.glob(data_path + "*.csv")
        csv_paths = sorted(csv_paths)
        print("Number of topics: ", len(csv_paths))

        names = ['question', 'A', 'B', 'C', 'D', 'correct_answer']

        total_df = pd.DataFrame(columns=names)
        for path in csv_paths:
            single_df = pd.read_csv(path, header=None,
                                    names=names, encoding='utf-8')
            total_df = pd.concat([total_df, single_df])

        total_df = total_df.reset_index(drop=True)

        # Pseudorandom shuffle
        total_df = total_df.reindex(rng.permutation(total_df.index))

        print("Total number of questions: ", len(total_df))

        return total_df

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.DataFrame:
        record = self._total_df.iloc[index]
        assert isinstance(record, pd.DataFrame) or isinstance(record, pd.Series)
        return record

    @staticmethod
    def record_to_input(record: pd.DataFrame) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
        )
        input_dict = {"task": demo_question}
        return input_dict

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            if len(answer) > 0:
                answer = answer[0]
            else:
                answer = ""
        if not isinstance(answer, str):
            raise Exception("Expected string")
        if len(answer) > 0:
            ans_pos = answer.find("answer is")
            if ans_pos != -1:
                answer = answer[ans_pos + len("answer is"):].strip(":").strip().strip("Option").strip()
            answer = answer[0]  # Try to format the answer by taking the first letter
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.DataFrame) -> str:
        correct_answer = record['correct_answer']
        assert isinstance(correct_answer, str), (
            f"String expected but got {correct_answer} "
            f"of type {type(correct_answer)} (2)" \
            f" record={record}")
        return correct_answer
