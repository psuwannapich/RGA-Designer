"""
Cold-start dataset generator using a local Gemma model (via Ollama) or a
HuggingFace model (via HFChat, suitable for HPC / Slurm).

Generates agent-graph training data for autoregressive graph generation.
For each task, several graph topologies are tried; graphs that correctly
solve the task are saved as .pt files consumed by the ARGDesigner trainer.

Usage:
    python experiment/cold_start_gemma.py \
        --dataset gsm8k \
        --dataset_json datasets/gsm8k/gsm8k.jsonl \
        --llm_name gemma3 \
        --output_dir ColdStartData_gemma_gsm8k \
        --num_tasks 0 \
        --batch_size 2

Supported datasets: gsm8k, aqua, multiarith, svamp, humaneval, mmlu
Supported models  : any Ollama short name (gemma3, llama3.2, ...)
                    any HuggingFace Hub ID  (Qwen/Qwen3-8B, ...)
"""

import os
import sys
import json
import math
import copy
import asyncio
import random
import argparse
import csv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from experiment.utils import get_kwargs, save_graph_with_features


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_MATH_DATASETS = {'gsm8k', 'multiarith', 'svamp'}
_MCQ_DATASETS  = {'aqua', 'mmlu'}
_CODE_DATASETS = {'humaneval'}
_ALL_DATASETS  = _MATH_DATASETS | _MCQ_DATASETS | _CODE_DATASETS


def _load_dataset(dataset: str, dataset_json: str):
    if dataset == 'gsm8k':
        from benchmark_datasets.gsm8k_dataset import gsm_data_process
        raw = JSONLReader.parse_file(dataset_json)
        return gsm_data_process(raw)
    elif dataset == 'aqua':
        from benchmark_datasets.aqua_dataset import aqua_data_process
        raw = JSONLReader.parse_file(dataset_json)
        return aqua_data_process(raw)
    elif dataset == 'multiarith':
        from benchmark_datasets.gsm8k_dataset import multiarith_data_process
        with open(dataset_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return multiarith_data_process(raw)
    elif dataset == 'svamp':
        from benchmark_datasets.gsm8k_dataset import svamp_data_process
        with open(dataset_json, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        return svamp_data_process(raw)
    elif dataset == 'humaneval':
        from benchmark_datasets.humaneval_dataset import humaneval_data_process
        raw = JSONLReader.parse_file(dataset_json)
        return humaneval_data_process(raw)
    elif dataset == 'mmlu':
        from benchmark_datasets.MMLU.download import download as mmlu_download
        from benchmark_datasets.mmlu_dataset import mmlu_data_process
        mmlu_download()   # no-op if already downloaded
        return mmlu_data_process(dataset_json, split='test')
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}. Choose from: {sorted(_ALL_DATASETS)}")


def _get_predict(dataset: str, pred_str: str) -> str:
    if dataset in _MATH_DATASETS:
        from benchmark_datasets.gsm8k_dataset import gsm_get_predict
        return gsm_get_predict(pred_str)
    elif dataset == 'aqua':
        from benchmark_datasets.aqua_dataset import aqua_get_predict
        return aqua_get_predict(pred_str)
    elif dataset == 'mmlu':
        from benchmark_datasets.mmlu_dataset import mmlu_get_predict
        return mmlu_get_predict(pred_str)
    elif dataset in _CODE_DATASETS:
        from benchmark_datasets.humaneval_dataset import humaneval_get_predict
        return humaneval_get_predict(pred_str)
    return pred_str


def _is_correct(dataset: str, predicted: str, true_answer: str) -> bool:
    if dataset in _MATH_DATASETS:
        try:
            return float(predicted) == float(true_answer)
        except (ValueError, TypeError):
            return False
    elif dataset in _MCQ_DATASETS:
        return predicted.strip().upper() == true_answer.strip().upper()
    elif dataset in _CODE_DATASETS:
        # true_answer is the test suite; predicted is the extracted code
        from mas_framework.tools.coding.python_executor import PyExecutor
        executor = PyExecutor()
        is_solved, _, _ = executor.execute(predicted, [true_answer], timeout=10)
        return bool(is_solved)
    return False


def _get_role_description(dataset: str) -> dict:
    if dataset in _MATH_DATASETS:
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
    elif dataset == 'aqua':
        from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
    elif dataset == 'mmlu':
        from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
    elif dataset in _CODE_DATASETS:
        from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
    return ROLE_DESCRIPTION


def _get_agent_name(dataset: str) -> str:
    if dataset in _CODE_DATASETS:
        return 'CodeWriting'
    if dataset == 'mmlu':
        return 'AnalyzeAgent'
    return 'MathSolver'


def _get_decision_method(dataset: str) -> str:
    if dataset in _CODE_DATASETS:
        return 'FinalWriteCode'
    return 'FinalRefer'


# ---------------------------------------------------------------------------
# Graph topology configs
# ---------------------------------------------------------------------------

def get_configs(min_agents: int = 3, max_agents: int = 4) -> list:
    """Return (mode, agent_count) pairs for cold-start generation."""
    configs = []
    for n in range(min_agents, max_agents + 1):
        configs.append(('FullConnected', n))
        if n >= 4:
            configs.append(('Mesh', n))
    return configs


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------

def _write_csv(output_dir: str, rows: list, dataset: str):
    filepath = os.path.join(output_dir, f"records_{dataset}.csv")
    fieldnames = ["dataset", "id", "question", "answer", "mode", "size", "is_correct"]
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def evaluate_and_save(
    graph: Graph,
    dataset_records: list,
    args,
    mode: str,
    agent_num: int,
    output_dir: str,
    solved_counter: dict,
):
    num_batches = math.ceil(len(dataset_records) / args.batch_size)
    total_solved = 0

    for i_batch in tqdm(range(num_batches), desc=f"{mode}-{agent_num}"):
        batch = dataset_records[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not batch:
            continue

        tasks = []
        for rec_idx, record in enumerate(batch):
            realized = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized.to_pyg_graph(input_dict)
            tg = TestGraph(
                domain=args.dataset,
                llm_name=args.llm_name,
                decision_method=_get_decision_method(args.dataset),
                pyg_data=flow_graph,
            )
            global_idx = i_batch * args.batch_size + rec_idx
            metadata = {
                "record": record,
                "flow_graph": flow_graph,
                "question": record["task"],
                "global_idx": global_idx,
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        results = await asyncio.gather(
            *[t for t, _ in tasks], return_exceptions=True
        )

        csv_rows = []
        for i, result in enumerate(results):
            meta = tasks[i][1]
            record = meta["record"]

            if isinstance(result, Exception):
                print(f"  [error] task {meta['global_idx']}: {result}")
                continue

            raw_answer = result[0] if isinstance(result, (list, tuple)) else result
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]
            if not isinstance(raw_answer, str):
                raw_answer = str(raw_answer)

            predicted = _get_predict(args.dataset, raw_answer)
            correct = _is_correct(args.dataset, predicted, record["answer"])

            if correct:
                total_solved += 1
                solved_counter['total'] += 1
                name = "_".join([
                    args.dataset,
                    str(meta["global_idx"]),
                    mode,
                    str(agent_num),
                    "True",
                ])
                filepath = os.path.join(output_dir, f"{name}.pt")
                save_graph_with_features(meta["flow_graph"], filepath, {
                    "mode": mode,
                    "agent_nums": agent_num,
                    "is_correct": True,
                    "question": meta["question"],
                })

            csv_rows.append({
                "dataset": args.dataset,
                "id": meta["global_idx"],
                "question": meta["question"],
                "answer": record["answer"],
                "mode": mode,
                "size": agent_num,
                "is_correct": correct,
            })

        _write_csv(output_dir, csv_rows, args.dataset)

    print(f"  {mode}-{agent_num}: solved {total_solved}/{len(dataset_records)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate cold-start graph data using a local Gemma model (Ollama)"
    )
    parser.add_argument('--dataset', type=str, default='gsm8k',
                        choices=sorted(_ALL_DATASETS),
                        help='Task dataset name')
    parser.add_argument('--dataset_json', type=str,
                        default='datasets/gsm8k/gsm8k.jsonl',
                        help='Path to the dataset JSONL file')
    parser.add_argument('--llm_name', type=str, default='gemma3',
                        help='Ollama model name (e.g. gemma3, gemma:2b, llama3.2)')
    parser.add_argument('--output_dir', type=str, default='ColdStartData_gemma',
                        help='Directory to save generated .pt graph files')
    parser.add_argument('--num_tasks', type=int, default=0,
                        help='Number of tasks to sample for cold-start generation. 0 for using all data')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Async batch size (keep small for local models)')
    parser.add_argument('--num_rounds', type=int, default=1,
                        help='Number of agent interaction rounds per task')
    parser.add_argument('--min_agents', type=int, default=3,
                        help='Minimum number of agents per graph')
    parser.add_argument('--max_agents', type=int, default=4,
                        help='Maximum number of agents per graph')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help='Number of training iterations (used to compute train/test split size)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset split constants (mirrors original cold_start_gsm8k.py)
# ---------------------------------------------------------------------------

BASE_RATE = 0.4  # fraction of train set used for cold-start graph generation

_TASK_SPLIT_DIRS = {
    'gsm8k':      'experiment/gsm8k',
    'aqua':       'experiment/aqua',
    'multiarith': 'experiment/multiarith',
    'svamp':      'experiment/svamp',
    'humaneval':  'experiment/humaneval',
    'mmlu':       'experiment/mmlu',
}


def _save_task_split(dataset: str, base_task_indices: list,
                     finetune_task_indices: list, test_indices: list,
                     project_root: str):
    subdir = _TASK_SPLIT_DIRS.get(dataset, f'experiment/{dataset}')
    split_dir = os.path.join(project_root, subdir)
    os.makedirs(split_dir, exist_ok=True)
    split_path = os.path.join(split_dir, f'task_split_{dataset}.json')

    if os.path.exists(split_path):
        print(f"Task split already exists, loading from: {split_path}")
        with open(split_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        base_task_indices     = existing.get('base_tasks_indices', base_task_indices)
        finetune_task_indices = existing.get('finetune_tasks_indices', finetune_task_indices)
        test_indices          = existing.get('test_indices', test_indices)
    else:
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump({
                'base_tasks_indices':     base_task_indices,
                'finetune_tasks_indices': finetune_task_indices,
                'test_indices':           test_indices,
            }, f)
        print(f"Task split saved to: {split_path}")

    print(f"  base (cold-start): {len(base_task_indices)}")
    print(f"  finetune         : {len(finetune_task_indices)}")
    print(f"  test             : {len(test_indices)}")
    return split_path, base_task_indices, finetune_task_indices, test_indices


async def main():
    args = parse_args()
    random.seed(args.seed)

    # Resolve project root (one level above experiment/)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load full dataset
    print(f"\nLoading {args.dataset} dataset from {args.dataset_json} ...")
    all_records = _load_dataset(args.dataset, args.dataset_json)
    print(f"Total records: {len(all_records)}")

    # ---- Train / test split (mirrors original cold_start_gsm8k.py) ----------
    train_set_size = args.num_iterations * args.batch_size
    train_set_size = min(train_set_size, len(all_records))

    all_indices = list(range(len(all_records)))
    train_indices = all_indices[:train_set_size]
    test_indices  = all_indices[train_set_size:]

    # Shuffle train to randomly assign base vs finetune
    finetune_candidates = train_indices.copy()
    random.shuffle(finetune_candidates)
    base_task_count     = int(BASE_RATE * len(finetune_candidates))
    base_task_indices   = finetune_candidates[:base_task_count]
    finetune_task_indices = finetune_candidates[base_task_count:]

    _, base_task_indices, finetune_task_indices, test_indices = _save_task_split(
        args.dataset, base_task_indices, finetune_task_indices, test_indices, project_root)

    # Cold-start generation uses only base_task_indices (not full dataset)
    # num_tasks == 0 means "use all base tasks"
    if args.num_tasks == 0:
        sampled_indices = base_task_indices
    else:
        num_tasks = min(args.num_tasks, len(base_task_indices))
        sampled_indices = random.sample(base_task_indices, num_tasks)
    sampled = [all_records[i] for i in sampled_indices]
    print(f"\nUsing {len(sampled)} base tasks for cold-start generation.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}\n")

    role_desc = _get_role_description(args.dataset)
    available_roles = list(role_desc.keys())
    agent_name = _get_agent_name(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    configs = get_configs(args.min_agents, args.max_agents)

    print(f"LLM: {args.llm_name}")
    print(f"Topologies: {configs}")
    print(f"Roles: {available_roles}\n")

    solved_counter = {'total': 0}

    for mode, agent_num in configs:
        print(f"=== Config: {mode}, {agent_num} agents ===")
        kwargs = get_kwargs(mode, agent_num)

        # Assign random roles to each agent node
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': r} for r in random_roles]
        print(f"  Roles assigned: {random_roles}")

        graph = Graph(
            domain=args.dataset,
            llm_name=args.llm_name,
            agent_names=[agent_name] * agent_num,
            decision_method=decision_method,
            **kwargs,
        )

        await evaluate_and_save(
            graph=graph,
            dataset_records=sampled,
            args=args,
            mode=mode,
            agent_num=agent_num,
            output_dir=args.output_dir,
            solved_counter=solved_counter,
        )

    print(f"\nDone. Total graphs saved: {solved_counter['total']}")
    print(f"Dataset saved to: {args.output_dir}/")
    print(f"\nTo train ARGDesigner on this data, run:")
    print(f"  python experiment/pretrain.py --dataset {args.dataset} --data_dir {args.output_dir} --output_dir checkpoints/{args.dataset}")


def main_cli():
    """Entry point for `uv run cold-start` (defined in pyproject.toml)."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())


if __name__ == "__main__":
    main_cli()
