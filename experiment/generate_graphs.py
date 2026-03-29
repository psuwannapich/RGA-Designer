"""
Stage 1 of the split benchmark pipeline.

Generates inference graphs from a trained ARGDesigner model and saves them
to a JSONL file so LLM inference (Stage 2, benchmark_pregraph.py) can run
on a separate machine / job.

Output JSONL line format:
    {task_id, task_text, true_answer, num_nodes, num_edges, graph: {nodes, edges}}
"""

import os
import sys
import json
import time
import math
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from experiment.utils import load_model, generate_graph


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate ARGDesigner inference graphs (Stage 1)")
    p.add_argument('--model_path', type=str, required=True,
                   help="Trained ARGDesigner checkpoint directory")
    p.add_argument('--dataset', type=str, required=True,
                   choices=['gsm8k', 'aqua', 'humaneval', 'mmlu', 'multiarith', 'svamp'])
    p.add_argument('--dataset_path', type=str, required=True,
                   help="Path to dataset file (or data_dir for MMLU)")
    p.add_argument('--task_split_path', type=str, default=None,
                   help="Task split JSON; if absent, all samples are used")
    p.add_argument('--output_file', type=str, required=True,
                   help="JSONL output path for generated graphs")
    p.add_argument('--limit', type=int, default=None,
                   help="Cap number of test samples")
    p.add_argument('--batch_size', type=int, default=64,
                   help="Flush checkpoint to disk every N samples")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no_ef', action='store_true',
                   help="Use best_model.pth instead of ef_best_model.pth")
    p.add_argument('--model_type', type=str, default='arg_designer',
                   choices=['arg_designer', 'rlhf'],
                   help="Label of the graph generator model (logged in output)")
    return p.parse_args()


def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, dataset_path: str, task_split_path: str):
    """Return (records, test_indices, role_description).

    Each record dict has at minimum:
        task_text   : str
        true_answer : str
    """
    if dataset == 'gsm8k':
        from mas_framework.tools.reader.readers import JSONLReader
        from benchmark_datasets.gsm8k_dataset import gsm_data_process
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
        raw = JSONLReader.parse_file(dataset_path)
        records = [{'task_text': r['task'], 'true_answer': str(r['answer'])}
                   for r in gsm_data_process(raw)]

    elif dataset == 'aqua':
        from mas_framework.tools.reader.readers import JSONLReader
        from benchmark_datasets.aqua_dataset import aqua_data_process
        from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
        raw = JSONLReader.parse_file(dataset_path)
        records = [{'task_text': r['task'], 'true_answer': str(r['answer'])}
                   for r in aqua_data_process(raw)]

    elif dataset == 'humaneval':
        from mas_framework.tools.reader.readers import JSONLReader
        from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
        raw = JSONLReader.parse_file(dataset_path)
        records = [{'task_text': r['prompt'],
                    'true_answer': r.get('test', ''),
                    '_task_id_raw': r.get('task_id', '')}
                   for r in raw]

    elif dataset == 'mmlu':
        from experiment.cold_start_gemma import _load_dataset
        from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
        ds = _load_dataset('mmlu', dataset_path)
        records = []
        for r in ds:
            records.append({'task_text': r['task'],
                            'true_answer': r["answer"]})

    elif dataset == 'multiarith':
        from benchmark_datasets.gsm8k_dataset import multiarith_data_process
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        records = [{'task_text': r['task'], 'true_answer': str(r['answer'])}
                   for r in multiarith_data_process(raw)]

    elif dataset == 'svamp':
        from benchmark_datasets.gsm8k_dataset import svamp_data_process
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
        with open(dataset_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        records = [{'task_text': r['task'], 'true_answer': str(r['answer'])}
                   for r in svamp_data_process(raw)]

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Resolve test indices
    if task_split_path and os.path.exists(task_split_path):
        with open(task_split_path, 'r') as f:
            split = json.load(f)
        test_indices = split.get('test_indices', list(range(len(records))))
    else:
        if task_split_path:
            print(f"No task split at '{task_split_path}' — using all {len(records)} samples.")
        test_indices = list(range(len(records)))

    return records, test_indices, ROLE_DESCRIPTION  # noqa: F821 (assigned in each branch)


# ---------------------------------------------------------------------------
# Graph serialisation
# ---------------------------------------------------------------------------

def serialize_graph(g) -> dict:
    """Convert a NetworkX graph to a JSON-serialisable dict."""
    return {
        'nodes': [{'id': n, 'role': g.nodes[n].get('role', 'Unknown')}
                  for n in sorted(g.nodes())],
        'edges': [[int(u), int(v)] for u, v in g.edges()],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    setup_seed(args.seed)

    ef = not args.no_ef
    ckpt_name = 'ef_best_model.pth' if ef else 'best_model.pth'
    print("=" * 60)
    print(f"  Graph generator  : {args.model_type.upper()}")
    print(f"  Checkpoint       : {args.model_path}/{ckpt_name}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Output           : {args.output_file}")
    print("=" * 60)
    print(f"Loading ARGDesigner ({ckpt_name}) ...")
    model = load_model(args.model_path, ef=ef)
    model.eval()

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    all_records, test_indices, role_description = load_dataset(
        args.dataset, args.dataset_path, args.task_split_path
    )
    dataset = [all_records[i] for i in test_indices]
    if args.limit:
        dataset = dataset[:args.limit]
        test_indices = test_indices[:args.limit]
    print(f"Loaded {len(dataset)} {args.dataset} test samples.")

    # Checkpoint resume
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    done_ids: set = set()
    results = []
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    results.append(r)
                    done_ids.add(r['task_id'])
        if done_ids:
            print(f"[Resume] {len(done_ids)} graphs already generated — skipping.")

    total = len(dataset)
    _wall = time.time()

    pbar = tqdm(enumerate(dataset), total=total,
                desc=f"Generating graphs [{args.dataset}]")

    for i, record in pbar:
        task_id = record.get('_task_id_raw') or f"task_{test_indices[i]}"
        if task_id in done_ids:
            continue

        task_text = record['task_text']
        true_answer = record['true_answer']

        try:
            emb = torch.tensor(
                sentence_model.encode(task_text), device=model.args.device
            ).float()
            graphs = generate_graph(model, emb, role_description, i)
            if not graphs:
                raise RuntimeError("generate_graph returned empty list")
            g = graphs[0]
            results.append({
                'task_id': task_id,
                'task_text': task_text,
                'true_answer': true_answer,
                'model_type': args.model_type,
                'model_path': args.model_path,
                'graph': serialize_graph(g),
                'num_nodes': g.number_of_nodes(),
                'num_edges': g.number_of_edges(),
            })
        except Exception as e:
            print(f"\nError on {task_id}: {e}")
            results.append({
                'task_id': task_id,
                'task_text': task_text,
                'true_answer': true_answer,
                'model_type': args.model_type,
                'model_path': args.model_path,
                'graph': None,
                'error': str(e),
            })

        # Periodic checkpoint flush
        done_count = i + 1
        if done_count % args.batch_size == 0 or done_count == total:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            elapsed = time.time() - _wall
            rate = done_count / elapsed if elapsed > 0 else 0
            valid = sum(1 for r in results if r.get('graph') is not None)
            pbar.set_postfix({'rate': f'{rate:.1f}/s', 'valid': valid, 'saved': len(results)})

    valid_total = sum(1 for r in results if r.get('graph') is not None)
    print(f"\nDone. {valid_total}/{len(results)} graphs saved to {args.output_file}")


if __name__ == '__main__':
    main()
