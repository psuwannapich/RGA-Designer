"""
Stage 1 of the split benchmark pipeline.

Generates inference graphs from a trained ARGDesigner model and saves them
to a JSONL file so LLM inference (Stage 2, benchmark_pregraph.py) can run
on a separate machine / job.

Output JSONL line format:
    {task_id, task_text, true_answer, num_nodes, num_edges, graph: {nodes, edges}}

Best-of-N sampling (--best_of_n N --rm_checkpoint PATH --role_emb_path PATH):
    Generate N candidate graphs per task, score each with the reward model,
    keep the graph with the highest reward score.  Only one graph per task is
    written to the output file (the selected best), so the downstream
    benchmark_pregraph.py step is unchanged.
"""

import os
import sys
import json
import time
import math
import pickle
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from mas_framework.llm.profile_embedding import get_sentence_model
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
    p.add_argument('--split_key', type=str, default='test_indices',
                   help="Comma-separated key(s) from task_split_path to select tasks "
                        "(union when multiple). E.g. 'base_tasks_indices,finetune_tasks_indices' "
                        "for the training set. (default: test_indices)")
    p.add_argument('--output_file', type=str, required=True,
                   help="JSONL output path for generated graphs")
    p.add_argument('--limit', type=int, default=None,
                   help="Cap number of test samples")
    p.add_argument('--batch_size', type=int, default=64,
                   help="Flush checkpoint to disk every N samples")
    p.add_argument('--seed', type=int, default=42)
    # ---- Best-of-N arguments ------------------------------------------------
    p.add_argument('--best_of_n', type=int, default=1,
                   help="Number of candidate graphs to sample per task. "
                        "The one with the highest reward model score is kept. "
                        "Requires --rm_checkpoint and --role_emb_path. (default: 1 = disabled)")
    p.add_argument('--rm_checkpoint', type=str, default=None,
                   help="Path to trained GraphRewardModel checkpoint (.pth). "
                        "Required when --best_of_n > 1.")
    p.add_argument('--role_emb_path', type=str, default=None,
                   help="Path to precomputed_role_embeddings.pkl for this dataset. "
                        "Required when --best_of_n > 1.")
    p.add_argument('--size_lambda', type=float, default=0.0,
                   help="Weight of the size term added to the reward-model score in "
                        "Best-of-N (same form as the policy's lambda_eff). 0 = none.")
    p.add_argument('--reward_squash', action='store_true',
                   help="Apply a sigmoid to the reward model output before adding the size term.")
    p.add_argument('--v_max', type=int, default=0,
                   help="V_max of the size term (0 = largest candidate).")
    p.add_argument('--bon_temperature', type=float, default=1.2,
                   help="Sampling temperature used when generating the N candidates. "
                        "Values > 1.0 increase diversity across candidates. (default: 1.2)")
    p.add_argument('--no_ef', action='store_true',
                   help="Use best_model.pth instead of ef_best_model.pth")
    p.add_argument('--model_type', type=str, default='arg_designer',
                   help="Label of the graph generator model (logged in output)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Best-of-N helpers
# ---------------------------------------------------------------------------

def load_best_of_n_components(rm_checkpoint: str, role_emb_path: str, device: torch.device):
    """Load reward model and role embeddings for Best-of-N scoring."""
    from mas_framework.rga.reward_trainer import load_reward_model

    print(f"  [BoN] Loading reward model from {rm_checkpoint}")
    rm = load_reward_model(rm_checkpoint, device)
    rm.eval()

    print(f"  [BoN] Loading role embeddings from {role_emb_path}")
    with open(role_emb_path, "rb") as f:
        role_embs = pickle.load(f)

    return rm, role_embs


def score_graph_with_rm(rm, nx_graph, task_emb_np: np.ndarray,
                        role_embs: dict, device: torch.device) -> float:
    """Score a single NetworkX graph with the reward model.

    Returns the scalar reward.  Returns -inf on any error so the graph
    is never selected as best.
    """
    from mas_framework.rga.preference_data import GraphSnapshot

    EMB_DIM = 384
    nodes_raw = []
    for n in sorted(nx_graph.nodes()):
        role = nx_graph.nodes[n].get("role", "Unknown")
        emb  = role_embs.get(role)
        if isinstance(emb, torch.Tensor):
            emb = emb.numpy()
        emb = np.array(emb, dtype=np.float32) if emb is not None else np.zeros(EMB_DIM, dtype=np.float32)
        nodes_raw.append({"id": int(n), "role": role, "role_embedding": emb.tolist()})

    snap = GraphSnapshot(
        nodes=nodes_raw,
        edges=[[int(u), int(v)] for u, v in nx_graph.edges()],
        num_nodes=nx_graph.number_of_nodes(),
    )
    try:
        pyg = snap.to_pyg(task_emb_np)
        return rm.score_single(pyg.x.to(device), pyg.edge_index.to(device))
    except Exception:
        return float("-inf")


def select_best_graph(model, task_emb_tensor: torch.Tensor, task_emb_np: np.ndarray,
                      role_description: dict, rm, role_embs: dict,
                      device: torch.device, n: int, temperature: float,
                      question_id=None, size_lambda: float = 0.0,
                      v_max: int = 0, reward_squash: bool = False):
    """Generate *n* candidate graphs, score each, return the highest-reward one.

    Falls back to a single graph (temperature=1.0) if all scoring attempts fail.
    """
    candidates = []
    for _ in range(n):
        graphs = generate_graph(model, task_emb_tensor, role_description,
                                question_id=question_id, temperature=temperature)
        if graphs:
            candidates.append(graphs[0])

    if not candidates:
        # Absolute fallback: one graph at default temperature
        graphs = generate_graph(model, task_emb_tensor, role_description, question_id)
        return graphs[0] if graphs else None

    if len(candidates) == 1:
        return candidates[0]

    scores = [score_graph_with_rm(rm, g, task_emb_np, role_embs, device)
              for g in candidates]

    # Size term, as in the policy reward.
    if size_lambda and size_lambda > 0.0:
        if reward_squash:
            scores = torch.sigmoid(torch.tensor(scores, dtype=torch.float64)).tolist()
        v_max = float(v_max or max(g.number_of_nodes() for g in candidates) or 1)
        e_max = v_max * (v_max - 1) / 2
        sel = []
        for g, r in zip(candidates, scores):
            node_bonus = g.number_of_nodes() / v_max
            edge_bonus = (g.number_of_edges() / e_max) if e_max > 0 else 0.0
            sel.append(r + size_lambda * (1 - node_bonus - edge_bonus))
    else:
        sel = scores
    best_idx = int(np.argmax(sel))
    return candidates[best_idx], scores, best_idx


def setup_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset: str, dataset_path: str, task_split_path: str,
                 split_key: str = 'test_indices'):
    """Return (records, selected_indices, role_description).

    Each record dict has at minimum:
        task_text   : str
        true_answer : str

    split_key may be comma-separated (e.g. 'base_tasks_indices,finetune_tasks_indices')
    in which case the union of those index lists is used, preserving order.
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
        from experiment.cold_start import _load_dataset
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

    # Resolve selected indices from task split
    if task_split_path and os.path.exists(task_split_path):
        with open(task_split_path, 'r') as f:
            split = json.load(f)
        keys = [k.strip() for k in split_key.split(',') if k.strip()]
        # Union of all requested keys, preserving order, deduplicating.
        seen = set()
        selected_indices = []
        for k in keys:
            if k not in split:
                print(f"WARNING: split key '{k}' not found in {task_split_path} "
                      f"(available: {list(split.keys())})")
                continue
            for idx in split[k]:
                if idx not in seen:
                    seen.add(idx)
                    selected_indices.append(idx)
        if not selected_indices:
            print(f"WARNING: no indices found for split_key={split_key!r} — using all samples.")
            selected_indices = list(range(len(records)))
    else:
        if task_split_path:
            print(f"No task split at '{task_split_path}' — using all {len(records)} samples.")
        selected_indices = list(range(len(records)))

    return records, selected_indices, ROLE_DESCRIPTION  # noqa: F821 (assigned in each branch)


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

    # ---- Validate Best-of-N arguments ---------------------------------------
    use_bon = args.best_of_n > 1
    if use_bon:
        if not args.rm_checkpoint or not os.path.exists(args.rm_checkpoint):
            raise ValueError("--best_of_n > 1 requires a valid --rm_checkpoint path")
        if not args.role_emb_path or not os.path.exists(args.role_emb_path):
            raise ValueError("--best_of_n > 1 requires a valid --role_emb_path path")

    ef = not args.no_ef
    ckpt_name = 'ef_best_model.pth' if ef else 'best_model.pth'
    print("=" * 60)
    print(f"  Graph generator  : {args.model_type.upper()}")
    print(f"  Checkpoint       : {args.model_path}/{ckpt_name}")
    print(f"  Dataset          : {args.dataset}")
    print(f"  Output           : {args.output_file}")
    if use_bon:
        print(f"  Best-of-N        : N={args.best_of_n}  temperature={args.bon_temperature}")
        print(f"  Reward model     : {args.rm_checkpoint}")
    print("=" * 60)

    print(f"Loading ARGDesigner ({ckpt_name}) ...")
    model = load_model(args.model_path, ef=ef)
    model.eval()
    rm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Best-of-N components if needed
    rm, role_embs = (None, None)
    if use_bon:
        rm, role_embs = load_best_of_n_components(
            args.rm_checkpoint, args.role_emb_path, rm_device
        )

    sentence_model = get_sentence_model()

    all_records, test_indices, role_description = load_dataset(
        args.dataset, args.dataset_path, args.task_split_path,
        split_key=args.split_key,
    )
    dataset = [all_records[i] for i in test_indices]
    if args.limit:
        dataset = dataset[:args.limit]
        test_indices = test_indices[:args.limit]
    print(f"Loaded {len(dataset)} {args.dataset} samples (split_key={args.split_key!r}).")

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
                desc=f"Generating graphs [{args.dataset}]"
                     + (f" (BoN={args.best_of_n})" if use_bon else ""))

    for i, record in pbar:
        task_id = record.get('_task_id_raw') or f"task_{test_indices[i]}"
        if task_id in done_ids:
            continue

        task_text = record['task_text']
        true_answer = record['true_answer']

        try:
            emb_np  = sentence_model.encode(task_text).astype(np.float32)
            emb_tensor = torch.tensor(emb_np, device=model.args.device).float()

            if use_bon:
                result = select_best_graph(
                    model, emb_tensor, emb_np,
                    role_description, rm, role_embs,
                    rm_device, args.best_of_n,
                    args.bon_temperature, question_id=i,
                    size_lambda=args.size_lambda,
                    v_max=args.v_max,
                    reward_squash=args.reward_squash,
                )
                g, bon_scores, bon_idx = result if isinstance(result, tuple) else (result, None, None)
            else:
                graphs = generate_graph(model, emb_tensor, role_description, i)
                if not graphs:
                    raise RuntimeError("generate_graph returned empty list")
                g, bon_scores, bon_idx = graphs[0], None, None

            if g is None:
                raise RuntimeError("No valid graph generated")

            rec = {
                'task_id': task_id,
                'task_text': task_text,
                'true_answer': true_answer,
                'model_type': args.model_type,
                'model_path': args.model_path,
                'graph': serialize_graph(g),
                'num_nodes': g.number_of_nodes(),
                'num_edges': g.number_of_edges(),
            }
            if bon_scores is not None:
                rec['bon_scores'] = [round(float(s), 4) for s in bon_scores]
                if bon_idx is not None and 0 <= bon_idx < len(bon_scores):
                    rec['bon_selected'] = int(bon_idx)
                    if args.reward_squash:
                        rec['bon_prob'] = round(torch.sigmoid(torch.tensor(float(bon_scores[bon_idx]), dtype=torch.float64)).item(), 4)
                rec['bon_n'] = args.best_of_n
            results.append(rec)

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
