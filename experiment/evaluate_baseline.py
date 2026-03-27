"""
Baseline evaluator for ARG-Designer comparison methods.

Supports fixed-topology multi-agent baselines and single-agent methods
that do NOT require a trained ARGDesigner checkpoint.

Methods
-------
cot              Chain-of-Thought: single agent, direct answer
self_consistency Self-Consistency: N independent CoT samples, majority vote
chain            Linear chain of N agents
complete         Fully-connected graph of N agents (Complete Graph)
random           Randomly-connected graph of N agents
star             Star topology of N agents (one central hub)
llm_debate       Fully-connected graph with debate-style prompting (≈ LLM-Debate)

Usage
-----
python experiment/evaluate_baseline.py \
    --dataset    gsm8k \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --method     chain \
    --llm_name   Qwen/Qwen3-8B \
    --num_agents 4 \
    --limit      100 \
    --output_file results/gsm8k_chain.jsonl

Supported datasets: gsm8k, aqua, multiarith, svamp, humaneval, mmlu
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime
import json
import math
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from tqdm import tqdm

from mas_framework.graph.graph import Graph, TestGraph
from experiment.cold_start_gemma import (
    _load_dataset,
    _load_task_split,
    _get_predict,
    _is_correct,
    _get_role_description,
    _get_decision_method,
    _get_agent_name,
)
from experiment.utils import get_kwargs
from mas_framework.utils.globals import PromptTokens, CompletionTokens

# ---------------------------------------------------------------------------
# Method → topology mapping
# ---------------------------------------------------------------------------

# Methods that map directly to a graph topology mode.
# CoT uses Chain-1 (not DirectAnswer) because DirectAnswer hardcodes role='Normal'
# which is not in any dataset's ROLE_DESCRIPTION and causes a KeyError in
# construct_adj_matrix / construct_features.  Chain-1 produces a single
# isolated agent (spatial_mask=[[0]], temporal_mask=[[1]]) with a real role.
_TOPOLOGY_METHODS = {
    "cot":          ("Chain",        1),    # single agent, no connections
    "chain":        ("Chain",        None), # N from --num_agents
    "complete":     ("FullConnected",None),
    "random":       ("Random",       None),
    "star":         ("Star",         None),
    "llm_debate":   ("FullConnected",None), # LLM-Debate ≈ fully connected debate
}

SUPPORTED_METHODS = list(_TOPOLOGY_METHODS.keys()) + ["self_consistency", "vanilla"]

# Default number of agents per method (when --num_agents is not set)
_DEFAULT_AGENTS: Dict[str, int] = {
    "vanilla":         1,
    "cot":             1,
    "self_consistency":1,  # single agent run multiple times
    "chain":           4,
    "complete":        4,
    "random":          4,
    "star":            4,
    "llm_debate":      4,
}

# SC sample count (number of independent CoT runs before majority vote)
_SC_SAMPLES = 5


# ---------------------------------------------------------------------------
# Vanilla: direct LLM call — no agent framework, no few-shot, no CoT
# ---------------------------------------------------------------------------

def _vanilla_prompt(task: str) -> list:
    """Minimal system+user prompt that asks for a direct answer."""
    return [
        {"role": "system", "content": "You are a helpful assistant. Answer the following question directly and concisely."},
        {"role": "user",   "content": task},
    ]


async def _run_vanilla(
    llm_name: str,
    task: str,
) -> Optional[str]:
    """Call the LLM directly with a minimal prompt; return raw output or None."""
    from mas_framework.llm.llm_registry import LLMRegistry
    llm = LLMRegistry.get(llm_name)
    try:
        result = await llm.agen(_vanilla_prompt(task))
    except Exception as e:
        import traceback
        print(f"  [vanilla error] {type(e).__name__}: {e}")
        traceback.print_exc()
        return None
    return result if isinstance(result, str) else (result[0] if result else None)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(
    method: str,
    num_agents: int,
    domain: str,
    llm_name: str,
    role_descriptions: Dict[str, str],
) -> Graph:
    """Return a Graph instance for the given method."""
    mode, forced_n = _TOPOLOGY_METHODS[method]
    n = forced_n if forced_n is not None else num_agents

    kwargs = get_kwargs(mode, n)

    # Always assign roles from the domain's ROLE_DESCRIPTION so that
    # construct_adj_matrix / construct_features never see an unknown role.
    available_roles = list(role_descriptions.keys())
    kwargs["node_kwargs"] = [
        {"role": available_roles[i % len(available_roles)]}
        for i in range(n)
    ]

    agent_name = _get_agent_name(domain)
    decision_method = _get_decision_method(domain)

    return Graph(
        domain=domain,
        llm_name=llm_name,
        agent_names=[agent_name] * n,
        decision_method=decision_method,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Single-graph inference
# ---------------------------------------------------------------------------

async def _run_once(
    graph: Graph,
    task: str,
    domain: str,
    llm_name: str,
    decision_method: str,
) -> Optional[str]:
    """Run one graph on one task; return raw string output or None on error."""
    realized = copy.deepcopy(graph)
    input_dict = {"task": task}
    flow_graph = realized.to_pyg_graph(input_dict)
    tg = TestGraph(
        domain=domain,
        llm_name=llm_name,
        decision_method=decision_method,
        pyg_data=flow_graph,
    )
    try:
        result = await tg.arun(input_dict, 1)
    except Exception as e:
        import traceback
        print(f"  [error] {type(e).__name__}: {e}")
        traceback.print_exc()
        return None

    raw = result[0] if isinstance(result, (list, tuple)) and result else result
    if isinstance(raw, list) and raw:
        raw = raw[0]
    return str(raw) if not isinstance(raw, str) else raw


# ---------------------------------------------------------------------------
# Self-Consistency (majority vote over N CoT samples)
# ---------------------------------------------------------------------------

async def _run_self_consistency(
    graph: Graph,
    task: str,
    domain: str,
    llm_name: str,
    decision_method: str,
    n_samples: int = _SC_SAMPLES,
) -> Optional[str]:
    """Run N independent CoT passes and return the majority-vote answer."""
    raw_outputs = []
    for _ in range(n_samples):
        raw = await _run_once(graph, task, domain, llm_name, decision_method)
        if raw is not None:
            raw_outputs.append(raw)

    if not raw_outputs:
        return None

    # Extract predicted answers and take majority
    predicted = [_get_predict(domain, r) for r in raw_outputs]
    majority = Counter(predicted).most_common(1)[0][0]
    return majority   # return the winning predicted string directly


# ---------------------------------------------------------------------------
# Full dataset evaluation
# ---------------------------------------------------------------------------

async def evaluate(args) -> None:
    all_records = _load_dataset(args.dataset, args.dataset_json)
    if args.seed is not None:
        random.seed(args.seed)

    # Filter to test split to avoid data leakage
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        if args.task_split_path:
            with open(args.task_split_path, 'r', encoding='utf-8') as f:
                task_split = json.load(f)
        else:
            task_split = _load_task_split(args.dataset, project_root)
        test_indices = task_split.get('test_indices', list(range(len(all_records))))
        dataset_records = [all_records[i] for i in test_indices]
        print(f"Using test split: {len(dataset_records)}/{len(all_records)} records")
    except FileNotFoundError as e:
        print(f"Warning: {e}\nFalling back to full dataset — run cold-start to create a split.")
        dataset_records = all_records

    if args.limit:
        dataset_records = dataset_records[: args.limit]
    total = len(dataset_records)

    # ------------------------------------------------------------------
    # Resume from checkpoint: the output file itself is the checkpoint.
    # Any task_id already written to it is skipped on restart.
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    done_ids: set = set()
    results: List[dict] = []
    solved = 0

    if os.path.exists(args.output_file):
        with open(args.output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done_ids.add(r["task_id"])
                    results.append(r)
                    if r.get("is_solved"):
                        solved += 1
                except json.JSONDecodeError:
                    pass
        if done_ids:
            print(f"Resuming: {len(done_ids)} tasks already done, {total - len(done_ids)} remaining.")

    print(f"Evaluating {total} samples | dataset={args.dataset} | method={args.method} | llm={args.llm_name}")

    role_descriptions = _get_role_description(args.dataset)
    decision_method   = _get_decision_method(args.dataset)
    n_agents = args.num_agents or _DEFAULT_AGENTS.get(args.method, 4)

    # Pre-build the graph template (reused per task via deepcopy)
    # vanilla bypasses the graph framework entirely
    graph = None
    if args.method not in ("vanilla",):
        if args.method == "self_consistency":
            graph = _build_graph("cot", 1, args.dataset, args.llm_name, role_descriptions)
        else:
            graph = _build_graph(args.method, n_agents, args.dataset, args.llm_name, role_descriptions)

    # Open output file in append mode — new results are flushed after every batch
    out_f = open(args.output_file, "a", encoding="utf-8")

    try:
        num_batches = math.ceil(total / args.eval_batch_size)
        pbar = tqdm(range(num_batches), desc=f"{args.method}/{args.dataset}")
        for b in pbar:
            batch = dataset_records[b * args.eval_batch_size : (b + 1) * args.eval_batch_size]
            base_idx = b * args.eval_batch_size

            # Skip tasks already completed in a previous run
            pending = [
                (i, rec) for i, rec in enumerate(batch)
                if f"task_{base_idx + i}" not in done_ids
            ]
            if not pending:
                acc = solved / len(results) * 100 if results else 0
                pbar.set_postfix({"acc": f"{acc:.1f}%", "done": len(results), "skip": len(batch)})
                continue

            # Build one coroutine per record in the batch; fire them all concurrently.
            # For vanilla/cot: every coroutine is one LLM call → the HF batcher in
            # hf_chat.py groups them into a single model.generate(batch) → high GPU util.
            # For multi-agent methods: each task still runs its agents sequentially
            # inside the coroutine, but different tasks overlap → partial batching.
            async def _process(record: dict, idx: int) -> Tuple[int, dict]:
                task  = record["task"]
                truth = record["answer"]

                pt_before = PromptTokens.instance().value
                ct_before = CompletionTokens.instance().value

                if args.method == "vanilla":
                    raw = await _run_vanilla(args.llm_name, task)
                    predicted = _get_predict(args.dataset, raw or "")
                elif args.method == "self_consistency":
                    predicted = await _run_self_consistency(
                        graph, task, args.dataset, args.llm_name, decision_method,
                        n_samples=args.sc_samples,
                    )
                    raw = predicted or ""
                else:
                    raw = await _run_once(
                        graph, task, args.dataset, args.llm_name, decision_method,
                    )
                    predicted = _get_predict(args.dataset, raw or "")

                correct = _is_correct(args.dataset, predicted or "", truth)
                return idx, {
                    "task_id": f"task_{idx}",
                    "question": task,
                    "true_answer": truth,
                    "predicted_answer": predicted,
                    "raw_response": raw,
                    "is_solved": correct,
                    "prompt_tokens": int(PromptTokens.instance().value - pt_before),
                    "completion_tokens": int(CompletionTokens.instance().value - ct_before),
                }

            batch_coros = [_process(rec, base_idx + i) for i, rec in pending]
            batch_outputs = await asyncio.gather(*batch_coros, return_exceptions=True)

            for out in batch_outputs:
                if isinstance(out, Exception):
                    import traceback
                    print(f"  [batch error] {type(out).__name__}: {out}")
                    traceback.print_exc()
                    continue
                _, result = out
                results.append(result)
                if result["is_solved"]:
                    solved += 1
                # Flush to disk immediately — this is the checkpoint
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

            acc = solved / len(results) * 100 if results else 0
            pbar.set_postfix({"acc": f"{acc:.1f}%", "done": len(results)})
    finally:
        out_f.close()

    accuracy = solved / total * 100 if total else 0
    total_prompt = int(PromptTokens.instance().value)
    total_completion = int(CompletionTokens.instance().value)
    print(f"\nAccuracy: {accuracy:.2f}%  ({solved}/{total})")
    print(f"Prompt tokens: {total_prompt}  "
          f"Completion tokens: {total_completion}  "
          f"Total tokens: {total_prompt + total_completion}")
    print(f"Results saved to: {args.output_file}")

    # Append to summary log
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": args.dataset,
        "method": args.method,
        "llm_name": args.llm_name,
        "num_agents": n_agents,
        "total_tasks": total,
        "solved_tasks": solved,
        "accuracy": accuracy,
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "detail_file": args.output_file,
    }
    if args.summary_log_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_log_file)), exist_ok=True)
        with open(args.summary_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"Summary appended to: {args.summary_log_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate fixed-topology baselines for ARG-Designer comparison"
    )
    p.add_argument("--dataset", required=True,
                   choices=["gsm8k", "aqua", "multiarith", "svamp", "humaneval", "mmlu"],
                   help="Dataset to evaluate on")
    p.add_argument("--dataset_json", required=True,
                   help="Path to dataset file (JSONL or JSON)")
    p.add_argument("--method", required=True, choices=SUPPORTED_METHODS,
                   help="Baseline method to evaluate")
    p.add_argument("--llm_name", default="Qwen/Qwen3-8B",
                   help="HuggingFace model ID or Ollama model name")
    p.add_argument("--num_agents", type=int, default=None,
                   help="Number of agents in the graph (default: per-method default)")
    p.add_argument("--sc_samples", type=int, default=_SC_SAMPLES,
                   help="Number of CoT samples for self_consistency (default: 5)")
    p.add_argument("--eval_batch_size", type=int, default=4,
                   help="Tasks to run concurrently per batch. For vanilla/cot this "
                        "maps directly to the HF batcher batch size; for multi-agent "
                        "methods reduce if you hit OOM (default: 4)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of test samples (default: all)")
    p.add_argument("--output_file", default=None,
                   help="JSONL file to write per-task results")
    p.add_argument("--summary_log_file", default=None,
                   help="JSONL file to append summary metrics")
    p.add_argument("--task_split_path", default=None,
                   help="Path to task split JSON; if absent, auto-derived from dataset name")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def cli():
    args = parse_args()

    # Fill default output file
    if args.output_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_file = f"results/{args.dataset}_{args.method}_{ts}.jsonl"

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    cli()
