"""
RLHF pipeline entry point — supports all datasets.

Three phases (select with --phase):

  collect      — Run multiple graph configurations per task, score them, and
                 write preference-pair shards to --preference_dir.

  train_rm     — Train a GNN reward model on the collected preference pairs.
                 Saves best checkpoint to --rm_checkpoint.

  train_policy — Fine-tune a pretrained ARGDesigner using REINFORCE + KL penalty
                 guided by the trained reward model.
                 Saves policy checkpoint to --policy_checkpoint.

Example
-------
# Step 1 – collect data
python experiment/rlhf.py --dataset gsm8k --phase collect \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --llm_name Qwen/Qwen3-8B \
    --preference_dir rlhf_data/gsm8k \
    --num_tasks 100

# Step 2 – train reward model
python experiment/rlhf.py --dataset gsm8k --phase train_rm \
    --preference_dir rlhf_data/gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth

# Step 3 – fine-tune policy (requires pretrained ARGDesigner checkpoint)
python experiment/rlhf.py --dataset gsm8k --phase train_policy \
    --model_dir checkpoints/gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth \
    --policy_checkpoint rlhf_checkpoints/gsm8k/policy_rlhf.pth \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --num_tasks 100

Supported datasets: gsm8k, aqua, multiarith, svamp, humaneval, mmlu
"""

import argparse
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")

from experiment.cold_start import (
    _load_dataset,
    _load_task_split,
    _get_predict,
    _is_correct,
    _get_role_description,
    _get_decision_method,
    _get_agent_name,
)

SUPPORTED_DATASETS = ["gsm8k", "aqua", "multiarith", "svamp", "humaneval", "mmlu"]

# Per-dataset max agents (matches cold_start_all.sh)
_MAX_AGENTS = {
    "gsm8k": 4,
    "aqua": 4,
    "multiarith": 4,
    "svamp": 4,
    "humaneval": 5,
    "mmlu": 6,
}


def _answer_checker(dataset: str):
    """Return a dataset-aware answer checker compatible with RLHFDataCollector."""
    def checker(predicted: str, ground_truth: str) -> bool:
        return _is_correct(dataset, predicted, ground_truth)
    return checker


def _predict_fn(dataset: str):
    """Return a dataset-aware predict extractor."""
    def predict(raw_output: str) -> str:
        return _get_predict(dataset, raw_output)
    return predict


# ---------------------------------------------------------------------------
# Phase 1 — collect preference data
# ---------------------------------------------------------------------------

def _load_coldstart_pool(coldstart_dirs):
    """
    Load ColdStart / Finetune .pt files and return a pool keyed by task question.

    Pool format: {question_str: [{"nx_graph": nx.DiGraph, "is_correct": bool,
                                   "mode": str, "num_nodes": int}, ...]}

    Passed to collect_dataset() so coldstart graphs are injected into the same
    pairing pool as LLM-collected graphs, enabling cross-source preference pairs.
    """
    import glob
    from collections import defaultdict

    import networkx as nx
    import torch

    pool = defaultdict(list)
    skipped = 0
    for d in coldstart_dirs:
        found = glob.glob(os.path.join(d, "*.pt"))
        print(f"  {d}: {len(found)} .pt files")
        for path in found:
            try:
                try:
                    pyg = torch.load(path, weights_only=False)
                except TypeError:
                    pyg = torch.load(path)
            except Exception as e:
                print(f"  [skip] {os.path.basename(path)}: {e}")
                skipped += 1
                continue

            question   = getattr(pyg, "question",   "")
            is_correct = bool(getattr(pyg, "is_correct", False))
            mode       = getattr(pyg, "mode",       "Unknown")
            num_nodes  = int(pyg.num_nodes)

            nx_g = nx.DiGraph()
            for i in range(num_nodes):
                node_data = pyg.x[i] if (hasattr(pyg, "x") and pyg.x is not None) else {}
                role = node_data.get("role", "Unknown") if isinstance(node_data, dict) else "Unknown"
                nx_g.add_node(i, role=role)
            if hasattr(pyg, "edge_index") and pyg.edge_index.numel() > 0:
                for src, dst in pyg.edge_index.t().tolist():
                    nx_g.add_edge(int(src), int(dst))

            key = question if question else os.path.basename(path)
            pool[key].append({
                "nx_graph": nx_g, "is_correct": is_correct,
                "mode": mode, "num_nodes": num_nodes,
            })

    # ---- Summary stats -------------------------------------------------------
    total      = sum(len(v) for v in pool.values())
    n_correct  = sum(1 for graphs in pool.values() for g in graphs if     g["is_correct"])
    n_rejected = sum(1 for graphs in pool.values() for g in graphs if not g["is_correct"])

    # Tasks that have at least one correct AND one incorrect graph — these are
    # the only tasks that can form a preference pair from the cold-start pool.
    n_pairable = sum(
        1 for graphs in pool.values()
        if any(g["is_correct"] for g in graphs) and any(not g["is_correct"] for g in graphs)
    )

    # Topology breakdown
    from collections import Counter
    mode_counts = Counter(g["mode"] for graphs in pool.values() for g in graphs)

    # Node-count distribution
    node_counts = [g["num_nodes"] for graphs in pool.values() for g in graphs]
    avg_nodes   = sum(node_counts) / len(node_counts) if node_counts else 0

    print(f"\n  ── Cold-start pool statistics ──────────────────────")
    print(f"  Total graphs  : {total}  ({n_correct} correct, {n_rejected} rejected, {skipped} skipped)")
    print(f"  Tasks         : {len(pool)}  ({n_pairable} have both correct+rejected → pairable)")
    print(f"  Avg nodes/graph: {avg_nodes:.1f}")
    print(f"  By topology   :", "  ".join(f"{m}={c}" for m, c in sorted(mode_counts.items())))
    print(f"  ────────────────────────────────────────────────────\n")

    return dict(pool)


def _filter_by_difficulty(sample, coldstart_pool, min_fail_rate: float = 0.05):
    """Keep only tasks where at least *min_fail_rate* of cold-start runs were wrong.

    Tasks where every cold-start topology succeeded are "all-correct" tasks —
    they produce no correctness-differentiating preference pairs (every graph
    gets a correct answer regardless of structure). Dropping them focuses RLHF
    collection on tasks where topology choice actually affects correctness.
    """
    filtered, skipped = [], 0
    for rec in sample:
        graphs = coldstart_pool.get(rec["task"], [])
        if not graphs:
            filtered.append(rec)  # no cold-start data → keep, can't judge
            continue
        n_wrong = sum(1 for g in graphs if not g["is_correct"])
        if n_wrong / len(graphs) >= min_fail_rate:
            filtered.append(rec)
        else:
            skipped += 1
    print(f"  Difficulty filter (min_fail_rate={min_fail_rate}): "
          f"kept {len(filtered)}/{len(filtered)+skipped} tasks "
          f"({skipped} all-correct tasks dropped)")
    return filtered


# ---------------------------------------------------------------------------
# Phase 1a — generate ARGDesigner candidate graphs (no LLM, no vLLM)
# ---------------------------------------------------------------------------

def _gen_candidates(args):
    """Generate graph candidates for all tasks using ARGDesigner (GNN only, no LLM).

    Saves {preference_dir}/candidates.pkl — a list of per-task dicts:
      {"record": {...}, "candidates": [(nx.DiGraph, label), ...]}

    Called before vLLM is started so the GPU is free for ARGDesigner inference.
    """
    from mas_framework.rlhf.data_collector import RLHFDataCollector
    from mas_framework.rlhf.preference_data import PreferenceWeights
    import random

    os.makedirs(args.preference_dir, exist_ok=True)
    candidates_path = os.path.join(args.preference_dir, "candidates.pkl")

    all_records = _load_dataset(args.dataset, args.dataset_json)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    task_split = _load_task_split(args.dataset, project_root)
    train_indices = task_split['base_tasks_indices'] + task_split['finetune_tasks_indices']
    dataset = [all_records[i] for i in train_indices]
    print(f"Using base+finetune split: {len(dataset)}/{len(all_records)} records (test excluded)")

    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))
    num_sample_for_tasks = int(args.num_tasks / len(sample))
    print(f"Generating candidates for {len(sample)} tasks ({args.dataset}) ...")

    role_desc = _get_role_description(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    agent_name = _get_agent_name(args.dataset)

    # Optional difficulty filter (needs cold-start pool to compute fail rates)
    coldstart_pool = None
    if args.coldstart_dirs:
        print(f"\nLoading ColdStart pool from: {args.coldstart_dirs}")
        coldstart_pool = _load_coldstart_pool(args.coldstart_dirs)

    if args.difficulty_filter and coldstart_pool:
        sample = _filter_by_difficulty(sample, coldstart_pool, args.min_fail_rate)
        if not sample:
            print("WARNING: difficulty filter removed all tasks — disabling filter.")
            random.seed(args.seed)
            sample = random.sample(dataset, min(args.num_tasks, len(dataset)))

    arg_model = None
    if args.arg_model_dir:
        from experiment.utils import load_model
        print(f"Loading ARGDesigner model from {args.arg_model_dir} ...")
        arg_model = load_model(args.arg_model_dir, ef=True)
        arg_model.eval()

    weights = PreferenceWeights(
        correctness=args.w_correct,
        graph_size=args.w_size,
        edge_cost=args.w_edge,
        ref_max_nodes=args.max_agents,
    )

    collector = RLHFDataCollector(
        domain=args.dataset,
        llm_name=args.llm_name,          # stored but not called during gen_candidates
        answer_checker=_answer_checker(args.dataset),
        get_predict=_predict_fn(args.dataset),
        role_descriptions=role_desc,
        agent_name=agent_name,
        decision_method=decision_method,
        num_rounds=1,
        weights=weights,
        pair_margin=args.pair_margin,
        timeout=args.llm_timeout,
        arg_model=arg_model,
        sample_temperatures=args.sample_temperatures,
        arg_model_samples=args.arg_model_samples,
        pruning_ratio=0.0,               # pruning happens during score phase
        weak_baselines=args.weak_baselines,
        role_sweep=args.role_sweep,
        role_sweep_topology=args.role_sweep_topology,
        role_sweep_n_agents=args.role_sweep_n_agents,
        role_sweep_max_combos=args.role_sweep_max_combos,
    )

    collector.generate_all_candidates(
        task_records=sample,
        output_path=candidates_path,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        num_sample_for_tasks=num_sample_for_tasks,
    )
    print(f"\nGen-candidates phase complete. Saved → {candidates_path}")


# ---------------------------------------------------------------------------
# Phase 1b — LLM scoring of pre-generated candidates
# ---------------------------------------------------------------------------

async def _collect_llm(args):
    """Run LLM inference on pre-generated candidates and write preference-pair shards.

    Requires {preference_dir}/candidates.pkl written by _gen_candidates().
    Called after vLLM has been started so LLM inference is available.
    """
    from mas_framework.rlhf.data_collector import RLHFDataCollector
    from mas_framework.rlhf.preference_data import PreferenceWeights

    candidates_path = os.path.join(args.preference_dir, "candidates.pkl")
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"candidates.pkl not found: {candidates_path}\n"
            "Run --phase gen_candidates first."
        )

    role_desc = _get_role_description(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    agent_name = _get_agent_name(args.dataset)

    weights = PreferenceWeights(
        correctness=args.w_correct,
        graph_size=args.w_size,
        edge_cost=args.w_edge,
        ref_max_nodes=args.max_agents,
    )

    # Load cold-start pool for free preference pairs (pre-scored, no LLM calls)
    coldstart_pool = None
    if args.coldstart_dirs:
        print(f"\nLoading ColdStart pool from: {args.coldstart_dirs}")
        coldstart_pool = _load_coldstart_pool(args.coldstart_dirs)

    collector = RLHFDataCollector(
        domain=args.dataset,
        llm_name=args.llm_name,
        answer_checker=_answer_checker(args.dataset),
        get_predict=_predict_fn(args.dataset),
        role_descriptions=role_desc,
        agent_name=agent_name,
        decision_method=decision_method,
        num_rounds=1,
        weights=weights,
        pair_margin=args.pair_margin,
        timeout=args.llm_timeout,
        arg_model=None,                  # not needed for scoring
        pruning_ratio=args.pruning_ratio,
        inference_concurrency=args.inference_concurrency,
    )

    total = await collector.collect_from_candidates(
        candidates_path=candidates_path,
        output_dir=args.preference_dir,
        checkpoint_every=args.checkpoint_every,
        coldstart_pool=coldstart_pool,
        task_concurrency=args.task_concurrency,
    )
    print(f"\nCollect-LLM phase complete. Total preference pairs: {total}")


# ---------------------------------------------------------------------------
# Phase 1 (legacy) — collect preference data (combined gen + LLM in one call)
# ---------------------------------------------------------------------------

async def _collect(args):
    from mas_framework.rlhf.data_collector import RLHFDataCollector
    from mas_framework.rlhf.preference_data import PreferenceWeights
    import random

    all_records = _load_dataset(args.dataset, args.dataset_json)

    # Use only base + finetune splits — never touch test data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    task_split = _load_task_split(args.dataset, project_root)
    train_indices = task_split['base_tasks_indices'] + task_split['finetune_tasks_indices']
    dataset = [all_records[i] for i in train_indices]
    print(f"Using base+finetune split: {len(dataset)}/{len(all_records)} records (test excluded)")

    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))
    # Number of sample generated from ARG-Designer model for each temp.
    num_sample_for_tasks = int(args.num_tasks / len(sample))
    print(f"Collecting RLHF data for {len(sample)} tasks ({args.dataset}) with {args.llm_name} ...")

    role_desc = _get_role_description(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    agent_name = _get_agent_name(args.dataset)

    weights = PreferenceWeights(
        correctness=args.w_correct,
        graph_size=args.w_size,
        edge_cost=args.w_edge,
        ref_max_nodes=args.max_agents,  # match dataset-specific max agents
    )

    arg_model = None
    if args.arg_model_dir:
        from experiment.utils import load_model
        print(f"Loading ARGDesigner model from {args.arg_model_dir} ...")
        arg_model = load_model(args.arg_model_dir, ef=True)
        arg_model.eval()

    # Load cold-start pool early so difficulty filter can use it
    coldstart_pool = None
    if args.coldstart_dirs:
        print(f"\nLoading ColdStart pool from: {args.coldstart_dirs}")
        coldstart_pool = _load_coldstart_pool(args.coldstart_dirs)

    # Difficulty pre-screening: drop tasks where all cold-start topologies succeeded.
    # Focuses RLHF on tasks where topology choice affects correctness.
    if args.difficulty_filter and coldstart_pool:
        sample = _filter_by_difficulty(sample, coldstart_pool, args.min_fail_rate)
        if not sample:
            print("WARNING: difficulty filter removed all tasks — disabling filter.")
            sample = random.sample(dataset, min(args.num_tasks, len(dataset)))

    collector = RLHFDataCollector(
        domain=args.dataset,
        llm_name=args.llm_name,
        answer_checker=_answer_checker(args.dataset),
        get_predict=_predict_fn(args.dataset),
        role_descriptions=role_desc,
        agent_name=agent_name,
        decision_method=decision_method,
        num_rounds=1,
        weights=weights,
        pair_margin=args.pair_margin,
        timeout=args.llm_timeout,
        arg_model=arg_model,
        sample_temperatures=args.sample_temperatures,
        arg_model_samples=args.arg_model_samples,
        pruning_ratio=args.pruning_ratio,
        weak_baselines=args.weak_baselines,
        role_sweep=args.role_sweep,
        role_sweep_topology=args.role_sweep_topology,
        role_sweep_n_agents=args.role_sweep_n_agents,
        role_sweep_max_combos=args.role_sweep_max_combos,
        inference_concurrency=args.inference_concurrency,
    )

    total = await collector.collect_dataset(
        task_records=sample,
        output_dir=args.preference_dir,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        checkpoint_every=args.checkpoint_every,
        coldstart_pool=coldstart_pool,
        num_sample_for_tasks=num_sample_for_tasks,
        task_concurrency=args.task_concurrency,
    )
    print(f"\nCollect phase complete. Total preference pairs: {total}")


# ---------------------------------------------------------------------------
# Phase 2 — train reward model
# ---------------------------------------------------------------------------

def _train_rm(args):
    from mas_framework.rlhf.reward_model import GraphRewardModel
    from mas_framework.rlhf.reward_trainer import train_reward_model

    device = torch.device(args.device)
    model = GraphRewardModel(
        node_feat_dim=773,
        hidden_dim=args.rm_hidden_dim,
        output_dim=args.rm_output_dim,
    )

    # --preference_dirs (multi-dataset global RM) takes precedence over
    # --preference_dir (single-dataset per-dataset RM).
    if getattr(args, "preference_dirs", None):
        data_src = args.preference_dirs
        print(f"Training global reward model on {len(data_src)} preference dirs ...")
        for d in data_src:
            print(f"  {d}")
    else:
        data_src = args.preference_dir
        print(f"Training reward model on data from {data_src} ...")

    train_reward_model(
        model=model,
        data_dir=data_src,
        device=device,
        epochs=args.rm_epochs,
        lr=args.rm_lr,
        batch_size=args.rm_batch_size,
        val_fraction=args.rm_val_fraction,
        save_path=args.rm_checkpoint,
        both_wrong_weight=args.both_wrong_weight,
        both_correct_weight=args.both_correct_weight,
        loss_type=args.rm_loss,
    )


# ---------------------------------------------------------------------------
# Phase 3 — fine-tune policy
# ---------------------------------------------------------------------------

def _train_policy(args):
    from mas_framework.rlhf.reward_trainer import load_reward_model
    from mas_framework.rlhf.policy_trainer import RLHFPolicyTrainer
    from experiment.utils import load_model
    import random
    import numpy as np
    from mas_framework.llm.profile_embedding import get_sentence_model

    device = torch.device(args.device)

    print(f"Loading ARGDesigner from {args.model_dir} ...")
    policy = load_model(args.model_dir)
    policy = policy.to(device)

    print(f"Loading reward model from {args.rm_checkpoint} ...")
    reward_model = load_reward_model(args.rm_checkpoint, device)

    all_records = _load_dataset(args.dataset, args.dataset_json)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    task_split = _load_task_split(args.dataset, project_root)
    train_indices = task_split['base_tasks_indices'] + task_split['finetune_tasks_indices']
    dataset = [all_records[i] for i in train_indices]
    # dataset = all_records

    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))

    sent_model = get_sentence_model()
    task_records = []
    for rec in sample:
        task_records.append({
            "task": rec["task"],
            "answer": rec["answer"],
            "task_embedding": np.array(sent_model.encode(rec["task"]), dtype="float32"),
        })

    print(f"Fine-tuning policy on {len(task_records)} tasks ({args.dataset}) ...")
    trainer = RLHFPolicyTrainer(
        policy=policy,
        reward_model=reward_model,
        device=device,
        lr=args.policy_lr,
        kl_coeff=args.kl_coeff,
        lambda_eff=args.lambda_eff,
        ref_max_nodes=args.max_agents,
    )
    trainer.train(
        task_records=task_records,
        epochs=args.policy_epochs,
        samples_per_task=args.samples_per_task,
        save_path=args.policy_checkpoint,
        grad_accum_steps=args.grad_accum_steps,
    )

    # Save benchmark-compatible checkpoint (ef_best_model.pth) so that
    # benchmark.sh / load_model() can evaluate the RLHF policy directly.
    # load_model() requires keys: args, data_statistics, model_state_dict.
    policy_dir = os.path.dirname(args.policy_checkpoint) or "."
    orig_ckpt_file = os.path.join(args.model_dir, "ef_best_model.pth")
    orig_ckpt = torch.load(orig_ckpt_file, map_location=device, weights_only=False)
    bench_ckpt_path = os.path.join(policy_dir, "ef_best_model.pth")
    torch.save(
        {
            "args": orig_ckpt["args"],
            "data_statistics": orig_ckpt["data_statistics"],
            "model_state_dict": policy.state_dict(),
        },
        bench_ckpt_path,
    )
    print(f"Saved benchmark-compatible checkpoint → {bench_ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RLHF pipeline for ARG-Designer (all datasets)")
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                   help="Dataset to run RLHF on")
    p.add_argument("--phase",
                   choices=["collect", "gen_candidates", "collect_llm",
                            "train_rm", "train_policy"],
                   required=True,
                   help=(
                       "collect        — legacy: generate + score in one call (vLLM must run). "
                       "gen_candidates — Phase 1a: generate ARGDesigner graphs (no vLLM, GPU free). "
                       "collect_llm    — Phase 1b: LLM scoring of saved candidates (vLLM required). "
                       "train_rm       — train GNN reward model. "
                       "train_policy   — REINFORCE+KL fine-tuning of ARGDesigner policy."
                   ))

    # Shared
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_json", default=None,
                   help="Path to dataset file (required for collect and train_policy)")
    p.add_argument("--num_tasks", type=int, default=100)

    # Collect
    p.add_argument("--llm_name", default="Qwen/Qwen3-8B",
                   help="HuggingFace model ID or Ollama name")
    p.add_argument("--preference_dir", default=None,
                   help="Directory for preference pair shards (default: rlhf_data/<dataset>)")
    p.add_argument("--min_agents", type=int, default=2)
    p.add_argument("--max_agents", type=int, default=None,
                   help="Max agents per graph (default: per-dataset value)")
    p.add_argument("--w_correct", type=float, default=0.6,
                   help="Correctness weight in preference score")
    p.add_argument("--w_size", type=float, default=0.2,
                   help="Graph size penalty weight")
    p.add_argument("--w_edge", type=float, default=0.2,
                   help="Edge density penalty weight")
    p.add_argument("--pair_margin", type=float, default=0.05,
                   help="Minimum score gap to keep a preference pair")
    p.add_argument("--pruning_ratio", type=float, default=0.25,
                   help="Fraction of edges to remove from correct graphs during collect; "
                        "pruned variants are added to the preference pool (default: 0.25, "
                        "set to 0 to disable)")
    p.add_argument("--checkpoint_every", type=int, default=20)
    p.add_argument("--task_concurrency", type=int, default=4,
                   help="Number of tasks whose LLM inference runs concurrently during collect")
    p.add_argument("--inference_concurrency", type=int, default=8,
                   help="Max concurrent vLLM requests across all tasks (default: 8)")
    p.add_argument("--llm_timeout", type=int, default=600,
                   help="Seconds to wait for a single LLM graph run (default: 600)")
    p.add_argument("--arg_model_dir", default=None,
                   help="Optional: pretrained ARGDesigner checkpoint dir for model-based "
                        "candidate generation during collect phase")
    p.add_argument("--sample_temperatures", type=float, nargs="+",
                   default=None,
                   help="Temperatures for ARGDesigner candidate sampling "
                        "(default: 1.0 1.5 2.0). Higher T → more structural diversity.")
    p.add_argument("--arg_model_samples", type=int, default=None,
                   help="Number of unique graphs to generate per task via ARGDesigner. "
                        "Default: len(temperatures) when _default_configs also runs, "
                        "len(temperatures)*3 when it is skipped.")
    p.add_argument("--coldstart_dirs", nargs="+", default=None,
                   help="Optional: one or more ColdStart/Finetune .pt directories whose "
                        "graphs are converted to preference pairs without re-running LLM "
                        "inference (is_correct is already recorded in each .pt file).")
    p.add_argument("--difficulty_filter", action="store_true", default=False,
                   help="Drop tasks where all cold-start runs were correct before collecting. "
                        "Focuses preference data on tasks where topology choice matters "
                        "(requires --coldstart_dirs). Recommended for high-ceiling datasets "
                        "like multiarith and svamp.")
    p.add_argument("--min_fail_rate", type=float, default=0.05,
                   help="Minimum fraction of cold-start runs that must have been wrong "
                        "for a task to pass the difficulty filter (default: 0.05).")
    p.add_argument("--weak_baselines", action="store_true", default=False,
                   help="Add a 1-agent and an over-sized (max_agents+1) Chain config to "
                        "the default topology grid. Forces some incorrect runs even for "
                        "near-ceiling datasets, creating correctness-differentiating pairs.")
    p.add_argument("--role_sweep", action="store_true", default=False,
                   help="Enable role-assignment sweep: fix topology and exhaustively run "
                        "all ordered role combinations per task. Produces dense role-quality "
                        "signal for the reward model, which is more informative than topology "
                        "variation on near-ceiling datasets (GSM8K, SVAMP).")
    p.add_argument("--role_sweep_topology", default="Chain",
                   choices=["Chain", "Star", "FullConnected"],
                   help="Fixed topology used for role sweep configs (default: Chain).")
    p.add_argument("--role_sweep_n_agents", type=lambda s: [int(x) for x in s.split(",")],
                   default=[2],
                   metavar="N[,N...]",
                   help="Comma-separated agent counts for role sweep (default: 2). "
                        "E.g. '2,3' sweeps 2-node and 3-node chains.")
    p.add_argument("--role_sweep_max_combos", type=int, default=None,
                   help="Cap the number of role combinations per agent count to avoid "
                        "combinatorial explosion (default: no cap — 4 roles × n=2 = 16 configs).")

    # Reward model
    p.add_argument("--preference_dirs", nargs="+", default=None,
                   help="Multiple preference dirs to pool for global reward model training "
                        "(overrides --preference_dir for train_rm phase)")
    p.add_argument("--rm_checkpoint", default=None,
                   help="Reward model checkpoint path (default: rlhf_checkpoints/<dataset>/reward_model.pth)")
    p.add_argument("--rm_epochs", type=int, default=20)
    p.add_argument("--rm_lr", type=float, default=1e-4)
    p.add_argument("--rm_batch_size", type=int, default=32)
    p.add_argument("--rm_hidden_dim", type=int, default=256)
    p.add_argument("--rm_output_dim", type=int, default=128)
    p.add_argument("--rm_val_fraction", type=float, default=0.1)
    p.add_argument("--both_wrong_weight", type=float, default=0.2,
                   help="Loss weight for pairs where both candidates are incorrect "
                        "(default: 0.2). Set to 0 to remove them entirely; "
                        "set to 1.0 to disable down-weighting.")
    p.add_argument("--both_correct_weight", type=float, default=1.0,
                   help="Loss weight for pairs where both candidates are correct "
                        "(default: 1.0, full weight). These pairs carry only a "
                        "graph-efficiency signal (size/edge preference) with no "
                        "correctness information. Lower values (e.g. 0.1) reduce "
                        "their influence so the reward model focuses on correctness "
                        "differences. Set to 0 to remove them entirely.")
    p.add_argument("--rm_loss", default="bradley_terry",
                   choices=["bradley_terry", "bce"],
                   help="Reward model training loss. "
                        "'bradley_terry' (default): pairwise ranking loss "
                        "L = -log σ(r_chosen - r_rejected). "
                        "'bce': per-graph binary correctness loss "
                        "L = BCE(r_chosen, chosen_is_correct) + BCE(r_rejected, rejected_is_correct). "
                        "Use 'bce' when graph size has been removed from the preference score.")

    # Policy
    p.add_argument("--model_dir", default="",
                   help="Directory of pretrained ARGDesigner checkpoint")
    p.add_argument("--policy_checkpoint", default=None,
                   help="Policy checkpoint save path (default: rlhf_checkpoints/<dataset>/policy_rlhf.pth)")
    p.add_argument("--policy_epochs", type=int, default=10)
    p.add_argument("--policy_lr", type=float, default=1e-5)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--lambda_eff", type=float, default=0.0,
                   help="Weight for the efficiency bonus added directly to the RM reward "
                        "during GRPO (default: 0 = disabled). "
                        "Bonus = lambda_eff * mean(node_bonus, edge_bonus) where "
                        "node_bonus = 1 - num_nodes/ref_max_nodes and "
                        "edge_bonus = 1 - num_edges/ref_max_edges. "
                        "Recommended range: 0.1–0.3.")
    p.add_argument("--samples_per_task", type=int, default=4,
                   help="Graphs sampled per task per GRPO step; ≥4 recommended so "
                        "within-group advantage magnitude encodes reward differences")
    p.add_argument("--grad_accum_steps", type=int, default=8,
                   help="Accumulate gradients over this many tasks before each "
                        "optimizer.step() — reduces per-step variance (default: 8)")

    args = p.parse_args()

    # Fill in dataset-derived defaults
    if args.preference_dir is None:
        args.preference_dir = f"rlhf_data/{args.dataset}"
    if args.rm_checkpoint is None:
        args.rm_checkpoint = f"rlhf_checkpoints/{args.dataset}/reward_model.pth"
    if args.policy_checkpoint is None:
        args.policy_checkpoint = f"rlhf_checkpoints/{args.dataset}/policy_rlhf.pth"
    if args.max_agents is None:
        args.max_agents = _MAX_AGENTS.get(args.dataset, 4)

    return args


def cli():
    """Entry point for `uv run rlhf` (defined in pyproject.toml)."""
    args = parse_args()

    if args.phase == "collect":
        if args.dataset_json is None:
            raise ValueError("--dataset_json is required for the collect phase")
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_collect(args))
        os._exit(0)

    elif args.phase == "gen_candidates":
        # Synchronous — no vLLM needed; ARGDesigner GNN can use GPU freely.
        if args.dataset_json is None:
            raise ValueError("--dataset_json is required for the gen_candidates phase")
        _gen_candidates(args)

    elif args.phase == "collect_llm":
        # Async — vLLM must be running before calling this phase.
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_collect_llm(args))
        os._exit(0)

    elif args.phase == "train_rm":
        _train_rm(args)

    elif args.phase == "train_policy":
        if not args.model_dir:
            raise ValueError("--model_dir is required for train_policy phase")
        if args.dataset_json is None:
            raise ValueError("--dataset_json is required for the train_policy phase")
        _train_policy(args)


if __name__ == "__main__":
    cli()
