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
python experiment/rlhf_gsm8k.py --dataset gsm8k --phase collect \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --llm_name Qwen/Qwen3-8B \
    --preference_dir rlhf_data/gsm8k \
    --num_tasks 100

# Step 2 – train reward model
python experiment/rlhf_gsm8k.py --dataset gsm8k --phase train_rm \
    --preference_dir rlhf_data/gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth

# Step 3 – fine-tune policy (requires pretrained ARGDesigner checkpoint)
python experiment/rlhf_gsm8k.py --dataset gsm8k --phase train_policy \
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

from experiment.cold_start_gemma import (
    _load_dataset,
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

async def _collect(args):
    from mas_framework.rlhf.data_collector import RLHFDataCollector
    from mas_framework.rlhf.preference_data import PreferenceWeights
    import random

    dataset = _load_dataset(args.dataset, args.dataset_json)
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))
    print(f"Collecting RLHF data for {len(sample)} tasks ({args.dataset}) with {args.llm_name} ...")

    role_desc = _get_role_description(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    agent_name = _get_agent_name(args.dataset)

    weights = PreferenceWeights(
        correctness=args.w_correct,
        graph_size=args.w_size,
        token_cost=args.w_token,
    )

    arg_model = None
    if args.arg_model_dir:
        from experiment.utils import load_model
        print(f"Loading ARGDesigner model from {args.arg_model_dir} ...")
        arg_model = load_model(args.arg_model_dir, ef=True)
        arg_model.eval()

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
    )

    total = await collector.collect_dataset(
        task_records=sample,
        output_dir=args.preference_dir,
        min_agents=args.min_agents,
        max_agents=args.max_agents,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Done. {total} preference pairs saved to {args.preference_dir}/")


# ---------------------------------------------------------------------------
# Phase 2 — train reward model
# ---------------------------------------------------------------------------

def _train_rm(args):
    from mas_framework.rlhf.reward_model import GraphRewardModel
    from mas_framework.rlhf.reward_trainer import train_reward_model

    device = torch.device(args.device)
    model = GraphRewardModel(
        node_feat_dim=768,
        hidden_dim=args.rm_hidden_dim,
        output_dim=args.rm_output_dim,
    )

    print(f"Training reward model on data from {args.preference_dir} ...")
    train_reward_model(
        model=model,
        data_dir=args.preference_dir,
        device=device,
        epochs=args.rm_epochs,
        lr=args.rm_lr,
        batch_size=args.rm_batch_size,
        val_fraction=args.rm_val_fraction,
        save_path=args.rm_checkpoint,
        both_wrong_weight=args.both_wrong_weight,
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
    from sentence_transformers import SentenceTransformer

    device = torch.device(args.device)

    print(f"Loading ARGDesigner from {args.model_dir} ...")
    policy = load_model(args.model_dir)
    policy = policy.to(device)

    print(f"Loading reward model from {args.rm_checkpoint} ...")
    reward_model = load_reward_model(args.rm_checkpoint, device)

    dataset = _load_dataset(args.dataset, args.dataset_json)
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))

    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
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
    )
    trainer.train(
        task_records=task_records,
        epochs=args.policy_epochs,
        samples_per_task=args.samples_per_task,
        save_path=args.policy_checkpoint,
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
    p.add_argument("--phase", choices=["collect", "train_rm", "train_policy"],
                   required=True, help="Pipeline phase to run")

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
    p.add_argument("--w_token", type=float, default=0.2,
                   help="Token cost penalty weight")
    p.add_argument("--pair_margin", type=float, default=0.05,
                   help="Minimum score gap to keep a preference pair")
    p.add_argument("--checkpoint_every", type=int, default=20)
    p.add_argument("--llm_timeout", type=int, default=600,
                   help="Seconds to wait for a single LLM graph run (default: 600)")
    p.add_argument("--arg_model_dir", default=None,
                   help="Optional: pretrained ARGDesigner checkpoint dir for model-based "
                        "candidate generation during collect phase")
    p.add_argument("--sample_temperatures", type=float, nargs="+",
                   default=None,
                   help="Temperatures for ARGDesigner candidate sampling "
                        "(default: 1.0 1.5 2.0). Higher T → more structural diversity.")

    # Reward model
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

    # Policy
    p.add_argument("--model_dir", default="",
                   help="Directory of pretrained ARGDesigner checkpoint")
    p.add_argument("--policy_checkpoint", default=None,
                   help="Policy checkpoint save path (default: rlhf_checkpoints/<dataset>/policy_rlhf.pth)")
    p.add_argument("--policy_epochs", type=int, default=10)
    p.add_argument("--policy_lr", type=float, default=1e-5)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--samples_per_task", type=int, default=2)

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
