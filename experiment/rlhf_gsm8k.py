"""
RLHF pipeline entry point for GSM8K.

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
python experiment/rlhf_gsm8k.py --phase collect \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --llm_name gemma3 \
    --preference_dir rlhf_data/gsm8k \
    --num_tasks 100

# Step 2 – train reward model
python experiment/rlhf_gsm8k.py --phase train_rm \
    --preference_dir rlhf_data/gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth

# Step 3 – fine-tune policy (requires pretrained ARGDesigner checkpoint)
python experiment/rlhf_gsm8k.py --phase train_policy \
    --model_dir ColdStartData_gemma_gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth \
    --policy_checkpoint rlhf_checkpoints/gsm8k/policy_rlhf.pth \
    --dataset_json datasets/gsm8k/gsm8k.jsonl \
    --num_tasks 100
"""

import argparse
import asyncio
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gsm8k(dataset_json: str):
    from mas_framework.tools.reader.readers import JSONLReader
    from datasets.gsm8k_dataset import gsm_data_process
    raw = JSONLReader.parse_file(dataset_json)
    return gsm_data_process(raw)


def _answer_checker(predicted: str, ground_truth: str) -> bool:
    try:
        return float(predicted) == float(ground_truth)
    except (ValueError, TypeError):
        return False


def _get_predict(raw_output: str) -> str:
    from datasets.gsm8k_dataset import gsm_get_predict
    return gsm_get_predict(raw_output)


# ---------------------------------------------------------------------------
# Phase 1 — collect preference data
# ---------------------------------------------------------------------------

async def _collect(args):
    from mas_framework.rlhf.data_collector import RLHFDataCollector
    from mas_framework.rlhf.preference_data import PreferenceWeights
    from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
    import random

    dataset = _load_gsm8k(args.dataset_json)
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))
    print(f"Collecting RLHF data for {len(sample)} tasks with {args.llm_name} ...")

    weights = PreferenceWeights(
        correctness=args.w_correct,
        graph_size=args.w_size,
        token_cost=args.w_token,
    )

    collector = RLHFDataCollector(
        domain="gsm8k",
        llm_name=args.llm_name,
        answer_checker=_answer_checker,
        get_predict=_get_predict,
        role_descriptions=ROLE_DESCRIPTION,
        decision_method="FinalRefer",
        num_rounds=1,
        weights=weights,
        pair_margin=args.pair_margin,
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

    # Load pretrained ARGDesigner
    print(f"Loading ARGDesigner from {args.model_dir} ...")
    policy = load_model(args.model_dir)
    policy = policy.to(device)

    # Load trained reward model
    print(f"Loading reward model from {args.rm_checkpoint} ...")
    reward_model = load_reward_model(args.rm_checkpoint, device)

    # Build task records with pre-computed embeddings
    dataset = _load_gsm8k(args.dataset_json)
    random.seed(args.seed)
    sample = random.sample(dataset, min(args.num_tasks, len(dataset)))

    sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    task_records = []
    for rec in sample:
        task_records.append({
            "task": rec["task"],
            "answer": rec["answer"],
            "task_embedding": np.array(sent_model.encode(rec["task"]), dtype="float32"),
        })

    print(f"Fine-tuning policy on {len(task_records)} tasks ...")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="RLHF pipeline for GSM8K")
    p.add_argument("--phase", choices=["collect", "train_rm", "train_policy"],
                   required=True, help="Pipeline phase to run")

    # Shared
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset_json", default="datasets/gsm8k/gsm8k.jsonl")
    p.add_argument("--num_tasks", type=int, default=100)

    # Collect
    p.add_argument("--llm_name", default="gemma3",
                   help="Local Ollama model (e.g. gemma3, llama3.2)")
    p.add_argument("--preference_dir", default="rlhf_data/gsm8k")
    p.add_argument("--min_agents", type=int, default=2)
    p.add_argument("--max_agents", type=int, default=4)
    p.add_argument("--w_correct", type=float, default=0.6,
                   help="Correctness weight in preference score")
    p.add_argument("--w_size", type=float, default=0.2,
                   help="Graph size penalty weight")
    p.add_argument("--w_token", type=float, default=0.2,
                   help="Token cost penalty weight")
    p.add_argument("--pair_margin", type=float, default=0.05,
                   help="Minimum score gap to keep a preference pair")
    p.add_argument("--checkpoint_every", type=int, default=20)

    # Reward model
    p.add_argument("--rm_checkpoint", default="rlhf_checkpoints/gsm8k/reward_model.pth")
    p.add_argument("--rm_epochs", type=int, default=20)
    p.add_argument("--rm_lr", type=float, default=1e-4)
    p.add_argument("--rm_batch_size", type=int, default=32)
    p.add_argument("--rm_hidden_dim", type=int, default=256)
    p.add_argument("--rm_output_dim", type=int, default=128)
    p.add_argument("--rm_val_fraction", type=float, default=0.1)

    # Policy
    p.add_argument("--model_dir", default="",
                   help="Directory of pretrained ARGDesigner checkpoint")
    p.add_argument("--policy_checkpoint",
                   default="rlhf_checkpoints/gsm8k/policy_rlhf.pth")
    p.add_argument("--policy_epochs", type=int, default=10)
    p.add_argument("--policy_lr", type=float, default=1e-5)
    p.add_argument("--kl_coeff", type=float, default=0.1)
    p.add_argument("--samples_per_task", type=int, default=2)

    return p.parse_args()


def cli():
    """Entry point for `uv run rlhf` (defined in pyproject.toml)."""
    args = parse_args()

    if args.phase == "collect":
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(_collect(args))

    elif args.phase == "train_rm":
        _train_rm(args)

    elif args.phase == "train_policy":
        if not args.model_dir:
            raise ValueError("--model_dir is required for train_policy phase")
        _train_policy(args)


if __name__ == "__main__":
    cli()
