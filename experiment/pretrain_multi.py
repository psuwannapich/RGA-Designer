"""
Multi-dataset ARGDesigner pretraining / fine-tuning for LOO/LGO experiments.

Pools existing cold-start or D_eff artifacts from multiple source datasets and
trains (or fine-tunes) a single ARGDesigner without any LLM re-inference.

Two usage modes, driven by --phase:

  pretrain  (Stage 2a equivalent)
    Reads *.pt files from each source dataset's ColdStartData directory,
    pools them, and trains a fresh ARGDesigner → best_model.pth.

  finetune  (Stage 2b/2c equivalent)
    Reads *.pt files from each source dataset's FinetuneData directory,
    pools them, and fine-tunes from an existing checkpoint → ef_best_model.pth.

Usage — LOO pretrain for gsm8k (trained on everything except gsm8k):
    uv run python experiment/pretrain_multi.py \\
        --phase          pretrain \\
        --src_datasets   aqua multiarith svamp humaneval mmlu \\
        --src_data_dirs  path/ColdStartData/aqua ... \\
        --target_dataset gsm8k \\
        --output_dir     checkpoints_loo/gsm8k \\
        --epochs         30

Usage — LOO finetune for gsm8k (continues from best_model.pth above):
    uv run python experiment/pretrain_multi.py \\
        --phase          finetune \\
        --src_datasets   aqua multiarith svamp humaneval mmlu \\
        --src_data_dirs  path/checkpoints/aqua/FinetuneData_aqua ... \\
        --target_dataset gsm8k \\
        --init_checkpoint checkpoints_loo/gsm8k/best_model.pth \\
        --output_dir     checkpoints_loo/gsm8k \\
        --epochs         30

The saved checkpoint format is identical to pretrain.py / finetune.py
output, so generate_graphs.py and all downstream scripts work unmodified.
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from experiment.args import Args
from experiment.model import ARGDesigner
from experiment import process_dataset as gdata
from experiment.train_ARGDesigner import train


SUPPORTED_DATASETS = ["gsm8k", "aqua", "multiarith", "svamp", "humaneval", "mmlu"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-dataset ARGDesigner pretraining / fine-tuning (LOO/LGO)"
    )
    p.add_argument("--phase", required=True, choices=["pretrain", "finetune"],
                   help="pretrain: train from scratch on cold-start data. "
                        "finetune: continue from --init_checkpoint on D_eff data.")
    p.add_argument("--src_datasets", nargs="+", required=True,
                   choices=SUPPORTED_DATASETS,
                   help="Source dataset names (in the same order as --src_data_dirs)")
    p.add_argument("--src_data_dirs", nargs="+", required=True,
                   help="Data directory for each source dataset. "
                        "For pretrain: ColdStartData/{ds}. "
                        "For finetune: checkpoints/{ds}/FinetuneData_{ds}.")
    p.add_argument("--target_dataset", required=True, choices=SUPPORTED_DATASETS,
                   help="Target dataset (excluded from training). Stored in checkpoint "
                        "so generate_graphs.py knows which dataset this model is for.")
    p.add_argument("--output_dir", required=True,
                   help="Directory where checkpoints are written.")
    p.add_argument("--init_checkpoint", default=None,
                   help="(finetune only) Path to best_model.pth to initialise from.")
    p.add_argument("--model_name", default=None,
                   help="Output checkpoint filename. "
                        "Defaults to 'best_model.pth' for pretrain, "
                        "'ef_best_model.pth' for finetune.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate. Defaults: 1e-4 (pretrain), 5e-5 (finetune).")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample_size", type=int, default=0,
                   help="Cap graphs per source dataset (0 = use all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_environment(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graphs_for_dataset(dataset: str, data_dir: str,
                             sample_size: int, pretrain: bool) -> tuple:
    """Load graph list and role mapping for one source dataset via its adapter."""
    pre = Args().update_args()
    pre.dataset = dataset
    pre.data_dir = data_dir
    pre.data_dir_ef = data_dir   # for finetune (pretrain=False) the adapter reads from here
    pre.dataset_sample_size = sample_size
    ds, _ = gdata.load_graph_dataset(pre, pretrain=pretrain)
    role_to_id = getattr(ds, "role_to_id", {})
    id_to_role  = getattr(ds, "id_to_role",  {})
    return ds.graph_list, role_to_id, id_to_role


def build_unified_role_mapping(*role_to_id_dicts):
    """Sorted union of all per-dataset role vocabularies."""
    all_roles: set = set()
    for r2id in role_to_id_dicts:
        all_roles.update(r2id.keys())
    sorted_roles = sorted(all_roles)
    role_to_id = {r: i for i, r in enumerate(sorted_roles)}
    id_to_role  = {i: r for i, r in enumerate(sorted_roles)}
    return role_to_id, id_to_role


def build_unified_role_embeddings(datasets: list, save_path: str) -> dict:
    """Merge ROLE_DESCRIPTION from every source dataset, embed all roles, save pkl."""
    from mas_framework.llm.profile_embedding import get_sentence_model

    all_descriptions: dict = {}
    for ds in datasets:
        if ds == "mmlu":
            from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
        elif ds == "humaneval":
            from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
        elif ds in ("svamp", "multiarith", "gsm8k"):
            from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
        elif ds == "aqua":
            from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
        else:
            raise ValueError(f"Unknown dataset for role lookup: {ds}")
        all_descriptions.update(ROLE_DESCRIPTION)

    sent_model = get_sentence_model()
    role_embeddings = {}
    for role, description in all_descriptions.items():
        vec = sent_model.encode(f"{role}: {description.strip()}")
        role_embeddings[role] = torch.tensor(vec)   # shape [384]

    with open(save_path, "wb") as f:
        pickle.dump(role_embeddings, f)
    print(f"Built unified role embeddings: {len(role_embeddings)} roles → {save_path}")
    return role_embeddings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cli = parse_args()

    if len(cli.src_datasets) != len(cli.src_data_dirs):
        raise ValueError(
            f"--src_datasets ({len(cli.src_datasets)}) and "
            f"--src_data_dirs ({len(cli.src_data_dirs)}) must have the same length"
        )

    is_pretrain = cli.phase == "pretrain"
    lr           = cli.lr or (1e-4 if is_pretrain else 5e-5)
    model_name   = cli.model_name or ("best_model.pth" if is_pretrain else "ef_best_model.pth")

    if not is_pretrain and cli.init_checkpoint is None:
        raise ValueError("--init_checkpoint is required for --phase finetune")

    setup_environment(cli.seed)
    os.makedirs(cli.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"  Multi-dataset ARGDesigner {'pretrain' if is_pretrain else 'finetune'} (LOO/LGO)")
    print(f"  Source datasets : {cli.src_datasets}")
    print(f"  Target dataset  : {cli.target_dataset}")
    print(f"  Output dir      : {cli.output_dir}")
    print(f"  Output name     : {model_name}")
    print(f"  Epochs          : {cli.epochs}  lr={lr}")
    if not is_pretrain:
        print(f"  Init checkpoint : {cli.init_checkpoint}")
    print("=" * 60)

    # ---- Load graphs from each source dataset --------------------------------
    all_graphs = []
    role_dicts = []

    for ds_name, ds_dir in zip(cli.src_datasets, cli.src_data_dirs):
        print(f"\nLoading '{ds_name}' from {ds_dir} ...")
        graphs, r2id, _ = load_graphs_for_dataset(
            ds_name, ds_dir, cli.sample_size, pretrain=is_pretrain
        )
        print(f"  {len(graphs)} graphs  |  roles: {list(r2id.keys())}")
        all_graphs.extend(graphs)
        role_dicts.append(r2id)

    if not all_graphs:
        raise RuntimeError(
            "No graphs loaded. Ensure pipeline_arg.sh has completed for all source datasets "
            f"and the data directories exist: {cli.src_data_dirs}"
        )

    # ---- Build unified role vocabulary ---------------------------------------
    unified_r2id, unified_id2r = build_unified_role_mapping(*role_dicts)
    print(f"\nUnified role vocabulary ({len(unified_r2id)} roles): {list(unified_r2id.keys())}")

    # Build a merged precomputed_role_embeddings.pkl covering every role in the
    # unified vocab so ARGDesigner.__init__ finds proper embeddings for all roles.
    unified_emb_path = os.path.join(cli.output_dir, "precomputed_role_embeddings.pkl")
    build_unified_role_embeddings(cli.src_datasets, unified_emb_path)

    # ---- Build Args ----------------------------------------------------------
    pre = Args().update_args()
    pre.dataset         = cli.target_dataset   # stored in checkpoint for downstream scripts
    pre.data_dir        = cli.output_dir       # ARGDesigner loads pkl from here
    pre.experiment_path = cli.output_dir
    pre.epochs          = cli.epochs
    pre.lr              = lr
    pre.batch_size      = cli.batch_size
    pre.seed            = cli.seed
    pre.device          = cli.device
    pre.save_model      = True
    pre.model_name      = model_name
    pre.dataset_sample_size = cli.sample_size
    pre.max_prev_node   = 3
    pre.max_head_and_tail = None

    # Unified role vocabulary — GraphListDataset._map_node_labels uses this
    # to re-assign 'label' fields on every node consistently.
    pre.role_mapping    = unified_r2id
    pre.id_to_role      = unified_id2r
    pre.START_TOKEN_ID  = len(unified_r2id)
    pre.END_TOKEN_ID    = len(unified_r2id) + 1

    # ---- Train / val split ---------------------------------------------------
    correct   = [g for g in all_graphs if g.graph.get("is_correct")]
    incorrect = [g for g in all_graphs if not g.graph.get("is_correct")]
    random.shuffle(correct)
    random.shuffle(incorrect)

    ratio = 1.0 - cli.val_ratio
    train_graphs = (correct[: int(len(correct) * ratio)]
                    + incorrect[: int(len(incorrect) * ratio)])
    val_graphs   = (correct[int(len(correct) * ratio):]
                    + incorrect[int(len(incorrect) * ratio):])
    random.shuffle(train_graphs)

    print(f"\nTotal graphs : {len(all_graphs)}")
    print(f"  correct    : {len(correct)}")
    print(f"  incorrect  : {len(incorrect)}")
    print(f"  train      : {len(train_graphs)}")
    print(f"  val        : {len(val_graphs)}")

    # GraphListDataset._map_node_labels remaps 'label' using pre.role_mapping
    train_ds = gdata.GraphListDataset(train_graphs, pre)
    val_ds   = gdata.GraphListDataset(val_graphs,   pre)

    # ---- Data statistics -----------------------------------------------------
    stats = gdata.get_data_statistics(all_graphs)
    stats["num_node_labels"] = len(unified_r2id) + 2   # +2 for START / END
    stats["num_edge_labels"] = 1
    print(f"\nData statistics : {stats}")

    # ---- Build model ---------------------------------------------------------
    if is_pretrain:
        model = ARGDesigner(pre, stats).to(cli.device)
        print(f"\nBuilt fresh ARGDesigner | device={cli.device}")
    else:
        # Load the pretrained checkpoint and reconstruct the model from it
        print(f"\nLoading pretrained checkpoint: {cli.init_checkpoint}")
        ckpt = torch.load(cli.init_checkpoint,
                          map_location=cli.device,
                          weights_only=False)
        ckpt_args = Args()
        ckpt_args.update_args_from_dict(ckpt["args"])
        ckpt_stats = ckpt["data_statistics"]

        # Overwrite role mapping with the unified one (source datasets may differ)
        ckpt_args.role_mapping    = unified_r2id
        ckpt_args.id_to_role      = unified_id2r
        ckpt_args.START_TOKEN_ID  = len(unified_r2id)
        ckpt_args.END_TOKEN_ID    = len(unified_r2id) + 1
        ckpt_args.device          = cli.device
        ckpt_args.epochs          = cli.epochs
        ckpt_args.lr              = lr
        ckpt_args.batch_size      = cli.batch_size
        ckpt_args.save_model      = True
        ckpt_args.model_name      = model_name
        ckpt_args.experiment_path = cli.output_dir

        # Update stats with merged data
        ckpt_stats.update({
            "num_node_labels": len(unified_r2id) + 2,
            "num_edge_labels": 1,
            "max_num_nodes":   max(ckpt_stats.get("max_num_nodes", 0),
                                   stats.get("max_num_nodes", 0)),
            "min_num_nodes":   min(ckpt_stats.get("min_num_nodes", 99),
                                   stats.get("min_num_nodes", 99)),
        })
        stats  = ckpt_stats
        pre    = ckpt_args

        model = ARGDesigner(pre, stats).to(cli.device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded weights from checkpoint.")

    print(f"  num_node_types = {len(unified_r2id) + 2} "
          f"({len(unified_r2id)} roles + START + END)")

    # ---- DataLoaders ---------------------------------------------------------
    dataloader_train = torch.utils.data.DataLoader(
        train_ds, batch_size=cli.batch_size, shuffle=True,
        collate_fn=lambda x: x
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_ds, batch_size=cli.batch_size, shuffle=False,
        collate_fn=lambda x: x
    ) if val_graphs else None

    # ---- Train / fine-tune ---------------------------------------------------
    print(f"\nStarting {'pretraining' if is_pretrain else 'fine-tuning'}: "
          f"{cli.epochs} epochs, lr={lr}")
    train(pre, model, dataloader_train, dataloader_val)

    # Also save a 'final_model.pth' alongside the best checkpoint
    final_name = "final_model.pth" if is_pretrain else "ef_final_model.pth"
    final_path = os.path.join(cli.output_dir, final_name)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "data_statistics":  stats,
            "args":             {k: v for k, v in pre.__dict__.items()
                                 if k != "parser"},
        },
        final_path,
    )

    print(f"\nDone.")
    print(f"  Best checkpoint  : {os.path.join(cli.output_dir, model_name)}")
    print(f"  Final checkpoint : {final_path}")
    print(f"  Source datasets  : {cli.src_datasets}")
    print(f"  Target dataset   : {cli.target_dataset}")


if __name__ == "__main__":
    main()
