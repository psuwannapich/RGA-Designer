"""
Standalone ARGDesigner pretraining script.

Trains a fresh ARGDesigner on cold-start graph data (.pt files) produced by
cold_start.py.  Works for all supported datasets without needing the
dataset-specific finetune_*.py scripts (which have hardcoded paths and assume
a specific working directory).

Usage
-----
    python experiment/pretrain.py \\
        --dataset   gsm8k \\
        --data_dir  ColdStartData_hf_gsm8k \\
        --output_dir checkpoints/gsm8k \\
        --epochs    100

Run via uv from the project root:
    uv run python experiment/pretrain.py --dataset gsm8k --data_dir ...
"""

import argparse
import os
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


SUPPORTED_DATASETS = [
    "gsm8k", "aqua", "multiarith", "svamp", "humaneval", "mmlu"
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrain ARGDesigner on cold-start graph data"
    )
    p.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS,
                   help="Dataset name (must match the adapter in process_dataset.py)")
    p.add_argument("--data_dir", required=True,
                   help="Directory containing cold-start .pt graph files")
    p.add_argument("--output_dir", default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Learning rate")
    p.add_argument("--batch_size", type=int, default=32,
                   help="Training batch size")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample_size", type=int, default=0,
                   help="Cap on number of graphs loaded (0 = all)")
    p.add_argument("--model_pool", default=None,
                   help="Base-model pool enabling the per-node model-selection head: "
                        "a JSON file ({name: description}), a comma-separated list of "
                        "model names, or 'default' for the built-in pool. "
                        "Unset = single-model ARGDesigner (no head).")
    p.add_argument("--model_loss_weight", type=float, default=0.2,
                   help="Weight of the model-selection CE term in the training loss")
    p.add_argument("--default_node_model", default=None,
                   help="Backfill model label for graphs without per-node 'model' "
                        "attributes (e.g. legacy single-model cold-start data). "
                        "Typically the --llm_name the data was generated with. "
                        "Unlabeled nodes are excluded from the model loss otherwise.")
    return p.parse_args()


def setup_environment(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cli = parse_args()
    setup_environment(cli.seed)

    os.makedirs(cli.output_dir, exist_ok=True)

    # ---- Build Args object expected by process_dataset / ARGDesigner -------
    pre = Args().update_args()
    pre.dataset      = cli.dataset
    pre.data_dir     = cli.data_dir
    pre.experiment_path = cli.output_dir
    pre.epochs       = cli.epochs
    pre.lr           = cli.lr
    pre.batch_size   = cli.batch_size
    pre.seed         = cli.seed
    pre.device       = cli.device
    pre.save_model   = True
    pre.model_name   = "best_model.pth"
    pre.dataset_sample_size = cli.sample_size

    # ---- Load dataset -------------------------------------------------------
    print(f"Loading '{cli.dataset}' graphs from: {cli.data_dir}")
    ds, _ = gdata.load_graph_dataset(pre, pretrain=True)

    if len(ds) == 0:
        raise RuntimeError(
            f"No graphs loaded from '{cli.data_dir}'. "
            "Run cold-start first: uv run cold-start --dataset {cli.dataset} ..."
        )

    # ---- Multi-model pool ----------------------------------------------------
    if cli.model_pool:
        from mas_framework.llm.model_pool import load_model_pool
        pool = load_model_pool(None if cli.model_pool == "default" else cli.model_pool)
        pre.model_pool = list(pool.keys())
        pre.model_descriptions = pool
        pre.model_loss_weight = cli.model_loss_weight
        print(f"Model pool ({len(pre.model_pool)}): {pre.model_pool}")
        if cli.default_node_model:
            n_backfilled = 0
            for g in ds.graph_list:
                for _, data in g.nodes(data=True):
                    if not data.get("model"):
                        data["model"] = cli.default_node_model
                        n_backfilled += 1
            print(f"Backfilled 'model'={cli.default_node_model!r} on {n_backfilled} unlabeled nodes")
    else:
        pre.model_pool = []
        pre.model_descriptions = {}

    # ---- Role mapping -------------------------------------------------------
    role_to_id = ds.role_to_id
    id_to_role = ds.id_to_role
    num_node_types = len(role_to_id) + 2   # +2 for START / END tokens

    pre.role_mapping   = role_to_id
    pre.id_to_role     = id_to_role
    pre.START_TOKEN_ID = len(role_to_id)
    pre.END_TOKEN_ID   = len(role_to_id) + 1

    # ---- Data statistics ----------------------------------------------------
    stats = gdata.get_data_statistics(ds.graph_list)
    stats["num_node_labels"] = num_node_types
    stats["num_edge_labels"] = 1

    print(f"Loaded {len(ds)} graphs | roles: {list(role_to_id.keys())}")
    print(f"Stats: {stats}")

    # ---- Train / val split --------------------------------------------------
    correct   = [g for g in ds if g.graph.get("is_correct")]
    incorrect = [g for g in ds if not g.graph.get("is_correct")]
    random.shuffle(correct)
    random.shuffle(incorrect)

    ratio = 1.0 - cli.val_ratio
    train_graphs = (correct[:int(len(correct) * ratio)]
                    + incorrect[:int(len(incorrect) * ratio)])
    val_graphs   = (correct[int(len(correct) * ratio):]
                    + incorrect[int(len(incorrect) * ratio):])
    random.shuffle(train_graphs)

    print(f"Train graphs: {len(train_graphs)} | Val graphs: {len(val_graphs)}")

    train_dataset = gdata.GraphListDataset(train_graphs, pre)
    val_dataset   = gdata.GraphListDataset(val_graphs,   pre)

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=cli.batch_size, shuffle=True,
        collate_fn=lambda x: x
    )
    dataloader_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=cli.batch_size, shuffle=False,
        collate_fn=lambda x: x
    ) if val_graphs else None

    # ---- Build model --------------------------------------------------------
    model = ARGDesigner(pre, stats).to(cli.device)
    print(f"ARGDesigner built | device: {cli.device}")

    # ---- Train --------------------------------------------------------------
    print(f"\nStarting pretraining: {cli.epochs} epochs, lr={cli.lr}")
    train(pre, model, dataloader_train, dataloader_val)

    # Save final checkpoint alongside best checkpoint
    final_path = os.path.join(cli.output_dir, "final_model.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "data_statistics": stats,
            "args": pre.__dict__,
        },
        final_path,
    )
    print(f"\nPretraining complete.")
    print(f"  Best checkpoint : {os.path.join(cli.output_dir, 'best_model.pth')}")
    print(f"  Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
