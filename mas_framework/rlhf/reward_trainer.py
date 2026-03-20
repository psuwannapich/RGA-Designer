"""
Phase 1 — Reward model training.

Trains GraphRewardModel on PreferencePairDataset using Bradley-Terry loss.
Validation metric: accuracy = fraction of pairs where r_chosen > r_rejected.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from mas_framework.rlhf.preference_data import PreferencePairDataset
from mas_framework.rlhf.reward_model import GraphRewardModel


def train_reward_model(
    model: GraphRewardModel,
    data_dir: str,
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-4,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    save_path: Optional[str] = None,
    weight_decay: float = 1e-4,
) -> GraphRewardModel:
    """
    Train *model* on preference pairs stored as .pkl shards in *data_dir*.

    Parameters
    ----------
    model        : GraphRewardModel (will be modified in-place)
    data_dir     : directory containing .pkl shards from RLHFDataCollector
    device       : torch device
    epochs       : training epochs
    lr           : Adam learning rate
    batch_size   : mini-batch size
    val_fraction : fraction of data reserved for validation
    save_path    : if given, save best checkpoint here
    weight_decay : AdamW weight decay

    Returns
    -------
    Trained model (same object, best weights restored if save_path was set).
    """
    dataset = PreferencePairDataset(data_dir)
    n = len(dataset)
    val_size = max(1, int(n * val_fraction))
    train_size = n - val_size

    idx = np.random.permutation(n)
    train_idx, val_idx = idx[val_size:], idx[:val_size]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=PreferencePairDataset.collate_fn,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=PreferencePairDataset.collate_fn,
    )

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss, train_acc, n_batches = 0.0, 0.0, 0
        for batch in train_loader:
            chosen = batch["chosen"].to(device)
            rejected = batch["rejected"].to(device)

            r_chosen = model(chosen.x, chosen.edge_index, chosen.batch)
            r_rejected = model(rejected.x, rejected.edge_index, rejected.batch)

            loss = GraphRewardModel.bradley_terry_loss(r_chosen, r_rejected)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_acc += (r_chosen > r_rejected).float().mean().item()
            n_batches += 1

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                chosen = batch["chosen"].to(device)
                rejected = batch["rejected"].to(device)

                r_chosen = model(chosen.x, chosen.edge_index, chosen.batch)
                r_rejected = model(rejected.x, rejected.edge_index, rejected.batch)

                loss = GraphRewardModel.bradley_terry_loss(r_chosen, r_rejected)
                val_loss += loss.item()
                val_acc += (r_chosen > r_rejected).float().mean().item()
                n_val += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        avg_train_acc = train_acc / max(n_batches, 1)
        avg_val_loss = val_loss / max(n_val, 1)
        avg_val_acc = val_acc / max(n_val, 1)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train loss {avg_train_loss:.4f} acc {avg_train_acc:.2%} | "
            f"val loss {avg_val_loss:.4f} acc {avg_val_acc:.2%}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "val_loss": avg_val_loss,
                        "val_acc": avg_val_acc,
                    },
                    save_path,
                )
                print(f"  → Best model saved to {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"\nReward model training complete. Best val loss: {best_val_loss:.4f}")
    return model


def load_reward_model(
    checkpoint_path: str,
    device: torch.device,
    node_feat_dim: int = 768,
    hidden_dim: int = 256,
    output_dim: int = 128,
) -> GraphRewardModel:
    """Load a saved GraphRewardModel checkpoint."""
    model = GraphRewardModel(
        node_feat_dim=node_feat_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded reward model from {checkpoint_path} "
          f"(val_loss={ckpt.get('val_loss', 'n/a'):.4f})")
    return model
