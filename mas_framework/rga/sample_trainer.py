"""Correctness model training: per-graph BCE on mixed queries, with
Mantel-Haenszel query weights. Train/validation are split by query."""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mas_framework.rga.preference_data import GraphSampleDataset


def mantel_haenszel_weights(task_ids, labels) -> np.ndarray:
    """w_i = W / (2|P|) if graph i succeeds, W / (2|F|) if it fails,
    W = |P||F| / (|P| + |F|), where P / F are the successful / failed graphs of i's query."""
    cnt = defaultdict(lambda: [0, 0])             # query -> [|F|, |P|]
    for t, y in zip(task_ids, labels):
        cnt[t][int(y)] += 1
    w = np.zeros(len(labels))
    for i, (t, y) in enumerate(zip(task_ids, labels)):
        n_y, n_other = cnt[t][int(y)], cnt[t][1 - int(y)]
        w[i] = n_other / (2.0 * (n_y + n_other))
    return w


def train_reward_model_per_sample(
    model,
    data_dir,
    device,
    epochs: int = 30,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    save_path: str | None = None,
):
    dataset = GraphSampleDataset(data_dir, mixed_only=True)
    n = len(dataset)
    if n == 0:
        raise ValueError(f"GraphSampleDataset is empty for {data_dir}")

    by_task = defaultdict(list)
    for i, t in enumerate(dataset.task_ids()):
        by_task[t].append(i)
    tasks = sorted(by_task)
    if len(tasks) < 2:
        raise ValueError(f"need at least 2 mixed queries to split, found {len(tasks)}")
    perm = np.random.permutation(len(tasks))
    n_val_tasks = min(len(tasks) - 1, max(1, int(round(len(tasks) * val_fraction))))
    val_tasks = {tasks[j] for j in perm[:n_val_tasks]}
    val_idx = np.array([i for t in val_tasks for i in by_task[t]], dtype=int)
    train_idx = np.array([i for t in tasks if t not in val_tasks
                          for i in by_task[t]], dtype=int)
    print(f"  Split by task: {len(tasks) - n_val_tasks} train / {n_val_tasks} val "
          f"tasks ({len(train_idx)} / {len(val_idx)} graphs)")

    labels = np.array([float(y) for _, _, y in dataset._samples])
    w_all = mantel_haenszel_weights(dataset.task_ids(), labels)
    weights = np.zeros(n)
    for ids in (train_idx, val_idx):
        weights[ids] = w_all[ids] / w_all[ids].mean()

    class _Weighted(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            d = dataset[i]; d["weight"] = torch.tensor(weights[i], dtype=torch.float32); return d

    def _collate(batch):
        out = GraphSampleDataset.collate_fn(batch)
        out["weight"] = torch.stack([b["weight"] for b in batch]); return out

    mk = lambda ids, sh: DataLoader(Subset(_Weighted(), ids), batch_size=batch_size,
                                    shuffle=sh, collate_fn=_collate)
    train_loader, val_loader = mk(train_idx, True), mk(val_idx, False)

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val, best_state = float("inf"), None
    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0; nb = 0
        for b in train_loader:
            g = b["graph"].to(device); y = b["is_correct"].to(device)
            logits = model(g.x, g.edge_index, g.batch).view(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y, weight=b["weight"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); nb += 1
        sched.step()

        model.eval(); vl = 0.0; nv = 0; P = []; Y = []
        with torch.no_grad():
            for b in val_loader:
                g = b["graph"].to(device); y = b["is_correct"].to(device)
                logits = model(g.x, g.edge_index, g.batch).view(-1)
                vl += F.binary_cross_entropy_with_logits(logits, y, weight=b["weight"].to(device)).item(); nv += 1
                P.append(torch.sigmoid(logits).cpu().numpy()); Y.append(y.cpu().numpy())
        P = np.concatenate(P) if P else np.array([])
        Y = np.concatenate(Y) if Y else np.array([])
        acc = float(((P >= 0.5) == (Y >= 0.5)).mean()) if P.size else float("nan")
        avg_v = vl / max(nv, 1)
        print(f"  Epoch {ep:3d}/{epochs} | train {tl/max(nb,1):.4f} | val {avg_v:.4f} "
              f"| acc {acc:.2%}")
        if avg_v < best_val:
            best_val = avg_v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save({"model_state_dict": best_state, "epoch": ep,
                            "val_loss": float(best_val), "loss_type": "per_graph_bce",
                            "n_train_graphs": int(len(train_idx))}, save_path)
                print(f"    -> saved {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Per-sample training complete. Best val loss {best_val:.4f}")
    return model
