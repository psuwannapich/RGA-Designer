"""
Serialisable data structures for RLHF preference pairs.

PreferencePair stores two graph results for the same task together with a
composite score derived from correctness, graph size, and estimated token cost.
These are written to .pkl shards and loaded by PreferencePairDataset.

Preference score formula
------------------------
score = w_correct * float(is_correct)
      - w_size    * (num_nodes / ref_max_nodes)
      - w_token   * log1p(estimated_tokens / ref_token_scale) / log1p(1)

All three terms are bounded to [0, 1] before weighting.
The correctness term dominates by default (weight 0.6) while size and token
penalties gently prefer cheaper graphs when correctness is equal.
"""

from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


# ---------------------------------------------------------------------------
# Serialisable graph snapshot
# ---------------------------------------------------------------------------

@dataclass
class GraphSnapshot:
    """
    Pickle-safe representation of a nx.DiGraph.

    Stores only the adjacency structure and per-node role attributes.
    Role embeddings (np.ndarray) are kept as lists for portability.
    """
    nodes: List[Dict]           # [{'id': int, 'role': str, 'role_embedding': list[float]}]
    edges: List[Tuple[int, int]]
    num_nodes: int

    @classmethod
    def from_nx(cls, g: nx.DiGraph) -> "GraphSnapshot":
        nodes = []
        for n in g.nodes():
            data = g.nodes[n]
            emb = data.get("role_embedding", None)
            if isinstance(emb, (np.ndarray, torch.Tensor)):
                emb = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            nodes.append({
                "id": int(n),
                "role": data.get("role", "Unknown"),
                "role_embedding": emb,
            })
        edges = [(int(u), int(v)) for u, v in g.edges()]
        return cls(nodes=nodes, edges=edges, num_nodes=g.number_of_nodes())

    def to_nx(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for nd in self.nodes:
            attrs = {"role": nd["role"]}
            if nd["role_embedding"] is not None:
                attrs["role_embedding"] = np.array(nd["role_embedding"], dtype=np.float32)
            g.add_node(nd["id"], **attrs)
        for u, v in self.edges:
            g.add_edge(u, v, label=1)
        return g

    def to_pyg(self, task_embedding: Optional[np.ndarray] = None) -> Data:
        """Convert to a PyG Data object with 768-dim node features.

        Node feature = concat(role_embedding[384], task_embedding[384]).
        Falls back to zero vectors when embeddings are missing.
        """
        from mas_framework.rlhf.reward_model import build_node_features

        EMB_DIM = 384
        node_feats = []
        for nd in self.nodes:
            if nd["role_embedding"] is not None:
                role_emb = torch.tensor(nd["role_embedding"], dtype=torch.float32)
            else:
                role_emb = torch.zeros(EMB_DIM)
            node_feats.append(role_emb)

        x_role = torch.stack(node_feats) if node_feats else torch.zeros(1, EMB_DIM)

        if task_embedding is not None:
            t_emb = torch.tensor(task_embedding, dtype=torch.float32)
        else:
            t_emb = torch.zeros(EMB_DIM)

        x = build_node_features(x_role, t_emb)          # [N, 768]

        if self.edges:
            edge_index = torch.tensor(self.edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, num_nodes=self.num_nodes)


# ---------------------------------------------------------------------------
# Preference weights and scoring
# ---------------------------------------------------------------------------

@dataclass
class PreferenceWeights:
    """Relative importance of each preference signal."""
    correctness: float = 0.6
    graph_size: float = 0.2
    token_cost: float = 0.2

    # Reference values for normalisation
    ref_max_nodes: int = 6
    ref_token_scale: int = 2000   # tokens at which the penalty is ~0.5


def compute_preference_score(
    is_correct: bool,
    num_nodes: int,
    estimated_tokens: int,
    weights: Optional[PreferenceWeights] = None,
) -> float:
    """
    Compute a scalar preference score in [0, 1].

    correctness_term  = float(is_correct)                              ∈ {0, 1}
    size_term         = max(0, 1 - num_nodes / ref_max_nodes)         ∈ [0, 1]
    token_term        = 1 / (1 + log1p(tokens / ref_token_scale))     ∈ (0, 1]
    """
    if weights is None:
        weights = PreferenceWeights()

    correctness_term = 1.0 if is_correct else 0.0

    size_term = max(0.0, 1.0 - num_nodes / max(weights.ref_max_nodes, 1))

    token_scale = max(weights.ref_token_scale, 1)
    token_term = 1.0 / (1.0 + math.log1p(estimated_tokens / token_scale))

    return (
        weights.correctness * correctness_term
        + weights.graph_size * size_term
        + weights.token_cost * token_term
    )


# ---------------------------------------------------------------------------
# Preference pair
# ---------------------------------------------------------------------------

@dataclass
class PreferencePair:
    """
    A (chosen, rejected) preference pair for one task.

    All heavy fields (embeddings) are stored as plain Python lists / numpy
    arrays so pickle / torch.save work without capturing any nn.Module state.
    """
    task_question: str
    task_embedding: np.ndarray          # shape (384,)

    chosen_graph: GraphSnapshot
    rejected_graph: GraphSnapshot

    chosen_score: float
    rejected_score: float
    chosen_is_correct: bool
    rejected_is_correct: bool

    chosen_num_nodes: int
    rejected_num_nodes: int
    chosen_estimated_tokens: int
    rejected_estimated_tokens: int

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["task_embedding"] = self.task_embedding.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PreferencePair":
        d = dict(d)
        d["task_embedding"] = np.array(d["task_embedding"], dtype=np.float32)
        return cls(**d)


def create_preference_pairs(
    results: List[Dict],
    weights: Optional[PreferenceWeights] = None,
    margin: float = 0.05,
) -> List[PreferencePair]:
    """
    Build all (chosen, rejected) pairs from a list of result dicts.

    Each dict must have keys:
      graph_snapshot, task_question, task_embedding,
      is_correct, num_nodes, estimated_tokens
    Pairs are only kept when score_chosen − score_rejected ≥ margin.
    """
    if weights is None:
        weights = PreferenceWeights()

    scored = [
        (r, compute_preference_score(
            r["is_correct"], r["num_nodes"], r["estimated_tokens"], weights
        ))
        for r in results
    ]

    pairs = []
    for i, (r_a, s_a) in enumerate(scored):
        for j, (r_b, s_b) in enumerate(scored):
            if i == j:
                continue
            if s_a - s_b >= margin:
                pairs.append(PreferencePair(
                    task_question=r_a["task_question"],
                    task_embedding=r_a["task_embedding"],
                    chosen_graph=r_a["graph_snapshot"],
                    rejected_graph=r_b["graph_snapshot"],
                    chosen_score=s_a,
                    rejected_score=s_b,
                    chosen_is_correct=r_a["is_correct"],
                    rejected_is_correct=r_b["is_correct"],
                    chosen_num_nodes=r_a["num_nodes"],
                    rejected_num_nodes=r_b["num_nodes"],
                    chosen_estimated_tokens=r_a["estimated_tokens"],
                    rejected_estimated_tokens=r_b["estimated_tokens"],
                    metadata={
                        "chosen_mode": r_a.get("mode", ""),
                        "rejected_mode": r_b.get("mode", ""),
                        "domain": r_a.get("domain", ""),
                    },
                ))
    return pairs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PreferencePairDataset(Dataset):
    """
    Loads all .pkl shards from a directory and exposes them as a flat dataset
    of (chosen_pyg, rejected_pyg, task_embedding) triples.

    Shards are memory-mapped lazily: only the requested shard is unpickled.
    """

    def __init__(self, data_dir: str, exclude_both_wrong: bool = False):
        self.data_dir = data_dir
        shard_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".pkl")
        )
        if not shard_paths:
            raise FileNotFoundError(f"No .pkl shard files found in {data_dir}")

        # Build flat index: (shard_path, local_idx)
        self._index: List[Tuple[str, int]] = []
        self._shard_cache: Dict[str, List[PreferencePair]] = {}
        n_excluded = 0

        for path in shard_paths:
            with open(path, "rb") as f:
                shard: List[PreferencePair] = pickle.load(f)
            self._shard_cache[path] = shard
            for i, pair in enumerate(shard):
                if exclude_both_wrong and not pair.chosen_is_correct and not pair.rejected_is_correct:
                    n_excluded += 1
                    continue
                self._index.append((path, i))

        msg = f"PreferencePairDataset: {len(self._index)} pairs from {len(shard_paths)} shards"
        if exclude_both_wrong and n_excluded:
            msg += f" ({n_excluded} both-wrong pairs excluded)"
        print(msg)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, local_idx = self._index[idx]
        pair: PreferencePair = self._shard_cache[path][local_idx]

        task_emb = torch.tensor(pair.task_embedding, dtype=torch.float32)
        chosen_pyg = pair.chosen_graph.to_pyg(pair.task_embedding)
        rejected_pyg = pair.rejected_graph.to_pyg(pair.task_embedding)

        return {
            "chosen": chosen_pyg,
            "rejected": rejected_pyg,
            "task_embedding": task_emb,
            "chosen_score": torch.tensor(pair.chosen_score, dtype=torch.float32),
            "rejected_score": torch.tensor(pair.rejected_score, dtype=torch.float32),
            "chosen_is_correct": torch.tensor(pair.chosen_is_correct, dtype=torch.bool),
            "rejected_is_correct": torch.tensor(pair.rejected_is_correct, dtype=torch.bool),
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        chosen_batch = Batch.from_data_list([item["chosen"] for item in batch])
        rejected_batch = Batch.from_data_list([item["rejected"] for item in batch])
        return {
            "chosen": chosen_batch,
            "rejected": rejected_batch,
            "task_embedding": torch.stack([item["task_embedding"] for item in batch]),
            "chosen_score": torch.stack([item["chosen_score"] for item in batch]),
            "rejected_score": torch.stack([item["rejected_score"] for item in batch]),
            "chosen_is_correct": torch.stack([item["chosen_is_correct"] for item in batch]),
            "rejected_is_correct": torch.stack([item["rejected_is_correct"] for item in batch]),
        }
