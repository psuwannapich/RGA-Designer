"""
Serialisable data structures for RLHF preference pairs.

PreferencePair stores two graph results for the same task together with a
composite score derived from correctness, graph size, and edge density.
These are written to .pkl shards and loaded by PreferencePairDataset.

Preference score formula
------------------------
score = w_correct * float(is_correct)
      + w_size    * (1 - num_nodes / ref_max_nodes)
      + w_edge    * edge_term

edge_term = 1 - (num_edges - min_edges) / (max_dag_edges - min_edges)
  where min_edges     = num_nodes - 1   (chain / spanning-tree topology)
        max_dag_edges = num_nodes * (num_nodes - 1) / 2

edge_term = 1.0 when num_edges == num_nodes - 1  (sparse, gets full credit)
edge_term = 0.0 when num_edges == max_dag_edges  (fully connected, no credit)

All three terms are bounded to [0, 1] before weighting.
The correctness term dominates by default (weight 0.6) while size and edge
penalties gently prefer simpler graphs when correctness is equal.
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
        """Convert to a PyG Data object with 773-dim node features.

        Node feature = concat(role_embedding[384], task_embedding[384],
                              structural_features[5]).
        Falls back to zero vectors when embeddings are missing.

        Structural features break the symmetry that would otherwise make all
        nodes in a single-role graph (e.g. all MathSolver) identical, which
        would cause the GNN to assign the same reward to any graph size.
        """
        from mas_framework.rlhf.reward_model import build_node_features

        EMB_DIM = 384
        MAX_NODES = 6    # maximum agents across all datasets (MMLU uses 6)
        MAX_EDGES = 15   # max DAG edges for 6 nodes: 6*5/2 = 15

        N = self.num_nodes
        E = len(self.edges)

        # Compute per-node degree from edge list
        in_deg  = [0] * N
        out_deg = [0] * N
        for u, v in self.edges:
            if 0 <= u < N and 0 <= v < N:
                out_deg[u] += 1
                in_deg[v]  += 1

        node_feats = []
        struct_feats = []
        for i, nd in enumerate(self.nodes):
            if nd["role_embedding"] is not None:
                role_emb = torch.tensor(nd["role_embedding"], dtype=torch.float32)
            else:
                role_emb = torch.zeros(EMB_DIM)
            node_feats.append(role_emb)

            # Structural scalars — all in [0, 1]
            norm_denom = max(N - 1, 1)
            struct_feats.append([
                N / MAX_NODES,             # graph size
                E / MAX_EDGES,             # edge density
                i / norm_denom,            # node position in generation order
                in_deg[i]  / norm_denom,  # in-degree
                out_deg[i] / norm_denom,  # out-degree
            ])

        x_role   = torch.stack(node_feats) if node_feats else torch.zeros(1, EMB_DIM)
        x_struct = torch.tensor(struct_feats, dtype=torch.float32)  # [N, 5]

        if task_embedding is not None:
            t_emb = torch.tensor(task_embedding, dtype=torch.float32)
        else:
            t_emb = torch.zeros(EMB_DIM)

        x = build_node_features(x_role, t_emb, x_struct)   # [N, 773]

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
    edge_cost: float = 0.2

    # Reference values for normalisation
    ref_max_nodes: int = 6


def compute_preference_score(
    is_correct: bool,
    num_nodes: int,
    num_edges: int,
    weights: Optional[PreferenceWeights] = None,
) -> float:
    """
    Compute a scalar preference score in [0, 1].

    correctness_term  = float(is_correct)                              ∈ {0, 1}
    size_term         = max(0, 1 - (num_nodes-1) / (ref_max_nodes-1))  ∈ [0, 1]
    edge_term         = 1 - (num_edges - min_edges) / (max_dag_edges - min_edges)
                        clamped to [0, 1]
                        where min_edges = num_nodes - 1
                              max_dag_edges = num_nodes * (num_nodes - 1) / 2
    """
    if weights is None:
        weights = PreferenceWeights()

    correctness_term = 1.0 if is_correct else 0.0

    size_term = max(0.0, 1.0 - (num_nodes - 1) / max(weights.ref_max_nodes - 1, 1))

    min_edges = max(num_nodes - 1, 0)
    max_dag_edges = num_nodes * (num_nodes - 1) // 2
    if max_dag_edges <= min_edges:
        edge_term = 1.0
    else:
        edge_term = 1.0 - (num_edges - min_edges) / (max_dag_edges - min_edges)
        edge_term = max(0.0, min(1.0, edge_term))

    return (
        weights.correctness * correctness_term
        + weights.graph_size * size_term
        + weights.edge_cost * edge_term
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
    chosen_num_edges: int
    rejected_num_edges: int

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
      is_correct, num_nodes, num_edges
    Pairs are only kept when score_chosen − score_rejected ≥ margin.
    """
    if weights is None:
        weights = PreferenceWeights()

    scored = [
        (r, compute_preference_score(
            r["is_correct"], r["num_nodes"], r["num_edges"], weights
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
                    chosen_num_edges=r_a["num_edges"],
                    rejected_num_edges=r_b["num_edges"],
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
    Loads all .pkl shards from one or more directories and exposes them as a
    flat dataset of (chosen_pyg, rejected_pyg, task_embedding) triples.

    Pass a list of directories to pool data across multiple datasets (e.g. for
    training a single global reward model).
    """

    def __init__(
        self,
        data_dir: "Union[str, List[str]]",
        exclude_both_wrong: bool = False,
        exclude_both_correct: bool = False,
    ):
        dirs: List[str] = [data_dir] if isinstance(data_dir, str) else list(data_dir)

        all_shard_paths: List[str] = []
        for d in dirs:
            paths = sorted(
                os.path.join(d, f) for f in os.listdir(d)
                if f.startswith("shard_") and f.endswith(".pkl")
            )
            if not paths:
                print(f"  [warn] No .pkl shards found in {d}")
            all_shard_paths.extend(paths)

        if not all_shard_paths:
            raise FileNotFoundError(
                f"No .pkl shard files found in any of: {dirs}"
            )

        # Build flat index: (shard_path, local_idx)
        self._index: List[Tuple[str, int]] = []
        self._shard_cache: Dict[str, List[PreferencePair]] = {}
        n_excluded_wrong = 0
        n_excluded_correct = 0

        # Stats for logging — per-directory and overall
        from collections import defaultdict, Counter
        dir_stats: Dict[str, dict] = {}

        for path in all_shard_paths:
            d = os.path.dirname(path)
            with open(path, "rb") as f:
                shard: List[PreferencePair] = pickle.load(f)
            self._shard_cache[path] = shard

            if d not in dir_stats:
                dir_stats[d] = {
                    "shards": 0, "pairs": 0, "excluded": 0,
                    "correct_vs_wrong": 0, "wrong_vs_correct": 0,
                    "both_correct": 0, "both_wrong": 0,
                    "domains": Counter(),
                }
            s = dir_stats[d]
            s["shards"] += 1

            for i, pair in enumerate(shard):
                both_wrong   = not pair.chosen_is_correct and not pair.rejected_is_correct
                both_correct = pair.chosen_is_correct and pair.rejected_is_correct
                if exclude_both_wrong and both_wrong:
                    n_excluded_wrong += 1
                    s["excluded"] += 1
                    continue
                if exclude_both_correct and both_correct:
                    n_excluded_correct += 1
                    s["excluded"] += 1
                    continue
                self._index.append((path, i))
                s["pairs"] += 1
                domain = pair.metadata.get("domain", "unknown")
                s["domains"][domain] += 1
                if pair.chosen_is_correct and not pair.rejected_is_correct:
                    s["correct_vs_wrong"] += 1
                elif not pair.chosen_is_correct and pair.rejected_is_correct:
                    s["wrong_vs_correct"] += 1
                elif both_correct:
                    s["both_correct"] += 1
                else:
                    s["both_wrong"] += 1

        # ── Logging ────────────────────────────────────────────────────────
        print("\n  ── Preference pair dataset ─────────────────────────────")
        total_pairs = len(self._index)
        for d, s in dir_stats.items():
            domain_str = "  ".join(f"{k}={v}" for k, v in sorted(s["domains"].items()))
            print(
                f"  Dir  : {d}\n"
                f"         shards={s['shards']}  pairs={s['pairs']}"
                + (f"  excluded={s['excluded']}" if s["excluded"] else "")
                + f"\n"
                f"         correct→wrong={s['correct_vs_wrong']}  "
                f"wrong→correct={s['wrong_vs_correct']}  "
                f"both-correct={s['both_correct']}  "
                f"both-wrong={s['both_wrong']}\n"
                + (f"         domains: {domain_str}" if domain_str else "")
            )
        excluded_parts = []
        if n_excluded_wrong:
            excluded_parts.append(f"{n_excluded_wrong} both-wrong excluded")
        if n_excluded_correct:
            excluded_parts.append(f"{n_excluded_correct} both-correct excluded")
        print(
            f"  Total: {total_pairs} pairs from {len(all_shard_paths)} shards"
            + (f"  ({', '.join(excluded_parts)})" if excluded_parts else "")
        )
        print("  ────────────────────────────────────────────────────────\n")

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
