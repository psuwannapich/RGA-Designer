"""
GNN-based reward model for scoring (graph, task) pairs.

Architecture
------------
Node features : concat(role_embedding[384], task_embedding_broadcast[384],
                       structural_features[5]) → 773-dim
                Structural features (per node, all normalized to [0,1]):
                  num_nodes / 6, num_edges / 15, node_position / (N-1),
                  in_degree / (N-1), out_degree / (N-1)
GNN           : SAGEConv(773→256) + residual → SAGEConv(256→128)
                SAGEConv handles empty edge_index gracefully (falls back to self-transform)
Pooling       : global mean pool → [B, 128]
Reward head   : MLP(128→64→1) → scalar reward per graph

Training loss : Bradley-Terry  L = -log σ(r_chosen − r_rejected)

Note on structural features
---------------------------
Without structural features, all nodes in a single-role dataset (e.g. all
MathSolver) carry identical 768-dim vectors. SAGEConv then produces the same
output for any N, so the reward model cannot distinguish a 2-node chain from
a 4-node chain. The 5 structural features break this symmetry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import Optional


class GraphRewardModel(nn.Module):

    def __init__(
        self,
        node_feat_dim: int = 773,   # 384 role-emb + 384 task-emb + 5 structural per node
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Two-layer GraphSAGE encoder
        self.conv1 = SAGEConv(node_feat_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, output_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Residual projection when dims differ
        self.res_proj1 = nn.Linear(node_feat_dim, hidden_dim, bias=False)
        self.res_proj2 = nn.Linear(hidden_dim, output_dim, bias=False)

        # Scalar reward head
        self.reward_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def encode_graph(
        self,
        x: torch.Tensor,            # [N_total, node_feat_dim]
        edge_index: torch.Tensor,   # [2, E]  — may be shape [2, 0]
        batch: torch.Tensor,        # [N_total] graph assignment
    ) -> torch.Tensor:              # [B, output_dim]
        """Encode a batch of graphs into fixed-size embeddings."""
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.norm1(h)
        h = F.relu(h) + self.res_proj1(x)
        h = self.dropout(h)

        # Layer 2
        h2 = self.conv2(h, edge_index)
        h2 = self.norm2(h2)
        h2 = F.relu(h2) + self.res_proj2(h)

        # Global mean pooling → one vector per graph
        graph_emb = global_mean_pool(h2, batch)  # [B, output_dim]
        return graph_emb

    def forward(
        self,
        x: torch.Tensor,            # [N_total, node_feat_dim]
        edge_index: torch.Tensor,   # [2, E]
        batch: torch.Tensor,        # [N_total]
    ) -> torch.Tensor:              # [B] scalar rewards
        graph_emb = self.encode_graph(x, edge_index, batch)
        reward = self.reward_head(graph_emb).squeeze(-1)  # [B]
        return reward

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def bradley_terry_loss(
        chosen_reward: torch.Tensor,            # [B]
        rejected_reward: torch.Tensor,          # [B]
        weights: Optional[torch.Tensor] = None, # [B] per-pair importance weights
    ) -> torch.Tensor:
        """
        L = -mean( w_i * log σ(r_chosen_i − r_rejected_i) )

        weights down-scales uninformative pairs (e.g. both wrong, 0 < w < 1).
        When weights is None reduces to the standard unweighted mean.
        """
        per_pair = -F.logsigmoid(chosen_reward - rejected_reward)   # [B]
        if weights is not None:
            per_pair = per_pair * weights
        return per_pair.mean()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_single(
        self,
        x: torch.Tensor,            # [N, node_feat_dim]
        edge_index: torch.Tensor,   # [2, E]
    ) -> float:
        """Score a single graph (no batch dimension needed)."""
        self.eval()
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        r = self.forward(x, edge_index, batch)
        return r.item()


def build_node_features(
    role_embeddings: torch.Tensor,          # [N, 384]
    task_embedding: torch.Tensor,           # [384]  (single task, broadcast to all nodes)
    structural_features: torch.Tensor,      # [N, 5]  per-node structural scalars
) -> torch.Tensor:                          # [N, 773]
    """
    Concatenate role embedding, broadcast task embedding, and per-node
    structural features.

    Structural features (5 scalars, all in [0, 1]):
      0: num_nodes / 6          — graph size (same for all nodes in graph)
      1: num_edges / 15         — edge count (same for all nodes in graph)
      2: node_position / (N-1)  — position in generation order
      3: in_degree  / (N-1)     — incoming neighbours
      4: out_degree / (N-1)     — outgoing neighbours

    Without these, all nodes in a single-role dataset have identical 768-dim
    vectors. SAGEConv + mean-pool then outputs the same reward regardless of
    graph size, making the reward model useless for size-aware learning.
    """
    N = role_embeddings.size(0)
    task_broadcast = task_embedding.unsqueeze(0).expand(N, -1)
    return torch.cat([role_embeddings, task_broadcast, structural_features], dim=-1)
