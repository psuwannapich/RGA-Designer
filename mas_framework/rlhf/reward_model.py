"""
GNN-based reward model for scoring (graph, task) pairs.

Architecture
------------
Node features : concat(role_embedding[384], task_embedding_broadcast[384]) → 768-dim
GNN           : SAGEConv(768→256) + residual → SAGEConv(256→128)
                SAGEConv handles empty edge_index gracefully (falls back to self-transform)
Pooling       : global mean pool → [B, 128]
Reward head   : MLP(128→64→1) → scalar reward per graph

Training loss : Bradley-Terry  L = -log σ(r_chosen − r_rejected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from typing import Optional


class GraphRewardModel(nn.Module):

    def __init__(
        self,
        node_feat_dim: int = 768,   # 384 role-emb + 384 task-emb per node
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
        chosen_reward: torch.Tensor,    # [B]
        rejected_reward: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        L = -mean( log σ(r_chosen − r_rejected) )

        Equivalent to binary cross-entropy on the "chosen is better" label.
        """
        return -F.logsigmoid(chosen_reward - rejected_reward).mean()

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
    role_embeddings: torch.Tensor,  # [N, 384]
    task_embedding: torch.Tensor,   # [384]  (single task, broadcast to all nodes)
) -> torch.Tensor:                  # [N, 768]
    """
    Concatenate role embedding with broadcast task embedding per node.
    This puts graph structure and task context in the same feature space
    used by graph.py's construct_new_features().
    """
    task_broadcast = task_embedding.unsqueeze(0).expand(role_embeddings.size(0), -1)
    return torch.cat([role_embeddings, task_broadcast], dim=-1)
