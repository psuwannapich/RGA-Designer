"""
Phase 2 — Policy (ARGDesigner) fine-tuning via REINFORCE + KL penalty.

Key additions to ARGDesigner
-----------------------------
sample_with_logprob() mirrors sample() but accumulates log π_θ(G | task)
over every node-type and edge decision so that REINFORCE gradients flow back
through the generator.

REINFORCE + KL objective
-------------------------
For each sampled graph G:

    r      = reward_model(G, task_emb)          — scalar, no grad
    KL     = log π_θ(G) − log π_ref(G)          — per-trajectory KL estimate
    loss   = −(r − kl_coeff × KL) × log π_θ(G)

The reference policy (π_ref) is a frozen copy of the initial ARGDesigner
checkpoint, preventing the policy from collapsing towards reward hacking.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from mas_framework.rlhf.reward_model import GraphRewardModel
from mas_framework.rlhf.preference_data import GraphSnapshot

EPS = 1e-9


# ---------------------------------------------------------------------------
# sample_with_logprob
# ---------------------------------------------------------------------------

def sample_with_logprob(
    model,                              # ARGDesigner instance
    task_embedding: torch.Tensor,       # [384] or [1, 384]
) -> Tuple[nx.DiGraph, torch.Tensor]:  # (graph, scalar log-prob tensor with grad)
    """
    Run one autoregressive sample from *model* and return the sampled graph
    together with the total trajectory log-probability (with gradient).

    Mirrors ARGDesigner.sample() with batch_size=1 but accumulates
    log π_θ at every multinomial (node type) and Bernoulli (edge) step.
    """
    device = model.args.device
    if task_embedding.dim() == 1:
        task_embedding = task_embedding.unsqueeze(0)   # [1, 384]
    task_embedding = task_embedding.to(device)

    role_embeddings_dict_full = model.precomputed_embeddings
    selected_roles_names = list(role_embeddings_dict_full.keys())
    selected_role_ids = [model.role_to_id[r] for r in selected_roles_names]

    end_embedding = model.full_embedding_matrix[model.END_TOKEN].unsqueeze(0)
    candidate_embs = torch.cat(
        [model.full_embedding_matrix[selected_role_ids], end_embedding], dim=0
    )
    temp_end_idx = len(selected_roles_names)

    min_num_node = model.data_statistics.get("min_num_nodes", 2)
    max_num_node = model.data_statistics.get("max_num_nodes", 10)
    HAS_EDGE_TOKEN = 1
    feature_len = model.embedding_dim + model.num_nodes_to_consider * model.len_edge_vec
    is_dag = getattr(model.args, "is_dag", False)

    processed_role_embs: Dict[str, torch.Tensor] = {
        r: torch.tensor(e, device=device).float()
        for r, e in role_embeddings_dict_full.items()
    }

    # Accumulated log-prob (scalar tensor, keeps grad_fn)
    log_prob = torch.zeros(1, device=device)

    # ---- Node GRU init ----
    h_node = torch.zeros(1, 1, model.args.hidden_size_node_level_transformer, device=device)
    t_proc = model.task_processor(task_embedding)           # [1, 384]

    start_input = torch.zeros(1, 1, feature_len, device=device)
    start_input[:, 0, :model.embedding_dim] = t_proc
    start_input[:, 0, model.embedding_dim + model.len_edge_vec - 2] = 1
    start_input = model.node_project(start_input)
    _, h_node = model.node_gru(start_input, h_node)

    generated_roles: List[str] = []
    node_embeddings: List[torch.Tensor] = []
    x_pred_edge = np.zeros((max_num_node, model.num_nodes_to_consider), dtype=np.int32)
    real_num_nodes = max_num_node
    finished = False

    for i in range(max_num_node):
        # Build current node input
        cur_input = torch.zeros(1, 1, feature_len, device=device)
        if i > 0 and node_embeddings:
            prev_embs = torch.stack(node_embeddings, dim=0)
            _, h_agg = model.prev_nodes_aggregator(prev_embs.unsqueeze(0))
            h_node_hist = h_agg.squeeze(0).squeeze(0)
            gate = torch.sigmoid(
                torch.sum(h_node_hist * t_proc[0]) / model.embedding_dim
            )
            combined = (1 - gate) * h_node_hist + gate * t_proc[0]
            cur_input[0, 0, :model.embedding_dim] = combined

        proj_in = model.node_project(cur_input)
        node_out, h_node = model.node_gru(proj_in, h_node)   # [1,1,H]

        # Node type prediction
        pred_emb = model.output_node(node_out)                # [1,1,384]
        proc_cand = model.role_processor(candidate_embs)
        scores = torch.matmul(pred_emb.squeeze(1), proc_cand.t())  # [1, n_cands]
        probs = F.softmax(scores, dim=-1)                          # [1, n_cands]

        if i < min_num_node:
            probs[:, temp_end_idx] = 0.0
            probs = probs / (probs.sum(dim=1, keepdim=True) + EPS)

        sampled_id = torch.multinomial(probs, 1).reshape(-1)   # [1]
        # Accumulate log π(node_type | context)
        log_prob = log_prob + torch.log(probs[0, sampled_id[0]] + EPS)

        if sampled_id[0].item() == temp_end_idx:
            real_num_nodes = i
            finished = True
            break

        cand_idx = sampled_id[0].item()
        node_type_id = selected_role_ids[cand_idx]
        role = model.id_to_role[node_type_id]
        generated_roles.append(role)
        role_emb = processed_role_embs.get(role, torch.zeros(model.embedding_dim, device=device))
        node_embeddings.append(role_emb.clone())

        # ---- Edge generation ----
        active_out = model.embedding_node_to_edge(node_out)   # [1,1,H_edge]
        edge_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
        edge_input[:, 0, model.len_edge_vec - 2] = 1
        edge_input = model.edge_project(edge_input)
        h_edge = active_out

        for j in range(min(model.num_nodes_to_consider, i)):
            if j > 0:
                edge_input = model.edge_project(edge_input)
            edge_out, h_edge = model.edge_gru(edge_input, h_edge)
            edge_pred = model.output_edge(edge_out).view(1, model.len_edge_vec)  # [1, E]
            p_edge = edge_pred[0, HAS_EDGE_TOKEN]                # scalar
            exists = torch.bernoulli(p_edge.detach()).long()
            # Accumulate log π(edge | context)
            log_prob = log_prob + (
                exists * torch.log(p_edge + EPS)
                + (1 - exists) * torch.log(1 - p_edge + EPS)
            )
            next_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
            next_input[:, 0, exists] = 1
            edge_input = next_input
            x_pred_edge[i, j] = exists.item()

        # Force at least one edge after first node
        if i > 0 and not x_pred_edge[i].any():
            j_forced = np.random.randint(0, min(i, model.num_nodes_to_consider))
            x_pred_edge[i, j_forced] = HAS_EDGE_TOKEN

    if not finished:
        real_num_nodes = max_num_node

    # ---- Build NetworkX graph ----
    G = nx.DiGraph() if is_dag else nx.Graph()
    for n in range(real_num_nodes):
        role = generated_roles[n] if n < len(generated_roles) else "Unknown"
        G.add_node(n, label=model.get_role_id(role), role=role)
        if role in processed_role_embs:
            G.nodes[n]["role_embedding"] = processed_role_embs[role].detach().cpu().numpy()

    for n in range(real_num_nodes):
        for j in range(min(model.num_nodes_to_consider, n)):
            if x_pred_edge[n, j] == HAS_EDGE_TOKEN:
                u, v = n - j - 1, n
                if 0 <= u < real_num_nodes:
                    G.add_edge(u, v, label=1)

    return G, log_prob.squeeze()


# ---------------------------------------------------------------------------
# Policy trainer
# ---------------------------------------------------------------------------

class RLHFPolicyTrainer:
    """
    Fine-tunes ARGDesigner (the policy) using REINFORCE with KL penalty.

    Parameters
    ----------
    policy          : ARGDesigner to be fine-tuned (modified in-place)
    reward_model    : frozen GraphRewardModel
    device          : torch device
    lr              : learning rate
    kl_coeff        : weight of KL divergence penalty
    """

    def __init__(
        self,
        policy,                         # ARGDesigner
        reward_model: GraphRewardModel,
        device: torch.device,
        lr: float = 1e-5,
        kl_coeff: float = 0.1,
    ):
        self.policy = policy.to(device)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.kl_coeff = kl_coeff

        # Frozen reference policy for KL penalty
        self.ref_policy = copy.deepcopy(policy).to(device)
        self.ref_policy.eval()
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def _graph_to_reward_input(
        self,
        g: nx.DiGraph,
        task_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert nx.DiGraph + task embedding into (x, edge_index, batch) for the reward model."""
        snapshot = GraphSnapshot.from_nx(g)
        pyg = snapshot.to_pyg(task_embedding.detach().cpu().numpy())
        x = pyg.x.to(self.device)
        edge_index = pyg.edge_index.to(self.device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        return x, edge_index, batch

    def train(
        self,
        task_records: List[Dict[str, Any]],
        epochs: int = 10,
        samples_per_task: int = 2,
        save_path: Optional[str] = None,
    ):
        """
        Main REINFORCE training loop.

        For each task:
          - Draw *samples_per_task* graphs from the policy (with log-probs)
          - Score each with the reward model
          - Also compute reference log-probs for KL penalty
          - Update policy via REINFORCE

        Parameters
        ----------
        task_records     : list of {'task': str, 'task_embedding': np.ndarray}
        epochs           : number of full passes over task_records
        samples_per_task : graphs sampled per task per step
        save_path        : path to save policy checkpoint after each epoch
        """
        from sentence_transformers import SentenceTransformer
        sent_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        for epoch in range(1, epochs + 1):
            epoch_losses, epoch_rewards = [], []

            for record in tqdm(task_records, desc=f"Policy epoch {epoch}/{epochs}"):
                # Task embedding
                task_text = record["task"]
                if "task_embedding" in record:
                    t_emb = torch.tensor(
                        record["task_embedding"], dtype=torch.float32
                    ).to(self.device)
                else:
                    t_emb = torch.tensor(
                        sent_model.encode(task_text), dtype=torch.float32
                    ).to(self.device)

                step_losses = []
                for _ in range(samples_per_task):
                    # Policy sample with log-prob (gradient attached)
                    self.policy.train()
                    g, logprob_policy = sample_with_logprob(self.policy, t_emb)

                    # Reference log-prob (no gradient)
                    with torch.no_grad():
                        _, logprob_ref = sample_with_logprob(self.ref_policy, t_emb)

                    # Reward model score (no gradient)
                    with torch.no_grad():
                        x, edge_index, batch = self._graph_to_reward_input(g, t_emb)
                        reward = self.reward_model(x, edge_index, batch)   # scalar tensor

                    # Per-trajectory KL estimate: log π_θ − log π_ref
                    kl = logprob_policy.detach() - logprob_ref

                    # REINFORCE loss: −(r − β KL) × log π_θ
                    loss = -(reward.detach() - self.kl_coeff * kl) * logprob_policy
                    step_losses.append(loss)
                    epoch_rewards.append(reward.item())

                if step_losses:
                    total_loss = torch.stack(step_losses).mean()
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()
                    epoch_losses.append(total_loss.item())

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            print(
                f"Policy epoch {epoch}/{epochs} | "
                f"loss {avg_loss:.4f} | mean reward {avg_reward:.4f}"
            )

            if save_path:
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.policy.state_dict(),
                        "avg_loss": avg_loss,
                        "avg_reward": avg_reward,
                    },
                    save_path,
                )

        print("Policy fine-tuning complete.")
        return self.policy
