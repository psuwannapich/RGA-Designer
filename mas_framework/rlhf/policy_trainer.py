"""
Phase 2 — Policy (ARGDesigner) fine-tuning via GRPO + KL penalty.

Key additions to ARGDesigner
-----------------------------
sample_with_logprob() mirrors sample() but accumulates log π_θ(G | task)
over every node-type and edge decision so that GRPO gradients flow back
through the generator.

GRPO + KL objective
--------------------
For a group of G graphs sampled per task:

    r_i    = reward_model(G_i, task_emb)          — scalar, no grad
    A_i    = (r_i − mean_group(r)) / std_group(r)  — within-group advantage
    KL_i   = log π_θ(G_i) − log π_ref(G_i)        — per-trajectory KL, same graph G_i
    loss   = −A_i × log π_θ(G_i) + kl_coeff × KL_i

KL_i is estimated by teacher-forcing G_i through the frozen reference policy
so that both log-probs are evaluated on the same trajectory.  The additive KL
penalty is stable: it does not multiply by log π_θ (which can be large negative
for long sequences).  Gradient flows through log π_θ in both terms.

Normalising within the group removes the reward model's arbitrary absolute
offset and scale, making gradients robust to reward model miscalibration.
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
        if i < min_num_node:
            # Mask END token by adding -inf to its logit before softmax
            # (avoids in-place writes that would invalidate the autograd graph)
            inf_mask = scores.new_zeros(scores.shape)
            inf_mask[:, temp_end_idx] = float('-inf')
            scores = scores + inf_mask
        probs = F.softmax(scores, dim=-1)                          # [1, n_cands]

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


def compute_logprob_for_graph(
    model,
    graph: nx.DiGraph,
    task_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Teacher-force *graph* through *model* and return log π(G | task).

    Mirrors sample_with_logprob exactly but, instead of sampling at each
    decision point, reads the actual choice from *graph*.  Call inside
    torch.no_grad() when evaluating the frozen reference policy.
    """
    device = model.args.device
    if task_embedding.dim() == 1:
        task_embedding = task_embedding.unsqueeze(0)
    task_embedding = task_embedding.to(device)

    num_nodes = graph.number_of_nodes()
    node_roles = [graph.nodes[i].get("role", "Unknown") for i in range(num_nodes)]
    edge_set = set(graph.edges())

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

    processed_role_embs: Dict[str, torch.Tensor] = {
        r: torch.tensor(e, device=device).float()
        for r, e in role_embeddings_dict_full.items()
    }

    log_prob = torch.zeros(1, device=device)

    h_node = torch.zeros(1, 1, model.args.hidden_size_node_level_transformer, device=device)
    t_proc = model.task_processor(task_embedding)

    start_input = torch.zeros(1, 1, feature_len, device=device)
    start_input[:, 0, :model.embedding_dim] = t_proc
    start_input[:, 0, model.embedding_dim + model.len_edge_vec - 2] = 1
    start_input = model.node_project(start_input)
    _, h_node = model.node_gru(start_input, h_node)

    node_embeddings: List[torch.Tensor] = []

    for i in range(max_num_node + 1):
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
        node_out, h_node = model.node_gru(proj_in, h_node)

        pred_emb = model.output_node(node_out)
        proc_cand = model.role_processor(candidate_embs)
        scores = torch.matmul(pred_emb.squeeze(1), proc_cand.t())
        if i < min_num_node:
            inf_mask = scores.new_zeros(scores.shape)
            inf_mask[:, temp_end_idx] = float("-inf")
            scores = scores + inf_mask
        probs = F.softmax(scores, dim=-1)

        # Teacher-forcing: use the actual token from the graph
        if i >= num_nodes:
            actual_idx = temp_end_idx
        else:
            role = node_roles[i]
            actual_idx = (
                selected_roles_names.index(role)
                if role in selected_roles_names
                else temp_end_idx
            )

        log_prob = log_prob + torch.log(probs[0, actual_idx] + EPS)

        if actual_idx == temp_end_idx:
            break

        role = node_roles[i]
        role_emb = processed_role_embs.get(
            role, torch.zeros(model.embedding_dim, device=device)
        )
        node_embeddings.append(role_emb.clone())

        active_out = model.embedding_node_to_edge(node_out)
        edge_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
        edge_input[:, 0, model.len_edge_vec - 2] = 1
        edge_input = model.edge_project(edge_input)
        h_edge = active_out

        for j in range(min(model.num_nodes_to_consider, i)):
            if j > 0:
                edge_input = model.edge_project(edge_input)
            edge_out, h_edge = model.edge_gru(edge_input, h_edge)
            edge_pred = model.output_edge(edge_out).view(1, model.len_edge_vec)
            p_edge = edge_pred[0, HAS_EDGE_TOKEN]

            u, v = i - j - 1, i
            exists = 1 if (u, v) in edge_set else 0
            log_prob = log_prob + (
                exists * torch.log(p_edge + EPS)
                + (1 - exists) * torch.log(1 - p_edge + EPS)
            )
            next_input = torch.zeros(1, 1, model.len_edge_vec, device=device)
            next_input[:, 0, exists] = 1
            edge_input = next_input

    return log_prob.squeeze()


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
        lambda_eff: float = 0.0,
        ref_max_nodes: int = 4,
    ):
        self.policy = policy.to(device)
        self.reward_model = reward_model.to(device)
        self.reward_model.eval()
        for p in self.reward_model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.kl_coeff = kl_coeff
        self.lambda_eff = lambda_eff
        self.ref_max_nodes = ref_max_nodes

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
        samples_per_task: int = 4,
        save_path: Optional[str] = None,
        grad_accum_steps: int = 8,
    ):
        """
        Main GRPO training loop.

        For each task:
          - Draw *samples_per_task* graphs from the policy (with log-probs)
          - Score each with the reward model
          - Compute within-group advantages: A_i = (r_i - mean) / std
          - Also compute reference log-probs for KL penalty
          - Accumulate gradients over *grad_accum_steps* tasks, then update

        Parameters
        ----------
        task_records      : list of {'task': str, 'task_embedding': np.ndarray}
        epochs            : number of full passes over task_records
        samples_per_task  : graphs sampled per task per step (≥2 required for std;
                            ≥4 recommended so advantage magnitude encodes reward gap)
        save_path         : path to save policy checkpoint after each epoch
        grad_accum_steps  : number of tasks to accumulate gradients over before
                            calling optimizer.step() (reduces per-step variance)
        """
        from mas_framework.llm.profile_embedding import get_sentence_model
        sent_model = get_sentence_model()

        best_reward = float("-inf")
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            epoch_losses, epoch_rewards = [], []

            # Gradient accumulation state — reset at the start of each epoch.
            self.optimizer.zero_grad()
            accum_count = 0

            for task_idx, record in enumerate(tqdm(task_records, desc=f"Policy epoch {epoch}/{epochs}")):
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

                # Collect all samples for this task first.
                self.policy.train()
                sample_logprobs, sample_rewards, sample_logprob_refs = [], [], []

                # Pre-compute max possible edges for an autoregressive DAG
                # with ref_max_nodes nodes: each node i can connect to all i
                # prior nodes → sum(0..N-1) = N*(N-1)/2.
                ref_max_edges = self.ref_max_nodes * (self.ref_max_nodes - 1) / 2

                for _ in range(samples_per_task):
                    g, logprob_policy = sample_with_logprob(self.policy, t_emb)

                    with torch.no_grad():
                        # Teacher-force the same graph G_i through the frozen reference
                        # policy so KL_i = log π_θ(G_i) − log π_ref(G_i) is on-trajectory.
                        logprob_ref = compute_logprob_for_graph(self.ref_policy, g, t_emb)
                        x, edge_index, batch = self._graph_to_reward_input(g, t_emb)
                        reward = self.reward_model(x, edge_index, batch)   # scalar tensor

                        if self.lambda_eff > 0.0:
                            num_nodes = g.number_of_nodes()
                            num_edges = g.number_of_edges()
                            node_bonus = num_nodes / self.ref_max_nodes
                            edge_bonus = (num_edges / ref_max_edges if ref_max_edges > 0 else 0.0)
                            efficiency_bonus = 1 - node_bonus - edge_bonus
                            reward = reward + self.lambda_eff * efficiency_bonus

                    sample_logprobs.append(logprob_policy)
                    sample_rewards.append(reward.detach())
                    sample_logprob_refs.append(logprob_ref)  # detached (from no_grad)

                # GRPO: normalise within the group (per-task mean and std).
                # Clamp std from below at 0.01 so near-identical rewards produce
                # proportionally small advantages rather than full-strength gradients.
                rewards_tensor = torch.stack(sample_rewards)                       # [G]
                group_mean = rewards_tensor.mean()
                group_std  = torch.clamp(rewards_tensor.std(), min=0.01)           # [fix2]
                advantages  = (rewards_tensor - group_mean) / group_std            # [G]

                # Log raw rewards for monitoring (advantages always average to 0).
                for r in sample_rewards:
                    epoch_rewards.append(r.item())

                step_losses = []
                for logprob_policy, advantage, logprob_ref in zip(
                    sample_logprobs, advantages, sample_logprob_refs
                ):
                    # KL_i = log π_θ(G_i) − log π_ref(G_i), gradient through logprob_policy.
                    # Additive penalty: loss = −A_i × log π_θ + β × KL_i
                    kl = logprob_policy - logprob_ref
                    loss = -advantage * logprob_policy + self.kl_coeff * kl
                    step_losses.append(loss)

                if step_losses:
                    # Divide by grad_accum_steps so the effective loss magnitude
                    # is independent of how many tasks are accumulated.
                    task_loss = torch.stack(step_losses).mean() / grad_accum_steps
                    task_loss.backward()                                            # [fix1]
                    epoch_losses.append(task_loss.item() * grad_accum_steps)
                    accum_count += 1

                # Step the optimizer every grad_accum_steps tasks, or at the end
                # of the epoch so no gradients are silently discarded.
                is_last_task = (task_idx + 1) == len(task_records)
                if accum_count > 0 and (accum_count % grad_accum_steps == 0 or is_last_task):
                    nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    accum_count = 0

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
            # avg_reward is the mean raw reward from the reward model.
            # A rising trend indicates the policy generates higher-scored graphs.
            is_best = avg_reward > best_reward
            if is_best:
                best_reward = avg_reward
                best_epoch = epoch
            print(
                f"Policy epoch {epoch}/{epochs} | "
                f"loss {avg_loss:.4f} | mean reward {avg_reward:.4f}"
                + (" [best]" if is_best else "")
            )

            if save_path and is_best:
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

        # Load the best checkpoint back into the policy.
        if save_path and os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ckpt["model_state_dict"])
            print(
                f"GRPO policy fine-tuning complete. "
                f"Loaded best model from epoch {best_epoch} "
                f"(mean reward {best_reward:.4f})."
            )
        else:
            print("GRPO policy fine-tuning complete.")
        return self.policy
