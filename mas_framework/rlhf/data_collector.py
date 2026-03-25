"""
Async RLHF preference-data collection pipeline.

For each task the collector:
  1. Tries several (topology, agent-count, roles) configurations
  2. Runs each as a TestGraph via arun() concurrently
  3. Scores every result with the composite preference score
  4. Builds all-pairs PreferencePair objects where score_chosen − score_rejected ≥ margin
  5. Writes shards of List[PreferencePair] to disk

The collector is intentionally dataset-agnostic: callers inject the
answer-checker and role-description dicts so the same class works for
gsm8k, aqua, mmlu, etc.
"""

from __future__ import annotations

import asyncio
import copy
import os
import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.rlhf.preference_data import (
    GraphSnapshot,
    PreferencePair,
    PreferenceWeights,
    create_preference_pairs,
)
from mas_framework.rlhf.token_estimator import estimate_tokens
from experiment.utils import get_kwargs, generate_graph, convert_to_pyg_graph


# ---------------------------------------------------------------------------
# Topology configurations
# ---------------------------------------------------------------------------

def _default_configs(
    available_roles: List[str],
    min_agents: int = 2,
    max_agents: int = 4,
) -> List[Tuple[str, int, List[str]]]:
    """Return (mode, num_agents, roles) triples to evaluate per task."""
    topologies = ["FullConnected", "Chain", "Star"]
    configs = []
    for mode in topologies:
        for n in range(min_agents, max_agents + 1):
            roles = random.choices(available_roles, k=n)
            configs.append((mode, n, roles))
    return configs


# Default temperatures used for model-based candidate generation.
# T=1.0 reproduces the model's trained distribution; higher values flatten
# the role/edge distributions to encourage structural diversity, analogous
# to temperature sampling in autoregressive LLMs.
DEFAULT_SAMPLE_TEMPERATURES: List[float] = [1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class RLHFDataCollector:
    """
    Parameters
    ----------
    domain            : task domain string ('gsm8k', 'aqua', …)
    llm_name          : model name passed to LLMRegistry (e.g. 'gemma3')
    answer_checker    : (predicted_str, ground_truth_str) → bool
    get_predict       : (raw_llm_output) → predicted_str
    role_descriptions : {role_name: description_str}
    decision_method   : agent-graph decision node name
    num_rounds        : spatial rounds per inference
    weights           : PreferenceWeights for scoring
    pair_margin       : minimum score gap to keep a pair
    """

    def __init__(
        self,
        domain: str,
        llm_name: str,
        answer_checker: Callable[[str, str], bool],
        get_predict: Callable[[str], str],
        role_descriptions: Dict[str, str],
        agent_name: str = "MathSolver",
        decision_method: str = "FinalRefer",
        num_rounds: int = 1,
        weights: Optional[PreferenceWeights] = None,
        pair_margin: float = 0.05,
        timeout: int = 600,
        arg_model=None,
        sample_temperatures: Optional[List[float]] = None,
    ):
        self.domain = domain
        self.llm_name = llm_name
        self.answer_checker = answer_checker
        self.get_predict = get_predict
        self.role_descriptions = role_descriptions
        self.agent_name = agent_name
        self.decision_method = decision_method
        self.num_rounds = num_rounds
        self.weights = weights or PreferenceWeights()
        self.pair_margin = pair_margin
        self.timeout = timeout
        # Optional trained ARGDesigner model for richer candidate generation.
        # One graph is sampled per temperature value; higher T → more varied structure.
        self.arg_model = arg_model
        self.sample_temperatures = sample_temperatures if sample_temperatures is not None \
            else DEFAULT_SAMPLE_TEMPERATURES

        self._sentence_model = None   # lazy-loaded once

    def _get_sentence_model(self):
        if self._sentence_model is None:
            from sentence_transformers import SentenceTransformer
            # Force CPU: all-MiniLM-L6-v2 is tiny (~90 MB) so CPU inference is
            # fast, and keeping it off the GPU prevents CUDA resource conflicts
            # with the main LLM that is already loaded on the accelerator.
            self._sentence_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
        return self._sentence_model

    def _encode_task(self, task: str) -> np.ndarray:
        return self._get_sentence_model().encode(task)

    # ------------------------------------------------------------------
    # Single graph execution
    # ------------------------------------------------------------------

    async def _run_graph(
        self,
        graph: Graph,
        record: Dict[str, Any],
        mode: str,
        agent_num: int,
        task_embedding: np.ndarray,
    ) -> Optional[Dict]:
        """Run one graph on one task; return a result dict or None on error."""
        realized = copy.deepcopy(graph)
        input_dict = {"task": record["task"]}

        flow_graph = realized.to_pyg_graph(input_dict)
        tg = TestGraph(
            domain=self.domain,
            llm_name=self.llm_name,
            decision_method=self.decision_method,
            pyg_data=flow_graph,
        )

        try:
            result = await asyncio.wait_for(
                tg.arun(input_dict, self.num_rounds), timeout=self.timeout
            )
        except Exception as e:
            import traceback
            print(f"  [skip] {mode}-{agent_num}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

        raw = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(raw, list) and raw:
            raw = raw[0]
        raw = str(raw) if not isinstance(raw, str) else raw

        predicted = self.get_predict(raw)
        is_correct = self.answer_checker(predicted, record["answer"])

        # Build nx.DiGraph from pyg data for GraphSnapshot
        nx_g = nx.DiGraph()
        num_nodes = flow_graph.num_nodes
        for i, node_data in enumerate(flow_graph.x):
            role = node_data.get("role", "Unknown") if isinstance(node_data, dict) else "Unknown"
            nx_g.add_node(i, role=role)
        if flow_graph.edge_index.numel() > 0:
            for src, dst in flow_graph.edge_index.t().numpy():
                nx_g.add_edge(int(src), int(dst))

        # Attach role embeddings from the sentence model
        model = self._get_sentence_model()
        for n in nx_g.nodes():
            role = nx_g.nodes[n].get("role", "Unknown")
            emb = model.encode(role)
            nx_g.nodes[n]["role_embedding"] = emb

        return {
            "task_question": record["task"],
            "task_embedding": task_embedding,
            "graph_snapshot": GraphSnapshot.from_nx(nx_g),
            "is_correct": is_correct,
            "num_nodes": num_nodes,
            "estimated_tokens": estimate_tokens(nx_g, self.num_rounds),
            "mode": mode,
            "domain": self.domain,
        }

    async def _run_nx_graph(
        self,
        nx_g: nx.DiGraph,
        record: Dict[str, Any],
        label: str,
        task_embedding: np.ndarray,
    ) -> Optional[Dict]:
        """Run one NetworkX graph on one task; return a result dict or None."""
        task_text = record["task"]

        # Apply role constraints then convert to PyG
        model = self._get_sentence_model()
        for n in nx_g.nodes():
            role = nx_g.nodes[n].get("role", "Unknown")
            nx_g.nodes[n]["constraint"] = self.role_descriptions.get(role, "")
            nx_g.nodes[n]["role_embedding"] = model.encode(role)

        pyg_data = convert_to_pyg_graph(nx_g, task_text)
        tg = TestGraph(
            domain=self.domain,
            llm_name=self.llm_name,
            decision_method=self.decision_method,
            pyg_data=pyg_data,
        )

        try:
            result = await asyncio.wait_for(
                tg.arun({"task": task_text}, self.num_rounds), timeout=self.timeout
            )
        except Exception as e:
            import traceback
            print(f"  [skip] {label}: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

        raw = result[0] if isinstance(result, (list, tuple)) else result
        if isinstance(raw, list) and raw:
            raw = raw[0]
        raw = str(raw) if not isinstance(raw, str) else raw

        predicted = self.get_predict(raw)
        is_correct = self.answer_checker(predicted, record["answer"])

        return {
            "task_question": task_text,
            "task_embedding": task_embedding,
            "graph_snapshot": GraphSnapshot.from_nx(nx_g),
            "is_correct": is_correct,
            "num_nodes": nx_g.number_of_nodes(),
            "estimated_tokens": estimate_tokens(nx_g, self.num_rounds),
            "mode": label,
            "domain": self.domain,
        }

    # ------------------------------------------------------------------
    # Per-task collection
    # ------------------------------------------------------------------

    async def collect_for_task(
        self,
        record: Dict[str, Any],
        min_agents: int = 2,
        max_agents: int = 4,
    ) -> List[PreferencePair]:
        available_roles = list(self.role_descriptions.keys())
        configs = _default_configs(available_roles, min_agents, max_agents)
        task_embedding = self._encode_task(record["task"])

        # Build all graph instances first (cheap)
        graph_runs = []
        for mode, n, roles in configs:
            kwargs = get_kwargs(mode, n)
            kwargs["node_kwargs"] = [{"role": r} for r in roles]
            try:
                g = Graph(
                    domain=self.domain,
                    llm_name=self.llm_name,
                    agent_names=[self.agent_name] * n,
                    decision_method=self.decision_method,
                    **kwargs,
                )
                graph_runs.append((g, mode, n))
            except Exception as e:
                import traceback
                print(f"  [skip build] {mode}-{n}: {type(e).__name__}: {e}")
                traceback.print_exc()

        # Run graphs sequentially — local HF models process one request at a
        # time, so concurrent calls just queue up and hit the timeout.
        raw_results = []
        for g, m, n in graph_runs:
            result = await self._run_graph(g, record, m, n, task_embedding)
            raw_results.append(result)

        # --- ARGDesigner model-based candidates (temperature sampling) ---
        # Each entry in sample_temperatures produces one independently sampled graph.
        # Higher T flattens the role/edge distributions → more structural diversity,
        # analogous to temperature sampling in autoregressive LLMs.
        if self.arg_model is not None:
            emb_tensor = torch.tensor(
                task_embedding, device=self.arg_model.args.device
            ).float()
            for temp in self.sample_temperatures:
                try:
                    graphs = generate_graph(
                        self.arg_model, emb_tensor, self.role_descriptions,
                        temperature=temp,
                    )
                    if graphs:
                        result = await self._run_nx_graph(
                            graphs[0], record, f"arg_model_T{temp:.2f}", task_embedding
                        )
                        raw_results.append(result)
                except Exception as e:
                    print(f"  [skip arg_model T={temp}] {type(e).__name__}: {e}")

        results = [r for r in raw_results if isinstance(r, dict)]
        if len(results) < 2:
            return []

        return create_preference_pairs(results, self.weights, self.pair_margin)

    # ------------------------------------------------------------------
    # Full dataset collection
    # ------------------------------------------------------------------

    async def collect_dataset(
        self,
        task_records: List[Dict[str, Any]],
        output_dir: str,
        min_agents: int = 2,
        max_agents: int = 4,
        checkpoint_every: int = 20,
    ) -> int:
        """
        Collect preference pairs for all tasks, writing .pkl shards to
        *output_dir* every *checkpoint_every* tasks.

        Returns the total number of preference pairs collected.
        """
        os.makedirs(output_dir, exist_ok=True)
        shard_idx = 0
        buffer: List[PreferencePair] = []
        total_pairs = 0

        for i, record in enumerate(tqdm(task_records, desc="RLHF collection")):
            pairs = await self.collect_for_task(record, min_agents, max_agents)
            buffer.extend(pairs)
            total_pairs += len(pairs)

            if (i + 1) % checkpoint_every == 0 or (i + 1) == len(task_records):
                if buffer:
                    path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(buffer, f)
                    print(f"  Shard {shard_idx}: saved {len(buffer)} pairs → {path}")
                    shard_idx += 1
                    buffer = []

            print(
                f"  Task {i+1}/{len(task_records)}: "
                f"+{len(pairs)} pairs  (total so far: {total_pairs})"
            )

        print(f"\nCollection complete. Total pairs: {total_pairs} in {shard_idx} shards.")
        return total_pairs
