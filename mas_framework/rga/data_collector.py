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
import itertools
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.rga.preference_data import (
    GraphSnapshot,
    PreferencePair,
    PreferenceWeights,
    create_preference_pairs,
)
from experiment.utils import get_kwargs, generate_graph, convert_to_pyg_graph


# ---------------------------------------------------------------------------
# Topology configurations
# ---------------------------------------------------------------------------

def _role_sweep_configs(
    available_roles: List[str],
    n_agents_list: Optional[List[int]] = None,
    fixed_topology: str = "Chain",
    max_combos: Optional[int] = None,
) -> List[Tuple[str, int, List[str]]]:
    """Return all ordered role-combination configs for a fixed topology.

    Instead of varying topology (Chain/Star/Full) with random roles, this fixes
    the topology and exhaustively enumerates every ordered role assignment for
    the given agent counts.  The reward model then learns *which role pairs work
    best for which task types* rather than just which size is smallest.

    For 4 GSM8K roles × n=2 → 4²=16 configs.
    For 4 roles × n=3  → 4³=64 configs (cap with max_combos to stay tractable).

    Parameters
    ----------
    available_roles : list of role name strings
    n_agents_list   : agent counts to sweep (default [2])
    fixed_topology  : topology to fix for all configs (default 'Chain')
    max_combos      : if set, randomly subsample to at most this many configs
                      (applied per-n_agents to preserve coverage)
    """
    if n_agents_list is None:
        n_agents_list = [2]

    configs: List[Tuple[str, int, List[str]]] = []
    for n in n_agents_list:
        combos = list(itertools.product(available_roles, repeat=n))
        if max_combos is not None and len(combos) > max_combos:
            random.shuffle(combos)
            combos = combos[:max_combos]
        for combo in combos:
            configs.append((fixed_topology, n, list(combo)))

    return configs


def _build_topology_graph(topology: str, roles: List[str]) -> nx.DiGraph:
    """Build a nx.DiGraph for the given topology and role sequence.

    Supported topologies: Chain, Star, FullConnected.
    Node *i* is assigned roles[i].  Edge direction: earlier → later nodes.
    """
    n = len(roles)
    g = nx.DiGraph()
    for i, role in enumerate(roles):
        g.add_node(i, role=role)

    if topology == "Chain":
        for i in range(n - 1):
            g.add_edge(i, i + 1)
    elif topology == "Star":
        # Node 0 fans out to all others
        for i in range(1, n):
            g.add_edge(0, i)
    elif topology in ("FullConnected", "Full"):
        for i in range(n):
            for j in range(n):
                if i != j:
                    g.add_edge(i, j)
    else:
        # Default: chain
        for i in range(n - 1):
            g.add_edge(i, i + 1)

    return g


def _default_configs(
    available_roles: List[str],
    min_agents: int = 2,
    max_agents: int = 4,
    weak_baselines: bool = False,
) -> List[Tuple[str, int, List[str]]]:
    """Return (mode, num_agents, roles) triples to evaluate per task.

    When *weak_baselines* is True two deliberately suboptimal configs are
    added to ensure some runs will fail even on high-ceiling datasets:
      - A single-agent Chain (no collaboration at all).
      - An over-sized Chain (max_agents + 1 nodes, redundant roles).
    These create correctness-differentiating preference pairs for datasets
    where most multi-agent topologies already achieve near-perfect accuracy.
    """
    topologies = ["FullConnected", "Chain", "Star"]
    configs = []
    for mode in topologies:
        for n in range(min_agents, max_agents + 1):
            roles = random.choices(available_roles, k=n)
            configs.append((mode, n, roles))

    if weak_baselines:
        # Single-agent: no peer discussion, often misses multi-step errors
        configs.append(("Chain", 1, random.choices(available_roles, k=1)))
        # Over-sized chain: extra redundant agent increases verbosity / confusion
        oversized_n = max_agents + 1
        configs.append(("Chain", oversized_n, random.choices(available_roles, k=oversized_n)))

    return configs


# Default temperatures used for model-based candidate generation.
# T=1.0 reproduces the model's trained distribution; higher values flatten
# the role/edge distributions to encourage structural diversity, analogous
# to temperature sampling in autoregressive LLMs.
DEFAULT_SAMPLE_TEMPERATURES: List[float] = [1.0, 1.5, 2.0]


def _prune_graph(nx_g: nx.DiGraph, pruning_ratio: float) -> nx.DiGraph:
    """Remove a random fraction of edges, reconnecting any isolated components."""
    g = copy.deepcopy(nx_g)
    n_remove = int(g.number_of_edges() * pruning_ratio)
    if n_remove > 0:
        edges = list(g.edges())
        random.shuffle(edges)
        g.remove_edges_from(edges[:n_remove])
        if not nx.is_weakly_connected(g):
            comps = list(nx.weakly_connected_components(g))
            main = max(comps, key=len)
            for c in comps:
                if c != main:
                    g.add_edge(
                        random.choice(list(c)),
                        random.choice(list(main)),
                    )
    return g


def _graph_fingerprint(nx_g: nx.DiGraph) -> tuple:
    """Stable fingerprint for deduplication: (num_nodes, sorted_edges, sorted_roles)."""
    roles = tuple(nx_g.nodes[n].get("role", "Unknown") for n in sorted(nx_g.nodes()))
    edges = tuple(sorted((int(u), int(v)) for u, v in nx_g.edges()))
    return (nx_g.number_of_nodes(), edges, roles)


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class RGADataCollector:
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
        arg_model_samples: Optional[int] = None,
        pruning_ratio: float = 0.0,
        weak_baselines: bool = False,
        role_sweep: bool = False,
        role_sweep_topology: str = "Chain",
        role_sweep_n_agents: Optional[List[int]] = None,
        role_sweep_max_combos: Optional[int] = None,
        inference_concurrency: int = 8,
        model_pool: Optional[Dict[str, str]] = None,
    ):
        self.domain = domain
        self.llm_name = llm_name
        # {model_name: description} pool for multi-model graphs.  Used to
        # attach semantic model embeddings to snapshot nodes so the reward
        # model can condition on per-node base-model identity.
        self.model_pool = model_pool or {}
        self._model_emb_cache: Dict[str, np.ndarray] = {}
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
        # Temperatures are cycled when more samples are needed than len(temperatures).
        self.arg_model = arg_model
        self.sample_temperatures = sample_temperatures if sample_temperatures is not None \
            else DEFAULT_SAMPLE_TEMPERATURES
        # Number of unique ARGDesigner graphs to generate per task.
        # None = auto: len(temperatures) when _default_configs also runs,
        #              len(temperatures)*3 when _default_configs is skipped.
        self.arg_model_samples = arg_model_samples
        # Edge pruning ratio: if > 0, each correct graph is also run after
        # removing this fraction of its edges, injecting pruned variants into
        # the preference pool for additional diversity.
        self.pruning_ratio = pruning_ratio
        self.weak_baselines = weak_baselines
        self.role_sweep = role_sweep
        self.role_sweep_topology = role_sweep_topology
        self.role_sweep_n_agents = role_sweep_n_agents if role_sweep_n_agents is not None else [2]
        self.role_sweep_max_combos = role_sweep_max_combos
        self.inference_concurrency = inference_concurrency

        self._sentence_model = None   # lazy-loaded once
        self._inference_sem: Optional[asyncio.Semaphore] = None  # created on first use

    def _get_inference_sem(self) -> asyncio.Semaphore:
        if self._inference_sem is None:
            self._inference_sem = asyncio.Semaphore(self.inference_concurrency)
        return self._inference_sem

    def _get_sentence_model(self):
        if self._sentence_model is None:
            from mas_framework.llm.profile_embedding import get_sentence_model
            self._sentence_model = get_sentence_model()
        return self._sentence_model

    def _encode_task(self, task: str) -> np.ndarray:
        return self._get_sentence_model().encode(task)

    def _attach_model_embeddings(self, nx_g: nx.DiGraph) -> None:
        """Attach semantic base-model embeddings to nodes carrying a 'model'
        attribute (no-op without a model pool)."""
        if not self.model_pool:
            return
        for n in nx_g.nodes():
            m = nx_g.nodes[n].get("model")
            if m and "model_embedding" not in nx_g.nodes[n]:
                if m not in self._model_emb_cache:
                    description = self.model_pool.get(m, m)
                    self._model_emb_cache[m] = self._get_sentence_model().encode(
                        f"{m}: {description}")
                nx_g.nodes[n]["model_embedding"] = self._model_emb_cache[m]

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
            async with self._get_inference_sem():
                result = await asyncio.wait_for(
                    tg.arun(input_dict, self.num_rounds), timeout=2400
                )
        except asyncio.CancelledError:
            raise
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
            node_model = node_data.get("model") if isinstance(node_data, dict) else None
            nx_g.add_node(i, role=role)
            if node_model:
                nx_g.nodes[i]["model"] = node_model
        if flow_graph.edge_index.numel() > 0:
            for src, dst in flow_graph.edge_index.t().numpy():
                nx_g.add_edge(int(src), int(dst))

        # Attach role embeddings from the sentence model
        model = self._get_sentence_model()
        for n in nx_g.nodes():
            role = nx_g.nodes[n].get("role", "Unknown")
            emb = model.encode(role)
            nx_g.nodes[n]["role_embedding"] = emb
        self._attach_model_embeddings(nx_g)

        return {
            "task_question": record["task"],
            "task_embedding": task_embedding,
            "graph_snapshot": GraphSnapshot.from_nx(nx_g),
            "is_correct": is_correct,
            "num_nodes": num_nodes,
            "num_edges": nx_g.number_of_edges(),
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
        self._attach_model_embeddings(nx_g)

        pyg_data = convert_to_pyg_graph(nx_g, task_text)
        tg = TestGraph(
            domain=self.domain,
            llm_name=self.llm_name,
            decision_method=self.decision_method,
            pyg_data=pyg_data,
        )

        try:
            async with self._get_inference_sem():
                result = await asyncio.wait_for(
                    tg.arun({"task": task_text}, self.num_rounds), timeout=self.timeout
                )
        except asyncio.CancelledError:
            # Propagate cancellation so the parent gather can clean up correctly.
            raise
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
            "num_edges": nx_g.number_of_edges(),
            "mode": label,
            "domain": self.domain,
        }

    # ------------------------------------------------------------------
    # Two-phase collection helpers (split vLLM on/off)
    # ------------------------------------------------------------------

    def generate_candidates_for_task(
        self,
        record: Dict[str, Any],
        min_agents: int = 2,
        max_agents: int = 4,
        num_sample_for_tasks: int = 1,
    ) -> List[Tuple[nx.DiGraph, str]]:
        """Generate candidate graph structures for one task (CPU/GPU only, no LLM).

        Returns a list of (nx_graph, label) pairs that need LLM scoring.
        Call this phase *before* starting vLLM; call score_candidates_for_task
        *after* vLLM is running to score them and build preference pairs.

        Note: cold-start pool graphs are not generated here — they already have
        correctness labels and are injected directly during score_candidates_for_task.
        """
        available_roles = list(self.role_descriptions.keys())
        has_arg_model = self.arg_model is not None

        candidates: List[Tuple[nx.DiGraph, str]] = []
        seen_fps: Set[tuple] = set()

        # --- Default topology grid (used only when no ARGDesigner model) ---
        if not has_arg_model:
            configs = _default_configs(available_roles, min_agents, max_agents,
                                       weak_baselines=self.weak_baselines)
            for mode, n, roles in configs:
                nx_g = _build_topology_graph(mode, roles)
                fp = _graph_fingerprint(nx_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                candidates.append((nx_g, f"{mode}_{n}"))

        # --- ARGDesigner model-based candidates (synchronous GPU/CPU work) ---
        if self.arg_model is not None:
            task_embedding = self._encode_task(record["task"])
            n_samples = self.arg_model_samples
            if n_samples is None:
                # No coldstart context here → treat as if coldstart is absent.
                # use_default_configs = False (arg_model exists) → multiply by factor.
                n_samples = len(self.sample_temperatures) * num_sample_for_tasks

            emb_tensor = torch.tensor(
                task_embedding, device=self.arg_model.args.device
            ).float()

            arg_count = 0
            max_attempts = n_samples * 4
            temp_cycle = itertools.cycle(self.sample_temperatures)
            for _ in range(max_attempts):
                if arg_count >= n_samples:
                    break
                temp = next(temp_cycle)
                try:
                    graphs = generate_graph(
                        self.arg_model, emb_tensor, self.role_descriptions,
                        temperature=temp,
                    )
                    if not graphs:
                        continue
                    fp = _graph_fingerprint(graphs[0])
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    candidates.append((graphs[0], f"arg_model_T{temp:.2f}"))
                    arg_count += 1
                except Exception as e:
                    print(f"  [skip arg_model T={temp}] {type(e).__name__}: {e}")

            if arg_count < n_samples:
                print(f"  [warn] ARGDesigner: requested {n_samples} unique graphs, "
                      f"got {arg_count} after {max_attempts} attempts")

        # --- Role sweep: exhaustive role-combination configs ---
        if self.role_sweep:
            sweep_configs = _role_sweep_configs(
                available_roles,
                n_agents_list=self.role_sweep_n_agents,
                fixed_topology=self.role_sweep_topology,
                max_combos=self.role_sweep_max_combos,
            )
            for mode, n, roles in sweep_configs:
                nx_g = _build_topology_graph(mode, roles)
                fp = _graph_fingerprint(nx_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                label = f"role_sweep_{'_'.join(r.replace(' ', '') for r in roles)}"
                candidates.append((nx_g, label))

        return candidates

    def generate_all_candidates(
        self,
        task_records: List[Dict[str, Any]],
        output_path: str,
        min_agents: int = 2,
        max_agents: int = 4,
        num_sample_for_tasks: int = 1,
    ) -> None:
        """Generate and save candidate graphs for all tasks (no LLM, no vLLM required).

        Saves a list of {"record": ..., "candidates": [(nx_graph, label), ...]}
        dicts to *output_path* as a pickle file.  Load with collect_from_candidates().
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        all_task_data = []
        for record in tqdm(task_records, desc="Generating candidates (no LLM)"):
            candidates = self.generate_candidates_for_task(
                record, min_agents, max_agents, num_sample_for_tasks,
            )
            all_task_data.append({"record": record, "candidates": candidates})

        with open(output_path, "wb") as fh:
            pickle.dump(all_task_data, fh)

        total_cands = sum(len(d["candidates"]) for d in all_task_data)
        print(f"\nCandidate generation complete.")
        print(f"  Tasks     : {len(all_task_data)}")
        print(f"  Candidates: {total_cands} total ({total_cands / max(1, len(all_task_data)):.1f} per task)")
        print(f"  Saved to  : {output_path}")

    async def score_candidates_for_task(
        self,
        record: Dict[str, Any],
        candidates: List[Tuple[nx.DiGraph, str]],
        task_embedding: np.ndarray,
        extra_results: Optional[List[Dict]] = None,
    ) -> Tuple[List[PreferencePair], dict]:
        """Run LLM inference on pre-generated candidate graphs, build preference pairs.

        candidates     : output of generate_candidates_for_task() — graphs needing LLM.
        task_embedding : pre-computed sentence embedding for the task.
        extra_results  : pre-scored cold-start graphs injected without LLM calls
                         (same format as collect_for_task's extra_results param).
        """
        raw_results: List[Dict] = []
        seen_fps: Set[tuple] = set()

        # --- Inject pre-scored cold-start / finetune graphs (no LLM needed) ---
        if extra_results:
            sent_model = self._get_sentence_model()
            for r in extra_results:
                nx_g = r["nx_graph"]
                fp = _graph_fingerprint(nx_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                for n in nx_g.nodes():
                    if "role_embedding" not in nx_g.nodes[n]:
                        role = nx_g.nodes[n].get("role", "Unknown")
                        nx_g.nodes[n]["role_embedding"] = sent_model.encode(role)
                self._attach_model_embeddings(nx_g)
                raw_results.append({
                    "task_question":  record["task"],
                    "task_embedding": task_embedding,
                    "graph_snapshot": GraphSnapshot.from_nx(nx_g),
                    "is_correct":     r["is_correct"],
                    "num_nodes":      r["num_nodes"],
                    "num_edges":      nx_g.number_of_edges(),
                    "mode":           r.get("mode", "coldstart"),
                    "domain":         self.domain,
                })

        # Deduplicate new candidates against cold-start fingerprints
        unique_candidates: List[Tuple[nx.DiGraph, str]] = []
        for nx_g, label in candidates:
            fp = _graph_fingerprint(nx_g)
            if fp in seen_fps:
                continue
            seen_fps.add(fp)
            unique_candidates.append((nx_g, label))

        # --- Run LLM inference on unique candidates concurrently ---
        if unique_candidates:
            llm_results = await asyncio.gather(
                *(self._run_nx_graph(g, record, label, task_embedding)
                  for g, label in unique_candidates),
                return_exceptions=True,
            )
            for r in llm_results:
                if r is not None and not isinstance(r, Exception):
                    raw_results.append(r)

        # --- Pruning phase: edge-pruned variants of correct graphs ---
        if self.pruning_ratio > 0:
            correct_so_far = [r for r in raw_results if isinstance(r, dict) and r["is_correct"]]
            to_prune: List[Tuple[nx.DiGraph, str]] = []
            for r in correct_so_far:
                pruned_g = _prune_graph(r["graph_snapshot"].to_nx(), self.pruning_ratio)
                fp = _graph_fingerprint(pruned_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                to_prune.append((pruned_g, f"pruned_{r['mode']}"))

            if to_prune:
                pruned_results = await asyncio.gather(
                    *(self._run_nx_graph(g, record, label, task_embedding)
                      for g, label in to_prune),
                    return_exceptions=True,
                )
                for r in pruned_results:
                    if r is not None and not isinstance(r, Exception):
                        raw_results.append(r)

        results = [r for r in raw_results if isinstance(r, dict)]
        skipped = len(raw_results) - len(results)
        n_correct   = sum(1 for r in results if     r["is_correct"])
        n_incorrect = sum(1 for r in results if not r["is_correct"])

        stats = {
            "total":     len(results),
            "correct":   n_correct,
            "incorrect": n_incorrect,
            "skipped":   skipped,
        }
        if len(results) < 2:
            return [], stats
        return create_preference_pairs(results, self.weights, self.pair_margin), stats

    async def collect_from_candidates(
        self,
        candidates_path: str,
        output_dir: str,
        checkpoint_every: int = 20,
        coldstart_pool: Optional[Dict[str, List[Dict]]] = None,
        task_concurrency: int = 4,
    ) -> int:
        """Load pre-generated candidates, run LLM scoring, write preference-pair shards.

        candidates_path : pickle file written by generate_all_candidates().
        output_dir      : directory for shard_XXXX.pkl preference-pair files.
        coldstart_pool  : pre-scored graphs keyed by task question (no LLM needed).

        Returns the total number of preference pairs written.
        """
        with open(candidates_path, "rb") as fh:
            all_task_data: List[Dict] = pickle.load(fh)

        os.makedirs(output_dir, exist_ok=True)
        sem = asyncio.Semaphore(task_concurrency)
        progress = tqdm(total=len(all_task_data), desc="RLHF collect (LLM scoring)")

        async def _run_one(item: Dict):
            record    = item["record"]
            candidates = item["candidates"]
            task_emb  = self._encode_task(record["task"])
            extra     = (coldstart_pool or {}).get(record["task"], [])
            async with sem:
                result = await self.score_candidates_for_task(
                    record, candidates, task_emb, extra_results=extra or None,
                )
            progress.update(1)
            return result

        all_results = await asyncio.gather(
            *[_run_one(item) for item in all_task_data],
            return_exceptions=True,
        )
        progress.close()

        # Cancel leftover connection-pool tasks (same pattern as collect_dataset)
        _current = asyncio.current_task()
        _pending = [t for t in asyncio.all_tasks() if t is not _current]
        if _pending:
            for t in _pending:
                t.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*_pending, return_exceptions=True), timeout=30.0,
                )
            except asyncio.TimeoutError:
                pass

        shard_idx  = 0
        buffer: List[PreferencePair] = []
        total_pairs = 0
        cum_total = cum_correct = cum_incorrect = cum_skipped = 0

        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"  Task {i+1} error: {type(result).__name__}: {result}")
                continue

            pairs, stats = result
            buffer.extend(pairs)
            total_pairs += len(pairs)
            cum_total     += stats["total"]
            cum_correct   += stats["correct"]
            cum_incorrect += stats["incorrect"]
            cum_skipped   += stats["skipped"]

            print(
                f"  Task {i+1}/{len(all_task_data)}: "
                f"graphs={stats['total']} "
                f"(correct={stats['correct']}, incorrect={stats['incorrect']}"
                + (f", skipped={stats['skipped']}" if stats["skipped"] else "")
                + f")  +{len(pairs)} pairs  (total pairs: {total_pairs})"
            )

            if (i + 1) % checkpoint_every == 0 or (i + 1) == len(all_task_data):
                if buffer:
                    path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pkl")
                    with open(path, "wb") as fh:
                        pickle.dump(buffer, fh)
                    print(f"  Shard {shard_idx}: saved {len(buffer)} pairs → {path}")
                    shard_idx += 1
                    buffer = []

        correct_rate = cum_correct / cum_total * 100 if cum_total else 0.0
        print(
            f"\n── Collection complete ──────────────────────────────\n"
            f"  Tasks processed  : {len(all_task_data)}\n"
            f"  Graphs evaluated : {cum_total}  "
            f"(correct={cum_correct} [{correct_rate:.1f}%], "
            f"incorrect={cum_incorrect}, skipped={cum_skipped})\n"
            f"  Preference pairs : {total_pairs} in {shard_idx} shards\n"
            f"────────────────────────────────────────────────────"
        )
        return total_pairs

    # ------------------------------------------------------------------
    # Per-task collection
    # ------------------------------------------------------------------

    async def collect_for_task(
        self,
        record: Dict[str, Any],
        min_agents: int = 2,
        max_agents: int = 4,
        extra_results: Optional[List[Dict]] = None,
        num_sample_for_tasks: int = 1,
    ) -> List[PreferencePair]:
        available_roles = list(self.role_descriptions.keys())
        task_embedding = self._encode_task(record["task"])

        # _default_configs is skipped when richer candidates are already available:
        # either the ARGDesigner model can generate them, or the coldstart pool
        # already covers this task.  In those cases _default_configs would just
        # add expensive LLM inference for graphs already explored during cold-start.
        has_coldstart = bool(extra_results)
        use_default_configs = (self.arg_model is None) and (not has_coldstart)

        raw_results = []
        # Fingerprints of all graphs already in raw_results — used to deduplicate
        # ARGDesigner candidates against both coldstart graphs and each other.
        seen_fps: Set[tuple] = set()

        # --- ColdStart / Finetune pre-collected graphs (loaded first) ---
        # Processed before any new generation so their fingerprints seed seen_fps
        # and ARGDesigner cannot produce duplicate structures.
        if extra_results:
            sent_model = self._get_sentence_model()
            for r in extra_results:
                nx_g = r["nx_graph"]
                fp = _graph_fingerprint(nx_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                # Add role embeddings if absent (needed by GraphSnapshot)
                for n in nx_g.nodes():
                    if "role_embedding" not in nx_g.nodes[n]:
                        role = nx_g.nodes[n].get("role", "Unknown")
                        nx_g.nodes[n]["role_embedding"] = sent_model.encode(role)
                self._attach_model_embeddings(nx_g)
                raw_results.append({
                    "task_question":  record["task"],
                    "task_embedding": task_embedding,
                    "graph_snapshot": GraphSnapshot.from_nx(nx_g),
                    "is_correct":     r["is_correct"],
                    "num_nodes":      r["num_nodes"],
                    "num_edges":      nx_g.number_of_edges(),
                    "mode":           r.get("mode", "coldstart"),
                    "domain":         self.domain,
                })

        # --- Default topology grid (fallback only) ---
        if use_default_configs:
            configs = _default_configs(available_roles, min_agents, max_agents,
                                       weak_baselines=self.weak_baselines)
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

            # Run all topology configs concurrently — vLLM handles batching.
            default_results = await asyncio.gather(
                *(self._run_graph(g, record, m, n, task_embedding)
                  for g, m, n in graph_runs),
                return_exceptions=True,
            )
            for r in default_results:
                if not isinstance(r, Exception):
                    raw_results.append(r)

        # --- ARGDesigner model-based candidates (temperature cycling) ---
        # When _default_configs is disabled, generate more samples to compensate.
        # Temperatures are cycled so we get structural diversity even when
        # n_samples > len(temperatures).  seen_fps is pre-seeded with coldstart
        # fingerprints so no duplicate of an existing graph is ever run.
        #
        # Graph *generation* is synchronous CPU work (ARGDesigner forward pass),
        # so we collect all unique structures first, then run LLM inference
        # for all of them concurrently.
        if self.arg_model is not None:
            n_samples = self.arg_model_samples
            if n_samples is None:
                n_samples = (len(self.sample_temperatures)
                             if use_default_configs
                             else len(self.sample_temperatures) * num_sample_for_tasks)

            emb_tensor = torch.tensor(
                task_embedding, device=self.arg_model.args.device
            ).float()

            # Phase A — collect unique graph structures (no I/O, all synchronous).
            unique_arg_graphs: List[Tuple] = []   # (nx_g, label)
            max_attempts = n_samples * 4
            temp_cycle = itertools.cycle(self.sample_temperatures)
            for _ in range(max_attempts):
                if len(unique_arg_graphs) >= n_samples:
                    break
                temp = next(temp_cycle)
                try:
                    graphs = generate_graph(
                        self.arg_model, emb_tensor, self.role_descriptions,
                        temperature=temp,
                    )
                    if not graphs:
                        continue
                    fp = _graph_fingerprint(graphs[0])
                    if fp in seen_fps:
                        continue
                    seen_fps.add(fp)
                    unique_arg_graphs.append((graphs[0], f"arg_model_T{temp:.2f}"))
                except Exception as e:
                    print(f"  [skip arg_model T={temp}] {type(e).__name__}: {e}")

            if len(unique_arg_graphs) < n_samples:
                print(f"  [warn] ARGDesigner: requested {n_samples} unique graphs, "
                      f"got {len(unique_arg_graphs)} after {max_attempts} attempts")

            # Phase B — run all unique graphs concurrently.
            arg_results = await asyncio.gather(
                *(self._run_nx_graph(g, record, label, task_embedding)
                  for g, label in unique_arg_graphs),
                return_exceptions=True,
            )
            for r in arg_results:
                if r is not None and not isinstance(r, Exception):
                    raw_results.append(r)

        # --- Role sweep: exhaustive role-combination configs on a fixed topology ---
        # Runs regardless of coldstart/ARGDesigner availability.  Produces dense
        # role-assignment signal: the reward model learns *which roles work best
        # for each task type*, not just which graph size is smallest.
        # This is especially useful for near-ceiling datasets (GSM8K, SVAMP) where
        # topology barely differentiates correctness but role assignments do.
        if self.role_sweep:
            sweep_configs = _role_sweep_configs(
                available_roles,
                n_agents_list=self.role_sweep_n_agents,
                fixed_topology=self.role_sweep_topology,
                max_combos=self.role_sweep_max_combos,
            )

            # Build unique nx.DiGraph objects (synchronous CPU work).
            sweep_graphs: List[Tuple[nx.DiGraph, str]] = []
            for mode, n, roles in sweep_configs:
                nx_g = _build_topology_graph(mode, roles)
                fp = _graph_fingerprint(nx_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                label = f"role_sweep_{'_'.join(r.replace(' ', '') for r in roles)}"
                sweep_graphs.append((nx_g, label))

            if sweep_graphs:
                sweep_results = await asyncio.gather(
                    *(self._run_nx_graph(g, record, label, task_embedding)
                      for g, label in sweep_graphs),
                    return_exceptions=True,
                )
                for r in sweep_results:
                    if r is not None and not isinstance(r, Exception):
                        raw_results.append(r)

        # --- Pruning phase: run edge-pruned variants of every correct graph ---
        # Pruned graph structures are derived synchronously first (CPU-only),
        # then all LLM inferences are dispatched concurrently.
        if self.pruning_ratio > 0:
            correct_so_far = [r for r in raw_results if isinstance(r, dict) and r["is_correct"]]
            to_prune: List[Tuple] = []   # (pruned_nx_g, label)
            for r in correct_so_far:
                pruned_g = _prune_graph(r["graph_snapshot"].to_nx(), self.pruning_ratio)
                fp = _graph_fingerprint(pruned_g)
                if fp in seen_fps:
                    continue
                seen_fps.add(fp)
                to_prune.append((pruned_g, f"pruned_{r['mode']}"))

            if to_prune:
                pruned_results = await asyncio.gather(
                    *(self._run_nx_graph(g, record, label, task_embedding)
                      for g, label in to_prune),
                    return_exceptions=True,
                )
                for r in pruned_results:
                    if r is not None and not isinstance(r, Exception):
                        raw_results.append(r)

        results = [r for r in raw_results if isinstance(r, dict)]
        skipped = len(raw_results) - len(results)
        n_correct   = sum(1 for r in results if     r["is_correct"])
        n_incorrect = sum(1 for r in results if not r["is_correct"])

        stats = {
            "total":     len(results),
            "correct":   n_correct,
            "incorrect": n_incorrect,
            "skipped":   skipped,
        }

        if len(results) < 2:
            return [], stats

        return create_preference_pairs(results, self.weights, self.pair_margin), stats

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
        coldstart_pool: Optional[Dict[str, List[Dict]]] = None,
        num_sample_for_tasks: int = 1,
        task_concurrency: int = 4,
    ) -> int:
        """
        Collect preference pairs for all tasks, writing .pkl shards to
        *output_dir* every *checkpoint_every* tasks.

        coldstart_pool maps task question → list of pre-loaded graph dicts
        (from ColdStart / Finetune .pt files).  When provided, those graphs
        are injected into the same pairing pool as the LLM-collected graphs,
        enabling cross-source preference pairs for the same task.

        task_concurrency controls how many tasks run concurrently.  Each task
        already issues multiple concurrent LLM requests internally, so this
        multiplies the vLLM throughput without overwhelming the server.

        Returns the total number of preference pairs collected.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Semaphore limits how many tasks run their LLM inference in parallel.
        sem = asyncio.Semaphore(task_concurrency)
        progress = tqdm(total=len(task_records), desc="RLHF collection")

        async def _run_one(record: Dict[str, Any]) -> Tuple[List[PreferencePair], dict]:
            async with sem:
                extra = (coldstart_pool or {}).get(record["task"], [])
                result = await self.collect_for_task(
                    record, min_agents, max_agents,
                    extra_results=extra or None,
                    num_sample_for_tasks=num_sample_for_tasks,
                )
            progress.update(1)
            return result

        # Dispatch all tasks concurrently; results arrive in original order.
        all_results = await asyncio.gather(
            *[_run_one(r) for r in task_records],
            return_exceptions=True,
        )
        progress.close()

        # Cancel any tasks left pending by timed-out HTTP connections.
        # asyncio.wait_for() cancels the inner coroutine on timeout, but httpx /
        # anyio connection-pool tasks can remain alive, causing asyncio.run() to
        # hang indefinitely during event-loop teardown.  Explicitly cancelling
        # them here — before we return — lets the loop close cleanly.
        # Note: by this point all LLM inference is finished; any remaining tasks
        # are lightweight connection-cleanup tasks (milliseconds at most).
        # The 30 s cap prevents a stuck socket close from blocking indefinitely.
        _current = asyncio.current_task()
        _pending = [t for t in asyncio.all_tasks() if t is not _current]
        if _pending:
            for t in _pending:
                t.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*_pending, return_exceptions=True),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                pass  # abandon tasks that didn't respond to cancel in time

        # Process results in order so sharding mirrors checkpoint_every grouping.
        shard_idx  = 0
        buffer: List[PreferencePair] = []
        total_pairs = 0
        cum_total = cum_correct = cum_incorrect = cum_skipped = 0

        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                print(f"  Task {i+1} error: {type(result).__name__}: {result}")
                continue

            pairs, stats = result
            buffer.extend(pairs)
            total_pairs += len(pairs)
            cum_total     += stats["total"]
            cum_correct   += stats["correct"]
            cum_incorrect += stats["incorrect"]
            cum_skipped   += stats["skipped"]

            print(
                f"  Task {i+1}/{len(task_records)}: "
                f"graphs={stats['total']} "
                f"(correct={stats['correct']}, incorrect={stats['incorrect']}"
                + (f", skipped={stats['skipped']}" if stats["skipped"] else "")
                + f")  +{len(pairs)} pairs  (total pairs: {total_pairs})"
            )

            if (i + 1) % checkpoint_every == 0 or (i + 1) == len(task_records):
                if buffer:
                    path = os.path.join(output_dir, f"shard_{shard_idx:04d}.pkl")
                    with open(path, "wb") as f:
                        pickle.dump(buffer, f)
                    print(f"  Shard {shard_idx}: saved {len(buffer)} pairs → {path}")
                    shard_idx += 1
                    buffer = []

        correct_rate = cum_correct / cum_total * 100 if cum_total else 0.0
        print(
            f"\n── Collection complete ──────────────────────────────\n"
            f"  Tasks processed  : {len(task_records)}\n"
            f"  Graphs evaluated : {cum_total}  "
            f"(correct={cum_correct} [{correct_rate:.1f}%], "
            f"incorrect={cum_incorrect}, skipped={cum_skipped})\n"
            f"  Preference pairs : {total_pairs} in {shard_idx} shards\n"
            f"────────────────────────────────────────────────────"
        )
        return total_pairs
