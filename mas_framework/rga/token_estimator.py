"""
Heuristic token cost estimator for multi-agent graph executions.

Each agent call costs approximately:
  system_prompt_tokens + user_prompt_base_tokens
  + in_degree * per_predecessor_context_tokens
  + response_tokens

These constants are calibrated from typical prompt/response lengths observed
with instruction-tuned 7-13B models on math reasoning tasks.
"""

import math
import networkx as nx
from typing import Dict

# Per-role token budget estimates (system + base user prompt + typical response)
_ROLE_ESTIMATES: Dict[str, Dict[str, int]] = {
    "Math Solver": {
        "system": 80,
        "user_base": 150,
        "per_predecessor": 100,
        "response": 200,
    },
    "Mathematical Analyst": {
        "system": 120,
        "user_base": 150,
        "per_predecessor": 150,
        "response": 350,
    },
    "Programming Expert": {
        "system": 200,
        "user_base": 150,
        "per_predecessor": 200,
        "response": 500,
    },
    "Inspector": {
        "system": 150,
        "user_base": 150,
        "per_predecessor": 200,
        "response": 300,
    },
    # Catch-all
    "_default": {
        "system": 100,
        "user_base": 150,
        "per_predecessor": 150,
        "response": 300,
    },
}

# Decision / aggregation node cost
_DECISION_ESTIMATES = {
    "system": 100,
    "user_base": 100,
    "per_agent_output": 150,
    "response": 100,
}


def estimate_tokens(graph: nx.DiGraph, num_rounds: int = 1) -> int:
    """
    Estimate total token usage for one complete execution of *graph*.

    Args:
        graph:      NetworkX DiGraph with ``role`` attribute on each node.
        num_rounds: Number of spatial-communication rounds.

    Returns:
        Integer estimate of total tokens consumed.
    """
    total = 0
    for node in graph.nodes():
        role = graph.nodes[node].get("role", "_default")
        est = _ROLE_ESTIMATES.get(role, _ROLE_ESTIMATES["_default"])
        in_deg = graph.in_degree(node)
        node_tokens = (
            est["system"]
            + est["user_base"]
            + in_deg * est["per_predecessor"]
            + est["response"]
        )
        total += node_tokens * num_rounds

    # Decision / final-refer node (aggregates all agent outputs)
    num_agents = graph.number_of_nodes()
    decision_tokens = (
        _DECISION_ESTIMATES["system"]
        + _DECISION_ESTIMATES["user_base"]
        + num_agents * _DECISION_ESTIMATES["per_agent_output"]
        + _DECISION_ESTIMATES["response"]
    )
    total += decision_tokens
    return total


def estimate_tokens_from_snapshot(snapshot, num_rounds: int = 1) -> int:
    """Estimate tokens directly from a :class:`GraphSnapshot`."""
    return estimate_tokens(snapshot.to_nx(), num_rounds)
