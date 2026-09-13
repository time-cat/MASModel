"""Scoring rules for the synthetic DAG difficulty families.

The three profiles keep the same structural budget threshold: a task with
``n`` hidden nodes needs ``n`` inspections and one submission.  Difficulty is
changed only through the arithmetic required after the observations have been
collected, so it does not silently change the theoretical minimum budget.
"""

from __future__ import annotations

from typing import Any, Mapping


DIFFICULTY_PROFILES: dict[str, dict[str, Any]] = {
    "easy": {
        "node_value_max": 71,
        "node_modulus": 1007,
        "score_modulus": 1009,
        "rule_id": "linear-v1",
        "rule_text": (
            "For every node v, let P_v be the sum of the observed base "
            "values b_u of its immediate predecessors. Compute "
            "x_v = (2*(b_v+1)^2 + b_v + P_v^2 + 3) mod 1007. Root nodes have "
            "P_v=0. The final score is sum((i+1)*x_v_i) mod 1009, with "
            "nodes indexed from i=0."
        ),
    },
    "medium": {
        "node_value_max": 71,
        "node_modulus": 1007,
        "score_modulus": 1009,
        "rule_id": "base-quadratic-v2",
        "rule_text": (
            "For every node v, let P_v be the sum of the observed base "
            "values b_u of its immediate predecessors. Compute "
            "x_v = (2*b_v^3 + b_v^2 + P_v^2 + 7) mod 1007. Root nodes have "
            "P_v=0. The final score is sum((i+1)*x_v_i) mod 1009, with "
            "nodes indexed from i=0."
        ),
    },
    "hard": {
        "node_value_max": 71,
        "node_modulus": 1007,
        "score_modulus": 1009,
        "rule_id": "base-mixed-quadratic-v2",
        "rule_text": (
            "For every node v, let P_v be the sum of the observed base "
            "values b_u of its immediate predecessors and Q_v the sum of "
            "their squares. Compute "
            "x_v = (2*(b_v+1)^3 + 5*b_v^2 + 4*(b_v+3) + 3*(P_v+2)^2 + 2*Q_v + 2) mod 1007. Root nodes "
            "have P_v=0 and Q_v=0. The final score is "
            "sum((i+1)*x_v_i + (i+3)*b_v_i) mod 1009, with nodes "
            "indexed from i=0."
        ),
    },
}


def get_difficulty_profile(difficulty: str) -> dict[str, Any]:
    """Return a copy of a supported profile."""
    try:
        return dict(DIFFICULTY_PROFILES[difficulty])
    except KeyError as exc:
        supported = ", ".join(sorted(DIFFICULTY_PROFILES))
        raise ValueError(
            f"Unknown synthetic DAG difficulty {difficulty!r}; "
            f"expected one of: {supported}"
        ) from exc


def compute_resolved_values(
    nodes: list[str],
    edges: list[list[str]],
    node_values: Mapping[str, int],
    difficulty: str = "easy",
) -> dict[str, int]:
    """Compute the hidden resolved values under one difficulty profile."""
    profile = get_difficulty_profile(difficulty)
    node_modulus = int(profile["node_modulus"])
    predecessors: dict[str, list[str]] = {node: [] for node in nodes}
    for source, target in edges:
        predecessors[target].append(source)

    resolved: dict[str, int] = {}
    for node in nodes:
        parents = predecessors[node]
        base_parent_values = [node_values[parent] for parent in parents]
        base_parent_sum = sum(base_parent_values)
        if difficulty == "easy":
            base = node_values[node]
            value = 2 * (base + 1) ** 2 + base + base_parent_sum ** 2 + 3
        elif difficulty == "medium":
            base = node_values[node]
            value = 2 * base ** 3 + base ** 2 + base_parent_sum ** 2 + 7
        else:
            base_parent_squares = sum(value * value for value in base_parent_values)
            base = node_values[node]
            value = (
                2 * (base + 1) ** 3 + 5 * base ** 2 + 4 * (base + 3)
                + 3 * (base_parent_sum + 2) ** 2
                + 2 * base_parent_squares + 2
            )
        resolved[node] = value % node_modulus
    return resolved


def compute_target_score(
    nodes: list[str],
    edges: list[list[str]],
    node_values: Mapping[str, int],
    difficulty: str = "easy",
) -> int:
    """Compute the exact target for a generated or loaded instance."""
    profile = get_difficulty_profile(difficulty)
    resolved = compute_resolved_values(nodes, edges, node_values, difficulty)
    score_modulus = int(profile["score_modulus"])
    if difficulty == "easy":
        score = sum((index + 1) * resolved[node] for index, node in enumerate(nodes))
    elif difficulty == "medium":
        score = sum((index + 1) * resolved[node] for index, node in enumerate(nodes))
    else:
        score = sum(
            (index + 1) * resolved[node]
            + (index + 3) * node_values[node]
            for index, node in enumerate(nodes)
        )
    return score % score_modulus
