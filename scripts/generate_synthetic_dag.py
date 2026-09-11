"""Generate reproducible synthetic DAG instances for coordination studies.

The default corpus contains 240 instances, exceeding the project's minimum
200-instance requirement. ``--seed`` and ``--num-instances`` allow controlled
replication and structural ablations.

For threshold experiments, ``--budget-distribution`` can be used to prescribe
the distribution of the theoretical minimum single-agent budget.  A task with
``n`` nodes requires ``n`` inspections plus one submission, so its structural
threshold is ``B0=n+1``.  For example::

    --budget-distribution 8:0.2,12:0.5,16:0.3

generates a family whose theoretical minimum budgets are sampled from those
three values with the given probabilities.  The distribution is recorded in
each instance's metadata, making the true threshold available for simulation
studies without leaking any model outcome into the generator.

The ``--difficulty`` option selects an arithmetic profile (``easy``,
``medium``, or ``hard``). All profiles preserve the same structural threshold
for a fixed node count; only the computation after node observations changes.
"""

import argparse
import importlib.util
import json
import random
from pathlib import Path
from typing import Mapping

# Keep the generator directly executable as ``python scripts/...`` from the
# project checkout, where Python otherwise only adds ``scripts/`` to sys.path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES_PATH = PROJECT_ROOT / "agent_scaling" / "datasets" / "synthetic_dag_rules.py"
RULES_SPEC = importlib.util.spec_from_file_location(
    "synthetic_dag_rules_standalone",
    RULES_PATH,
)
if RULES_SPEC is None or RULES_SPEC.loader is None:
    raise ImportError(f"Unable to load synthetic DAG rules from {RULES_PATH}")
RULES_MODULE = importlib.util.module_from_spec(RULES_SPEC)
RULES_SPEC.loader.exec_module(RULES_MODULE)
compute_target_score = RULES_MODULE.compute_target_score
get_difficulty_profile = RULES_MODULE.get_difficulty_profile


def _longest_path(nodes: list[str], edges: list[list[str]]) -> int:
    preds = {n: [] for n in nodes}
    for u, v in edges:
        preds[v].append(u)
    depth: dict[str, int] = {}
    for node in nodes:
        depth[node] = 1 + max((depth[p] for p in preds[node]), default=0)
    return max(depth.values(), default=1)


def _parse_distribution(spec: str, *, name: str) -> dict[str, float]:
    """Parse ``key:weight,key:weight`` into normalized nonnegative weights."""
    result: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            key, weight = item.split(":", 1)
            value = float(weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {name} item {item!r}; expected key:weight"
            ) from exc
        if value < 0:
            raise ValueError(f"{name} weights must be nonnegative")
        result[key.strip()] = result.get(key.strip(), 0.0) + value
    if not result or sum(result.values()) <= 0:
        raise ValueError(f"{name} must contain at least one positive weight")
    total = sum(result.values())
    return {key: value / total for key, value in result.items()}


def _sample_keys(
    rng: random.Random,
    distribution: Mapping[str, float],
    count: int,
) -> list[str]:
    """Create an exact-size stratified sample from a categorical distribution.

    The largest-remainder allocation makes the requested family proportions
    deterministic up to the final integer rounding, avoiding an unnecessary
    source of variation between independently generated families.
    """
    keys = list(distribution)
    expected = [distribution[key] * count for key in keys]
    counts = [int(value) for value in expected]
    for index in sorted(
        range(len(keys)),
        key=lambda i: expected[i] - counts[i],
        reverse=True,
    )[: count - sum(counts)]:
        counts[index] += 1
    sampled = [key for key, amount in zip(keys, counts) for _ in range(amount)]
    rng.shuffle(sampled)
    return sampled


def _layer_partition(num_nodes: int, width: int) -> list[list[str]]:
    """Partition topologically ordered node names into nonempty layers."""
    width = max(1, min(width, num_nodes))
    levels = (num_nodes + width - 1) // width
    layers: list[list[str]] = []
    cursor = 0
    for level in range(levels):
        remaining = num_nodes - cursor
        slots_left = levels - level
        layer_size = min(width, remaining - (slots_left - 1))
        layers.append([f"N{i}" for i in range(cursor, cursor + layer_size)])
        cursor += layer_size
    return layers


def _choose_structure(index: int, rng: random.Random, requested: str | None) -> str:
    if requested is not None:
        if requested not in {"chain", "wide", "balanced"}:
            raise ValueError("structure must be chain, wide, or balanced")
        return requested
    return ["chain", "wide", "balanced"][index % 3]


def make_instance(
    index: int,
    rng: random.Random,
    *,
    num_nodes: int | None = None,
    structure: str | None = None,
    difficulty: str = "easy",
    family_name: str = "budgeted-dag-v1",
) -> dict:
    profile = get_difficulty_profile(difficulty)
    mode_name = _choose_structure(index, rng, structure)
    mode = {"chain": 0, "wide": 1, "balanced": 2}[mode_name]
    # Chain, wide, and balanced families make parallelism and critical path
    # independently observable in the final scaling analysis.
    if num_nodes is None:
        if mode == 0:
            width, levels = 1, rng.randint(6, 10)
        elif mode == 1:
            width, levels = rng.randint(3, 5), rng.randint(2, 3)
        else:
            width, levels = rng.randint(2, 4), rng.randint(3, 5)
        num_nodes = width * levels
    else:
        if num_nodes < 1:
            raise ValueError("num_nodes must be positive")
        if mode == 0:
            width = 1
        elif mode == 1:
            width = rng.randint(3, 5)
        else:
            width = rng.randint(2, 4)
    nodes = [f"N{i}" for i in range(num_nodes)]
    level_nodes = _layer_partition(num_nodes, width)
    edges: list[list[str]] = []
    for level in range(1, len(level_nodes)):
        previous = level_nodes[level - 1]
        for node in level_nodes[level]:
            parent = rng.choice(previous)
            edges.append([parent, node])
            if len(previous) > 1 and rng.random() < 0.45:
                alternatives = [p for p in previous if p != parent]
                edges.append([rng.choice(alternatives), node])
    values = {
        node: rng.randint(1, int(profile["node_value_max"]))
        for node in nodes
    }
    target = compute_target_score(nodes, edges, values, difficulty)
    critical_path = _longest_path(nodes, edges)
    return {
        "task_id": f"dag-{index:04d}",
        "nodes": nodes,
        "edges": edges,
        "node_values": values,
        "target_score": target,
        "parallelism": round(1.0 - critical_path / len(nodes), 4),
        "critical_path": critical_path,
        "metadata": {
            "generator": family_name,
            "difficulty": difficulty,
            "difficulty_rule": profile["rule_id"],
            "instance_index": index,
            "structure": mode_name,
            "num_tools": 2,
            "num_nodes": len(nodes),
            # Single-agent exact-score solving requires one inspection per
            # node and one final submit call.  This is a structural lower
            # bound, not an observed model trajectory statistic.
            "theoretical_min_budget": len(nodes) + 1,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-instances", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic_dag_240.json"))
    parser.add_argument(
        "--budget-distribution",
        type=str,
        default=None,
        help=(
            "Optional B0:weight distribution, e.g. "
            "8:0.2,12:0.5,16:0.3. B0 must be at least 2."
        ),
    )
    parser.add_argument(
        "--structure-distribution",
        type=str,
        default=None,
        help="Optional structure:weight distribution over chain,wide,balanced.",
    )
    parser.add_argument(
        "--family-name",
        type=str,
        default="budgeted-dag-v1",
        help="Generator/family identifier stored in metadata.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="easy",
        choices=("easy", "medium", "hard"),
        help="Arithmetic difficulty profile; does not change the structural B0.",
    )
    args = parser.parse_args()
    if args.num_instances < 200:
        raise ValueError("Synthetic DAG corpus must contain at least 200 instances")
    rng = random.Random(args.seed)
    budget_distribution = (
        _parse_distribution(args.budget_distribution, name="budget-distribution")
        if args.budget_distribution
        else None
    )
    if budget_distribution is not None:
        for key in budget_distribution:
            if not key.isdigit() or int(key) < 2:
                raise ValueError("B0 values must be integer budgets >= 2")
    structure_distribution = (
        _parse_distribution(args.structure_distribution, name="structure-distribution")
        if args.structure_distribution
        else None
    )
    if structure_distribution is not None:
        unknown = set(structure_distribution) - {"chain", "wide", "balanced"}
        if unknown:
            raise ValueError(f"Unknown structures: {sorted(unknown)}")
    b0_keys = (
        _sample_keys(rng, budget_distribution, args.num_instances)
        if budget_distribution
        else [None] * args.num_instances
    )
    structure_keys = (
        _sample_keys(rng, structure_distribution, args.num_instances)
        if structure_distribution
        else [None] * args.num_instances
    )
    instances = []
    for i, (b0_key, requested_structure) in enumerate(zip(b0_keys, structure_keys)):
        b0 = int(b0_key) if b0_key is not None else None
        instance = make_instance(
            i,
            rng,
            num_nodes=b0 - 1 if b0 is not None else None,
            structure=requested_structure,
            difficulty=args.difficulty,
            family_name=args.family_name,
        )
        if budget_distribution is not None:
            instance["metadata"]["budget_family"] = args.family_name
            instance["metadata"]["budget_distribution"] = budget_distribution
        instances.append(instance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"dataset_id": "synthetic_dag", "instances": instances}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {len(instances)} instances to {args.output}")


if __name__ == "__main__":
    main()
