import json
import subprocess
from pathlib import Path

from agent_scaling.datasets.base import DatasetInstanceOutput
from agent_scaling.datasets.synthetic_dag import SyntheticDAGDataset, SyntheticDAGInstance
from agent_scaling.env.synthetic_dag import SyntheticDAGEnvironment


def test_generator_has_reproducible_200_plus_corpus(tmp_path):
    output = tmp_path / "synthetic.json"
    subprocess.run(
        ["python", "scripts/generate_synthetic_dag.py", "--output", str(output)],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["dataset_id"] == "synthetic_dag"
    assert len(payload["instances"]) == 240
    assert len({x["task_id"] for x in payload["instances"]}) == 240
    assert {x["metadata"]["structure"] for x in payload["instances"]} == {
        "chain", "wide", "balanced"
    }


def test_generator_can_control_theoretical_budget_distribution(tmp_path):
    output = tmp_path / "threshold.json"
    subprocess.run(
        [
            "python",
            "scripts/generate_synthetic_dag.py",
            "--num-instances",
            "200",
            "--seed",
            "7",
            "--budget-distribution",
            "8:0.25,16:0.75",
            "--structure-distribution",
            "chain:0.5,wide:0.5",
            "--output",
            str(output),
        ],
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    b0 = [x["metadata"]["theoretical_min_budget"] for x in payload["instances"]]
    assert set(b0) == {8, 16}
    assert sum(x == 8 for x in b0) + sum(x == 16 for x in b0) == 200
    assert all(x["metadata"]["num_nodes"] + 1 == x["metadata"]["theoretical_min_budget"] for x in payload["instances"])


def test_difficulty_profiles_preserve_b0_and_change_rule(tmp_path):
    outputs = {}
    for difficulty in ("easy", "medium", "hard"):
        output = tmp_path / f"{difficulty}.json"
        subprocess.run(
            [
                "python",
                "scripts/generate_synthetic_dag.py",
                "--num-instances",
                "200",
                "--seed",
                "11",
                "--budget-distribution",
                "8:1.0",
                "--difficulty",
                difficulty,
                "--family-name",
                f"threshold-8-{difficulty}",
                "--output",
                str(output),
            ],
            check=True,
        )
        outputs[difficulty] = json.loads(output.read_text(encoding="utf-8"))

    assert all(
        {x["metadata"]["theoretical_min_budget"] for x in payload["instances"]}
        == {8}
        for payload in outputs.values()
    )
    assert {
        x["metadata"]["difficulty"] for x in outputs["medium"]["instances"]
    } == {"medium"}
    assert {
        x["metadata"]["difficulty_rule"] for x in outputs["hard"]["instances"]
    } == {"nonlinear-product-v1"}
    assert {
        x["metadata"]["difficulty_rule"] for x in outputs["easy"]["instances"]
    } == {"linear-v1"}


def test_threshold_family_batch_generator(tmp_path):
    output_dir = tmp_path / "families"
    subprocess.run(
        [
            "python",
            "scripts/generate_threshold_families.py",
            "--output-dir",
            str(output_dir),
            "--num-instances",
            "200",
            "--families",
            "small=8:1.0;large=16:1.0",
        ],
        check=True,
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["families"]) == {"small", "large"}
    for name, expected in [("small", 8), ("large", 16)]:
        payload = json.loads((output_dir / f"{name}.json").read_text(encoding="utf-8"))
        assert {x["metadata"]["theoretical_min_budget"] for x in payload["instances"]} == {expected}


def test_dataset_exact_integer_grader():
    instance = SyntheticDAGInstance(
        task_id="dag-test",
        nodes=["N0"],
        edges=[],
        node_values={"N0": 7},
        target_score=7,
    )
    dataset = SyntheticDAGDataset(dataset_id="synthetic_dag", instances=[instance])
    correct = dataset.get_instance_eval_metrics(
        DatasetInstanceOutput(data_instance=instance, agent_output="Final Answer: 7")
    )
    wrong = dataset.get_instance_eval_metrics(
        DatasetInstanceOutput(data_instance=instance, agent_output="Final Answer: 8")
    )
    assert correct["success"] is True
    assert correct["theoretical_min_budget"] == 2
    assert correct["budget_family"] == "default"
    assert wrong["success"] is False

    noisy = dataset.get_instance_eval_metrics(
        DatasetInstanceOutput(
            data_instance=instance,
            agent_output="Inspected N0 with value 7. Final Answer: 7",
        )
    )
    assert noisy["submitted_score"] == 7
    assert noisy["success"] is True

    missing = dataset.get_instance_eval_metrics(
        DatasetInstanceOutput(data_instance=instance, agent_output="ERROR: no submission")
    )
    assert missing["submitted_score"] == -1
    assert missing["submitted"] is False
    assert missing["success"] is False

    explicit_invalid = dataset.get_instance_eval_metrics(
        DatasetInstanceOutput(
            data_instance=instance,
            agent_output="I inspected N3 but have not finished.",
            canonical_submission=None,
        )
    )
    assert explicit_invalid["submitted_score"] == -1
    assert explicit_invalid["submitted"] is False
    assert explicit_invalid["success"] is False


def test_environment_enforces_dependencies_and_submit():
    instance = SyntheticDAGInstance(
        task_id="dag-test",
        nodes=["N0", "N1"],
        edges=[["N0", "N1"]],
        node_values={"N0": 3, "N1": 4},
        target_score=17,
    )
    env = SyntheticDAGEnvironment(dataset_instance=instance, tools=["inspect_node", "submit"])
    blocked = env.inspect_node.invoke({"node_id": "N1"})
    assert "BLOCKED" in str(blocked)
    first = env.inspect_node.invoke({"node_id": "N0"})
    assert "value=3" in str(first)
    final = env.submit.invoke({"answer": 17})
    assert "FINAL_SCORE: 17" in str(final)
    assert env.env_done() is True

    auto = SyntheticDAGEnvironment(dataset_instance=instance, tools=["inspect_node", "submit"])
    auto_result = auto.submit.invoke({"reasoning": "Auto-submit: step budget exhausted"})
    assert "FINAL_SCORE: -1" in str(auto_result)
    assert auto.env_done() is True
    assert auto.success is False

    labelled = SyntheticDAGEnvironment(
        dataset_instance=instance, tools=["inspect_node", "submit"]
    )
    labelled_result = labelled.submit.invoke(
        {"reasoning": "Observed b0=3. Final Answer: 17"}
    )
    assert "FINAL_SCORE: 17" in str(labelled_result)
    assert labelled.success is True


def test_worker_environments_do_not_leak_state():
    instance = SyntheticDAGInstance(
        task_id="dag-shared",
        nodes=["N0", "N1"],
        edges=[["N0", "N1"]],
        node_values={"N0": 3, "N1": 4},
        target_score=17,
    )
    worker_a = SyntheticDAGEnvironment(dataset_instance=instance, tools=["inspect_node", "submit"])
    worker_b = SyntheticDAGEnvironment(dataset_instance=instance, tools=["inspect_node", "submit"])
    worker_a.inspect_node.invoke({"node_id": "N0"})
    result = worker_b.inspect_node.invoke({"node_id": "N1"})
    assert "BLOCKED" in str(result)
