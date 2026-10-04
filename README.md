# MASModel

`MASModel` is a framework for measuring how single-agent and multi-agent LLM systems scale on tool-use, reasoning, and partially observable computation-graph tasks. The canonical repository is [time-cat/MASModel](https://github.com/time-cat/MASModel).

The current checkout declares package version `0.1.0`, requires Python 3.11 or newer, and pins the development interpreter to 3.11 in `.python-version`. Earlier documentation referred to `ybkim95/agent-scaling` and `v2.1.3`.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first, then run these commands from the `MASModel` directory:

```bash
git clone https://github.com/time-cat/MASModel.git
cd MASModel
uv sync --locked

# Optional: install the extra dependency used by the SWE-bench loader.
uv sync --locked --extra swebench

cp .env.example .env       # PowerShell: Copy-Item .env.example .env
# Edit .env and add at least one LLM provider key.
```

Use `uv run ...` below so activation is not required. If you prefer an activated environment, use `source .venv/bin/activate` on Linux/macOS or `.venv\\Scripts\\Activate.ps1` in PowerShell. The base project does not declare `flash-attn`; installing it is not part of the supported base setup. BrowseComp-Plus additionally needs a compatible FAISS/Tevatron/PyTorch retrieval setup and local index, as described in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

Required keys are provider-specific. The checked-in `.env.example` documents `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `ANTHROPIC_API_KEY`; `TAVILY_API_KEY` is required by Finance-Agent's `web_search` tool. LangFuse keys are optional and only needed when `log_langfuse=true`.

## Quick start

Run a one-instance smoke test against the checked-in synthetic DAG data:

```bash
uv run python scripts/run_experiment.py \
  agent=single-agent \
  dataset=synthetic-dag \
  dataset.local_path=datasets/synthetic_dag_240.json \
  llm.model=openai/gpt-5-mini \
  max_instances=1 \
  num_workers=1
```

The default configuration in `run_conf/run_exp.yaml` is centralized multi-agent on PlanCraft with `debug=true` and `max_instances=3`. Any Hydra value can be overridden on the command line:

```bash
uv run python scripts/run_experiment.py agent=single-agent dataset=plancraft-test max_instances=5
uv run python scripts/run_experiment.py agent=multi-agent-centralized dataset=synthetic-dag dataset.local_path=datasets/synthetic_dag_240.json agent.n_base_agents=3
uv run python scripts/run_experiment.py llm.model=openai/gpt-5-mini num_workers=4 max_instances=10
```

The available agent configs are:

| Config | Behavior |
|---|---|
| `single-agent` | One tool-using agent |
| `multi-agent-independent` | Independent workers with `synthesis_only` aggregation |
| `multi-agent-centralized` | Lead agent with orchestrated subagents |
| `multi-agent-decentralized` | Peer debate with a 70% consensus threshold |
| `multi-agent-hybrid` | Lead coordination with peer communication |
| `direct-prompt` | Direct prompt baseline |

The available dataset config names are `plancraft-test`, `browsecomp-plus`, `synthetic-dag`, `swebench-verified`, `terminalbench`, and (after setup) `finance-agent` and `workbench`. See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for exact local paths, counts, and acquisition steps.

## Reproducing the multi-agent sweeps

The files in `bash/` are executable shell scripts despite their `.txt` suffix. They invoke `python` directly, so activate `.venv` first (or otherwise put the project environment on `PATH`). Run them from the repository root with the `bash/` prefix:

```bash
bash bash/multiagent_budget_experiments.txt
bash bash/multiagent_budget_experiments_plancraft_.txt
bash bash/multiagent_budget_experiments_plancraft.txt
bash bash/multiagent_budget_experiments_dag_stru_wide.txt
bash bash/multiagent_budget_experiments_dag_stru_chain.txt
bash bash/multiagent_budget_experiments_dag_stru_balance.txt
```

Their current roles are:

- `multiagent_budget_experiments.txt`: synthetic DAG budget law, verification, horizon, and arithmetic-difficulty controls; it uses `datasets/threshold_families/threshold-3-5-7-spread.json`.
- `multiagent_budget_experiments_plancraft_.txt`: the larger PlanCraft budget sweep (`max_instances=100`). The trailing underscore is part of the filename.
- `multiagent_budget_experiments_plancraft.txt`: a shorter PlanCraft sweep (`max_instances=60`).
- `multiagent_budget_experiments_dag_stru_wide.txt`, `_chain.txt`, and `_balance.txt`: structure-specific synthetic DAG sweeps using `threshold-wide.json`, `threshold-chain.json`, and `threshold-balanced.json` respectively (`max_instances=60`).

Every sweep sets `agent.total_decision_budget` explicitly. For a team of `n` workers, each worker receives `floor(total_decision_budget / n)` lifetime decision calls; this budget is not reset between communication rounds. Planning, coordination, synthesis, and optional verification calls are tracked separately. The scripts use `openai/qwen-flash` as their configured model identifier; replace it with a model available through your LiteLLM provider if that identifier is not available in your account.

## Dataset setup

For the two upstream-converted datasets, activate `.venv` first because these scripts invoke `python` directly:

```bash
bash scripts/setup_finance_agent.sh
bash scripts/setup_workbench.sh
```

For SWE-bench Verified and Terminal-Bench, obtain the upstream JSON files and place them at the paths in [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md). Both environments require Docker. BrowseComp-Plus also requires its retrieval index and the optional retrieval stack; the checked-in JSON alone is not sufficient to initialize the FAISS search environment.

## Outputs and configuration

Hydra writes each run below:

```text
exp_outputs/{dataset_id}/{agent_name}/{llm.model}/{date}/{time}/
├── run_config.yaml
├── run.log
├── dataset_eval_metrics.json
└── instance_runs/
    └── 0000/
        └── instance_save.yaml
```

The resolved configuration is saved in `run_config.yaml`; `run.log` contains the execution trace; `dataset_eval_metrics.json` contains aggregate metrics; and `instance_runs/` contains per-instance records. Keep the generated directory outside version control.

Key configuration files are under `run_conf/agent/`, `run_conf/dataset/`, and `run_conf/run_exp.yaml`. The canonical multi-agent parameters are `n_base_agents`, `total_decision_budget`, `min_iterations_per_agent`, and, for coordinated protocols, `max_rounds`. The centralized, decentralized, and hybrid defaults use three workers and a ten-round communication horizon; independent workers do not communicate.

## Development checks

Run the repository tests after installation:

```bash
uv run pytest -q
```

The current test suite covers the decentralized debate, independent synthesis, result selection, synthetic DAG generator/environment, and Finance-Agent/WorkBench adapters. The changelog records 28 tests for the current integration set.

## Citation and license

If you use the research code, cite the accompanying work and record the exact Git commit used for your experiments. The repository is released under the MIT License; see [`LICENSE`](LICENSE).
