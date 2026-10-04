# Reproduction Guide

This guide reproduces the experiments supported by the current `time-cat/MASModel` checkout. Run every command from the repository root (`MASModel/`). The code uses Hydra; command-line assignments such as `dataset=synthetic-dag` override the YAML files under `run_conf/`.

## 1. Pin the code and install dependencies

The project is currently version `0.1.0` and requires Python 3.11+.

```bash
git clone https://github.com/time-cat/MASModel.git
cd MASModel
# For a reproducible record, save the output of this command.
git rev-parse HEAD

uv sync --locked
# Needed only for SWE-bench support:
uv sync --locked --extra swebench

cp .env.example .env       # PowerShell: Copy-Item .env.example .env
```

The lock file and `pyproject.toml` define the supported base environment. `flash-attn` is not a project dependency and is not required by the base setup. BrowseComp-Plus is an optional retrieval environment whose source imports FAISS, Tevatron, PyTorch, and Transformers; install a compatible retrieval stack and prepare its local index separately before using that benchmark.

Use `uv run python ...` and `uv run pytest ...` in the commands below. This works on Linux, macOS, and Windows without relying on `source`. If you activate manually, use `.venv/bin/activate` on POSIX systems or `.venv\\Scripts\\Activate.ps1` in PowerShell.

## 2. Configure API keys

Copying `.env.example` creates the supported variable names:

```dotenv
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
TAVILY_API_KEY=
# Optional LangFuse tracing:
# LANGFUSE_HOST=https://us.cloud.langfuse.com
# LANGFUSE_SECRET_KEY=
# LANGFUSE_PUBLIC_KEY=
```

At least one LLM provider key is needed. `TAVILY_API_KEY` is required for Finance-Agent's web-search tool. Set `log_langfuse=true` only when the LangFuse variables are configured.

## 3. Verify a minimal run

The checked-in synthetic DAG corpus is self-contained and avoids Docker and external benchmark downloads:

```bash
uv run python scripts/run_experiment.py \
  agent=single-agent \
  dataset=synthetic-dag \
  dataset.local_path=datasets/synthetic_dag_240.json \
  llm.model=openai/gpt-5-mini \
  max_instances=1 \
  num_workers=1
```

For a multi-agent smoke test:

```bash
uv run python scripts/run_experiment.py \
  agent=multi-agent-centralized \
  dataset=synthetic-dag \
  dataset.local_path=datasets/synthetic_dag_240.json \
  agent.n_base_agents=3 \
  agent.total_decision_budget=9 \
  agent.max_rounds=3 \
  llm.model=openai/gpt-5-mini \
  max_instances=1
```

The default synthetic config points to `datasets/synthetic_dag_240.json`, so the `dataset.local_path` override is shown explicitly for clarity.

## 4. Prepare datasets

The current checkout contains:

- `datasets/plancraft-test.json` (580 PlanCraft test instances);
- `datasets/browsecomp_plus_sampled_100.json` (100 BrowseComp-Plus instances);
- `datasets/synthetic_dag_240.json` (240 synthetic DAG instances);
- threshold-family JSON files under `datasets/threshold_families/` and `datasets/threshold_families_structure/`.

The following files are expected but are not checked in:

| Dataset | Config | Expected file | Extra requirement |
|---|---|---|---|
| SWE-bench Verified | `run_conf/dataset/swebench-verified.yaml` | `datasets/swebench-verified.json` | Docker and `uv sync --extra swebench` |
| Terminal-Bench | `run_conf/dataset/terminalbench.yaml` | `datasets/terminalbench.json` | Docker |

Obtain those two JSON files from their upstream benchmark releases. For Finance-Agent and WorkBench, activate `.venv` first because the setup scripts invoke `python` directly, then run the repository's converters instead of manually copying JSON:

```bash
bash scripts/setup_finance_agent.sh
bash scripts/setup_workbench.sh
```

The first script writes `datasets/finance_agent.json` and an active `run_conf/dataset/finance-agent.yaml` from its template. The second writes the 100-task `datasets/workbench.json`, the full `datasets/workbench_full_690.json`, and an active WorkBench config. Both scripts clone into `third_party/` by default and accept `--upstream-dir` and `--out` overrides.

BrowseComp-Plus requires more than its checked-in question JSON: the FAISS searcher also loads a precomputed index, an embedding model, and the BrowseComp corpus. Its default index path in the current code is machine-specific, so configure a valid local retrieval environment before attempting that benchmark.

## 5. Reproduce the checked-in sweeps

The multi-agent sweeps are shell scripts stored in `bash/` with `.txt` extensions. From `MASModel/`, run:

```bash
bash bash/multiagent_budget_experiments.txt
bash bash/multiagent_budget_experiments_plancraft_.txt
bash bash/multiagent_budget_experiments_plancraft.txt
bash bash/multiagent_budget_experiments_dag_stru_wide.txt
bash bash/multiagent_budget_experiments_dag_stru_chain.txt
bash bash/multiagent_budget_experiments_dag_stru_balance.txt
```

The scripts currently use these datasets and caps:

| Script | Dataset path | Config name | `max_instances` | Purpose |
|---|---|---|---:|---|
| `multiagent_budget_experiments.txt` | `datasets/threshold_families/threshold-3-5-7-spread.json` | `synthetic-dag` | 100 | Fixed per-agent/system budgets, horizon, verification, and arithmetic-difficulty controls |
| `multiagent_budget_experiments_plancraft_.txt` | `datasets/plancraft-test.json` | `plancraft-test` | 100 | Larger PlanCraft budget sweep and ablations |
| `multiagent_budget_experiments_plancraft.txt` | `datasets/plancraft-test.json` | `plancraft-test` | 60 | Shorter PlanCraft sweep |
| `multiagent_budget_experiments_dag_stru_wide.txt` | `datasets/threshold_families_structure/threshold-wide.json` | `synthetic-dag` | 60 | Wide DAG structure |
| `multiagent_budget_experiments_dag_stru_chain.txt` | `datasets/threshold_families_structure/threshold-chain.json` | `synthetic-dag` | 60 | Chain DAG structure |
| `multiagent_budget_experiments_dag_stru_balance.txt` | `datasets/threshold_families_structure/threshold-balanced.json` | `synthetic-dag` | 60 | Balanced DAG structure |

These scripts call `python scripts/run_experiment.py` internally, so they should be launched only after the environment is installed and activated (or after making `python` resolve to the project virtual environment). To use uv explicitly, replace that invocation in a private copy with `uv run python scripts/run_experiment.py`.

The budget variable is a system lifetime budget. With `n_base_agents=n`, each worker receives `floor(total_decision_budget / n)` decision calls across all rounds; the ledger is not reset. `max_rounds` controls communication opportunities, while planning, coordination, synthesis, and optional verification calls are auxiliary calls.

## 6. Agent and dataset configurations

Agent YAML files are in `run_conf/agent/`:

| Config | Default behavior |
|---|---|
| `single-agent` | One tool-using agent |
| `multi-agent-independent` | Independent workers and `synthesis_only` aggregation |
| `multi-agent-centralized` | Lead plus orchestrated subagents |
| `multi-agent-decentralized` | Peer debate and 70% consensus threshold |
| `multi-agent-hybrid` | Lead plus peer communication |

Dataset YAML files are in `run_conf/dataset/`. `synthetic-dag` has `output_type: scalar_exact`; PlanCraft has `executable_plan`; BrowseComp-Plus and Finance-Agent/WorkBench have `free_text`; SWE-bench and Terminal-Bench have `patch_or_state`.

## 7. Outputs and validation

Hydra writes runs to:

```text
exp_outputs/{dataset_id}/{agent_name}/{llm.model}/{date}/{time}/
├── run_config.yaml
├── run.log
├── dataset_eval_metrics.json
└── instance_runs/{instance_id}/instance_save.yaml
```

The resolved YAML, execution log, aggregate metrics, and per-instance records are the artifacts needed to audit a run. Save the exact Git commit, model identifier, overrides, dataset file hash, and relevant environment variables (excluding secrets) with the results.

Run the unit tests after installation:

```bash
uv run pytest -q
```

The tests cover synthetic DAG generation/environment, decentralized debate, independent synthesis, result selection, and the Finance-Agent/WorkBench adapters. The current integration set contains 28 tests as recorded in `CHANGELOG.md`.

## 8. Reproducibility notes

- API responses are stochastic unless the selected provider/model honors the configured `temperature=0.0`; cache settings and provider-side changes can still affect outcomes.
- Docker must be running before SWE-bench or Terminal-Bench experiments. The runner registers cleanup handlers for benchmark containers.
- Do not commit `.env`, upstream checkouts, caches, or `exp_outputs/`.
- Keep the exact dataset files: PlanCraft is 580 instances in this checkout, while the synthetic threshold and structure suites are separate 200-instance families. Do not substitute the old 100-instance PlanCraft description.
