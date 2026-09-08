# Agent Scaling

A framework for studying scaling behaviors of LLM-based single-agent and multi-agent systems on complex reasoning tasks. Code release accompanying the Nature Machine Intelligence manuscript *"Beyond more agents: quantifying when multi-agent collaboration benefits large language model agents"* (the arXiv preprint at arXiv:2512.08296 retains the earlier title *"Towards a science of scaling agent systems"*).

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### Installation

```bash
# Clone the repository and check out the release used for the manuscript
git clone https://github.com/ybkim95/agent-scaling.git
cd agent-scaling
git checkout v2.1.3   # release tag cited in the Code Availability section of the manuscript

# Install dependencies
uv sync --prerelease=allow

# Install flash-attn (needed for BrowseComp+ environment)
uv pip install --no-build-isolation flash-attn

# Activate the virtual environment
source .venv/bin/activate
```

### Setting Environment Variables

Create a `.env` file with your LLM API keys. See [LiteLLM providers](https://docs.litellm.ai/docs/providers) for supported providers.

```bash
# Required: At least one LLM provider API key
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-gemini-key"
ANTHROPIC_API_KEY="your-anthropic-key"

# Optional: LangFuse for LLM call tracing
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_SECRET_KEY="your-secret-key"
LANGFUSE_PUBLIC_KEY="your-public-key"
```

## Running Experiments

### Basic Usage

Run an experiment with default configuration:

```bash
python scripts/run_experiment.py
```

Run in debug mode (processes fewer instances):

```bash
python scripts/run_experiment.py debug=true
```

### Configuring Experiments

The framework uses [Hydra](https://hydra.cc/docs/intro/) for configuration management. Override parameters via command line:

```bash
# Run single-agent on PlanCraft dataset
python scripts/run_experiment.py agent=single-agent dataset=plancraft-test

# Run multi-agent centralized system
python scripts/run_experiment.py agent=multi-agent-centralized dataset=plancraft-test

# Run with different LLM (use any model from the paper pool: gpt-5, gpt-5-mini, gpt-5-nano,
# gemini/gemini-2.0-flash, gemini/gemini-2.5-pro, anthropic/claude-sonnet-4-5, etc.)
python scripts/run_experiment.py llm.model=openai/gpt-5-mini

# Run with parallel workers
python scripts/run_experiment.py num_workers=4

# Process more instances
python scripts/run_experiment.py max_instances=10
```

### Available Configurations

#### Agent Types

| Agent | Config Name | Description |
|-------|-------------|-------------|
| Single Agent | `single-agent` | Single LLM agent with tool use |
| Multi-Agent Centralized | `multi-agent-centralized` | Orchestrated multi-agent system with lead agent |
| Multi-Agent Decentralized | `multi-agent-decentralized` | Peer-to-peer multi-agent coordination |
| Multi-Agent Hybrid | `multi-agent-hybrid` | Hybrid coordination approach |
| Multi-Agent Independent | `multi-agent-independent` | Independent parallel agents |

#### Datasets

The paper evaluates on six benchmarks. All six are runnable from this repository. Four ship with direct config files; two (Workbench and Finance Agent) ship with setup scripts that download the upstream tasks and generate the dataset config in one command — see `DATA_AVAILABILITY.md` and `REPRODUCTION.md` for the per-benchmark workflow.

| Dataset | Config Name | Description |
|---------|-------------|-------------|
| BrowseComp-Plus | `browsecomp-plus` | Web browsing / multi-hop question answering |
| PlanCraft | `plancraft-test` | Minecraft crafting planning tasks |
| SWE-bench Verified | `swebench-verified` | Real-world GitHub issue resolution (Docker; 7 tools) |
| Terminal-Bench | `terminalbench` | CLI task execution (Docker; 2 tools) |
| Workbench | `workbench` | Common business tool-use tasks. Run `bash scripts/setup_workbench.sh` to download upstream and generate the dataset config (upstream: https://github.com/olly-styles/WorkBench). |
| Finance Agent | `finance-agent` | Multi-step financial reasoning. Run `bash scripts/setup_finance_agent.sh` to download upstream and generate the dataset config (upstream: https://github.com/vals-ai/finance-agent). |

#### Supported LLMs

| Provider | Models |
|----------|--------|
| OpenAI | GPT-5, GPT-5-mini, GPT-5-nano |
| Google | Gemini-2.5 Pro, Gemini-2.5 Flash, Gemini-2.0 Flash |
| Anthropic | Claude Sonnet 4.5, Claude Sonnet 4, Claude Sonnet 3.7 (original 4 benchmarks only; deprecated February 2026 and therefore unavailable for SWE-bench Verified and Terminal-Bench) |

## Example Experiments

### Single-Agent on PlanCraft

```bash
python scripts/run_experiment.py \
    agent=single-agent \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Multi-Agent Centralized on PlanCraft

```bash
python scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=plancraft-test \
    llm.model=gemini/gemini-2.0-flash \
    max_instances=5
```

### Multi-Agent Centralized on BrowseComp-Plus

```bash
python scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=browsecomp-plus \
    llm.model=openai/gpt-5-mini \
    max_instances=5
```

### Single-Agent on SWE-bench Verified (Docker required)

```bash
python scripts/run_experiment.py \
    agent=single-agent \
    dataset=swebench-verified \
    llm.model=openai/gpt-5-mini \
    max_instances=5
```

### Multi-Agent Centralized on Terminal-Bench (Docker required)

```bash
python scripts/run_experiment.py \
    agent=multi-agent-centralized \
    dataset=terminalbench \
    llm.model=openai/gpt-5-mini \
    max_instances=5
```

### Scaling Number of Agents

The multi-agent centralized system supports configuring the number of agents:

```bash
# Run with 5 agents
python scripts/run_experiment.py \
    agent=multi-agent-centralized \
    agent.n_base_agents=5 \
    dataset=plancraft-test

# Run with 10 agents
python scripts/run_experiment.py \
    agent=multi-agent-centralized \
    agent.n_base_agents=10 \
    dataset=plancraft-test
```

## Output Structure

Experiment outputs are saved to `exp_outputs/{dataset}/{agent}/{model}/{date}/{time}/`:

```
exp_outputs/
└── plancraft-test/
    └── multi-agent-centralized/
        └── gemini/
            └── gemini-2.0-flash/
                └── 2025-01-21/
                    └── 12-30-45/
                        ├── run_config.yaml        # Experiment configuration
                        ├── run.log                # Detailed execution logs
                        ├── dataset_eval_metrics.json  # Aggregated metrics
                        └── instance_runs/         # Per-instance outputs
                            ├── 0000/
                            ├── 0001/
                            └── ...
```

### Output Files

- **`run_config.yaml`**: Full configuration used for the experiment
- **`run.log`**: Detailed logs including prompts, LLM responses, and tool calls
- **`dataset_eval_metrics.json`**: Aggregated evaluation metrics
  ```json
  {
    "avg_success": 0.85,
    "avg_num_steps": 7.2,
    "num_instances": 100
  }
  ```

## Example Traces

See `example_traces/` directory for sanitized sample experiment outputs demonstrating:
- Single-agent execution traces
- Multi-agent coordination logs
- Per-instance and aggregated evaluation metrics

## Project Structure

```
agent-scaling/
├── agent_scaling/           # Main Python package
│   ├── agents/              # Agent implementations
│   │   ├── single_agent.py
│   │   ├── multiagent_centralized.py
│   │   ├── multiagent_decentralized.py
│   │   ├── multiagent_hybrid.py
│   │   └── multiagent_independent.py
│   ├── datasets/            # Dataset loaders (SWE-bench, Terminal-Bench, etc.)
│   ├── env/                 # Environment & tools (includes Docker environments)
│   ├── llm/                 # LLM integration
│   └── config/              # Configuration classes
├── scripts/                 # Entry point + analysis scripts
│   ├── run_experiment.py
│   └── (regression and scaling-principle analysis scripts)
├── run_conf/                # Hydra configurations
│   ├── agent/               # Agent configs (single, centralized, decentralized, hybrid, independent)
│   ├── dataset/             # Dataset configs
│   └── run_exp.yaml         # Master config
├── prompts/                 # Prompt templates
├── example_traces/          # Sanitized sample execution traces
├── datasets/                # (user-populated) Dataset files; see DATA_AVAILABILITY.md
├── REPRODUCTION.md          # Step-by-step reproduction guide
└── DATA_AVAILABILITY.md     # Benchmark source URLs and subset selection methodology
```

## Configuration Reference

### Master Config (`run_conf/run_exp.yaml`)

```yaml
defaults:
  - agent: multi-agent-centralized  # Agent type
  - dataset: plancraft-test         # Dataset

llm:
  model: gemini/gemini-2.0-flash    # LLM model
  params:
    temperature: 0.0                # Generation temperature

log_langfuse: false                 # Enable LangFuse tracing
use_disk_cache: true                # Cache LLM calls
num_workers: 1                      # Parallel workers
debug: true                         # Debug mode
max_instances: 3                    # Max instances to process
```

### Multi-Agent Config (`run_conf/agent/multi-agent-centralized.yaml`)

```yaml
name: multi-agent-centralized
total_decision_budget: 32             # Lifetime worker decision budget B
n_base_agents: 3                    # Number of agents
min_iterations_per_agent: 0         # No artificial minimum in budget-controlled runs
max_rounds: 10                       # Fixed communication horizon (not the worker budget)
consensus_threshold: 0.7            # Decentralized only: agreement fraction for consensus
communication:
  strategy: orchestrated            # Communication strategy
```

### Lifetime decision budgets

Each agent configuration exposes `total_decision_budget` (default `32`). This
is a per-instance lifetime budget for worker decision calls: a single-agent
run receives `B` decisions, while an `n_base_agents` multi-agent run gives each
worker `floor(B / n_base_agents)` decisions across all rounds. Planning,
coordination, debate-summary, and synthesis calls are auxiliary calls and do
not consume the worker decision budget. `max_steps` and
`max_iterations_per_agent` are optional legacy safety caps and are omitted from
the canonical budget-controlled configurations. Multi-round protocols retain a
fixed `max_rounds: 10` communication horizon; it controls when agents exchange
information, while each worker's per-round quota is derived as
`ceil((B / n) / max_rounds)` and the lifetime ledger remains authoritative.
`min_iterations_per_agent` is capped by the worker's allocated lifetime budget.
The ledger also records prompt/completion tokens and auxiliary-call counts for
reporting compute separately from the decision-call budget.

For example, with `B=32`, `n=3`, and `max_rounds=10`, each worker receives
`floor(32/3)=10` lifetime decisions and at most one decision per communication
round; the budget is never reset between rounds.

Wall-clock safeguards are implementation-level controls, not experimental
variables: dataset-specific `time_limit` values (or the internal 600-second
fallback) stop a run if it hangs. The old configuration fields
`worker_timeout` and `max_findings` are not part of the active orchestration
path and are intentionally omitted from the canonical YAML files.

## Citation

This work is currently under revision at *Nature Machine Intelligence* under the title *"Beyond more agents: quantifying when multi-agent collaboration benefits large language model agents"*. While the revision is in review, please cite the arXiv preprint (which retains the original title):

```bibtex
@article{kim2025towards,
  title={Towards a science of scaling agent systems},
  author={Kim, Yubin and Gu, Ken and Park, Chanwoo and Park, Chunjong and Schmidgall, Samuel and Heydari, A Ali and Yan, Yao and Zhang, Zhihan and Zhuang, Yuchen and Liu, Yun and others},
  journal={arXiv preprint arXiv:2512.08296},
  year={2025}
}
```

For the archived code release accompanying this work, please cite the version-specific Zenodo DOI given in the accompanying manuscript's Code Availability section, or the persistent concept DOI `10.5281/zenodo.20144433` which always resolves to the latest archived release of this repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
