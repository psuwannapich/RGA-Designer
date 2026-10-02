# RGA-Designer

Official implementation of **RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback**.

RGA-Designer introduces a preference-based alignment stage for autoregressive multi-agent graph generation, training a reward model from execution feedback and fine-tuning the graph generator to produce topologies that better match downstream task performance while using fewer agents and tokens. It builds on graph generation backbone from [ARG-Designer](https://arxiv.org/abs/2507.18224).

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/psuwannapich/RGA-Designer.git
cd RGA-Designer
uv sync
```

## Environment Setup

### LLM Backend

Export `LOCAL_BASE_URL` to use any OpenAI-compatible API — a local vLLM server, Ollama, or a commercial provider. Leave it unset to use the HuggingFace transformers backend directly.

```bash
# Option A: OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.)
export LOCAL_BASE_URL="http://localhost:8000/v1"
export LOCAL_API_KEY="YOUR_API_KEY"   # default: EMPTY

# Option B: HuggingFace transformers (no server needed, slower)
# just leave LOCAL_BASE_URL unset
```

## Quick Start

All pipeline scripts are in `scripts/`. Pass the dataset as an environment variable
(`gsm8k`, `aqua`, `multiarith`, `svamp`, `humaneval`, `mmlu`).

### Stage 1: Train ARG-Designer

```bash
DATASET=gsm8k bash scripts/train.sh
```

### Stage 2: RGA fine-tuning

Requires Stage 1 to have completed first.

```bash
DATASET=gsm8k bash scripts/rga.sh
```

`rga.sh` runs these steps, and skips any step that has already finished:

1. **Collect** (1a, 1b): sample candidate graphs for each training query, execute them with
   the LLM and label each graph by task success. Queries come from `rlhf_tasks_indices` in
   the dataset's task split (GSM8K, MultiArith, SVAMP, MMLU), which never overlaps the
   test set; AQuA and HumanEval use their base and finetune queries.
2. **Correctness model** (2): a GNN trained with per-graph BCE on queries that have both
   successful and failed graphs. Mantel-Haenszel weights give a query's successful and
   failed graphs equal total weight.
3. **Policy** (3): GRPO fine-tuning of ARG-Designer with reward
   `sigmoid(c) + lambda_size * (1 - |V|/V_max - |E|/E_max)`, where `c` is the correctness
   model's score and `V_max` is the dataset's maximum number of agents.
4. **Graphs and benchmark** (4a, 4b): Best-of-N picks one graph per test query with the same
   reward, then the graphs are run on the test set.

Results go to `<MODEL_SLUG>/benchmark_results/pregraph/rga/summary.jsonl`; `MODEL_SLUG`
defaults to the model name, e.g. `Qwen-Qwen3-4B-no_thinking`.

Main settings (environment variables, defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `HF_MODEL` | `Qwen/Qwen3-4B` | LLM used by the agents |
| `RLHF_NUM_TASKS` | `100` | training queries to collect |
| `RM_LOSS` | `per_graph_bce` | correctness-model loss (`bradley_terry` for pairwise preferences) |
| `LAMBDA_EFF` | `0.6` | `lambda_size` in the policy reward |
| `SIZE_LAMBDA` | `0.6` | `lambda_size` in Best-of-N |
| `BEST_OF_N` | `5` | candidate graphs per test query |
| `RGA_DECISION_METHOD` | `FinalReferTurns` | referee agent; its few-shot demo is sent as a separate turn |
| `RGA_SOLVER_FEWSHOT` | `turns` | solver few-shot demos as separate turns (`inline` = original prompt) |
| `MAX_AGENT_TOKENS` | `2048` | maximum tokens per agent reply |

To train with the original merged reward instead, where graph size is part of the
preference label:

```bash
RM_LOSS=bradley_terry LAMBDA_EFF=0 SIZE_LAMBDA=0 REWARD_SQUASH=0 W_SIZE=0.3 W_EDGE=0.1 \
DATASET=gsm8k bash scripts/rga.sh
```

### Global reward model

`rga_global_rm.sh` trains one Bradley-Terry reward model on the preference pairs of all six
datasets, then a policy per dataset. It needs the collection step of `rga.sh` to have run
for every dataset.

```bash
bash scripts/rga_global_rm.sh
```

## Repository Structure

```
RGA-Designer/
├── experiment/
│   ├── model.py               # ARG-Designer autoregressive model
│   ├── cold_start.py          # ARG-Designer cold-start data generation
│   ├── finetune.py            # ARG-Designer D_eff fine-tuning
│   ├── rga.py                 # RGA pipeline: collection, correctness model, policy
│   ├── generate_graphs.py     # Graph sampling and Best-of-N selection
│   ├── benchmark_pregraph.py  # LLM inference on pre-generated graphs
│   └── evaluate_baseline.py   # Baselines, `uv run baseline` (CoT, fixed topologies, ...)
├── mas_framework/
│   ├── agents/                # Agents, including the FinalReferTurns referee
│   ├── graph/                 # Async multi-agent execution engine
│   ├── llm/                   # LLM backends
│   ├── tools/coding/          # Generated code runs in a killable subprocess
│   └── rga/
│       ├── data_collector.py  # Executes candidate graphs, labels them, resumable
│       ├── sample_trainer.py  # Correctness model (per-graph BCE)
│       ├── reward_trainer.py  # Bradley-Terry reward model (merged reward)
│       └── policy_trainer.py  # GRPO policy fine-tuning
├── benchmark_datasets/        # Dataset loaders, raw data and task splits
└── scripts/
    ├── train.sh               # ARG-Designer training
    ├── rga.sh                 # RGA per-dataset fine-tuning
    └── rga_global_rm.sh       # RGA with one global reward model
```

## Acknowledgments

This codebase builds on [ARG-Designer](https://github.com/Shiy-Li/ARG-Designer), [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), and [GDesigner](https://github.com/yanweiyue/GDesigner).
