# RGA-Designer

Official implementation of **RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback**.

This work extends [ARG-Designer](https://arxiv.org/abs/2507.18224) (AAAI 2026 oral) — which frames multi-agent system design as a conditional autoregressive graph generation task — by applying **Reinforcement Learning from Human Feedback (RLHF)** to align the generated topologies with downstream task performance.

## Overview

Designing effective multi-agent systems requires carefully selecting which agent roles to include and how they communicate. ARG-Designer automates this by learning to generate directed graphs (roles + edges) conditioned on a task description. However, the supervised policy is trained purely on cold-start data and has no mechanism to directly optimise for task accuracy.

**RGA-Designer** closes this gap with a three-component RLHF pipeline:

1. **Preference pair collection** — Execute sampled graphs on training tasks using a live LLM backend. Score each execution with a composite reward balancing correctness, graph size, and edge sparsity. Pairs with sufficiently different scores form preference data.

2. **GNN reward model** — Train a GraphSAGE network on the preference pairs with a Bradley-Terry ranking loss. Node features combine role embeddings, task embeddings, and normalised structural scalars (degree, position, graph size) to handle single-role datasets where role identity alone is degenerate.

3. **GRPO policy fine-tuning** — Fine-tune the ARG-Designer policy with Group Relative Policy Optimization under a KL divergence penalty. Within-group advantage normalisation removes the reward model's arbitrary absolute scale, making training stable without a separate critic.

Additionally, **Best-of-N topology sampling** uses the trained reward model at inference time to select the best graph from N candidates, providing an orthogonal inference-time scaling axis.

## Method

### Three-Stage Training Pipeline

```
Stage 1: Cold-Start (experiment/cold_start_gemma.py)
  ├── Try multiple fixed topologies × agent counts per task
  ├── Execute each via LLM and record correctness
  └── Save correct graphs → ColdStartData/{dataset}/

Stage 2: Curriculum Fine-tuning (experiment/finetune_gemma.py)
  ├── Phase 2a: Pre-train ARGDesigner on cold-start graphs
  ├── Phase 2b: Build D_eff = D_pruned ∪ D_simple ∪ D_replay
  └── Phase 2c: Fine-tune → ef_best_model.pth

Stage 3: RLHF (experiment/rlhf_gsm8k.py)
  ├── gen_candidates: Sample diverse graphs (temperature sweep + weak baselines)
  ├── collect_llm:    Execute candidates via LLM, score, form preference pairs
  ├── train_rm:       Bradley-Terry loss on GNN reward model → reward_model.pth
  └── train_policy:   GRPO + KL penalty → policy_rlhf.pth

Evaluation
  ├── generate_graphs.py:    Sample graphs from policy → graphs.jsonl
  └── benchmark_pregraph.py: Run LLM on graphs, score accuracy → results.jsonl
```

### Preference Scoring

Each graph execution is scored as:

```
s(G, q) = 0.6 × correct + 0.2 × (1 - size_ratio) + 0.2 × (1 - edge_ratio)
```

Correctness dominates: a correct graph always scores higher than an incorrect one.
Among correct graphs, smaller and sparser topologies are preferred (Occam's razor).

### Reward Model Architecture

- **Node features**: `[role_emb (384-d) | task_emb (384-d) | struct_feats (5-d)]` = 773-d
- **Structural features**: normalised graph size, edge density, node position, in/out degree
- **Architecture**: 2-layer SAGEConv (773→256→128) + global mean pooling + MLP head
- **Loss**: Bradley-Terry — `L = −log σ(r_chosen − r_rejected)`, with both-correct pairs down-weighted

### GRPO Policy Fine-tuning

```python
# Within-group advantage normalisation
rewards = [reward_model(graph, task_emb) for graph in group]
A = (rewards - mean(rewards)) / (std(rewards) + 1e-8)

# Policy loss with KL penalty
loss = -E[(A - kl_coeff * KL(π || π_ref)) * log π(graph | task)]
```

## Results

Accuracy (%) across 10 independent seeds — Qwen/Qwen3-4B backbone, no-thinking mode.

| Dataset    | ARG-Designer      | RGA-Designer      | Δ      |
|------------|-------------------|-------------------|--------|
| GSM8K      | 88.2 ± 0.8        | 88.3 ± 1.4        | +0.2   |
| AQuA       | 81.0 ± 0.8        | **83.4 ± 1.6**    | **+2.3** |
| MultiArith | **98.5** ± 0.1    | 98.3 ± 0.2        | −0.2   |
| SVAMP      | **94.7** ± 0.5    | 94.3 ± 0.4        | −0.3   |
| HumanEval  | 79.6 ± 3.0        | **81.3 ± 1.2**    | **+1.7** |
| MMLU       | 78.8 ± 1.0        | **79.5 ± 1.2**    | **+0.7** |

RLHF improves accuracy on 4 of 6 benchmarks. The largest gains appear on AQuA (+2.3%) and HumanEval (+1.7%), where tasks are hard enough that topology quality meaningfully affects correctness. Small regressions on MultiArith and SVAMP are due to both-correct preference pair dominance at high base accuracy, which biases the reward model toward smaller graphs.

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/RGA-Designer.git
cd RGA-Designer
uv sync
```

### LLM Backend

Three backends are supported, selected via environment variables:

| Variable | Backend | When to use |
|----------|---------|-------------|
| `USE_VLLM_SERVER=1` | vLLM HTTP server | Production; fastest |
| `USE_VLLM=1` | vLLM in-process | Single-node, no server |
| neither | HuggingFace transformers | Dev / debugging |

For the vLLM server backend, install vLLM in a separate virtual environment and set:

```bash
export VLLM_SERVE_DIR=/path/to/your/vllm_env   # must contain .venv/bin/vllm
export VLLM_CHAT_TEMPLATE=/path/to/qwen3_nonthinking.jinja  # optional; Qwen3 only
```

## Quick Start

All pipeline scripts are in `scripts/`. They accept the dataset as a positional argument (0–5) or via `DATASET_IDX` / `DATASET` environment variables.

**Dataset index mapping**: `0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu`

### Full Training Pipeline (Stages 1–2)

```bash
# Run on GSM8K (index 0)
DATASET=gsm8k bash scripts/train.sh

# Or by index
bash scripts/train.sh 0

# Custom model
HF_MODEL=Qwen/Qwen3-8B DATASET=aqua bash scripts/train.sh
```

### RLHF Pipeline (Stage 3) — Per-Dataset Reward Model

```bash
# Prerequisites: scripts/train.sh must have completed first
DATASET=gsm8k bash scripts/rlhf.sh

# With Best-of-N enabled (N=5 at inference)
DATASET=humaneval BEST_OF_N=5 bash scripts/rlhf.sh
```

### RLHF Pipeline — Global Reward Model

Trains a single reward model on preference data pooled across all datasets, then fine-tunes per-dataset policies. Requires `scripts/rlhf.sh` (collect phases 1a+1b) to have completed for each dataset.

```bash
# Default run
bash scripts/rlhf_global_rm.sh

# Custom run name (namespaces all outputs)
RUN_NAME=kl02_bon5 KL_COEFF=0.2 BEST_OF_N=5 bash scripts/rlhf_global_rm.sh
```

### Baseline Benchmarks

```bash
# Single method + dataset
METHOD=chain DATASET=gsm8k bash scripts/benchmark_baselines.sh

# All combinations
for m in vanilla cot self_consistency chain complete random star llm_debate; do
  for d in 0 1 2 3 4 5; do
    METHOD=$m bash scripts/benchmark_baselines.sh $d
  done
done
```

## Individual Commands

For finer control, run individual pipeline steps directly:

```bash
# Stage 1: Cold-start data generation
uv run cold-start --dataset gsm8k \
    --dataset_json benchmark_datasets/gsm8k/gsm8k.jsonl \
    --llm_name Qwen/Qwen3-4B --num_tasks 100

# Stage 3: Collect preference pairs
uv run rlhf --dataset gsm8k --phase collect \
    --dataset_json benchmark_datasets/gsm8k/gsm8k.jsonl \
    --num_tasks 100

# Stage 3: Train reward model
uv run rlhf --dataset gsm8k --phase train_rm \
    --preference_dir rlhf_data/gsm8k

# Stage 3: Fine-tune policy
uv run rlhf --dataset gsm8k --phase train_policy \
    --model_dir checkpoints/gsm8k \
    --rm_checkpoint rlhf_checkpoints/gsm8k/reward_model.pth

# Generate graphs from trained policy
uv run python experiment/generate_graphs.py \
    --model_path checkpoints/gsm8k \
    --dataset gsm8k \
    --dataset_path benchmark_datasets/gsm8k/gsm8k.jsonl \
    --output_file graphs.jsonl

# Benchmark pre-generated graphs
uv run python experiment/benchmark_pregraph.py \
    --graphs_file graphs.jsonl \
    --dataset gsm8k \
    --llm_name Qwen/Qwen3-4B \
    --output_file results.jsonl
```

## Repository Structure

```
RGA-Designer/
├── experiment/
│   ├── model.py                   # ARGDesigner autoregressive model (GRU)
│   ├── cold_start_gemma.py        # Stage 1: cold-start data generation
│   ├── finetune_gemma.py          # Stage 2: curriculum fine-tuning
│   ├── rlhf_gsm8k.py             # Stage 3: RLHF pipeline entry point
│   ├── generate_graphs.py         # Graph sampling from policy
│   ├── benchmark_pregraph.py      # LLM inference on pre-generated graphs
│   └── args.py                    # Shared argparse configuration
│
├── mas_framework/
│   ├── agents/                    # Agent implementations (AgentRegistry)
│   ├── graph/graph.py             # Async multi-agent execution engine
│   ├── llm/                       # LLM backends (vLLM server / in-process / HF)
│   └── rlhf/
│       ├── data_collector.py      # Async preference pair collection
│       ├── preference_data.py     # PreferencePair dataclass + dataset
│       ├── reward_model.py        # GNN GraphRewardModel (SAGEConv)
│       ├── reward_trainer.py      # Bradley-Terry training loop
│       └── policy_trainer.py      # GRPO policy fine-tuning
│
├── benchmark_datasets/            # Dataset loaders and raw data
│   ├── gsm8k/  aqua/  multiarith/  svamp/  humaneval/  mmlu/
│
└── scripts/
    ├── train.sh                   # Stages 1–2: cold-start + training
    ├── rlhf.sh                    # Stage 3: per-dataset RLHF
    ├── rlhf_global_rm.sh          # Stage 3: global reward model variant
    └── benchmark_baselines.sh     # Fixed-topology baseline evaluation
```

## Dataset-Specific Settings

| Dataset    | Max Agents | Decision Method  | Agent Role    |
|------------|-----------|------------------|---------------|
| GSM8K      | 4         | FinalRefer       | MathSolver    |
| AQuA       | 4         | FinalRefer       | MathSolver    |
| MultiArith | 4         | FinalRefer       | MathSolver    |
| SVAMP      | 4         | FinalRefer       | MathSolver    |
| HumanEval  | 5         | FinalWriteCode   | CodeWriting   |
| MMLU       | 6         | FinalRefer       | AnalyzeAgent  |

## Checkpoint Structure

```
{MODEL_SLUG}/
├── ColdStartData/{dataset}/           # Stage 1 graphs
├── checkpoints/{dataset}/
│   ├── best_model.pth                 # After Stage 2a pre-training
│   └── ef_best_model.pth             # After Stage 2c fine-tuning
├── rlhf_data/{dataset}/              # Preference pair shards (*.pkl)
├── rlhf_checkpoints/{dataset}/
│   ├── reward_model.pth              # Trained GNN reward model
│   └── policy_rlhf.pth              # RLHF fine-tuned policy
├── graphs/{model_type}/{dataset}_graphs.jsonl
├── benchmark_results/pregraph/{model_type}/{dataset}.jsonl
└── state/{dataset}/                  # Stage completion flags (*.done)
```

Stage flags are stored as empty `.done` files under `state/`. Delete a flag to re-run that stage without restarting the full pipeline.

## Citation

If you use RGA-Designer, please cite both papers:

```bibtex
@article{suwannapichat2025rga,
  title     = {RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback},
  author    = {Suwannapichat, Poomphob and others},
  year      = {2025}
}

@inproceedings{li2026assemble,
  title     = {Assemble Your Crew: Automatic Multi-Agent Communication Topology Design via Autoregressive Graph Generation},
  author    = {Li, Shiyuan and Liu, Yixin and Wen, Qingsong and Zhang, Chengqi and Pan, Shirui},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

## Acknowledgments

This codebase builds on [ARG-Designer](https://github.com/your-org/ARG-Designer), [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), and [GDesigner](https://github.com/yanweiyue/GDesigner).
