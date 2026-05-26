# RGA-Designer

Official implementation of **RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback**.

This work extends [ARG-Designer](https://arxiv.org/abs/2507.18224) (AAAI 2026 oral) by applying RLHF to align generated multi-agent topologies with downstream task performance.

## Results

Accuracy (%) across 10 independent seeds — Qwen/Qwen3-4B backbone.

| Dataset    | ARG-Designer      | RGA-Designer      | Δ        |
|------------|-------------------|-------------------|----------|
| GSM8K      | 88.2 ± 0.8        | 88.3 ± 1.4        | +0.2     |
| AQuA       | 81.0 ± 0.8        | **83.4 ± 1.6**    | **+2.3** |
| MultiArith | **98.5** ± 0.1    | 98.3 ± 0.2        | −0.2     |
| SVAMP      | **94.7** ± 0.5    | 94.3 ± 0.4        | −0.3     |
| HumanEval  | 79.6 ± 3.0        | **81.3 ± 1.2**    | **+1.7** |
| MMLU       | 78.8 ± 1.0        | **79.5 ± 1.2**    | **+0.7** |

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/your-org/RGA-Designer.git
cd RGA-Designer
uv sync
```

## Environment Setup

### API Keys

Copy `template.env` to `.env` and fill in your credentials:

```bash
cp template.env .env
```

```
BASE_URL = ""   # OpenAI-compatible API base URL
API_KEY  = ""   # API key
```

### LLM Backend

Set `LOCAL_BASE_URL` to use any OpenAI-compatible API — a local vLLM server, Ollama, or a commercial provider. Leave it unset to use the HuggingFace transformers backend directly.

```bash
# Option A: OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.)
export LOCAL_BASE_URL="http://localhost:8000/v1"
export LOCAL_API_KEY="EMPTY"   # or your actual API key

# Option B: HuggingFace transformers (no server needed, slower)
# just leave LOCAL_BASE_URL unset
```

## Quick Start

All pipeline scripts are in `scripts/`. Pass the dataset as a positional argument or via environment variables.

**Dataset index**: `0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu`

### Stage 1–2: Train ARG-Designer

```bash
DATASET=gsm8k bash scripts/train.sh

# Or by index
bash scripts/train.sh 0

# Custom model
HF_MODEL=Qwen/Qwen3-8B DATASET=aqua bash scripts/train.sh
```

### Stage 3: RLHF Fine-tuning

Requires Stage 1–2 to have completed first.

```bash
# Per-dataset reward model
DATASET=gsm8k bash scripts/rga.sh

# With Best-of-N inference (N=5)
DATASET=humaneval BEST_OF_N=5 bash scripts/rga.sh

# Global reward model (pooled across all datasets)
RUN_NAME=kl02_bon5 KL_COEFF=0.2 BEST_OF_N=5 bash scripts/rga_global_rm.sh
```

### Baseline Evaluation

```bash
METHOD=chain DATASET=gsm8k bash scripts/benchmark_baselines.sh
```

Available methods: `vanilla`, `cot`, `self_consistency`, `chain`, `complete`, `random`, `star`, `llm_debate`

## Individual Commands

```bash
# Stage 1: Cold-start data generation
uv run cold-start --dataset gsm8k \
    --dataset_json benchmark_datasets/gsm8k/gsm8k.jsonl \
    --llm_name Qwen/Qwen3-4B --num_tasks 100

# Stage 3: Collect preference pairs
uv run rga --dataset gsm8k --phase collect \
    --dataset_json benchmark_datasets/gsm8k/gsm8k.jsonl --num_tasks 100

# Stage 3: Train reward model
uv run rga --dataset gsm8k --phase train_rm \
    --preference_dir rga_data/gsm8k

# Stage 3: Fine-tune policy
uv run rga --dataset gsm8k --phase train_policy \
    --model_dir checkpoints/gsm8k \
    --rm_checkpoint rga_checkpoints/gsm8k/reward_model.pth

# Generate graphs from trained policy
uv run python experiment/generate_graphs.py \
    --model_path checkpoints/gsm8k --dataset gsm8k \
    --dataset_path benchmark_datasets/gsm8k/gsm8k.jsonl \
    --output_file graphs.jsonl

# Benchmark pre-generated graphs
uv run python experiment/benchmark_pregraph.py \
    --graphs_file graphs.jsonl --dataset gsm8k \
    --llm_name Qwen/Qwen3-4B --output_file results.jsonl
```

## Repository Structure

```
RGA-Designer/
├── experiment/
│   ├── model.py               # ARGDesigner autoregressive model
│   ├── cold_start.py          # Stage 1: cold-start data generation
│   ├── finetune.py            # Stage 2: curriculum fine-tuning
│   ├── rga.py                # Stage 3: RLHF pipeline
│   ├── generate_graphs.py     # Graph sampling from policy
│   └── benchmark_pregraph.py  # LLM inference on pre-generated graphs
├── mas_framework/
│   ├── agents/                # Agent implementations
│   ├── graph/                 # Async multi-agent execution engine
│   ├── llm/                   # LLM backends
│   └── rga/                  # Reward model, policy trainer, data collector
├── benchmark_datasets/        # Dataset loaders and raw data
└── scripts/
    ├── train.sh               # Stages 1–2
    ├── rga.sh                 # Stage 3: per-dataset RGA
    ├── rga_global_rm.sh      # Stage 3: global reward model
    └── benchmark_baselines.sh # Baseline evaluation
```

## Citation

```bibtex
@article{suwannapichat2025rga,
  title  = {RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback},
  author = {Suwannapichat, Poomphob and others},
  year   = {2025}
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
