# RGA-Designer

Official implementation of **RGA-Designer: Aligning Autoregressive Multi-Agent Graph Generation with Human Feedback**.

RGA-Designer introduces a preference-based alignment stage for autoregressive multi-agent graph generation, training a reward model from execution feedback and fine-tuning the graph generator to produce topologies that better match downstream task performance while using fewer agents and tokens. It builds on graph generation backbone from [ARG-Designer](https://arxiv.org/abs/2507.18224).

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

### Local LLM Backend

Set `LOCAL_BASE_URL` to use any OpenAI-compatible API — a local vLLM server, Ollama, or a commercial provider. Leave it unset to use the HuggingFace transformers backend directly.

```bash
# Option A: OpenAI-compatible API (vLLM, Ollama, OpenAI, etc.)
export LOCAL_BASE_URL="http://localhost:8000/v1"
export LOCAL_API_KEY="YOUR_API_KEY"

# Option B: HuggingFace transformers (no server needed, slower)
# just leave LOCAL_BASE_URL unset
```

## Quick Start

All pipeline scripts are in `scripts/`. Pass the dataset as a positional argument or via environment variables.

**Dataset index**: `0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu`

### Stage 1: Train ARG-Designer

```bash
DATASET=gsm8k bash scripts/train.sh
```

### Stage 2: RGA Fine-tuning

Requires Stage 1 to have completed first.

```bash
# Per-dataset reward model
DATASET=gsm8k bash scripts/rga.sh
```

Or using global reward model fine-tuning across datasets:

```bash
# Global reward model (requires all datasets to have completed Stage 1)
bash scripts/rga_global_rm.sh
```

## Repository Structure

```
RGA-Designer/
├── experiment/
│   ├── model.py               # ARGDesigner autoregressive model
│   ├── cold_start.py          # ARGDesigner cold-start data generation
│   ├── finetune.py            # ARGDesigner D_eff fine-tuning
│   ├── rga.py                 # RGA pipeline
│   ├── generate_graphs.py     # Graph sampling from policy
│   └── benchmark_pregraph.py  # LLM inference on pre-generated graphs
├── mas_framework/
│   ├── agents/                # Agent implementations
│   ├── graph/                 # Async multi-agent execution engine
│   ├── llm/                   # LLM backends
│   └── rga/                   # Reward model, policy trainer, data collector
├── benchmark_datasets/        # Dataset loaders and raw data
└── scripts/
    ├── train.sh               # ARGDesigner training
    ├── rga.sh                 # RGA per-dataset fine-tuning
    ├── rga_global_rm.sh       # RGA global reward model fine-tuning
    └── benchmark_baselines.sh # Baseline evaluation
```

## Acknowledgments

This codebase builds on [ARG-Designer](https://github.com/Shiy-Li/ARG-Designer), [GPTSwarm](https://github.com/metauto-ai/GPTSwarm), and [GDesigner](https://github.com/yanweiyue/GDesigner).
