#!/bin/bash
#SBATCH --job-name=verify_rm
#SBATCH --output=logs/verify_rm_%A_%a.out
#SBATCH --error=logs/verify_rm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --array=0-5          # one task per dataset: 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p batch
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Verify reward model sanity: does the RM score RLHF-generated graphs higher
# than ARG-generated graphs on the same tasks?
#
# For each dataset, loads the trained reward model, scores both ARG and RLHF
# graphs on the held-out test set, and prints per-task comparison.
#
# Expected correct behaviour:
#   mean(RLHF score) > mean(ARG score) for all datasets
#
# Usage:
#   sbatch slurm/verify_reward_model.sh
#   sbatch --array=1 slurm/verify_reward_model.sh   # aqua only
#
# Optional env vars:
#   EXP_DIR   path to experiment dir  (default: see below)
#   EMB_DIR   path to role embedding  (default: see below)
#   N_TASKS   max tasks to score      (default: 500)
# ---------------------------------------------------------------------------

set -euo pipefail

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

EXP_DIR="${EXP_DIR:-/home/users/psuwannapichat/work_space/RLHF-Designer/Qwen-Qwen3-4B-exp_fix4_change_both_correct_w-no_thinking}"
EMB_DIR="${EMB_DIR:-/mnt/aiongpfs/users/psuwannapichat/work_space/RLHF-Designer/Qwen-Qwen3-4B-exp_fix_both_wrong-no_thinking/7/ColdStartData}"
N_TASKS="${N_TASKS:-500}"

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Dataset  : $DATASET"
echo "Exp dir  : $EXP_DIR"
echo "Started  : $(date)"
echo "========================================"

uv run python3 - << PYEOF
import json, glob, os, pickle, statistics, sys
import torch, numpy as np
from sentence_transformers import SentenceTransformer
from mas_framework.rlhf.reward_trainer import load_reward_model
from mas_framework.rlhf.preference_data import GraphSnapshot

DATASET  = "${DATASET}"
EXP_DIR  = "${EXP_DIR}"
EMB_DIR  = "${EMB_DIR}"
N_TASKS  = int("${N_TASKS}")
DEVICE   = torch.device("cpu")

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def make_snapshot(rec, role_embs):
    nodes_raw = []
    for n in rec["graph"]["nodes"]:
        role = n["role"]
        emb  = role_embs.get(role)
        if isinstance(emb, torch.Tensor): emb = emb.numpy()
        emb  = np.array(emb, dtype=np.float32) if emb is not None else np.zeros(384, dtype=np.float32)
        nodes_raw.append({"id": n["id"], "role": role, "role_embedding": emb.tolist()})
    return GraphSnapshot(nodes=nodes_raw, edges=[list(e) for e in rec["graph"]["edges"]],
                        num_nodes=len(nodes_raw))

def score(rm, snap, task_emb_np):
    pyg = snap.to_pyg(task_emb_np)
    return rm.score_single(pyg.x.to(DEVICE), pyg.edge_index.to(DEVICE))

# ---- collect all (run_dir, dataset) pairs that have all required files ----
runs = sorted(glob.glob(f"{EXP_DIR}/*"),
              key=lambda p: int(p.split("/")[-1]) if p.split("/")[-1].isdigit() else 99)

emb_file = f"{EMB_DIR}/{DATASET}/precomputed_role_embeddings.pkl"
if not os.path.exists(emb_file):
    print(f"[ERROR] role embeddings not found: {emb_file}", flush=True)
    sys.exit(1)
with open(emb_file, "rb") as f:
    role_embs = pickle.load(f)

print(f"\n{'Run':<6} {'N':>5} {'ARG_mean':>10} {'RLHF_mean':>11} "
      f"{'Delta':>8} {'RLHF>ARG%':>11} {'Verdict'}", flush=True)
print("=" * 62, flush=True)

all_deltas = []

for run_dir in runs:
    run_num   = run_dir.split("/")[-1]
    rm_path   = f"{run_dir}/rlhf_checkpoints/{DATASET}/reward_model.pth"
    arg_file  = f"{run_dir}/graphs/arg_designer/{DATASET}_graphs.jsonl"
    rlhf_file = f"{run_dir}/graphs/rlhf/{DATASET}_graphs.jsonl"

    if not all(os.path.exists(p) for p in [rm_path, arg_file, rlhf_file]):
        print(f"{run_num:<6} skip (missing files)", flush=True)
        continue

    rm = load_reward_model(rm_path, DEVICE)

    arg_gs, rlhf_gs = {}, {}
    with open(arg_file)  as f:
        for line in f: r = json.loads(line); arg_gs[r["task_id"]] = r
    with open(rlhf_file) as f:
        for line in f: r = json.loads(line); rlhf_gs[r["task_id"]] = r

    shared = sorted(set(arg_gs) & set(rlhf_gs))[:N_TASKS]
    a_sc, r_sc = [], []
    for tid in shared:
        te = encoder.encode(arg_gs[tid]["task_text"]).astype(np.float32)
        try:
            a_sc.append(score(rm, make_snapshot(arg_gs[tid],  role_embs), te))
            r_sc.append(score(rm, make_snapshot(rlhf_gs[tid], role_embs), te))
        except Exception as e:
            pass  # skip malformed records

    if not a_sc:
        print(f"{run_num:<6} no valid tasks", flush=True)
        continue

    diffs  = [r - a for r, a in zip(r_sc, a_sc)]
    wins   = sum(1 for d in diffs if d > 0)
    mean_a = statistics.mean(a_sc)
    mean_r = statistics.mean(r_sc)
    mean_d = statistics.mean(diffs)
    pct    = 100 * wins / len(diffs)
    verdict = "PASS ✓" if mean_d > 0 else "FAIL ✗"
    all_deltas.extend(diffs)

    print(f"{run_num:<6} {len(diffs):>5} {mean_a:>10.4f} {mean_r:>11.4f} "
          f"{mean_d:>+7.4f}  {pct:>9.1f}%  {verdict}", flush=True)

if all_deltas:
    print("\n" + "=" * 62, flush=True)
    overall_wins = sum(1 for d in all_deltas if d > 0)
    print(f"OVERALL [{DATASET}]:  N={len(all_deltas)}  "
          f"mean_delta={statistics.mean(all_deltas):+.4f}  "
          f"RLHF>ARG={100*overall_wins/len(all_deltas):.1f}%", flush=True)
    if statistics.mean(all_deltas) > 0:
        print(f"=> REWARD MODEL PASSES SANITY CHECK for {DATASET}", flush=True)
    else:
        print(f"=> REWARD MODEL FAILS SANITY CHECK for {DATASET} — "
              f"RM assigns higher score to ARG than RLHF", flush=True)
PYEOF

echo "========================================"
echo "Finished : $(date)"
echo "========================================"
