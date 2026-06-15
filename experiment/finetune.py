"""
Unified D_eff generator + Phase-2 fine-tuner for ARG-Designer.

Implements the second phase of the curriculum learning strategy from the paper:
    D_eff = D_simple ∪ D_pruned ∪ D_replay

  • D_pruned  — Phase-1 model generates graphs; edges are pruned; only
                graphs that still correctly solve the task are kept.
  • D_simple  — Simple topologies (Chain / Star / Layered) evaluated with
                the LLM; only correct solutions are kept.
  • D_replay  — Random subset of the original cold-start graphs (D_exp)
                to prevent catastrophic forgetting.

After building D_eff the script reloads the Phase-1 checkpoint and trains
it on D_eff at a lower learning rate, producing `ef_best_model.pth`.

Usage:
    python experiment/finetune.py \\
        --dataset        gsm8k \\
        --dataset_json   datasets/gsm8k/gsm8k.jsonl \\
        --cold_start_dir ColdStartData_hf_gsm8k \\
        --checkpoint_dir checkpoints/gsm8k \\
        --output_dir     checkpoints/gsm8k \\
        --llm_name       Qwen/Qwen3-8B

Supported datasets: gsm8k, aqua, multiarith, svamp, humaneval, mmlu
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import random
import shutil
import sys
import argparse

import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.llm.profile_embedding import get_sentence_model
from experiment.args import Args
from experiment.model import ARGDesigner
from experiment import process_dataset as gdata
from experiment.train_ARGDesigner import train
from experiment.utils import (
    get_kwargs, save_graph_with_features,
    load_model, generate_graph, convert_to_pyg_graph,
)
# Re-use all dataset helpers from the cold-start script
from experiment.cold_start import (
    _load_dataset,
    _get_predict,
    _is_correct,
    _get_role_description,
    _get_agent_name,
    _get_decision_method,
    _TASK_SPLIT_DIRS,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build D_eff and fine-tune ARGDesigner (Phase 2)"
    )
    p.add_argument('--phase', choices=['all', 'build_deff', 'finetune'], default='all',
                   help=(
                       'all        — build D_eff then run NN fine-tuning (default, legacy). '
                       'build_deff — only build D_eff via LLM inference (vLLM must be running). '
                       'finetune   — only run NN fine-tuning on existing D_eff (no vLLM needed, '
                       '             can use GPU freely since vLLM is stopped).'
                   ))
    p.add_argument('--dataset', required=True,
                   choices=['gsm8k', 'aqua', 'multiarith', 'svamp', 'humaneval', 'mmlu'])
    p.add_argument('--dataset_json', required=True,
                   help='Path to dataset file (same as used for cold-start)')
    p.add_argument('--cold_start_dir', required=True,
                   help='Directory containing Phase-1 .pt graphs (D_exp)')
    p.add_argument('--checkpoint_dir', required=True,
                   help='Directory containing Phase-1 best_model.pth + final_model.pth')
    p.add_argument('--output_dir', required=True,
                   help='Directory to save D_eff data and ef_best_model.pth')
    p.add_argument('--llm_name', type=str, default='Qwen/Qwen3-8B')
    p.add_argument('--model_pool', type=str, default=None,
                   help='Base-model pool for per-node model assignment in D_simple '
                        'generation (JSON file, comma-separated names, or "default"). '
                        'D_pruned graphs already carry the models chosen by the '
                        'Phase-1 policy. Unset = single-model mode.')
    p.add_argument('--pruning_ratio', type=float, default=0.25,
                   help='Fraction of edges to prune per graph')
    p.add_argument('--replay_ratio', type=float, default=0.3,
                   help='Fraction of D_exp graphs to include as replay')
    p.add_argument('--finetune_epochs', type=int, default=200)
    p.add_argument('--finetune_lr', type=float, default=5e-5)
    p.add_argument('--batch_size', type=int, default=2,
                   help='Async inference batch size for D_eff generation')
    p.add_argument('--train_batch_size', type=int, default=32,
                   help='Training batch size for Phase-2 fine-tuning')
    p.add_argument('--num_rounds', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Simple topology configs (same across all datasets — matches originals)
# ---------------------------------------------------------------------------

def get_simple_configs() -> list:
    """Return Chain / Star / Layered configurations for D_simple generation."""
    configs = set()
    for n in range(2, 5):
        if n == 2:
            configs.add(('Chain', 2))
        elif n == 3:
            configs.add(('Chain', 3))
            configs.add(('Star', 3))
        else:
            configs.add(('Chain', n))
            configs.add(('Star', n))
            configs.add(('Layered', n))
    return list(configs)


# ---------------------------------------------------------------------------
# Graph pruning
# ---------------------------------------------------------------------------

def apply_efficiency_strategy(graphs: list, pruning_ratio: float = 0.25) -> list:
    """
    Prune a random fraction of edges from each graph.
    Re-connects any weakly-disconnected components to preserve DAG reachability.
    """
    out = []
    for g in graphs:
        n_remove = int(g.number_of_edges() * pruning_ratio)
        if n_remove > 0:
            edges = list(g.edges())
            random.shuffle(edges)
            g.remove_edges_from(edges[:n_remove])
            if not nx.is_weakly_connected(g):
                comps = list(nx.weakly_connected_components(g))
                main = max(comps, key=len)
                for c in comps:
                    if c != main:
                        g.add_edge(
                            random.choice(list(c)),
                            random.choice(list(main)),
                        )
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# D_pruned
# ---------------------------------------------------------------------------

async def generate_pruned_data(args, dataset: list, output_dir: str, rlhf_dir: str = "") -> int:
    """
    Use the Phase-1 model to generate graphs for each task, prune edges,
    then verify with the LLM.  Saves successful pruned graphs as .pt files.
    Incorrect graphs are saved to rlhf_dir (if provided) for RLHF pairing only.
    """
    print("\n" + "=" * 20 + f" D_pruned ({args.dataset}) " + "=" * 20)
    model = load_model(args.checkpoint_dir)
    model.eval()
    sentence_model = get_sentence_model()
    role_desc = _get_role_description(args.dataset)
    decision_method = _get_decision_method(args.dataset)

    saved = 0
    total = len(dataset)
    sem = asyncio.Semaphore(args.batch_size)
    pbar = tqdm(total=total, desc="Pruning")

    async def _prune_one(idx: int, record: dict) -> None:
        nonlocal saved
        task_text = record['task']
        emb = torch.tensor(
            sentence_model.encode(task_text),
            device=model.args.device,
        ).float()
        gens = generate_graph(model, emb, role_desc, idx)
        pruned = apply_efficiency_strategy(gens, args.pruning_ratio)
        if not pruned:
            pbar.update(1)
            return
        pyg_data = convert_to_pyg_graph(pruned[0], task_text)
        tg = TestGraph(
            domain=args.dataset,
            llm_name=args.llm_name,
            decision_method=decision_method,
            pyg_data=pyg_data,
        )
        async with sem:
            try:
                result = await asyncio.wait_for(
                    tg.arun({'task': task_text}, num_rounds=1),
                    timeout=1200,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pbar.update(1)
                return
        pbar.update(1)
        raw = result
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if not isinstance(raw, str):
            raw = str(raw)
        predicted = _get_predict(args.dataset, raw)
        correct = _is_correct(args.dataset, predicted, record['answer'])
        if correct:
            saved += 1
            fname = f"pruned_{args.dataset}_q{idx}_g0.pt"
            torch.save(pyg_data, os.path.join(output_dir, fname))
        elif rlhf_dir:
            fname = f"pruned_{args.dataset}_q{idx}_g0_False.pt"
            from experiment.utils import save_graph_with_features
            save_graph_with_features(
                pyg_data,
                os.path.join(rlhf_dir, fname),
                {'mode': 'pruned', 'agent_nums': pyg_data.num_nodes,
                 'is_correct': False, 'question': task_text},
            )

    await asyncio.gather(
        *(_prune_one(idx, r) for idx, r in enumerate(dataset)),
        return_exceptions=True,
    )
    pbar.close()
    print(f"D_pruned done: {saved} graphs saved")
    return saved


# ---------------------------------------------------------------------------
# D_simple
# ---------------------------------------------------------------------------

async def generate_simple_data(args, dataset: list, output_dir: str, rlhf_dir: str = "") -> int:
    """
    Evaluate simple topology graphs (Chain / Star / Layered).
    Saves correct graphs to output_dir and incorrect graphs to rlhf_dir (if provided).

    All (config × task) combinations are dispatched in a single asyncio.gather so
    the vLLM server receives requests from every topology at once rather than
    waiting for one config to finish before starting the next.
    """
    print("\n" + "=" * 20 + f" D_simple ({args.dataset}) " + "=" * 20)
    role_desc = _get_role_description(args.dataset)
    agent_name = _get_agent_name(args.dataset)
    decision_method = _get_decision_method(args.dataset)
    configs = get_simple_configs()
    saved = 0

    # Pre-build one graph template per config (roles sampled once per topology).
    config_graphs: dict = {}
    for mode, agent_num in configs:
        print(f"  Config: {mode}-{agent_num}")
        kwargs = get_kwargs(mode, agent_num)
        roles = random.choices(list(role_desc.keys()), k=agent_num)
        kwargs['node_kwargs'] = [{'role': r} for r in roles]
        config_graphs[(mode, agent_num)] = Graph(
            domain=args.dataset,
            llm_name=args.llm_name,
            agent_names=[agent_name] * agent_num,
            decision_method=decision_method,
            **kwargs,
        )

    # Single semaphore shared across ALL configs × tasks.
    sem = asyncio.Semaphore(args.batch_size)
    total = len(configs) * len(dataset)
    pbar = tqdm(total=total, desc="D_simple (all configs)")

    model_pool_names = None
    if getattr(args, 'model_pool', None):
        from mas_framework.llm.model_pool import load_model_pool
        pool = load_model_pool(None if args.model_pool == 'default' else args.model_pool)
        model_pool_names = list(pool.keys())
        print(f"  Model pool ({len(model_pool_names)}): {model_pool_names}")

    async def _simple_one(global_idx: int, record: dict, mode: str, agent_num: int) -> None:
        nonlocal saved
        realized = copy.deepcopy(config_graphs[(mode, agent_num)])
        input_dict = {'task': record['task']}
        flow_graph = realized.to_pyg_graph(input_dict)
        # Multi-model mode: re-roll each node's base model per task (as in
        # cold_start) so D_simple covers diverse (role, model) combinations.
        if model_pool_names:
            for node_feature in flow_graph.x:
                node_feature['model'] = random.choice(model_pool_names)
        tg = TestGraph(
            domain=args.dataset,
            llm_name=args.llm_name,
            decision_method=decision_method,
            pyg_data=flow_graph,
        )
        async with sem:
            try:
                result = await asyncio.wait_for(
                    tg.arun(input_dict, args.num_rounds),
                    timeout=1200,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                pbar.update(1)
                return
        raw = result
        if isinstance(raw, (list, tuple)) and raw:
            raw = raw[0]
        if not isinstance(raw, str):
            raw = str(raw)
        predicted = _get_predict(args.dataset, raw)
        correct = _is_correct(args.dataset, predicted, record['answer'])
        label = 'True' if correct else 'False'
        fname = "_".join([args.dataset, str(global_idx), mode, str(agent_num), label]) + '.pt'
        if correct:
            saved += 1
            save_graph_with_features(
                flow_graph,
                os.path.join(output_dir, fname),
                {'mode': mode, 'agent_nums': agent_num,
                 'is_correct': True, 'question': record['task']},
            )
        elif rlhf_dir:
            save_graph_with_features(
                flow_graph,
                os.path.join(rlhf_dir, fname),
                {'mode': mode, 'agent_nums': agent_num,
                 'is_correct': False, 'question': record['task']},
            )
        pbar.update(1)

    # All configs × all tasks in one gather — vLLM server sees the full request
    # fan-out immediately instead of one config's worth at a time.
    await asyncio.gather(*(
        _simple_one(i, r, mode, agent_num)
        for mode, agent_num in configs
        for i, r in enumerate(dataset)
    ), return_exceptions=True)
    pbar.close()
    print(f"D_simple done: {saved} graphs saved")
    return saved


# ---------------------------------------------------------------------------
# D_replay
# ---------------------------------------------------------------------------

def generate_replay_data(cold_start_dir: str, output_dir: str, replay_ratio: float) -> int:
    """
    Copy a random subset of successful Phase-1 graphs into the D_eff directory.
    """
    print("\n" + "=" * 20 + " D_replay " + "=" * 20)
    candidates = [
        f for f in os.listdir(cold_start_dir)
        if f.endswith('.pt') and ('True' in f or 'solved' in f.lower())
    ]
    n = min(int(len(candidates) * replay_ratio), len(candidates))
    chosen = random.sample(candidates, n)
    for fn in tqdm(chosen, desc="Copying replay"):
        src = os.path.join(cold_start_dir, fn)
        dst = os.path.join(output_dir, fn)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
    print(f"D_replay done: {n} graphs copied")
    return n


# ---------------------------------------------------------------------------
# Phase-2 fine-tuning
# ---------------------------------------------------------------------------

def run_finetuning(args, d_eff_dir: str, device: str = 'cpu'):
    """
    Load the Phase-1 ARGDesigner checkpoint and fine-tune it on D_eff.
    Saves ef_best_model.pth to args.output_dir.

    device: 'cpu' when vLLM is still running (phase='all'),
            args.device (CUDA) when vLLM has been stopped (phase='finetune').
    """
    print("\n" + "=" * 20 + f" Phase-2 Fine-tuning ({args.dataset}) " + "=" * 20)

    # Load Phase-1 config from final_model.pth (saved by pretrain.py)
    final_ckpt_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    best_ckpt_path  = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Phase-1 checkpoint not found: {best_ckpt_path}")

    finetune_args = Args().update_args()

    # Restore pre-trained configuration from final_model.pth if available
    if os.path.exists(final_ckpt_path):
        saved = torch.load(final_ckpt_path, map_location='cpu')
        for k, v in saved.get('args', {}).items():
            try:
                setattr(finetune_args, k, v)
            except Exception:
                pass

    # Override / set required fields
    finetune_args.dataset        = args.dataset
    finetune_args.data_dir       = args.cold_start_dir   # D_exp (for role mapping)
    finetune_args.data_dir_ef    = d_eff_dir             # D_eff
    finetune_args.experiment_path = args.output_dir
    finetune_args.lr             = args.finetune_lr
    finetune_args.epochs         = args.finetune_epochs
    finetune_args.batch_size     = args.train_batch_size
    finetune_args.seed           = args.seed
    # device is caller-controlled:
    #   'cpu'  — when vLLM server is still resident in GPU memory (phase='all').
    #   CUDA   — when vLLM has been stopped before this stage (phase='finetune').
    finetune_args.device         = device
    finetune_args.pretrain       = False
    finetune_args.model_name     = 'ef_best_model.pth'
    finetune_args.save_model     = True

    # Load D_eff graph dataset
    ef_dataset, _ = gdata.load_graph_dataset(finetune_args, pretrain=False)
    if not ef_dataset.graph_list:
        print("WARNING: No D_eff graphs found — skipping Phase-2 fine-tuning.")
        return

    # Rebuild role mapping from D_eff
    role_to_id = ef_dataset.role_to_id
    id_to_role = ef_dataset.id_to_role
    num_node_types = len(role_to_id) + 2
    finetune_args.role_mapping    = role_to_id
    finetune_args.id_to_role      = id_to_role
    finetune_args.START_TOKEN_ID  = len(role_to_id)
    finetune_args.END_TOKEN_ID    = len(role_to_id) + 1

    stats = gdata.get_data_statistics(ef_dataset.graph_list)
    stats['num_node_labels'] = num_node_types
    stats['num_edge_labels'] = 1

    dataset_ft = gdata.GraphListDataset(ef_dataset.graph_list, finetune_args)
    dataloader_ft = torch.utils.data.DataLoader(
        dataset_ft,
        batch_size=finetune_args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    # Build model and load Phase-1 weights.
    model = ARGDesigner(finetune_args, stats).to(finetune_args.device)
    ckpt = torch.load(best_ckpt_path, map_location=finetune_args.device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded Phase-1 weights from: {best_ckpt_path}")
    print(f"Fine-tuning for {args.finetune_epochs} epochs at lr={args.finetune_lr} ...")

    train(finetune_args, model, dataloader_ft)

    print(f"Phase-2 complete.  ef_best_model.pth saved to: {args.output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    d_eff_dir = os.path.join(args.output_dir, f'FinetuneData_{args.dataset}')
    os.makedirs(d_eff_dir, exist_ok=True)

    # Rejected graphs (incorrect during finetune inference) go here.
    # Kept separate from d_eff_dir so they are never loaded by run_finetuning()
    # or generate_replay_data().  Used only as RLHF rejected candidates.
    rlhf_dir = os.path.join(d_eff_dir, 'rga_rejected')
    os.makedirs(rlhf_dir, exist_ok=True)
    print(f"RLHF rejected dir : {rlhf_dir}")

    # Load finetune split from task_split file
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    split_subdir = _TASK_SPLIT_DIRS.get(args.dataset, f'experiment/{args.dataset}')
    split_path = os.path.join(project_root, split_subdir, f'task_split_{args.dataset}.json')
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Task split file not found: {split_path}\n"
            "Run cold-start first: uv run cold-start --dataset {args.dataset} ..."
        )
    with open(split_path, 'r') as f:
        task_split = json.load(f)
    finetune_indices = task_split.get('finetune_tasks_indices', [])

    # Load full dataset and select finetune subset
    print(f"Loading {args.dataset} dataset from {args.dataset_json} ...")
    all_records = _load_dataset(args.dataset, args.dataset_json)
    finetune_dataset = [all_records[i] for i in finetune_indices]
    print(f"Finetune subset: {len(finetune_dataset)} tasks")

    # ---- Build D_eff (phases: 'all' or 'build_deff') ------------------------
    # D_pruned and D_simple are independent — run them concurrently so the
    # vLLM server handles requests from both sources at the same time.
    if args.phase in ('all', 'build_deff'):
        _d_results = await asyncio.gather(
            generate_pruned_data(args, finetune_dataset, d_eff_dir, rlhf_dir),
            generate_simple_data(args, finetune_dataset, d_eff_dir, rlhf_dir),
            return_exceptions=True,
        )
        pruned_count = _d_results[0] if isinstance(_d_results[0], int) else 0
        simple_count = _d_results[1] if isinstance(_d_results[1], int) else 0
        if isinstance(_d_results[0], BaseException):
            print(f"WARNING: D_pruned raised {type(_d_results[0]).__name__}: {_d_results[0]}")
        if isinstance(_d_results[1], BaseException):
            print(f"WARNING: D_simple raised {type(_d_results[1]).__name__}: {_d_results[1]}")
        generate_replay_data(args.cold_start_dir, d_eff_dir, args.replay_ratio)

        pt_count = len([f for f in os.listdir(d_eff_dir) if f.endswith('.pt')])
        print(f"\nD_eff total: {pt_count} graphs in {d_eff_dir}")

    # ---- Phase-2 NN fine-tuning (phases: 'all' or 'finetune') ---------------
    if args.phase in ('all', 'finetune'):
        # When phase='finetune' the caller has already stopped vLLM, so we can
        # use CUDA freely.  When phase='all' vLLM may still be running, so
        # force CPU to avoid GPU memory contention.
        finetune_device = args.device if args.phase == 'finetune' else 'cpu'
        run_finetuning(args, d_eff_dir, device=finetune_device)

    # ---- Summary ------------------------------------------------------------
    if args.phase == 'build_deff':
        print(f"\nBuild D_eff complete for {args.dataset}.")
        print(f"  D_eff data : {d_eff_dir}")
    elif args.phase == 'finetune':
        print(f"\nNN fine-tuning complete for {args.dataset}.")
        print(f"  Final model: {os.path.join(args.output_dir, 'ef_best_model.pth')}")
    else:  # 'all'
        print(f"\nFine-tuning pipeline complete for {args.dataset}.")
        print(f"  D_eff data : {d_eff_dir}")
        print(f"  Final model: {os.path.join(args.output_dir, 'ef_best_model.pth')}")

    # Cancel any orphaned httpx connection-pool tasks left by asyncio.wait_for
    # timeouts, to avoid hanging in asyncio.run() shutdown.
    _current = asyncio.current_task()
    _pending = [t for t in asyncio.all_tasks() if t is not _current]
    if _pending:
        for t in _pending:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(*_pending, return_exceptions=True),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            pass


def main_cli():
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
    # Force-exit to bypass asyncio shutdown hang: httpx connection-pool background
    # tasks can keep the event loop alive indefinitely after all work is done,
    # preventing the pipeline script from proceeding to the next stage.
    os._exit(0)


if __name__ == '__main__':
    main_cli()
