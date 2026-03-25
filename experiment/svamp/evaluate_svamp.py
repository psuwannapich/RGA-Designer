import os
import json
import math
import time
import torch
import random
import argparse
import asyncio
import datetime
import numpy as np
import sys
from tqdm import tqdm
from typing import Iterator, List, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from mas_framework.graph.graph import TestGraph
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph
from benchmark_datasets.gsm8k_dataset import svamp_data_process, gsm_get_predict
from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
from experiment.eval_checkpoint import load_checkpoint, log_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ARG-Designer on SVAMP")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to trained ARGDesigner checkpoint directory")
    parser.add_argument('--dataset_path', type=str,
                        default='../../datasets/SVAMP/SVAMP.json',
                        help="Path to SVAMP JSON file")
    parser.add_argument('--task_split_path', type=str,
                        default='./task_split_svamp.json',
                        help="Task split JSON (optional — if absent, all samples are used as test set)")
    parser.add_argument('--llm_name', type=str, default='Qwen/Qwen3-8B',
                        help="LLM name (HuggingFace ID or Ollama short name)")
    parser.add_argument('--decision_method', type=str, default='FinalRefer')
    parser.add_argument('--output_file', type=str, default='svamp_eval_results.jsonl')
    parser.add_argument('--summary_log_file', type=str,
                        default='./res_logs/evaluation_summary.jsonl')
    parser.add_argument('--limit', type=int, default=None,
                        help="Cap number of test samples (default: all)")
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_environment(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


async def main(ef: bool = True):
    args = parse_args()
    setup_environment(args.seed)

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    print("Loading model ...")
    model = load_model(args.model_path, ef=ef)
    model.eval()
    args.model_name = 'ef_best' if ef else 'best'

    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Load dataset
    import json as _json
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        raw = _json.load(f)
    full_dataset = svamp_data_process(raw)

    # Determine test indices
    if os.path.exists(args.task_split_path):
        with open(args.task_split_path, 'r') as f:
            task_split = json.load(f)
        test_indices = task_split.get('test_indices', list(range(len(full_dataset))))
    else:
        print(f"No task split file found at '{args.task_split_path}' — using all {len(full_dataset)} samples.")
        test_indices = list(range(len(full_dataset)))

    dataset = [full_dataset[i] for i in test_indices]
    if args.limit:
        dataset = dataset[:args.limit]
        test_indices = test_indices[:args.limit]
    print(f"Loaded {len(dataset)} SVAMP test samples.")

    total_tasks = len(dataset)
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    results_list, done_ids, solved_tasks = load_checkpoint(args.output_file)
    _wall_start = time.time()

    def eval_loader(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
        buf = []
        for item in data:
            buf.append(item)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    num_batches = math.ceil(total_tasks / args.eval_batch_size)
    pbar = tqdm(enumerate(eval_loader(dataset, args.eval_batch_size)),
                total=num_batches, desc="Evaluating SVAMP")

    for i_batch, record_batch in pbar:
        answer_tasks = []
        metadata_list = []
        _batch_start = time.time()

        for i_record, record in enumerate(record_batch):
            task_text = record['task']
            true_answer = record['answer']
            global_idx = i_batch * args.eval_batch_size + i_record
            task_id = f"task_{test_indices[global_idx]}"

            if task_id in done_ids:
                continue

            try:
                task_embedding = torch.tensor(
                    sentence_model.encode(task_text),
                    device=model.args.device
                ).float()
                graphs = generate_graph(model, task_embedding, ROLE_DESCRIPTION, global_idx)
                if not graphs:
                    raise RuntimeError("Graph generation failed.")
                pyg_data = convert_to_pyg_graph(graphs[0], task_text)
                tg = TestGraph(domain='svamp', llm_name=args.llm_name,
                               decision_method=args.decision_method, pyg_data=pyg_data)
                answer_tasks.append(tg.arun({'task': task_text}, num_rounds=1))
                metadata_list.append({'task_id': task_id, 'task_text': task_text,
                                       'true_answer': true_answer, 'graph': graphs[0]})
            except Exception as e:
                print(f"Error preparing {task_id}: {e}")
                results_list.append({'task_id': task_id, 'question': task_text,
                                      'true_answer': true_answer, 'predicted_answer': None,
                                      'is_solved': False, 'error': str(e)})

        if not answer_tasks:
            continue

        all_results = await asyncio.gather(*answer_tasks, return_exceptions=True)

        for i, result in enumerate(all_results):
            meta = metadata_list[i]
            if isinstance(result, Exception):
                results_list.append({'task_id': meta['task_id'], 'question': meta['task_text'],
                                      'true_answer': meta['true_answer'], 'predicted_answer': None,
                                      'is_solved': False, 'error': str(result)})
                continue

            raw = result[0] if isinstance(result, list) and result else result
            predicted = gsm_get_predict(str(raw))
            is_solved = False
            try:
                is_solved = float(predicted) == float(meta['true_answer'])
            except (ValueError, TypeError):
                pass

            if is_solved:
                solved_tasks += 1
            results_list.append({
                'task_id': meta['task_id'], 'question': meta['task_text'],
                'true_answer': meta['true_answer'], 'predicted_answer': predicted,
                'raw_response': str(raw), 'is_solved': is_solved,
                'num_nodes': meta['graph'].number_of_nodes(),
                'num_edges': meta['graph'].number_of_edges(),
            })

        acc = solved_tasks / len(results_list) * 100 if results_list else 0
        pbar.set_postfix({'Accuracy': f'{acc:.2f}% ({solved_tasks}/{len(results_list)})'})

        with open(args.output_file, 'w', encoding='utf-8') as f:
            for r in results_list:
                f.write(json.dumps(r) + '\n')

        log_batch(i_batch, num_batches, solved_tasks, len(results_list), total_tasks,
                  time.time() - _batch_start, _wall_start)

    pass_at_1 = solved_tasks / total_tasks * 100 if total_tasks > 0 else 0
    print(f"\n{'='*50}\nSVAMP Evaluation Summary")
    print(f"Model: {args.model_path}  |  LLM: {args.llm_name}")
    print(f"Total: {total_tasks}  Solved: {solved_tasks}  Pass@1: {pass_at_1:.2f}%")
    print(f"Results saved to: {args.output_file}")

    log_record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'dataset': 'svamp', 'model_path': args.model_path + args.model_name,
        'llm_name': args.llm_name, 'total_tasks': total_tasks,
        'solved_tasks': solved_tasks, 'pass_at_1': pass_at_1,
        'detail_file': args.output_file,
    }
    try:
        os.makedirs(os.path.dirname(args.summary_log_file) or '.', exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
    except Exception as e:
        print(f"Failed to write summary log: {e}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(True))
