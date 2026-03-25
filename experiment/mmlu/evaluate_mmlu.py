import os
import json
import math
import asyncio
import time
import torch
import numpy as np
import argparse
import random
from tqdm import tqdm
from typing import List, Any, Iterator
import sys
import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens
from sentence_transformers import SentenceTransformer
from mas_framework.graph.graph import TestGraph
from experiment.utils import Accuracy, load_model, generate_graph, convert_to_pyg_graph
from benchmark_datasets.mmlu_dataset import MMLUDataset
from benchmark_datasets.MMLU.download import download
from experiment.eval_checkpoint import load_checkpoint, log_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on MMLU")
    parser.add_argument('--agent_nums', type=int, default=6,
                        help="number of agents")
    parser.add_argument('--data_dir', type=str, default='../../ColdStartData_mmlu',
                        help="data directory")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="number of inference rounds per query")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
                        help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="dataset name")
    parser.add_argument('--limit_questions', type=int, default=153,
                        help="limit number of questions to evaluate")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="decision method for final node")
    parser.add_argument('--model_path', type=str, default='output/xxx',
                        help="path to pretrained model")
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help="evaluation batch size")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed")
    parser.add_argument('--embedding_model', type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="model for task embeddings")
    parser.add_argument('--output_file', type=str, default='mmlu_eval_results.jsonl',
                        help="file to save per-sample evaluation results")
    parser.add_argument('--summary_log_file', type=str, default='./res_logs/evaluation_summary.jsonl',
                        help="log file to record evaluation summaries")

    return parser.parse_args()


async def evaluate(
        model,
        dataset,
        sentence_model,
        role_constraints_dict,
        args
) -> float:
    """Evaluate model on MMLU dataset"""
    print(f"Evaluating model on {dataset.__class__.__name__}")

    accuracy = Accuracy()
    limit_questions = args.limit_questions
    results_list, done_ids, solved_tasks = load_checkpoint(args.output_file, solved_key='is_correct')
    _wall_start = time.time()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / args.eval_batch_size))

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=args.eval_batch_size)), total=num_batches):
        print(f"{'-' * 80}")

        start_ts = time.time()
        answer_tasks = []
        metadata_list = []

        for i, record in enumerate(record_batch):
            input_dict = dataset.record_to_input(record)
            task_text = input_dict['task']
            true_answer = dataset.record_to_target_answer(record)
            question_id = i_batch * args.eval_batch_size + i + 1

            if str(question_id) in done_ids:
                continue

            task_embedding = torch.tensor(
                sentence_model.encode(task_text),
                device=model.args.device
            ).float()

            generated_graph = generate_graph(
                model,
                task_embedding,
                role_constraints_dict,
                question_id=question_id
            )

            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=convert_to_pyg_graph(generated_graph[0], task_text)
            )
            answer_tasks.append(asyncio.create_task(tg.arun(input_dict, args.num_rounds)))
            metadata_list.append({
                'question_id': question_id,
                'task_text': task_text,
                'true_answer': true_answer,
                'graph': generated_graph[0],
            })

        raw_results = await asyncio.gather(*answer_tasks)

        for meta, raw_answer in zip(metadata_list, raw_results):
            answer = dataset.postprocess_answer(raw_answer)
            is_correct = accuracy.update(answer, meta['true_answer'])
            if is_correct:
                solved_tasks += 1

            print(f"Accuracy: {accuracy.print()} | "
                  f"Cost: ${Cost.instance().value:.4f} | "
                  f"Tokens: P({int(PromptTokens.instance().value)}), C({int(CompletionTokens.instance().value)})")

            results_list.append({
                'id': meta['question_id'],
                'question': meta['task_text'],
                'true_answer': meta['true_answer'],
                'predicted_answer': answer,
                'raw_response': raw_answer,
                'is_correct': is_correct,
                'num_nodes': meta['graph'].number_of_nodes(),
                'num_edges': meta['graph'].number_of_edges(),
            })

        # Flush results to disk after every batch (overwrites with full list so far)
        os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for r in results_list:
                f.write(json.dumps(r) + '\n')

        log_batch(i_batch, num_batches, solved_tasks, len(results_list), data_len,
                  time.time() - start_ts, _wall_start)

    accuracy.print()
    print("Evaluation complete!")
    # Compute final accuracy across all results (includes any checkpoint-resumed items)
    total_correct = sum(1 for r in results_list if r.get('is_correct'))
    return total_correct / len(results_list) * 100 if results_list else 0.0


async def main(ef=True):
    args = parse_args()

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("Loading MMLU dataset...")
    download()
    dataset_test = MMLUDataset('val', data_dir=args.data_dir)

    print(f"Loading Sentence Transformer model: {args.embedding_model}")
    sentence_model = SentenceTransformer(args.embedding_model)

    from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
    role_constraints_dict = {role: desc for role, desc in ROLE_DESCRIPTION.items()}

    print(f"Loading pretrained model: {args.model_path}")
    args.model_name = 'ef_best' if ef else 'best'
    print('model_name', args.model_name)
    model = load_model(args.model_path, ef=ef)

    score = await evaluate(
        model=model,
        dataset=dataset_test,
        sentence_model=sentence_model,
        role_constraints_dict=role_constraints_dict,
        args=args
    )

    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value
    total_tasks = min(len(dataset_test), args.limit_questions) if args.limit_questions is not None else len(dataset_test)

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Model path: {args.model_path}")
    print(f"Total tasks: {total_tasks}\nFinal accuracy (Pass@1): {score:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)
    print(f"Detailed results saved to: {args.output_file}")

    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "mmlu",
        "model_path": args.model_path + args.model_name,
        "llm_name": args.llm_name,
        "total_tasks": total_tasks,
        "pass_at_1": score,
        "cost": final_cost,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "detail_file": args.output_file
    }

    try:
        os.makedirs(os.path.dirname(args.summary_log_file) or '.', exist_ok=True)
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
        print(f"Summary log appended to: {args.summary_log_file}")
    except Exception as e:
        print(f"Failed to write summary log file: {e}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main(True))
