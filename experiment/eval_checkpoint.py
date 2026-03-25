"""
Shared helpers for all evaluate_*.py scripts.

  - load_checkpoint : resume from a partial output file
  - log_batch       : per-batch progress line printed to stdout
"""

import json
import os
import time
from typing import List, Set, Tuple


def load_checkpoint(
    output_file: str,
    solved_key: str = "is_solved",
) -> Tuple[List[dict], Set[str], int]:
    """Load partial results so a re-submitted job continues where it left off.

    Returns
    -------
    results_list : list of already-completed result dicts
    done_ids     : set of task_id / id strings already in the file
    solved_count : number of those that were marked correct/solved
    """
    if not os.path.exists(output_file):
        return [], set(), 0

    results_list, done_ids, solved = [], set(), 0
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                results_list.append(r)
                tid = str(r.get("task_id") or r.get("id") or "")
                if tid:
                    done_ids.add(tid)
                if r.get(solved_key):
                    solved += 1
    except Exception as exc:
        print(f"[Checkpoint] Warning: could not read '{output_file}' ({exc}). Starting fresh.")
        return [], set(), 0

    if results_list:
        print(
            f"\n[Checkpoint] Resuming from existing output file:\n"
            f"  {output_file}\n"
            f"  Already done : {len(done_ids)} tasks  ({solved} solved)\n"
            f"  Skipping these and continuing from task {len(done_ids) + 1}.\n"
        )
    return results_list, done_ids, solved


def log_batch(
    i_batch: int,
    num_batches: int,
    solved: int,
    total_done: int,
    total_tasks: int,
    batch_time: float,
    wall_start: float,
) -> None:
    """Print a one-line progress summary after each batch."""
    acc = solved / total_done * 100 if total_done > 0 else 0.0
    elapsed = time.time() - wall_start
    rate = total_done / elapsed if elapsed > 0 else 0.0
    remaining = max(total_tasks - total_done, 0)
    eta_s = int(remaining / rate) if rate > 0 else 0
    eta_str = f"{eta_s // 60}m{eta_s % 60:02d}s"
    pct = total_done / total_tasks * 100 if total_tasks > 0 else 0.0
    print(
        f"[Batch {i_batch + 1:>3}/{num_batches}] "
        f"{total_done}/{total_tasks} ({pct:.1f}%) | "
        f"acc={acc:.1f}% ({solved}/{total_done}) | "
        f"batch={batch_time:.1f}s | "
        f"elapsed={elapsed:.0f}s | "
        f"ETA={eta_str}"
    )
