"""
run_repair_batch.py
Thesis â€” Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

Batch runner for the iterative self-repair loop. Runs a selected slice of
existing Round 0 samples through self_repair.py, prints pass@1 summaries by
repair round, and saves the full per-task trajectories for later analysis.

Usage
    python run_repair_batch.py --dataset humaneval \
        --jsonl_path ../samples/llama-33-70b-versatile_t02_cgo.jsonl \
        --backend groq --model llama-3.3-70b-versatile \
        --max_repair_rounds 2 --repair_strategy cot --first_n 30 \
        --start_index 0
"""

import argparse
import json
import os
import sys
from collections import Counter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_samples import ENV_PATH, get_client, load_env_file
from self_repair import _load_completion_from_jsonl, run_self_repair


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BACKEND = "groq"
DEFAULT_HUMANEVAL_JSONL = os.path.join(
    BASE_DIR,
    "samples",
    "llama-33-70b-versatile_t02_cgo.jsonl",
)
DEFAULT_MBPP_JSONL = os.path.join(
    BASE_DIR,
    "samples",
    "mbpp_llama-33-70b-versatile_t02_cgo.jsonl",
)


def load_task_ids_from_jsonl(jsonl_path: str) -> list[str]:
    """Load task IDs from a samples JSONL file, preserving file order."""
    task_ids = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "task_id" not in sample:
                raise KeyError(f"Line {line_num} has no task_id")
            task_ids.append(sample["task_id"])
    return task_ids


def parse_task_ids(raw_task_ids: Optional[str]) -> Optional[list[str]]:
    """Parse a comma-separated task ID list."""
    if raw_task_ids is None:
        return None
    task_ids = [task_id.strip() for task_id in raw_task_ids.split(",")]
    task_ids = [task_id for task_id in task_ids if task_id]
    if not task_ids:
        raise ValueError("--task_ids was provided but no task IDs were parsed")
    return task_ids


def parse_task_range(raw_task_range: Optional[str], all_task_ids: list[str]) -> Optional[list[str]]:
    """Parse a 0-based file-order slice in START:END form."""
    if raw_task_range is None:
        return None

    if ":" not in raw_task_range:
        raise ValueError("--task_range must use START:END syntax")

    start_text, end_text = raw_task_range.split(":", 1)
    start = int(start_text) if start_text else 0
    end = int(end_text) if end_text else len(all_task_ids)

    if start < 0 or end < start:
        raise ValueError("--task_range must satisfy 0 <= START <= END")

    return all_task_ids[start:end]


def select_task_ids(args: argparse.Namespace, all_task_ids: list[str]) -> list[str]:
    """Select task IDs by explicit list, file-order range, or start + first N."""
    explicit_task_ids = parse_task_ids(args.task_ids)
    range_task_ids = parse_task_range(args.task_range, all_task_ids)

    selectors = [
        explicit_task_ids is not None,
        range_task_ids is not None,
    ]
    if sum(selectors) > 1:
        raise ValueError("Use only one of --task_ids or --task_range")

    if args.start_index < 0:
        raise ValueError("--start_index must be at least 0")

    if args.start_index and (explicit_task_ids is not None or range_task_ids is not None):
        raise ValueError("--start_index can only be used with --first_n/default slicing")

    if explicit_task_ids is not None:
        missing = [task_id for task_id in explicit_task_ids if task_id not in all_task_ids]
        if missing:
            raise KeyError(f"Task IDs not found in JSONL: {missing}")
        return explicit_task_ids

    if range_task_ids is not None:
        return range_task_ids

    first_n = args.first_n
    if first_n is None:
        first_n = 30 if args.dataset == "humaneval" else len(all_task_ids)

    if first_n < 1:
        raise ValueError("--first_n must be at least 1")

    return all_task_ids[args.start_index:args.start_index + first_n]


def default_jsonl_path(dataset: str) -> str:
    """Return the default samples path for the selected dataset."""
    if dataset == "humaneval":
        return DEFAULT_HUMANEVAL_JSONL
    if dataset == "mbpp":
        return DEFAULT_MBPP_JSONL
    raise ValueError(f"Unsupported dataset: {dataset}")


def default_output_path(dataset: str, total: int, repair_strategy: str) -> str:
    """Build the default batch result path."""
    return os.path.join(
        BASE_DIR,
        "results",
        f"repair_batch_{dataset}_{total}_{repair_strategy}.json",
    )


def solved_by_round(result: dict, round_num: int) -> bool:
    """Return True if a task solved at or before round_num."""
    solved_at = result.get("solved_at_round")
    return solved_at is not None and solved_at <= round_num


def format_rate(count: int, total: int) -> str:
    """Format a count and percentage."""
    pct = (count / total * 100) if total else 0.0
    return f"{count}/{total} ({pct:.1f}%)"


def describe_result(result: dict, max_repair_rounds: int) -> str:
    """Format a one-line result for live progress output."""
    if result.get("solved"):
        return f"solved at round {result['solved_at_round']}"
    return (
        f"unsolved after {max_repair_rounds} rounds, "
        f"stalled={result.get('stalled', False)}"
    )


def summarize_results(results: list[dict], repair_strategy: str) -> None:
    """Print aggregate pass@1 and repair diagnostics."""
    total = len(results)
    round0_count = sum(1 for result in results if solved_by_round(result, 0))
    round1_count = sum(1 for result in results if solved_by_round(result, 1))
    round2_count = sum(1 for result in results if solved_by_round(result, 2))
    stalled_count = sum(1 for result in results if result.get("stalled", False))

    round0_rate = round0_count / total * 100 if total else 0.0
    round2_rate = round2_count / total * 100 if total else 0.0
    improvement = round2_rate - round0_rate

    round0_error_counts = Counter()
    cot_method_counts = Counter()

    for result in results:
        if result.get("rounds"):
            r0 = result["rounds"][0]
            if not r0.get("passed", False):
                round0_error_counts[r0.get("error_type")] += 1

        if repair_strategy == "cot":
            for round_entry in result.get("rounds", []):
                method = round_entry.get("cot_extraction_method")
                if round_entry.get("round", 0) > 0 and method:
                    cot_method_counts[method] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total tasks                : {total}")
    print(f"  Round 0 pass@1             : {format_rate(round0_count, total)}")
    print(f"  Cumulative Round 1 pass@1  : {format_rate(round1_count, total)}")
    print(f"  Cumulative Round 2 pass@1  : {format_rate(round2_count, total)}")
    print(f"  Total improvement          : {improvement:+.1f} pp")
    print(f"  Stalled tasks              : {stalled_count}")

    if round0_error_counts:
        print("\n  Round 0 error type breakdown:")
        for err_type, count in round0_error_counts.most_common():
            print(f"    {err_type}: {count}")

    if repair_strategy == "cot":
        print("\n  CoT extraction method breakdown:")
        for method in ["marker", "fence", "raw_fallback"]:
            print(f"    {method}: {cot_method_counts[method]}")


def run_batch(args: argparse.Namespace, task_ids: list[str]) -> list[dict]:
    """Run self-repair over the selected task IDs."""
    load_env_file(ENV_PATH)
    client = get_client(args.backend)

    results = []
    total = len(task_ids)

    for index, task_id in enumerate(task_ids, start=1):
        print(f"\n[{index}/{total}] {task_id}")
        completion = _load_completion_from_jsonl(
            args.jsonl_path,
            task_id,
            args.dataset,
        )
        result = run_self_repair(
            task_id=task_id,
            initial_completion=completion,
            dataset=args.dataset,
            client=client,
            model=args.model,
            backend=args.backend,
            max_repair_rounds=args.max_repair_rounds,
            repair_strategy=args.repair_strategy,
        )
        results.append(result)
        print(f"[{index}/{total}] {task_id} -> {describe_result(result, args.max_repair_rounds)}")

    return results


def save_results(results: list[dict], output_path: str) -> None:
    """Save full per-task repair trajectories."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run self-repair over a batch of existing samples."
    )
    parser.add_argument(
        "--jsonl_path",
        default=None,
        help="Path to an existing samples JSONL file.",
    )
    parser.add_argument(
        "--dataset",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset. Default: humaneval.",
    )
    parser.add_argument(
        "--task_ids",
        default=None,
        help="Comma-separated task IDs to process, e.g. HumanEval/0,HumanEval/4.",
    )
    parser.add_argument(
        "--task_range",
        default=None,
        help="0-based file-order slice START:END, e.g. 0:30.",
    )
    parser.add_argument(
        "--first_n",
        type=int,
        default=None,
        help="Process the first N task IDs in JSONL file order.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="0-based JSONL file-order index where the first-N slice begins. Default: 0.",
    )
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["ollama", "groq", "cerebras", "openrouter", "gemini"],
        help=f"Inference backend. Default: {DEFAULT_BACKEND}.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name for the inference backend. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--max_repair_rounds",
        "--max_rounds",
        dest="max_repair_rounds",
        type=int,
        default=2,
        help="Maximum repair rounds per task. Default: 2.",
    )
    parser.add_argument(
        "--repair_strategy",
        default="cot",
        choices=["minimal", "cot"],
        help="Repair prompt strategy to use. Default: cot.",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        help="Where to save full batch results JSON.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print selected task IDs without making API calls.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jsonl_path is None:
        args.jsonl_path = default_jsonl_path(args.dataset)

    all_task_ids = load_task_ids_from_jsonl(args.jsonl_path)
    task_ids = select_task_ids(args, all_task_ids)

    if not task_ids:
        raise RuntimeError("No task IDs selected")

    if args.output_path is None:
        args.output_path = default_output_path(
            args.dataset,
            len(task_ids),
            args.repair_strategy,
        )

    print(f"Dataset   : {args.dataset}")
    print(f"JSONL     : {args.jsonl_path}")
    print(f"Backend   : {args.backend}")
    print(f"Model     : {args.model}")
    print(f"Max rounds: {args.max_repair_rounds}")
    print(f"Strategy  : {args.repair_strategy}")
    print(f"Output    : {args.output_path}")
    print(f"Dry run   : {args.dry_run}")
    print()

    if args.dry_run:
        print(f"Selected {len(task_ids)} task IDs:")
        for index, task_id in enumerate(task_ids, start=1):
            print(f"  [{index}/{len(task_ids)}] {task_id}")
        return

    results = run_batch(args, task_ids)
    summarize_results(results, args.repair_strategy)
    save_results(results, args.output_path)
    print(f"\nSaved full batch results to: {args.output_path}")


if __name__ == "__main__":
    main()
