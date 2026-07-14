"""
run_reflexion_batch.py
Baseline - Reflexion (Shinn et al., 2023)
Author : Gurdiraj Bal (z5386590)

CLI batch driver for the Reflexion baseline. Parallel to
project/scripts/generate_samples.py's run_repair_generation(), but
standalone - reuses shared JSONL/trajectory/run-log plumbing from
generate_samples.py (which is already generic over repair_strategy)
rather than duplicating it, while the Reflexion-specific per-task loop
lives in reflexion_repair.run_reflexion_repair().

Output artifacts are written to the *shared* project/samples/ and
project/results/ directories, using the same naming convention as the
existing `cot`/`minimal` repair runs (prompt_strategy label "cop", since
that is what those runs used for path purposes even though their actual
Round-0 completions were reused from `..._cgo.jsonl` - see
decisions.md / research_log.md for that history), so the reflexion
trajectory sits side by side with `_cot`/`_minimal` and the existing
Docker EvalPlus command pattern, analyze_failures.py, and safety_oracle.py
all work unchanged.

Scope: both HumanEval+ and MBPP+ (select via --dataset). The
Reflexion-specific modules were already dataset-generic; only this driver
needed the dataset wired through, the matching Round-0 seed selected, and
the MBPP `solution` vs HumanEval `completion` seed key handled.

Usage
    # HumanEval+ (Groq, as originally run)
    python run_reflexion_batch.py --dataset humaneval --model llama-3.3-70b-versatile \
        --backend groq --max_repair_rounds 2 --reflexion_memory_size 1 \
        --resume-missing

    # MBPP+ on UNSW Katana (vLLM OpenAI-compatible server on the GPU node).
    # Serve Llama-3.3-70B-Instruct with --served-model-name llama-3.3-70b-versatile
    # so the model slug (and thus output filenames) match the existing runs.
    python run_reflexion_batch.py --dataset mbpp --model llama-3.3-70b-versatile \
        --backend katana --max_repair_rounds 2 --reflexion_memory_size 1 \
        --resume-missing
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Optional

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_samples import (  # noqa: E402
    BASE_DIR,
    SLEEP,
    apply_limit_to_path,
    get_client,
    get_output_path,
    get_problems,
    get_repair_round_path,
    get_repair_trajectory_path,
    load_env_file,
    load_existing_samples,
    load_existing_trajectories,
    print_repair_summary,
    save_repair_run_log,
    summarize_repair_trajectories,
    write_repair_round_jsonls,
    ENV_PATH,
)

from reflexion_repair import run_reflexion_repair  # noqa: E402


# Matches the prompt_strategy label the existing cot/minimal repair runs
# used for path purposes (see module docstring).
PROMPT_STRATEGY_LABEL = "cop"
REPAIR_STRATEGY_LABEL = "reflexion"

# Per-dataset Round-0 seed, matching what the existing cot/minimal repair runs
# reused (see their _repair_run_log.json round0_source_jsonl fields): the `cgo`
# generation for each dataset. HumanEval keeps the historical unprefixed name.
DEFAULT_ROUND0_SOURCE = {
    "humaneval": os.path.join(BASE_DIR, "samples", "llama-33-70b-versatile_t02_cgo.jsonl"),
    "mbpp": os.path.join(BASE_DIR, "samples", "mbpp_llama-33-70b-versatile_t02_cgo.jsonl"),
}


def default_round0_source(dataset: str) -> str:
    return DEFAULT_ROUND0_SOURCE[dataset]


def summarize_confusion(results: list) -> dict:
    """Aggregate a TP/FN/FP/TN confusion count between self_test_passed
    and ground-truth passed, across every round of every task.

    Mirrors Shinn et al. Table 2: TP = both pass, FN = self-test fails but
    ground truth passes, FP = self-test passes but ground truth fails,
    TN = both fail.
    """
    counts = Counter()
    for result in results:
        for round_entry in result.get("rounds", []):
            self_ok = round_entry.get("self_test_passed", False)
            real_ok = round_entry.get("passed", False)
            if self_ok and real_ok:
                counts["TP"] += 1
            elif not self_ok and real_ok:
                counts["FN"] += 1
            elif self_ok and not real_ok:
                counts["FP"] += 1
            else:
                counts["TN"] += 1
    total = sum(counts.values())
    return {
        "total_rounds": total,
        "TP": counts["TP"],
        "FN": counts["FN"],
        "FP": counts["FP"],
        "TN": counts["TN"],
    }


def print_confusion(confusion: dict) -> None:
    total = confusion["total_rounds"]
    print("\n  Self-test vs ground-truth confusion (all rounds):")
    for key in ("TP", "FN", "FP", "TN"):
        count = confusion[key]
        rate = round(count / total * 100, 1) if total else 0.0
        print(f"    {key}: {count}/{total} ({rate:.1f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Reflexion baseline over HumanEval+ or MBPP+."
    )
    parser.add_argument(
        "--dataset",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="Benchmark dataset. Default: humaneval.",
    )
    parser.add_argument(
        "--model",
        default="llama-3.3-70b-versatile",
        help="Model name for the inference backend.",
    )
    parser.add_argument(
        "--backend",
        default="groq",
        choices=["ollama", "groq", "cerebras", "openrouter", "gemini", "katana"],
        help="Inference backend. Default: groq. Use 'katana' for a vLLM "
             "OpenAI-compatible server on a UNSW Katana GPU node.",
    )
    parser.add_argument(
        "--max_repair_rounds",
        type=int,
        default=2,
        help="Maximum repair rounds (default: 2, matching the existing cot/minimal runs).",
    )
    parser.add_argument(
        "--reflexion_memory_size",
        type=int,
        default=1,
        help="Number of past self-reflections retained in memory (paper default: 1).",
    )
    parser.add_argument(
        "--round0_source_jsonl",
        default=None,
        help="Existing samples JSONL to reuse for Round 0 completions. "
             "Defaults to the matching cgo seed for the chosen --dataset.",
    )
    parser.add_argument(
        "--resume-missing",
        "--resume",
        dest="resume_missing",
        action="store_true",
        help="Keep existing trajectories and continue only missing task IDs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional task limit for smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(ENV_PATH)

    dataset = args.dataset
    round0_source_jsonl = args.round0_source_jsonl or default_round0_source(dataset)
    # MBPP seeds store Round-0 code under "solution"; HumanEval under "completion"
    # (see generate_samples.py get_output_path / write_samples_jsonl, decisions.md D5).
    sample_key = "solution" if dataset == "mbpp" else "completion"

    base_output_path = apply_limit_to_path(
        get_output_path(args.model, PROMPT_STRATEGY_LABEL, dataset),
        args.limit,
    )
    round_paths = {
        round_num: get_repair_round_path(
            base_output_path,
            REPAIR_STRATEGY_LABEL,
            round_num,
        )
        for round_num in range(args.max_repair_rounds + 1)
    }
    trajectory_path = get_repair_trajectory_path(
        model=args.model,
        prompt_strategy=PROMPT_STRATEGY_LABEL,
        dataset=dataset,
        repair_strategy=REPAIR_STRATEGY_LABEL,
        limit=args.limit,
    )

    os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    print(f"Baseline : Reflexion (Shinn et al., 2023)")
    print(f"Backend  : {args.backend}")
    print(f"Dataset  : {dataset}")
    print(f"Model    : {args.model}")
    print(f"Max rds  : {args.max_repair_rounds}")
    print(f"Memory Ω : {args.reflexion_memory_size}")
    print(f"Limit    : {args.limit}")
    print(f"R0 source: {round0_source_jsonl}")
    print(f"Traj     : {trajectory_path}")
    print()

    if not os.path.exists(round0_source_jsonl):
        raise FileNotFoundError(f"Round 0 source JSONL not found: {round0_source_jsonl}")

    client = get_client(args.backend)
    problems = get_problems(dataset)
    problem_items = list(problems.items())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        problem_items = problem_items[:args.limit]

    total = len(problem_items)
    existing_trajectories = (
        load_existing_trajectories(trajectory_path) if args.resume_missing else {}
    )
    round0_source_samples = load_existing_samples(round0_source_jsonl)

    trajectories_by_id = dict(existing_trajectories)
    skipped = []
    total_api_calls = 0
    sleep_s = SLEEP.get(args.backend, 1.0)

    if args.resume_missing:
        existing_count = sum(
            1 for task_id, _ in problem_items if task_id in existing_trajectories
        )
        print(f"Resume   : found {existing_count}/{total} existing trajectories")
        print()

    for index, (task_id, task) in enumerate(problem_items, start=1):
        if task_id in existing_trajectories:
            print(f"[{index:>3}/{total}] {task_id} ... RESUME")
            continue

        source_sample = round0_source_samples.get(task_id)
        if source_sample is None or sample_key not in source_sample:
            print(f"[{index:>3}/{total}] {task_id} ... SKIPPED (missing Round 0 source)")
            skipped.append(task_id)
            continue

        print(f"[{index:>3}/{total}] {task_id} ...", flush=True)

        result = run_reflexion_repair(
            task_id=task_id,
            initial_completion=source_sample[sample_key],
            dataset=dataset,
            client=client,
            model=args.model,
            backend=args.backend,
            max_repair_rounds=args.max_repair_rounds,
            reflexion_memory_size=args.reflexion_memory_size,
        )
        total_api_calls += result.get("api_calls", 0)

        # An API failure (e.g. Groq daily-quota exhaustion) means this task's
        # self-tests/reflections are degraded. Do NOT persist it — otherwise
        # --resume-missing would skip it and silently keep the corrupt data,
        # and every remaining task would churn through the same exhausted quota
        # saving garbage. Abort cleanly here; the already-completed tasks are
        # written below, and a later --resume-missing re-does this one.
        if result.get("api_failed"):
            print(
                f"[{index:>3}/{total}] {task_id} ... ABORT: API failure "
                f"(likely {args.backend} quota exhausted). Not saving this task. "
                f"Re-run with --resume-missing to continue."
            )
            break

        trajectories_by_id[task_id] = result

        if result.get("solved"):
            print(f"[{index:>3}/{total}] {task_id} -> solved at round {result['solved_at_round']}")
        else:
            flag = " (stopped early on self-test pass)" if result.get("stopped_early_self_test_pass") else ""
            print(f"[{index:>3}/{total}] {task_id} -> unsolved after {args.max_repair_rounds} rounds{flag}")

        ordered_partial = [
            trajectories_by_id[tid]
            for tid, _ in problem_items
            if tid in trajectories_by_id
        ]
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(ordered_partial, f, indent=2)

        time.sleep(sleep_s)

    task_ids = [task_id for task_id, _ in problem_items]
    ordered_results = [
        trajectories_by_id[task_id]
        for task_id in task_ids
        if task_id in trajectories_by_id
    ]

    with open(trajectory_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2)
    print(f"\n  Trajectories -> {trajectory_path} ({len(ordered_results)} rows)")

    write_repair_round_jsonls(
        round_paths=round_paths,
        trajectories_by_id=trajectories_by_id,
        task_ids=task_ids,
        dataset=dataset,
    )

    if skipped:
        print(f"  Skipped {len(skipped)} tasks: {skipped}")

    summary = summarize_repair_trajectories(ordered_results, REPAIR_STRATEGY_LABEL)
    print_repair_summary(summary, REPAIR_STRATEGY_LABEL)

    confusion = summarize_confusion(ordered_results)
    print_confusion(confusion)
    summary["self_test_confusion"] = confusion

    print()
    print(f"API calls: total={total_api_calls}")

    save_repair_run_log(
        round_paths=round_paths,
        trajectory_path=trajectory_path,
        round0_source_jsonl=round0_source_jsonl,
        model=args.model,
        backend=args.backend,
        dataset=dataset,
        prompt_strategy=PROMPT_STRATEGY_LABEL,
        repair_strategy=REPAIR_STRATEGY_LABEL,
        max_repair_rounds=args.max_repair_rounds,
        total_api_calls=total_api_calls,
        round0_generation_api_calls=0,
        round0_reused_count=len(ordered_results),
        repair_api_calls=total_api_calls,
        skipped=skipped,
        summary=summary,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
