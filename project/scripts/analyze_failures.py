"""
analyze_failures.py

Extract failed EvalPlus tasks into a compact JSON report for prompt-refinement
analysis. This script does not call an LLM; it prepares structured failure
context that can be reviewed manually or passed to a later diagnostic step.

Usage:
    python scripts/analyze_failures.py \
      --dataset mbpp \
      --samples samples/mbpp_llama-33-70b-versatile_t02_cgo.jsonl \
      --results samples/mbpp_llama-33-70b-versatile_t02_cgo_eval_results.json
"""

import argparse
import json
import os
from collections import Counter
from typing import Any

from evalplus.data import get_human_eval_plus, get_mbpp_plus


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract EvalPlus failures for prompt-refinement analysis."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset used for the run.",
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="Path to the generated samples JSONL file.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the EvalPlus *_eval_results.json file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <results>_failures.json.",
    )
    parser.add_argument(
        "--include-passing",
        action="store_true",
        help="Include passing tasks as well as failing tasks.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_problems(dataset: str) -> dict[str, dict[str, Any]]:
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_samples(path: str) -> dict[str, dict[str, Any]]:
    samples = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            task_id = sample.get("task_id")
            if not task_id:
                raise ValueError(f"Missing task_id in {path}:{line_no}")
            samples[task_id] = sample
    return samples


def load_results(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "eval" not in data:
        raise ValueError(f"Missing 'eval' key in {path}")
    return data


def first_result(result_entry: Any) -> dict[str, Any]:
    if isinstance(result_entry, list):
        if not result_entry:
            return {}
        return result_entry[0]
    if isinstance(result_entry, dict):
        return result_entry
    return {}


def solution_key_for_dataset(dataset: str) -> str:
    return "solution" if dataset == "mbpp" else "completion"


def classify_stage(base_status: str, plus_status: str) -> str:
    base_passed = base_status == "pass"
    plus_passed = plus_status == "pass"

    if base_passed and plus_passed:
        return "pass"
    if not base_passed and not plus_passed:
        return "base_and_plus"
    if not base_passed:
        return "base_only"
    return "plus_only"


def status_value(result: dict[str, Any], key: str) -> str:
    value = result.get(key)
    return value if isinstance(value, str) else "unknown"


def build_task_record(
    task_id: str,
    dataset: str,
    problem: dict[str, Any],
    sample: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Any]:
    base_status = status_value(result, "base_status")
    plus_status = status_value(result, "plus_status")
    solution_key = solution_key_for_dataset(dataset)

    return {
        "task_id": task_id,
        "failure_stage": classify_stage(base_status, plus_status),
        "base_status": base_status,
        "plus_status": plus_status,
        "prompt": problem.get("prompt"),
        "canonical_solution": problem.get("canonical_solution"),
        "generated_solution": (
            result.get(solution_key)
            or result.get("solution")
            or result.get("completion")
            or sample.get(solution_key)
            or sample.get("solution")
            or sample.get("completion")
        ),
        "base_fail_tests": result.get("base_fail_tests", []),
        "plus_fail_tests": result.get("plus_fail_tests", []),
    }


def summarize(records: list[dict[str, Any]], total_tasks: int) -> dict[str, Any]:
    stage_counts = Counter(record["failure_stage"] for record in records)
    failure_records = [
        record for record in records if record["failure_stage"] != "pass"
    ]
    base_passes = sum(record["base_status"] == "pass" for record in records)
    plus_passes = sum(
        record["base_status"] == "pass" and record["plus_status"] == "pass"
        for record in records
    )

    return {
        "total_tasks_in_results": total_tasks,
        "records_analyzed": len(records),
        "failure_count": len(failure_records),
        "stage_counts": dict(stage_counts),
        "base_pass_count": base_passes,
        "plus_pass_count": plus_passes,
        "base_pass_at_1": base_passes / total_tasks if total_tasks else None,
        "plus_pass_at_1": plus_passes / total_tasks if total_tasks else None,
        "robustness_gap": (
            (base_passes / total_tasks) - (plus_passes / total_tasks)
            if total_tasks
            else None
        ),
    }


def default_output_path(results_path: str) -> str:
    if results_path.endswith(".json"):
        return results_path[:-5] + "_failures.json"
    return results_path + "_failures.json"


def main() -> None:
    args = parse_args()

    samples_path = resolve_path(args.samples)
    results_path = resolve_path(args.results)
    output_path = resolve_path(args.output) if args.output else default_output_path(results_path)

    problems = load_problems(args.dataset)
    samples = load_samples(samples_path)
    results = load_results(results_path)
    eval_results = results["eval"]

    records = []
    for task_id, result_entry in eval_results.items():
        result = first_result(result_entry)
        problem = problems.get(task_id, {})
        sample = samples.get(task_id, {})
        record = build_task_record(task_id, args.dataset, problem, sample, result)

        if args.include_passing or record["failure_stage"] != "pass":
            records.append(record)

    records.sort(key=lambda record: record["task_id"])

    report = {
        "dataset": args.dataset,
        "samples": samples_path,
        "results": results_path,
        "summary": summarize(
            [
                build_task_record(
                    task_id,
                    args.dataset,
                    problems.get(task_id, {}),
                    samples.get(task_id, {}),
                    first_result(result_entry),
                )
                for task_id, result_entry in eval_results.items()
            ],
            total_tasks=len(eval_results),
        ),
        "records_written": len(records),
        "failures": records,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = report["summary"]
    print(f"Wrote {len(records)} records -> {output_path}")
    print(f"Failures: {summary['failure_count']}/{summary['total_tasks_in_results']}")
    print(f"Base pass@1: {summary['base_pass_at_1']:.6f}")
    print(f"Plus pass@1: {summary['plus_pass_at_1']:.6f}")
    print(f"Robustness gap: {summary['robustness_gap']:.6f}")


if __name__ == "__main__":
    main()
