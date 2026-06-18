"""
safety_eval_correlation.py

Joins Safety Oracle output (safety_oracle.py) against an EvalPlus
_eval_results.json file and tests whether guard presence predicts Plus-test
survival, conditioned on the task already passing the Base test suite. See
decisions.md D9 for why the comparison is conditioned on base_status=="pass"
rather than run unconditionally across all tasks, and D10 for the
significance-testing methodology (two-proportion z-test, no multiple-
comparisons correction).

Usage:
    python -m robustness.safety_eval_correlation \
      --safety-oracle samples/mbpp_llama-33-70b-versatile_t02_cgo_safety_oracle.json \
      --results samples/mbpp_llama-33-70b-versatile_t02_cgo_eval_results.json
"""

import argparse
import json
import math
import os
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUARD_CATEGORIES = (
    "any_guard_present",
    "has_none_check",
    "has_specific_value_check",
    "has_range_check",
    "has_type_check",
)

SMALL_SAMPLE_THRESHOLD = 10


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_safety_oracle(path: str) -> dict[str, dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {record["task_id"]: record for record in data["results"]}


def first_result(result_entry: Any) -> dict[str, Any]:
    if isinstance(result_entry, list):
        return result_entry[0] if result_entry else {}
    if isinstance(result_entry, dict):
        return result_entry
    return {}


def status_value(result: dict[str, Any], key: str) -> str:
    value = result.get(key)
    return value if isinstance(value, str) else "unknown"


def load_eval_results(path: str) -> dict[str, dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "eval" not in data:
        raise ValueError(f"Missing 'eval' key in {path}")

    statuses = {}
    for task_id, result_entry in data["eval"].items():
        result = first_result(result_entry)
        statuses[task_id] = {
            "base_status": status_value(result, "base_status"),
            "plus_status": status_value(result, "plus_status"),
        }
    return statuses


def join_records(
    safety_records: dict[str, dict[str, Any]],
    eval_statuses: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    task_ids = sorted(set(safety_records) & set(eval_statuses))
    joined = []
    for task_id in task_ids:
        record = {"task_id": task_id}
        record.update(safety_records[task_id])
        record.update(eval_statuses[task_id])
        joined.append(record)
    return joined


def plus_survival_rate(records: list[dict[str, Any]]) -> float | None:
    if not records:
        return None
    return sum(r["plus_status"] == "pass" for r in records) / len(records)


def two_proportion_z_test(
    n1: int, p1: float, n2: int, p2: float
) -> tuple[float, float] | None:
    if n1 == 0 or n2 == 0:
        return None
    x1, x2 = p1 * n1, p2 * n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))
    return z, p_value


def conditional_analysis(joined: list[dict[str, Any]]) -> dict[str, Any]:
    base_passing = [r for r in joined if r["base_status"] == "pass"]

    categories = {}
    for category in GUARD_CATEGORIES:
        guarded = [r for r in base_passing if r[category]]
        unguarded = [r for r in base_passing if not r[category]]

        guarded_rate = plus_survival_rate(guarded)
        unguarded_rate = plus_survival_rate(unguarded)
        rate_difference = (
            guarded_rate - unguarded_rate
            if guarded_rate is not None and unguarded_rate is not None
            else None
        )

        z_result = (
            two_proportion_z_test(len(guarded), guarded_rate, len(unguarded), unguarded_rate)
            if guarded_rate is not None and unguarded_rate is not None
            else None
        )
        z_score, p_value = z_result if z_result else (None, None)

        categories[category] = {
            "n_guarded": len(guarded),
            "n_unguarded": len(unguarded),
            "plus_survival_rate_guarded": guarded_rate,
            "plus_survival_rate_unguarded": unguarded_rate,
            "rate_difference": rate_difference,
            "z_score": z_score,
            "p_value": p_value,
            "significant": p_value is not None and p_value < 0.05,
            "small_sample": (
                len(guarded) < SMALL_SAMPLE_THRESHOLD
                or len(unguarded) < SMALL_SAMPLE_THRESHOLD
            ),
        }

    return {
        "n_total": len(joined),
        "n_base_passing": len(base_passing),
        "categories": categories,
    }


def default_output_path(results_path: str) -> str:
    if results_path.endswith(".json"):
        return results_path[: -len(".json")] + "_safety_correlation.json"
    return results_path + "_safety_correlation.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join Safety Oracle output against EvalPlus results and test "
            "whether guard presence predicts Plus survival among Base-passing tasks."
        )
    )
    parser.add_argument(
        "--safety-oracle",
        required=True,
        help="Path to a *_safety_oracle.json file produced by safety_oracle.py.",
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to the matching EvalPlus *_eval_results.json file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <results>_safety_correlation.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    safety_path = resolve_path(args.safety_oracle)
    results_path = resolve_path(args.results)
    output_path = (
        resolve_path(args.output) if args.output else default_output_path(results_path)
    )

    safety_records = load_safety_oracle(safety_path)
    eval_statuses = load_eval_results(results_path)
    joined = join_records(safety_records, eval_statuses)

    analysis = conditional_analysis(joined)

    report = {
        "safety_oracle": safety_path,
        "results": results_path,
        "analysis": analysis,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote correlation report -> {output_path}")
    print(
        f"Base-passing tasks: {analysis['n_base_passing']}/{analysis['n_total']}"
    )
    for category, stats in analysis["categories"].items():
        flag = " (small sample)" if stats["small_sample"] else ""
        guarded_rate = stats["plus_survival_rate_guarded"]
        unguarded_rate = stats["plus_survival_rate_unguarded"]
        diff = stats["rate_difference"]
        if guarded_rate is None or unguarded_rate is None:
            print(f"{category}: insufficient data{flag}")
            continue
        sig = " *significant*" if stats["significant"] else ""
        print(
            f"{category}: guarded={guarded_rate:.3f} (n={stats['n_guarded']})  "
            f"unguarded={unguarded_rate:.3f} (n={stats['n_unguarded']})  "
            f"diff={diff:+.3f}  z={stats['z_score']:+.2f}  p={stats['p_value']:.4f}{sig}{flag}"
        )


if __name__ == "__main__":
    main()
