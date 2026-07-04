"""Full verification of code_executor.py against existing MBPP samples."""

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_executor import run_executor


JSONL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "samples",
    "mbpp_llama-33-70b-versatile_t02_cgo.jsonl",
)


def load_samples(path: str) -> list:
    """Load MBPP samples from JSONL in file order."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def main() -> None:
    samples = load_samples(JSONL_PATH)
    print(f"Loaded {len(samples)} samples from {JSONL_PATH}\n")

    results = []
    for i, sample in enumerate(samples, 1):
        task_id = sample["task_id"]
        solution = sample["solution"]
        print(f"[{i:>3}/{len(samples)}] {task_id} ...", end=" ", flush=True)

        result = run_executor(task_id=task_id, completion=solution, dataset="mbpp")
        tag = "PASS" if result["passed"] else ("TIMEOUT" if result["timed_out"] else "FAIL")
        print(tag)
        results.append(result)

    passed = [r for r in results if r["passed"]]
    timed_out = [r for r in results if r["timed_out"]]
    failed = [r for r in results if not r["passed"] and not r["timed_out"]]

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total tasks : {len(results)}")
    print(f"  Passed      : {len(passed)}")
    print(f"  Failed      : {len(failed)}")
    print(f"  Timed out   : {len(timed_out)}")

    error_counts = Counter(r["error_type"] for r in failed)
    if error_counts:
        print("\n  Error type breakdown:")
        for err_type, count in error_counts.most_common():
            print(f"    {err_type}: {count}")

    all_not_passed = [r for r in results if not r["passed"]]
    if all_not_passed:
        print(f"\n{'=' * 60}")
        print(f"ALL {len(all_not_passed)} FAILED/TIMED-OUT TASKS (detail)")
        print("=" * 60)
        for r in all_not_passed:
            print(f"\n  task_id      : {r['task_id']}")
            print(f"  error_type   : {r['error_type']}")
            print(f"  error_message: {r['error_message']}")


if __name__ == "__main__":
    main()
