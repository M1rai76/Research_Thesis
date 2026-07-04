"""One-off verification of safety_check.py against an existing samples file."""

import json
import os
import sys
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from safety_check import check_safety

JSONL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "samples",
    "llama-33-70b-versatile_t02_cgo.jsonl",
)


def load_samples(path: str) -> list:
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify Bandit safety findings for a samples JSONL file."
    )
    parser.add_argument("--jsonl_path", default=JSONL_PATH)
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.jsonl_path)
    print(f"Loaded {len(samples)} samples from {args.jsonl_path}\n")

    results = []
    for i, sample in enumerate(samples, 1):
        task_id = sample["task_id"]
        completion = sample["solution"] if args.dataset == "mbpp" else sample["completion"]
        print(f"[{i:>3}/{len(samples)}] {task_id} ...", end=" ", flush=True)

        result = check_safety(task_id=task_id, completion=completion, dataset=args.dataset)
        tag = "CLEAN" if result["issue_count"] == 0 else f"{result['issue_count']} issues ({result['max_severity']})"
        print(tag)
        results.append(result)

    clean = [r for r in results if r["issue_count"] == 0]
    flagged = [r for r in results if r["issue_count"] > 0]
    total = len(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total tasks scanned : {total}")
    print(f"  Clean (zero issues) : {len(clean)}  ({100 * len(clean) / total:.1f}% Safety Rate)")
    print(f"  Flagged (>=1 issue) : {len(flagged)}")

    severity_counts = Counter(r["max_severity"] for r in flagged)
    if severity_counts:
        print("\n  Max severity breakdown (among flagged):")
        for sev in ["LOW", "MEDIUM", "HIGH"]:
            if sev in severity_counts:
                print(f"    {sev}: {severity_counts[sev]}")

    all_findings = []
    for r in flagged:
        all_findings.extend(r["findings"])
    finding_counts = Counter(all_findings)
    if finding_counts:
        print("\n  Bandit test_id frequency (all findings):")
        for test_id, count in finding_counts.most_common():
            print(f"    {test_id}: {count}")

    if flagged:
        show = flagged[:10]
        print(f"\n{'=' * 60}")
        print(f"FIRST {len(show)} FLAGGED TASKS (detail)")
        print("=" * 60)
        for r in show:
            print(f"\n  task_id      : {r['task_id']}")
            print(f"  issue_count  : {r['issue_count']}")
            print(f"  max_severity : {r['max_severity']}")
            print(f"  findings     : {r['findings']}")


if __name__ == "__main__":
    main()
