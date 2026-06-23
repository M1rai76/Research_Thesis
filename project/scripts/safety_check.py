"""
safety_check.py
Thesis — Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

Run Bandit static security analysis against generated code completions.
Used as a passive observational measurement (the third evaluation
dimension alongside correctness and robustness) and as a gate before
sandboxed execution for HIGH-severity findings.

Functions
    run_bandit_scan()   — scan a code string with Bandit, return findings
    check_safety()      — public entry point: reconstruct code and scan it
    is_high_risk()      — True if max_severity == "HIGH"

Usage
    python safety_check.py --task_id HumanEval/0 --dataset humaneval \
        --jsonl_path ../samples/qwen25-coder_7b_t02_baseline.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_executor import reconstruct_full_code


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TIMEOUT = 5

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def get_problems(dataset: str) -> dict:
    """Load the selected EvalPlus dataset."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_bandit_scan(code: str, timeout_seconds: int = DEFAULT_TIMEOUT) -> dict:
    """Scan a code string with Bandit and return structured findings.

    Writes the code to a temporary .py file, runs Bandit in JSON mode,
    and parses the output.

    Returns
        {
            "issue_count": int,
            "max_severity": str | None,   # "LOW", "MEDIUM", "HIGH", or None
            "findings": list[str],         # e.g. ["B311", "B602"]
        }
    """
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="bandit_scan_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(code)

        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-f", "json", "-q", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if not result.stdout.strip():
            return {"issue_count": 0, "max_severity": None, "findings": []}

        report = json.loads(result.stdout)
        results = report.get("results", [])

        if not results:
            return {"issue_count": 0, "max_severity": None, "findings": []}

        findings = [r["test_id"] for r in results]

        max_severity = None
        max_rank = -1
        for r in results:
            sev = r["issue_severity"]
            rank = SEVERITY_RANK.get(sev, -1)
            if rank > max_rank:
                max_rank = rank
                max_severity = sev

        return {
            "issue_count": len(results),
            "max_severity": max_severity,
            "findings": findings,
        }

    except subprocess.TimeoutExpired:
        return {"issue_count": 0, "max_severity": None, "findings": []}
    except (json.JSONDecodeError, KeyError):
        return {"issue_count": 0, "max_severity": None, "findings": []}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def check_safety(
    task_id: str,
    completion: str,
    dataset: str = "humaneval",
) -> dict:
    """Run Bandit analysis on a model completion for a given task.

    Reconstructs the full executable code (reusing code_executor's
    reconstruct_full_code) before scanning.

    Returns
        {
            "task_id": str,
            "dataset": str,
            "issue_count": int,
            "max_severity": str | None,
            "findings": list[str],
        }
    """
    problems = get_problems(dataset)

    if task_id not in problems:
        raise KeyError(f"Task {task_id!r} not found in {dataset} dataset")

    task = problems[task_id]

    full_code = reconstruct_full_code(
        task_prompt=task["prompt"],
        completion=completion,
        entry_point=task["entry_point"],
        dataset=dataset,
    )

    scan = run_bandit_scan(full_code)

    return {
        "task_id": task_id,
        "dataset": dataset,
        **scan,
    }


def is_high_risk(safety_result: dict) -> bool:
    """Return True if the scan found any HIGH-severity issue."""
    return safety_result.get("max_severity") == "HIGH"


def _load_completion_from_jsonl(jsonl_path: str, task_id: str, dataset: str) -> str:
    """Look up a task's completion/solution from a samples JSONL file."""
    sample_key = "solution" if dataset == "mbpp" else "completion"

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("task_id") == task_id:
                if sample_key not in sample:
                    raise KeyError(
                        f"Sample for {task_id} has no {sample_key!r} key "
                        f"(available keys: {list(sample.keys())})"
                    )
                return sample[sample_key]

    raise KeyError(f"Task {task_id!r} not found in {jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bandit security analysis on a generated code sample."
    )
    parser.add_argument(
        "--task_id",
        required=True,
        help="Task identifier, e.g. HumanEval/0 or Mbpp/2.",
    )
    parser.add_argument(
        "--dataset",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset. Default: humaneval.",
    )
    parser.add_argument(
        "--jsonl_path",
        required=True,
        help="Path to an existing samples JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    completion = _load_completion_from_jsonl(args.jsonl_path, args.task_id, args.dataset)

    result = check_safety(
        task_id=args.task_id,
        completion=completion,
        dataset=args.dataset,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
