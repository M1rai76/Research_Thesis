"""
code_executor.py
Thesis - Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

Execute generated code against visible (base) test cases for the
iterative self-repair loop. Only the original test assertions shipped
with HumanEval / MBPP are used - the augmented EvalPlus test inputs
are reserved for final evaluation.

Functions
    get_visible_tests()        - extract base test code from EvalPlus data
    reconstruct_full_code()    - combine prompt + completion into runnable code
    execute_in_sandbox()       - run code + tests in an isolated subprocess
    run_executor()             - public entry point tying it all together

Usage
    python code_executor.py --task_id HumanEval/0 --dataset humaneval \
        --jsonl_path ../samples/qwen25-coder_7b_t02_baseline.jsonl
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAX_ERROR_LENGTH = 300
DEFAULT_TIMEOUT = 5


def get_problems(dataset: str) -> dict:
    """Load the selected EvalPlus dataset."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_visible_tests(task_id: str, dataset: str) -> dict:
    """Extract the visible/base test code for a given task.

    HumanEval - the ``test`` field contains a ``check(candidate)`` function.
                Returns test_code that defines ``check`` and calls it with
                the task's entry point.

    MBPP      - the ``assertion`` field contains direct assert statements
                that reference the entry point by name.

    Returns
        {"test_code": str, "entry_point": str}
    """
    problems = get_problems(dataset)

    if task_id not in problems:
        raise KeyError(f"Task {task_id!r} not found in {dataset} dataset")

    task = problems[task_id]
    entry_point = task["entry_point"]

    if dataset == "humaneval":
        test_func = task["test"]
        test_code = f"{test_func}\ncheck({entry_point})\n"
    else:
        test_code = task["assertion"]

    return {"test_code": test_code, "entry_point": entry_point}


def reconstruct_full_code(
    task_prompt: str,
    completion: str,
    entry_point: str,
    dataset: str,
) -> str:
    """Combine task prompt and model completion into executable code.

    HumanEval - if the completion already contains ``def {entry_point}``,
                the model echoed the full function back; use it as-is to
                avoid creating a nested definition. Otherwise prepend the
                prompt (signature + docstring) as the completion is
                body-only.

    MBPP      - the completion is a standalone solution (stored under the
                ``solution`` key in the samples JSONL) and is used as-is.
    """
    if dataset == "humaneval":
        if f"def {entry_point}" in completion:
            return completion.lstrip("\n")
        return task_prompt + completion
    return completion


def execute_in_sandbox(
    full_code: str,
    test_code: str,
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict:
    """Run full_code + test_code in an isolated subprocess.

    The combined code is written to a temporary .py file and executed via
    ``subprocess.run`` with a strict timeout. Never uses exec/eval in the
    current process.

    Returns
        {
            "passed": bool,
            "error_type": str | None,
            "error_message": str | None,
            "timed_out": bool,
        }
    """
    combined = full_code + "\n" + test_code

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="executor_")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(combined)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )

        if result.returncode == 0:
            return {
                "passed": True,
                "error_type": None,
                "error_message": None,
                "timed_out": False,
            }

        error_type, error_message = _parse_error(result.stderr)
        if error_type == "AssertionError":
            error_message = _enhance_assert_message(result.stderr, error_message)
        return {
            "passed": False,
            "error_type": error_type,
            "error_message": error_message,
            "timed_out": False,
        }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error_type": "Timeout",
            "error_message": f"Execution exceeded {timeout_seconds}s limit",
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_assert_expression(stderr: str) -> Optional[str]:
    """Find the failing assertion expression line from a Python traceback.

    Python tracebacks print the source line that raised the exception
    (e.g. ``assert candidate([1, 3, -2, 1]) == True``) one or two lines
    before the final ``AssertionError`` line. This function returns that
    expression, stripped of leading whitespace, or None if not found.
    """
    match = re.search(r"^\s+(assert\s+.+)$", stderr, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _enhance_assert_message(
    stderr: str,
    original_message: Optional[str],
) -> Optional[str]:
    """Replace a bare or generic AssertionError message with the failing
    assertion expression from the traceback, which is more actionable
    for repair prompts.
    """
    expr = _extract_assert_expression(stderr)
    if expr is None:
        return original_message
    if len(expr) > MAX_ERROR_LENGTH:
        expr = expr[:MAX_ERROR_LENGTH] + "..."
    return expr


def _parse_error(stderr: str) -> tuple:
    """Extract exception class name and a short message from a traceback.

    Returns (error_type, error_message).
    """
    if not stderr.strip():
        return ("UnknownError", None)

    match = re.search(r"^(\w+Error|\w+Exception|KeyboardInterrupt):\s*(.+)", stderr, re.MULTILINE)
    if match:
        error_type = match.group(1)
        error_message = match.group(2).strip()
        if len(error_message) > MAX_ERROR_LENGTH:
            error_message = error_message[:MAX_ERROR_LENGTH] + "..."
        return (error_type, error_message)

    match_no_msg = re.search(r"^(\w+Error|\w+Exception|KeyboardInterrupt)\s*$", stderr, re.MULTILINE)
    if match_no_msg:
        return (match_no_msg.group(1), None)

    last_line = stderr.strip().splitlines()[-1].strip()
    if len(last_line) > MAX_ERROR_LENGTH:
        last_line = last_line[:MAX_ERROR_LENGTH] + "..."
    return ("UnknownError", last_line)


def run_executor(
    task_id: str,
    completion: str,
    dataset: str = "humaneval",
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict:
    """Execute a model completion against the visible tests for a task.

    Returns
        {
            "task_id": str,
            "dataset": str,
            "passed": bool,
            "error_type": str | None,
            "error_message": str | None,
            "timed_out": bool,
        }
    """
    problems = get_problems(dataset)

    if task_id not in problems:
        raise KeyError(f"Task {task_id!r} not found in {dataset} dataset")

    task = problems[task_id]
    visible = get_visible_tests(task_id, dataset)

    full_code = reconstruct_full_code(
        task_prompt=task["prompt"],
        completion=completion,
        entry_point=visible["entry_point"],
        dataset=dataset,
    )

    result = execute_in_sandbox(
        full_code=full_code,
        test_code=visible["test_code"],
        timeout_seconds=timeout_seconds,
    )

    return {
        "task_id": task_id,
        "dataset": dataset,
        **result,
    }


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
        description="Execute a generated sample against visible test cases."
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Per-task execution timeout in seconds. Default: {DEFAULT_TIMEOUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    completion = _load_completion_from_jsonl(args.jsonl_path, args.task_id, args.dataset)

    result = run_executor(
        task_id=args.task_id,
        completion=completion,
        dataset=args.dataset,
        timeout_seconds=args.timeout,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
