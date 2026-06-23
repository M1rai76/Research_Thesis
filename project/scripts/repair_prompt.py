"""
repair_prompt.py
Thesis — Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

Build repair prompts for the iterative self-repair loop. When a task
fails a round, this module constructs a new prompt containing the
original task, the broken code, and the error signal from
code_executor.py, then asks the model to fix its own attempt.

One fixed template is used across all tasks and rounds — only the
task content, broken code, and error info vary. This isolates the
effect of the repair loop design rather than per-task prompt tuning.

Functions
    build_repair_prompt()   — construct the full repair prompt text
    build_repair_context()  — extract error fields from an executor result

Usage
    python repair_prompt.py --task_id HumanEval/4 --dataset humaneval \
        --jsonl_path ../samples/llama-33-70b-versatile_t02_cgo.jsonl
"""

import argparse
import json
import os
import sys
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_executor import run_executor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_problems(dataset: str) -> dict:
    """Load the selected EvalPlus dataset."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_repair_prompt(
    task_prompt: str,
    broken_completion: str,
    error_type: Optional[str],
    error_message: Optional[str],
    dataset: str = "humaneval",
) -> str:
    """Construct a lean repair prompt for the self-repair loop.

    Deliberately omits the original task prompt/docstring from the repair
    text. Research on program repair shows that restating context which
    does not directly expose the defect adds prompt bloat that can hurt
    repair performance more than it helps — the broken code plus the
    concrete error signal is the minimal sufficient evidence for the model
    to localise and fix the fault. The task_prompt parameter is retained
    in the signature for future A/B ablation experiments.

    Uses one fixed template across all tasks and rounds. The output
    constraints mirror the COP prompt strategy in generate_samples.py
    so the repair prompt is consistent with the seed prompt's format.

    Parameters
        task_prompt        : retained for ablation (unused in rendered text)
        broken_completion  : the model's previous failing code
        error_type         : exception class name from code_executor, or None
        error_message      : truncated error message, or None
        dataset            : "humaneval" or "mbpp"

    Returns
        The full prompt string ready to send to the model.
    """
    if error_type is not None:
        if error_message:
            error_line = f"This attempt failed with: {error_type}: {error_message}"
        else:
            error_line = f"This attempt failed with: {error_type}"
    else:
        error_line = (
            "This attempt passed visible tests but failed broader validation."
        )

    if dataset == "humaneval":
        output_constraints = (
            "Output only the function body. No explanations. No markdown."
        )
    else:
        output_constraints = (
            "Output only Python code. Include the required function definition "
            "and any imports it needs. No explanations. No markdown."
        )

    return (
        "You are an expert Python programmer. "
        "The following code you wrote failed when executed. Fix it.\n\n"
        "--- Your previous attempt ---\n"
        f"{broken_completion}\n\n"
        "--- Error ---\n"
        f"{error_line}\n\n"
        f"Fix the code. {output_constraints}\n"
    )


def build_repair_context(executor_result: dict, dataset: str) -> dict:
    """Extract the fields needed by build_repair_prompt from an executor result.

    Handles the case where execution passed (returns neutral values rather
    than crashing).

    Returns
        {"error_type": str | None, "error_message": str | None}
    """
    if executor_result.get("passed", False):
        return {"error_type": None, "error_message": None}

    return {
        "error_type": executor_result.get("error_type"),
        "error_message": executor_result.get("error_message"),
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
        description="Build a repair prompt for a failed task and print it."
    )
    parser.add_argument(
        "--task_id",
        required=True,
        help="Task identifier, e.g. HumanEval/4 or Mbpp/2.",
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

    executor_result = run_executor(
        task_id=args.task_id,
        completion=completion,
        dataset=args.dataset,
    )

    problems = get_problems(args.dataset)
    task_prompt = problems[args.task_id]["prompt"]

    context = build_repair_context(executor_result, args.dataset)

    prompt = build_repair_prompt(
        task_prompt=task_prompt,
        broken_completion=completion,
        error_type=context["error_type"],
        error_message=context["error_message"],
        dataset=args.dataset,
    )

    print(prompt)


if __name__ == "__main__":
    main()
