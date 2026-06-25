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
import re
import sys
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from code_executor import run_executor


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _format_error_line(
    error_type: Optional[str],
    error_message: Optional[str],
) -> str:
    """Format executor error fields consistently across repair prompts."""
    if error_type is not None:
        if error_message:
            return f"This attempt failed with: {error_type}: {error_message}"
        return f"This attempt failed with: {error_type}"

    return "This attempt passed visible tests but failed broader validation."


def _output_constraints(dataset: str) -> str:
    """Return dataset-specific output constraints for repaired code."""
    if dataset == "humaneval":
        return "Output only the function body. No explanations. No markdown."
    return (
        "Output only Python code. Include the required function definition "
        "and any imports it needs. No explanations. No markdown."
    )


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
    error_line = _format_error_line(error_type, error_message)
    output_constraints = _output_constraints(dataset)

    return (
        "You are an expert Python programmer. "
        "The following code you wrote failed when executed. Fix it.\n\n"
        "--- Your previous attempt ---\n"
        f"{broken_completion}\n\n"
        "--- Error ---\n"
        f"{error_line}\n\n"
        f"Fix the code. {output_constraints}\n"
    )


def build_repair_prompt_cot(
    task_prompt: str,
    broken_completion: str,
    error_type: Optional[str],
    error_message: Optional[str],
    dataset: str = "humaneval",
) -> str:
    """Construct a chain-of-thought repair prompt for the self-repair loop.

    The model is asked to reason through the failure before emitting a final
    corrected code section. The final code must appear after the exact marker
    ``### Fixed Code`` so callers can separate reasoning from executable code.
    """
    error_line = _format_error_line(error_type, error_message)
    output_constraints = _output_constraints(dataset)

    return (
        "You are an expert Python programmer. "
        "The following code you wrote failed when executed. Fix it.\n\n"
        "--- Task ---\n"
        f"{task_prompt}\n\n"
        "--- Your previous attempt ---\n"
        f"{broken_completion}\n\n"
        "--- Error ---\n"
        f"{error_line}\n\n"
        "Before fixing the code, reason through the problem in three steps: "
        "1) What does the error tell us about the failure? "
        "2) What is the root cause of the bug? "
        "3) What is the correct fix?\n\n"
        "After your reasoning, put the final corrected code after a line "
        "reading EXACTLY:\n"
        "### Fixed Code\n\n"
        f"Under ### Fixed Code, {output_constraints}\n"
    )


def extract_code_from_cot_response(raw_response: str) -> tuple[str, str]:
    """Extract final code from a CoT repair response.

    Returns
        (code, extraction_method), where extraction_method is one of:
        ``marker``       - extracted after the required ``### Fixed Code`` line
        ``fence``        - extracted from the longest markdown code fence
        ``raw_fallback`` - no marker or fence was found, so raw text was used
    """
    marker = "### Fixed Code"
    if marker in raw_response:
        return raw_response.split(marker, 1)[1].lstrip(), "marker"

    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_response, flags=re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip(), "fence"

    code = re.sub(r"^```(?:python)?\s*\n", "", raw_response.strip())
    if code != raw_response.strip():
        return code, "fence"

    return raw_response, "raw_fallback"


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
