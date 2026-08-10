"""
self_repair.py
Thesis - Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)
Edited by: Gurdiraj Bal (z5386590)

Orchestrator for the iterative self-repair loop. Runs a single task
through up to N rounds of LLM-driven repair, tying together
code_executor, safety_check, repair_prompt, and the LLM-calling
logic from generate_samples.

Functions
    run_self_repair()   - full N-round repair loop for one task

Usage
    python self_repair.py --task_id HumanEval/40 --dataset humaneval \
        --jsonl_path ../samples/llama-33-70b-versatile_t02_cgo.jsonl \
        --backend groq --model llama-3.3-70b-versatile --max_rounds 2 \
        --repair_strategy cot
"""

import argparse
import json
import os
import sys
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code_executor import run_executor
from safety_check import check_safety
from repair_prompt import (
    FEEDBACK_MODES,
    build_repair_prompt,
    build_repair_prompt_cot,
    build_repair_context,
    extract_code_from_cot_response,
)
from generate_samples import (
    load_env_file,
    get_client,
    generate_raw_completion,
    build_prompt,
    post_process,
    post_process_solution,
    ENV_PATH,
    TEMPERATURE,
    MAX_TOKENS,
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_problems(dataset: str) -> dict:
    """Load the selected EvalPlus dataset."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_self_repair(
    task_id: str,
    initial_completion: str,
    dataset: str,
    client,
    model: str,
    backend: str,
    max_repair_rounds: int = 2,
    repair_strategy: str = "cot",
    feedback_mode: str = "full",
    prompt_strategy: str = "cop",
) -> dict:
    """Run the full N-round repair loop for a single task.

    Round 0 executes the initial_completion. If it fails, up to
    max_repair_rounds of LLM-driven repair are attempted. ``repair_strategy``
    selects the repair prompt: ``minimal`` preserves the original lean prompt,
    while ``cot`` asks the model to reason first and emit final code after
    ``### Fixed Code``.

    Each round entry (except round 0) includes ``completion_changed``
    indicating whether the model produced a different completion from
    the previous round. Round 0 sets this to None (no prior round). CoT repair
    rounds also include ``cot_extraction_method`` with one of ``marker``,
    ``fence``, or ``raw_fallback``.

    The top-level result includes ``stalled``: True when the task is
    unsolved AND the model never produced a different completion across
    any repair round - meaning it did not engage with the repair signal
    at all, as opposed to genuinely attempting different (but still
    wrong) fixes. This distinction matters for thesis analysis: a stall
    indicates the repair prompt failed to elicit a behavioural change,
    while a non-stalled failure means the model tried and couldn't fix
    the underlying bug.

    ``feedback_mode`` drives the feedback-content ablation (decisions.md
    D24). It varies ONLY the failure signal shown to the model, holding the
    number of attempts and the ground-truth stopping oracle fixed, so every
    arm gets the same repair budget and stops on the same condition:
        ``full``       - exception class and message (pre-ablation behaviour)
        ``error-type`` - exception class only
        ``binary``     - "This attempt failed." and nothing more
        ``blind``      - no failure signal at all: the Round 0 generation
                         prompt is re-run, making the arm a pure resampling
                         control. ``prompt_strategy`` selects that prompt and
                         must match the one that produced the Round 0 seed.

    Returns a trajectory dict capturing every round's completion,
    executor result, and safety scan for per-round thesis analysis.
    """
    if repair_strategy not in {"minimal", "cot"}:
        raise ValueError(
            "Unsupported repair_strategy: "
            f"{repair_strategy!r}. Expected 'minimal' or 'cot'."
        )
    if feedback_mode not in FEEDBACK_MODES:
        raise ValueError(
            "Unsupported feedback_mode: "
            f"{feedback_mode!r}. Expected one of {FEEDBACK_MODES}."
        )

    problems = get_problems(dataset)
    task = problems[task_id]
    task_prompt = task["prompt"]
    entry_point = task["entry_point"]

    rounds = []
    previous_completion = None
    current_completion = initial_completion
    current_cot_extraction_method = None
    api_failed = False

    total_rounds = 1 + max_repair_rounds

    for round_num in range(total_rounds):
        exec_result = run_executor(
            task_id=task_id,
            completion=current_completion,
            dataset=dataset,
            capture_failing_case=(feedback_mode == "grounded+"),
        )

        safety_result = check_safety(
            task_id=task_id,
            completion=current_completion,
            dataset=dataset,
        )

        if round_num == 0:
            completion_changed = None
        else:
            completion_changed = (current_completion != previous_completion)

        round_entry = {
            "round": round_num,
            "completion": current_completion,
            "completion_changed": completion_changed,
            "passed": exec_result["passed"],
            "error_type": exec_result.get("error_type"),
            "error_message": exec_result.get("error_message"),
            "timed_out": exec_result.get("timed_out", False),
            "safety": {
                "issue_count": safety_result["issue_count"],
                "max_severity": safety_result["max_severity"],
                "findings": safety_result["findings"],
            },
        }
        if repair_strategy == "cot":
            round_entry["cot_extraction_method"] = (
                None if round_num == 0 else current_cot_extraction_method
            )
        rounds.append(round_entry)

        tag = "PASS" if exec_result["passed"] else "FAIL"
        change_tag = ""
        if round_num > 0 and not completion_changed:
            change_tag = " [no change]"
        print(f"  Round {round_num}: {tag}", end="")
        if not exec_result["passed"]:
            print(f" ({exec_result['error_type']}){change_tag}", end="")
        print()

        if exec_result["passed"]:
            return {
                "task_id": task_id,
                "dataset": dataset,
                "repair_strategy": repair_strategy,
                "feedback_mode": feedback_mode,
                "solved": True,
                "solved_at_round": round_num,
                "stalled": False,
                "rounds": rounds,
            }

        if round_num >= max_repair_rounds:
            break

        context = build_repair_context(exec_result, dataset)
        if feedback_mode == "blind":
            # Zero-bit control: the model is not told it failed, or that a
            # previous attempt exists at all - it simply generates again from
            # the original task prompt. Isolates plain resampling from the
            # contribution of the feedback content.
            repair_prompt_text = build_prompt(task_prompt, prompt_strategy, dataset)
        elif repair_strategy == "minimal":
            repair_prompt_text = build_repair_prompt(
                task_prompt=task_prompt,
                broken_completion=current_completion,
                error_type=context["error_type"],
                error_message=context["error_message"],
                dataset=dataset,
                feedback_mode=feedback_mode,
                failing_case=context["failing_case"],
            )
        else:
            repair_prompt_text = build_repair_prompt_cot(
                task_prompt=task_prompt,
                broken_completion=current_completion,
                error_type=context["error_type"],
                error_message=context["error_message"],
                dataset=dataset,
                feedback_mode=feedback_mode,
                failing_case=context["failing_case"],
            )

        raw_response = generate_raw_completion(client, model, repair_prompt_text)

        if raw_response is None:
            # A quota/API failure is NOT a result - the task simply did not get
            # its repair attempt. Flagged so the caller can refuse to persist
            # this trajectory, otherwise --resume-missing would treat the task
            # as complete and permanently bake the failure into the data.
            print(f"  Round {round_num + 1}: SKIPPED (API error)")
            api_failed = True
            break

        response_for_post_process = raw_response
        next_cot_extraction_method = None
        # A blind round re-runs the plain generation prompt, so its response
        # has no "### Fixed Code" marker to extract from.
        if repair_strategy == "cot" and feedback_mode != "blind":
            response_for_post_process, next_cot_extraction_method = (
                extract_code_from_cot_response(raw_response)
            )
            if next_cot_extraction_method == "raw_fallback":
                print(
                    f"  Round {round_num + 1}: "
                    "CoT raw fallback used (missing marker and code fence)"
                )

        if dataset == "mbpp":
            new_completion = post_process_solution(response_for_post_process)
        else:
            new_completion = post_process(
                response_for_post_process,
                entry_point=entry_point,
            )

        if not new_completion:
            print(f"  Round {round_num + 1}: SKIPPED (empty after post-process)")
            break

        previous_completion = current_completion
        current_completion = new_completion
        current_cot_extraction_method = next_cot_extraction_method

    repair_rounds = [r for r in rounds if r["round"] > 0]
    stalled = (
        len(repair_rounds) > 0
        and all(r["completion_changed"] is False for r in repair_rounds)
    )

    return {
        "task_id": task_id,
        "dataset": dataset,
        "repair_strategy": repair_strategy,
        "feedback_mode": feedback_mode,
        "solved": False,
        "solved_at_round": None,
        "stalled": stalled,
        "api_failed": api_failed,
        "rounds": rounds,
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
        description="Run the self-repair loop for a single task."
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
        help="Path to an existing samples JSONL file for Round 0 completions.",
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["ollama", "groq", "cerebras", "openrouter", "gemini"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for the inference backend.",
    )
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=2,
        help="Maximum repair rounds (default: 2).",
    )
    parser.add_argument(
        "--repair_strategy",
        default="cot",
        choices=["minimal", "cot"],
        help="Repair prompt strategy to use. Default: cot.",
    )
    parser.add_argument(
        "--feedback_mode",
        default="full",
        choices=list(FEEDBACK_MODES),
        help=(
            "Feedback-content ablation rung (D24): how much of the execution "
            "failure the model is shown. Default: full."
        ),
    )
    parser.add_argument(
        "--prompt_strategy",
        default="cop",
        help=(
            "Round 0 generation strategy, re-used as the regeneration prompt "
            "when --feedback_mode blind. Must match the seed. Default: cop."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env_file(ENV_PATH)

    client = get_client(args.backend)
    completion = _load_completion_from_jsonl(args.jsonl_path, args.task_id, args.dataset)

    print(f"Task     : {args.task_id}")
    print(f"Dataset  : {args.dataset}")
    print(f"Backend  : {args.backend}")
    print(f"Model    : {args.model}")
    print(f"Max rds  : {args.max_rounds}")
    print(f"Strategy : {args.repair_strategy}")
    print(f"Feedback : {args.feedback_mode}")
    print()

    result = run_self_repair(
        task_id=args.task_id,
        initial_completion=completion,
        dataset=args.dataset,
        client=client,
        model=args.model,
        backend=args.backend,
        max_repair_rounds=args.max_rounds,
        repair_strategy=args.repair_strategy,
        feedback_mode=args.feedback_mode,
        prompt_strategy=args.prompt_strategy,
    )

    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
