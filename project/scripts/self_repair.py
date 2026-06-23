"""
self_repair.py
Thesis — Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

Orchestrator for the iterative self-repair loop. Runs a single task
through up to N rounds of LLM-driven repair, tying together
code_executor, safety_check, repair_prompt, and the LLM-calling
logic from generate_samples.

Functions
    run_self_repair()   — full N-round repair loop for one task

Usage
    python self_repair.py --task_id HumanEval/40 --dataset humaneval \
        --jsonl_path ../samples/llama-33-70b-versatile_t02_cgo.jsonl \
        --backend groq --model llama-3.3-70b-versatile --max_rounds 2
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
from repair_prompt import build_repair_prompt, build_repair_context
from generate_samples import (
    load_env_file,
    get_client,
    generate_raw_completion,
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


def _build_repair_messages(repair_prompt_text: str) -> list:
    """Wrap a repair prompt in the chat message format used by generate_raw_completion."""
    return repair_prompt_text


def run_self_repair(
    task_id: str,
    initial_completion: str,
    dataset: str,
    client,
    model: str,
    backend: str,
    max_repair_rounds: int = 2,
) -> dict:
    """Run the full N-round repair loop for a single task.

    Round 0 executes the initial_completion. If it fails, up to
    max_repair_rounds of LLM-driven repair are attempted, each time
    feeding the broken code and its error signal back to the model.

    Each round entry (except round 0) includes ``completion_changed``
    indicating whether the model produced a different completion from
    the previous round. Round 0 sets this to None (no prior round).

    The top-level result includes ``stalled``: True when the task is
    unsolved AND the model never produced a different completion across
    any repair round — meaning it did not engage with the repair signal
    at all, as opposed to genuinely attempting different (but still
    wrong) fixes. This distinction matters for thesis analysis: a stall
    indicates the repair prompt failed to elicit a behavioural change,
    while a non-stalled failure means the model tried and couldn't fix
    the underlying bug.

    Returns a trajectory dict capturing every round's completion,
    executor result, and safety scan for per-round thesis analysis.
    """
    problems = get_problems(dataset)
    task = problems[task_id]
    task_prompt = task["prompt"]
    entry_point = task["entry_point"]
    sample_key = "solution" if dataset == "mbpp" else "completion"

    rounds = []
    previous_completion = None
    current_completion = initial_completion

    total_rounds = 1 + max_repair_rounds

    for round_num in range(total_rounds):
        exec_result = run_executor(
            task_id=task_id,
            completion=current_completion,
            dataset=dataset,
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
                "solved": True,
                "solved_at_round": round_num,
                "stalled": False,
                "rounds": rounds,
            }

        if round_num >= max_repair_rounds:
            break

        context = build_repair_context(exec_result, dataset)
        repair_prompt_text = build_repair_prompt(
            task_prompt=task_prompt,
            broken_completion=current_completion,
            error_type=context["error_type"],
            error_message=context["error_message"],
            dataset=dataset,
        )

        raw_response = generate_raw_completion(client, model, repair_prompt_text)

        if raw_response is None:
            print(f"  Round {round_num + 1}: SKIPPED (API error)")
            break

        if dataset == "mbpp":
            new_completion = post_process_solution(raw_response)
        else:
            new_completion = post_process(raw_response, entry_point=entry_point)

        if not new_completion:
            print(f"  Round {round_num + 1}: SKIPPED (empty after post-process)")
            break

        previous_completion = current_completion
        current_completion = new_completion

    repair_rounds = [r for r in rounds if r["round"] > 0]
    stalled = (
        len(repair_rounds) > 0
        and all(r["completion_changed"] is False for r in repair_rounds)
    )

    return {
        "task_id": task_id,
        "dataset": dataset,
        "solved": False,
        "solved_at_round": None,
        "stalled": stalled,
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
    print()

    result = run_self_repair(
        task_id=args.task_id,
        initial_completion=completion,
        dataset=args.dataset,
        client=client,
        model=args.model,
        backend=args.backend,
        max_repair_rounds=args.max_rounds,
    )

    print()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
