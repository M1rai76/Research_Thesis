"""
reflexion_repair.py
Baseline - Reflexion (Shinn et al., 2023)
Author : Gurdiraj Bal (z5386590)

Orchestrator for the Reflexion baseline's per-task repair loop. Parallel
to project/scripts/self_repair.py's run_self_repair(), but not a branch
inside it - Reflexion's loop design differs enough (self-generated test
evaluator, a separate Self-Reflection call, a persisting memory buffer)
that keeping it as an independent module avoids entangling the thesis's
own `cot`/`minimal` repair strategies with this baseline's logic.

Ground truth (project/scripts/code_executor.run_executor) is still run
every round so `solved`/`solved_at_round` stay directly comparable to
the `cot`/`minimal` trajectories and so downstream EvalPlus grading is
unaffected - but it plays no role in the agent's own decision-making,
matching the paper's design where the agent never sees the hidden/real
verdict mid-loop.

Functions
    run_reflexion_repair()   - full N-round Reflexion loop for one task
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from code_executor import run_executor  # noqa: E402
from safety_check import check_safety  # noqa: E402
from generate_samples import (  # noqa: E402
    generate_raw_completion,
    get_problems,
    post_process,
    post_process_solution,
)

from reflexion_prompt import (  # noqa: E402
    build_reflexion_actor_prompt,
    build_self_reflection_prompt,
    build_test_generation_prompt,
    format_failure_summary,
    parse_and_filter_tests,
)
from reflexion_executor import run_self_generated_tests  # noqa: E402


def run_reflexion_repair(
    task_id: str,
    initial_completion: str,
    dataset: str,
    client,
    model: str,
    backend: str,
    max_repair_rounds: int = 2,
    reflexion_memory_size: int = 1,
) -> dict:
    """Run the full N-round Reflexion loop for a single task.

    Round 0 executes the initial_completion against both ground truth
    and a freshly self-generated test suite. On self-test failure, up to
    max_repair_rounds of Reflexion-style repair are attempted: a
    Self-Reflection call producing a natural-language critique, appended
    to a bounded memory buffer, followed by an Actor call producing the
    next fix conditioned on that memory.

    The loop's stop/continue decision is driven entirely by
    ``self_test_passed`` (the agent's only visible signal, per the
    paper), not by the ground-truth ``passed`` field. If self-tests pass
    while ground truth still fails, the loop stops early and this is
    recorded as ``stopped_early_self_test_pass`` - analogous to a false
    positive in the paper's Table 2.

    Returns a trajectory dict:
        {
            "task_id", "dataset", "repair_strategy": "reflexion",
            "self_test_code": str,
            "solved": bool,                    # ground-truth outcome
            "solved_at_round": int | None,
            "stopped_early_self_test_pass": bool,
            "reflection_history": list[str],   # full, untruncated
            "api_calls": int,
            "rounds": [
                {
                    "round", "completion",
                    "passed", "error_type", "error_message", "timed_out",
                    "self_test_passed", "self_test_error_type",
                    "self_test_error_message",
                    "safety": {...},
                },
                ...
            ],
        }
    """
    problems = get_problems(dataset)
    task = problems[task_id]
    task_prompt = task["prompt"]
    entry_point = task["entry_point"]

    api_calls = 0

    # api_failed distinguishes "generate_raw_completion exhausted its retries and
    # returned None" (e.g. Groq daily-quota exhaustion) from a legitimate model
    # response that simply produced no usable output. The batch driver uses it to
    # abort-and-not-persist so the task is retried on --resume-missing, rather
    # than churning through the rest of the run saving degraded trajectories.
    api_failed = False

    test_gen_prompt = build_test_generation_prompt(task_prompt, entry_point, dataset)
    test_gen_response = generate_raw_completion(client, model, test_gen_prompt)
    api_calls += 1
    if test_gen_response is None:
        api_failed = True
    self_test_code = (
        parse_and_filter_tests(test_gen_response, entry_point)
        if test_gen_response is not None
        else ""
    )

    rounds = []
    memory = []
    reflection_history = []
    current_completion = initial_completion
    stopped_early_self_test_pass = False
    # First round (if any) where ground truth passed, tracked independently
    # of the self-test verdict - self_test_passed only decides whether the
    # loop stops early, it must never gate whether "solved" reflects reality.
    ground_truth_solved_at = None

    total_rounds = 1 + max_repair_rounds

    for round_num in range(total_rounds):
        exec_result = run_executor(
            task_id=task_id,
            completion=current_completion,
            dataset=dataset,
        )
        if exec_result["passed"] and ground_truth_solved_at is None:
            ground_truth_solved_at = round_num
        self_test_result = run_self_generated_tests(
            task_prompt=task_prompt,
            completion=current_completion,
            entry_point=entry_point,
            test_code=self_test_code,
            dataset=dataset,
        )
        safety_result = check_safety(
            task_id=task_id,
            completion=current_completion,
            dataset=dataset,
        )

        round_entry = {
            "round": round_num,
            "completion": current_completion,
            "passed": exec_result["passed"],
            "error_type": exec_result.get("error_type"),
            "error_message": exec_result.get("error_message"),
            "timed_out": exec_result.get("timed_out", False),
            "self_test_passed": self_test_result["passed"],
            "self_test_error_type": self_test_result.get("error_type"),
            "self_test_error_message": self_test_result.get("error_message"),
            "safety": {
                "issue_count": safety_result["issue_count"],
                "max_severity": safety_result["max_severity"],
                "findings": safety_result["findings"],
            },
        }
        rounds.append(round_entry)

        tag = "PASS" if exec_result["passed"] else "FAIL"
        self_tag = "PASS" if self_test_result["passed"] else "FAIL"
        print(f"  Round {round_num}: ground-truth={tag}, self-test={self_tag}")

        if self_test_result["passed"]:
            stopped_early_self_test_pass = not exec_result["passed"]
            return {
                "task_id": task_id,
                "dataset": dataset,
                "repair_strategy": "reflexion",
                "self_test_code": self_test_code,
                "solved": ground_truth_solved_at is not None,
                "solved_at_round": ground_truth_solved_at,
                "stopped_early_self_test_pass": stopped_early_self_test_pass,
                "reflection_history": reflection_history,
                "api_calls": api_calls,
                "api_failed": api_failed,
                "rounds": rounds,
            }

        if round_num >= max_repair_rounds:
            break

        failure_summary = format_failure_summary(
            self_test_result.get("error_type"),
            self_test_result.get("error_message"),
        )

        reflection_prompt = build_self_reflection_prompt(
            task_prompt=task_prompt,
            broken_completion=current_completion,
            failure_summary=failure_summary,
            memory=memory,
        )
        reflection_text = generate_raw_completion(client, model, reflection_prompt)
        api_calls += 1
        if reflection_text is None:
            print(f"  Round {round_num + 1}: SKIPPED (self-reflection API error)")
            api_failed = True
            break
        reflection_text = reflection_text.strip()
        reflection_history.append(reflection_text)
        memory.append(reflection_text)
        if len(memory) > reflexion_memory_size:
            memory = memory[-reflexion_memory_size:]

        actor_prompt = build_reflexion_actor_prompt(
            task_prompt=task_prompt,
            broken_completion=current_completion,
            failure_summary=failure_summary,
            memory=memory,
            dataset=dataset,
        )
        raw_response = generate_raw_completion(client, model, actor_prompt)
        api_calls += 1
        if raw_response is None:
            print(f"  Round {round_num + 1}: SKIPPED (actor API error)")
            api_failed = True
            break

        new_completion = (
            post_process_solution(raw_response)
            if dataset == "mbpp"
            else post_process(raw_response, entry_point=entry_point)
        )
        if not new_completion:
            print(f"  Round {round_num + 1}: SKIPPED (empty after post-process)")
            break

        current_completion = new_completion

    return {
        "task_id": task_id,
        "dataset": dataset,
        "repair_strategy": "reflexion",
        "self_test_code": self_test_code,
        "solved": ground_truth_solved_at is not None,
        "solved_at_round": ground_truth_solved_at,
        "stopped_early_self_test_pass": False,
        "reflection_history": reflection_history,
        "api_calls": api_calls,
        "api_failed": api_failed,
        "rounds": rounds,
    }
