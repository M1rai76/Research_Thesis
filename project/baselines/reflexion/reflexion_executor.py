"""
reflexion_executor.py
Baseline - Reflexion (Shinn et al., 2023)
Author : Gurdiraj Bal (z5386590)

Execute a candidate completion against a self-generated test suite,
rather than the ground-truth visible tests used by
project/scripts/code_executor.py. This is Reflexion's in-loop Evaluator
(M_e) for programming tasks: the agent only sees whether its own tests
pass, never the real answer, until the trajectory is scored afterward.

Functions
    run_self_generated_tests()   - execute a completion against self-written tests
"""

import os
import sys

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)

from code_executor import (  # noqa: E402
    DEFAULT_TIMEOUT,
    execute_in_sandbox,
    reconstruct_full_code,
)


def run_self_generated_tests(
    task_prompt: str,
    completion: str,
    entry_point: str,
    test_code: str,
    dataset: str = "humaneval",
    timeout_seconds: int = DEFAULT_TIMEOUT,
) -> dict:
    """Execute a completion against a self-generated (not ground-truth) test suite.

    Mirrors code_executor.run_executor()'s shape and sandboxing exactly,
    substituting the self-written ``test_code`` for the visible tests.
    If ``test_code`` is empty (no valid self-tests survived filtering),
    treats the round as a failure with a dedicated error_type so callers
    can distinguish "no usable self-tests" from "self-tests ran and failed".

    Returns
        {
            "passed": bool,
            "error_type": str | None,
            "error_message": str | None,
            "timed_out": bool,
        }
    """
    if not test_code.strip():
        return {
            "passed": False,
            "error_type": "NoValidSelfTests",
            "error_message": "No syntactically valid self-generated tests were available.",
            "timed_out": False,
        }

    full_code = reconstruct_full_code(
        task_prompt=task_prompt,
        completion=completion,
        entry_point=entry_point,
        dataset=dataset,
    )

    return execute_in_sandbox(
        full_code=full_code,
        test_code=test_code,
        timeout_seconds=timeout_seconds,
    )
