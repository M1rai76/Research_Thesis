"""
reflexion_prompt.py
Baseline - Reflexion (Shinn et al., 2023)
Author : Gurdiraj Bal (z5386590)

Prompt builders for the Reflexion baseline - a faithful re-implementation
of "Reflexion: Language Agents with Verbal Reinforcement Learning"
(arXiv:2303.11366), Section 4.3 (Programming).

Unlike the thesis's own `cot`/`minimal` repair strategies
(project/scripts/repair_prompt.py), which repair against ground-truth
test execution, Reflexion's programming agent only has access to a
self-generated test suite as its in-loop evaluator, and separates the
"why did this fail" judgement (Self-Reflection model) from the "write
the fix" step (Actor model) into two distinct LLM calls, with the
reflection persisted across rounds in a bounded memory buffer.

Functions
    build_test_generation_prompt()   - ask the model to write its own test suite
    parse_and_filter_tests()         - AST-filter the model's proposed tests
    format_failure_summary()         - render a self-test failure as prompt text
    build_self_reflection_prompt()   - ask the model to critique its failed attempt
    build_reflexion_actor_prompt()   - ask the model to produce the next fix
"""

import ast
import os
import sys
from typing import Optional

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)

from repair_prompt import _output_constraints  # noqa: E402


MAX_SELF_TESTS = 6


def build_test_generation_prompt(
    task_prompt: str,
    entry_point: str,
    dataset: str = "humaneval",
) -> str:
    """Construct a CoT prompt asking the model to write its own test suite.

    Mirrors Shinn et al. Section 4.3: "we use Chain-of-Thought prompting
    to produce diverse, extensive tests with corresponding natural
    language descriptions." Tests are plain ``assert`` statements calling
    the entry point directly (matching MBPP's existing ``assertion``
    convention), so they slot into code_executor.execute_in_sandbox()
    without needing a wrapping ``check(candidate)`` function.
    """
    return (
        "You are an expert Python test writer. Given the following function "
        f"specification, write up to {MAX_SELF_TESTS} diverse unit tests that "
        "check its correctness, including typical cases, boundary values, and "
        "empty/edge-case inputs.\n\n"
        "For each test, write a one-line comment describing what it checks, "
        f"followed by a single `assert` statement that calls `{entry_point}` "
        "directly. For example:\n"
        "# Checks empty input returns the identity value\n"
        f"assert {entry_point}([]) == 0\n\n"
        "--- Task ---\n"
        f"{task_prompt}\n\n"
        "Output only the comment + assert pairs, one test per pair, "
        "with no other text, no markdown, and no explanations."
    )


def parse_and_filter_tests(
    raw_response: str,
    entry_point: str,
    max_tests: int = MAX_SELF_TESTS,
) -> str:
    """Filter a model's proposed test suite down to syntactically valid tests.

    Mirrors Shinn et al. Section 4.3: "filter for syntactically valid test
    statements by attempting to construct a valid abstract syntax tree
    (AST) for each proposed test." A line is kept only if it parses as
    valid Python, starts with ``assert``, and references the entry point
    (guards against the model asserting on an unrelated helper name).

    Returns
        A newline-joined block of up to ``max_tests`` valid assert
        statements, or an empty string if none were valid.
    """
    kept = []
    for raw_line in raw_response.splitlines():
        line = raw_line.strip()
        if not line.startswith("assert"):
            continue
        if entry_point not in line:
            continue
        try:
            ast.parse(line)
        except SyntaxError:
            continue
        kept.append(line)
        if len(kept) >= max_tests:
            break

    return "\n".join(kept)


def format_failure_summary(
    error_type: Optional[str],
    error_message: Optional[str],
) -> str:
    """Format a self-test execution result as prompt text.

    Deliberately kept local rather than reused from repair_prompt's
    error-formatting helper, since the Reflexion strategy's failure
    signal comes from the self-generated test suite, not ground-truth
    execution - a distinct data source that happens to share the same
    {error_type, error_message} shape.
    """
    if error_type is None:
        return "All self-written tests passed."
    if error_message:
        return f"The self-written tests failed with: {error_type}: {error_message}"
    return f"The self-written tests failed with: {error_type}"


def _format_memory(memory: list) -> str:
    """Render accumulated self-reflections as prompt text."""
    if not memory:
        return "(no prior reflections yet)"
    return "\n".join(f"- {entry}" for entry in memory)


def build_self_reflection_prompt(
    task_prompt: str,
    broken_completion: str,
    failure_summary: str,
    memory: list,
) -> str:
    """Construct the Self-Reflection call (Reflexion's M_sr).

    Given the current failing attempt, its self-test failure, and prior
    reflections, ask the model to produce ONE natural-language critique
    explaining what went wrong and what to try differently - not code.
    Mirrors the paper's Appendix C.3 Self-Reflection instruction.
    """
    return (
        "You are a Python writing assistant reflecting on a failed attempt. "
        "You will be given a task, your previous implementation, and the "
        "results of unit tests you wrote yourself. Explain in 2-4 sentences, "
        "in the first person, why the implementation failed and what you "
        "should do differently on the next attempt. Do not write code - "
        "output only the reflection text.\n\n"
        "--- Task ---\n"
        f"{task_prompt}\n\n"
        "--- Your previous attempt ---\n"
        f"{broken_completion}\n\n"
        "--- Self-test result ---\n"
        f"{failure_summary}\n\n"
        "--- Prior reflections from earlier attempts ---\n"
        f"{_format_memory(memory)}\n"
    )


def build_reflexion_actor_prompt(
    task_prompt: str,
    broken_completion: str,
    failure_summary: str,
    memory: list,
    dataset: str = "humaneval",
) -> str:
    """Construct the Actor call for the next attempt (Reflexion's M_a).

    Conditions the next fix on the accumulated self-reflection memory
    rather than the raw error alone, following the paper's Actor
    generation form: (Instruction)(Function implementation)(Unit test
    feedback)(Self-reflection)(Instruction for next implementation).
    """
    output_constraints = _output_constraints(dataset)

    return (
        "You are an expert Python programmer. The following code you wrote "
        "failed its self-written tests. Using your own past reflections as "
        "guidance, fix it.\n\n"
        "--- Task ---\n"
        f"{task_prompt}\n\n"
        "--- Your previous attempt ---\n"
        f"{broken_completion}\n\n"
        "--- Self-test result ---\n"
        f"{failure_summary}\n\n"
        "--- Your reflections from earlier attempts ---\n"
        f"{_format_memory(memory)}\n\n"
        f"Fix the code. {output_constraints}\n"
    )
