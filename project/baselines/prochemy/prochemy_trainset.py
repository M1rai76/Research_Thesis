"""
prochemy_trainset.py
Baseline - Prochemy (Ye et al., 2025, arXiv:2503.11085)
Author : Gurdiraj Bal (z5386590)

Training-set handling for the Prochemy search. Prochemy scores each candidate
prompt by pass@1 on a small (20-task) training set whose tasks are disjoint from
the HumanEval+/MBPP+ test sets (paper Sec. III-B; footnote 1).

Two paths:
    * load_training_set()      - default: reuse the authors' shipped 20-task set
                                 (prochemy_training_set.jsonl, copied verbatim from
                                 their repo). Most faithful, fully reproducible, and
                                 zero API cost - so it is the default.
    * generate_training_set()  - optional (--regenerate-trainset): reproduce their
                                 Sec. III-B recipe (K existing sampled from a disjoint
                                 dataset + K LLM-mutated entries, each execution-
                                 validated). Costs API calls; gated behind a flag.

Each training entry is HumanEval-style:
    {task_id, prompt, entry_point, canonical_solution, test}
where `test` defines `check(candidate)`. run_training_task() grades a candidate
program against that check via the shared code_executor sandbox - reproducing
Prochemy's execution-based pass@1 fitness signal (M_ij in {0,1}) using this
project's own executor rather than the authors' vendored human-eval harness, so
the fitness *signal* is faithful while the *implementation* stays in-project and
reuses shared utilities.

Data-quality note on the shipped set: it is the authors' *synthetic* training
data, and ~5/20 of the bundled canonical_solutions do not pass their own tests
(order-sensitive asserts, an off-by-one, and one unterminated docstring in
auto/9). This is a property of their released data, not a defect here - and it
does not affect the search, because the fitness oracle is each task's `test`
(the model generates fresh code that is graded against the test), never the
bundled canonical. The canonicals are only consulted by validate_generated_entry
on the optional regeneration path.
"""

import json
import os
import sys
from typing import Dict, List, Optional

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)

from code_executor import execute_in_sandbox  # noqa: E402

# Default shipped training set (verbatim copy of the authors'
# code_generation/code_generation_training_set.jsonl).
DEFAULT_TRAINING_SET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "prochemy_training_set.jsonl"
)

# Per-task execution timeout during the search (seconds).
TRAIN_TIMEOUT = 10


def load_training_set(path: Optional[str] = None) -> List[Dict]:
    """Load the training set JSONL as a list of task dicts.

    Validates that every entry carries the fields the search needs.
    """
    path = path or DEFAULT_TRAINING_SET
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training set not found: {path}")

    tasks: List[Dict] = []
    required = {"task_id", "prompt", "entry_point", "canonical_solution", "test"}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            missing = required - set(entry.keys())
            if missing:
                raise ValueError(
                    f"{path}:{line_no} training entry missing fields: {sorted(missing)}"
                )
            tasks.append(entry)

    if not tasks:
        raise ValueError(f"Training set {path} is empty")
    return tasks


def run_training_task(candidate_code: str, task: Dict, timeout: int = TRAIN_TIMEOUT) -> bool:
    """Execute a candidate program against one training task's tests.

    `candidate_code` is the full Python program extracted from the model's
    ```python``` block (it already defines `task['entry_point']`). We append the
    task's `check(candidate)` definition plus an explicit `check(<entry_point>)`
    call and run it in the shared sandbox. Returns True iff it passes (M_ij == 1).

    Any empty/None candidate scores False without executing.
    """
    if not candidate_code or not candidate_code.strip():
        return False

    entry_point = task["entry_point"]
    test_code = f"{task['test']}\n\ncheck({entry_point})\n"
    result = execute_in_sandbox(
        full_code=candidate_code,
        test_code=test_code,
        timeout_seconds=timeout,
    )
    return bool(result.get("passed"))


def validate_generated_entry(entry: Dict, timeout: int = TRAIN_TIMEOUT) -> bool:
    """Validate a *generated* training entry the way the authors do: the
    canonical_solution must pass the entry's own tests (two-step functional
    integrity check, 0_train_set_generate.py). Used only by generate_training_set.

    NB: the entry's `prompt` ends with the signature line plus a trailing partial
    indent (e.g. ``def f(...):\\n    ``), so a naive ``prompt + canonical_solution``
    double-indents the first body line. rstrip the prompt and rejoin with a single
    newline; the canonical carries its own 4-space body indent.
    """
    required = {"task_id", "prompt", "entry_point", "canonical_solution", "test"}
    if required - set(entry.keys()):
        return False
    full_code = entry["prompt"].rstrip() + "\n" + entry["canonical_solution"]
    return run_training_task(full_code, entry, timeout=timeout)


def generate_training_set(*args, **kwargs):
    """Optional reproduction of Prochemy's Sec. III-B training-set generation
    (K existing sampled from a disjoint dataset + K execution-validated mutated
    entries). Not implemented in this scaffold: the default and recommended path
    is to reuse the authors' shipped set (load_training_set), which is faithful,
    fully reproducible, and free. Wire this up only if a fresh training set is
    explicitly wanted for an ablation - it costs API calls and, being random,
    reduces comparability with the authors' reported run.
    """
    raise NotImplementedError(
        "Training-set regeneration is not implemented; reuse the shipped "
        "prochemy_training_set.jsonl (load_training_set). See module docstring."
    )


__all__ = [
    "DEFAULT_TRAINING_SET",
    "load_training_set",
    "run_training_task",
    "validate_generated_entry",
    "generate_training_set",
]
