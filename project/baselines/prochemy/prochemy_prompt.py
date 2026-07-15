"""
prochemy_prompt.py
Baseline - Prochemy / "Prompt Alchemy" (Ye et al., 2025, arXiv:2503.11085)
Author : Gurdiraj Bal (z5386590)

Verbatim prompts and parsers for the Prochemy baseline, transcribed from the
authors' reference implementation (github.com/buriyuanyou/Prochemy,
code_generation/) so the reproduction is byte-faithful rather than paraphrased
from the paper body:

    * INITIAL_PROMPT_ZERO_SHOT  - origin_prompt.jsonl (the seed S(0))
    * the prompt-mutation meta-prompt        - 1_prompt_mutate.py / 2+3+4_reinfocement.py
    * the data-mutation meta-prompt          - 0_train_set_generate.py (trainset regen only)
    * {{...}} / [Start]..[End] / ```python``` extraction helpers

Only the *prompts* live here; the search loop is in prochemy_optimize.py. The
initial/mutation prompts are the paper's own, NOT this project's hand-designed
templates: substituting ours would make it "Prochemy's optimiser applied to our
prompts", a hybrid rather than a faithful reproduction of the baseline.

Temperature convention for this baseline: solution generation runs at this
project's fixed 0.2 (matching every other run here, so results are comparable),
while the prompt-mutation step keeps the paper's 1.0 (its diversity mechanism).
"""

import re
from typing import List, Optional


# --------------------------------------------------------------------------- #
# 1. Initial prompt S(0)  (origin_prompt.jsonl)                                #
# --------------------------------------------------------------------------- #
# The zero-shot seed shipped verbatim in the authors' origin_prompt.jsonl.
INITIAL_PROMPT_ZERO_SHOT = (
    "You are a code generation assistant. Your task is to generate Python code "
    "based on the given task description and complete the work described in the task.\n"
)

# The authors ship only the zero-shot origin prompt. Their Table I also reports a
# "CoT + Prochemy" row, whose seed is a chain-of-thought prompt not included in the
# repo. This CoT seed is therefore *our* faithful construction (zero-shot seed +
# an explicit step-by-step instruction), clearly flagged as a reconstruction so it
# is never mistaken for the authors' verbatim text. The zero-shot seed is the
# primary, unambiguous baseline; use --initial cot only for the secondary CoT run.
INITIAL_PROMPT_COT = (
    "You are a code generation assistant. Your task is to generate Python code "
    "based on the given task description and complete the work described in the task. "
    "Let's think step by step: reason through the problem before writing the final code.\n"
)

INITIAL_PROMPTS = {
    "zero_shot": INITIAL_PROMPT_ZERO_SHOT,
    "cot": INITIAL_PROMPT_COT,
}


# --------------------------------------------------------------------------- #
# 2. Prompt-mutation meta-prompt  (1_prompt_mutate.py / 2+3+4_reinfocement.py) #
# --------------------------------------------------------------------------- #
MUTATION_SYSTEM = "\nYou are an expert prompt engineer.\n"

_MUTATION_INFORMATION = (
    "\nPlease help me improve the given prompt to get a more helpful and harmless response.\n"
    "Suppose I need to generate a Python program based on natural language descriptions.\n"
    "The generated Python program should be able to complete the tasks described in "
    "natural language and pass any test cases specific to those tasks.\n\n"
)

_MUTATION_FORMAT = (
    "\nYou may add any information you think will help improve the task's effectiveness "
    "during the prompt optimization process.\n"
    "If you find certain expressions and wording in the original prompt inappropriate, "
    "you can also modify these usages.\n"
    "Ensure that the optimized prompt includes a detailed task description and clear "
    "process guidance added to the original prompt.\n"
    "Wrap the optimized prompt in {{}}.\n"
)


def build_mutation_user(prompt_text: str) -> str:
    """Build the user message for one prompt-mutation call.

    Mirrors 2+3+4_reinfocement.py's generate_new_prompts(): the prompt under
    optimisation is wrapped in [] and sandwiched between the information and
    format blocks. The call is made with MUTATION_SYSTEM as the system message
    (see prochemy_optimize.mutate_prompts), temperature 1.0.
    """
    formatted = (
        "The prompt ready to be optimized are as follows and wrapped in []:\n"
        f"[{prompt_text}]\n"
    )
    return _MUTATION_INFORMATION + formatted + _MUTATION_FORMAT


def extract_mutated_prompt(text: str) -> Optional[str]:
    """Extract the {{...}}-wrapped optimised prompt from a mutation response.

    Returns None when no wrapped content is present, so the caller can retry
    (matching the authors' process_optimization_task retry loop).
    """
    match = re.search(r"\{\{(.*?)\}\}", text, re.DOTALL)
    return match.group(1).strip() if match else None


# --------------------------------------------------------------------------- #
# 3. Solution extraction  (2_prompt_evaluate.py)                              #
# --------------------------------------------------------------------------- #
def extract_python_block(text: str) -> Optional[str]:
    """Extract the ```python ... ``` fenced code block from a solution response.

    Matches the authors' evaluation-time extraction (2_prompt_evaluate.py). The
    Prochemy system prompt asks for a full fenced Python program, so during the
    search we grade whatever is inside the first ```python``` fence. Returns None
    when no fenced block is found (caller retries up to MAX_RETRIES).
    """
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: a bare ``` ... ``` fence with no language tag.
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else None


# --------------------------------------------------------------------------- #
# 4. Data-mutation meta-prompt  (0_train_set_generate.py)                      #
#    Only needed when regenerating the training set (--regenerate-trainset);   #
#    the default run reuses the authors' shipped prochemy_training_set.jsonl.  #
# --------------------------------------------------------------------------- #
DATA_MUTATION_SYSTEM = "\nYou are an expert in software engineering.\n"

DATA_MUTATION_INFORMATION = (
    "\nPlease help me generate similar data based on the format provided below.\n"
)

DATA_MUTATION_FORMAT = (
    "\nEnsure that the data you provide is consistent with the reference data format, "
    "and that all test cases included in the data are correct.\n"
    "Ensure that the generated data is different from the provided reference data.\n"
    "Return the data in the same Json format as the reference data and wrapped the data "
    "with [Start] and [End].\n"
)


def extract_data_entry(text: str) -> Optional[str]:
    """Extract the [Start]..[End]-wrapped JSON data entry from a data-mutation
    response (used only by the optional training-set regeneration path)."""
    match = re.search(r"\[Start\]\s*\n(.*?)\n\[End\]", text, re.DOTALL)
    return match.group(1).strip() if match else None


__all__ = [
    "INITIAL_PROMPT_ZERO_SHOT",
    "INITIAL_PROMPT_COT",
    "INITIAL_PROMPTS",
    "MUTATION_SYSTEM",
    "build_mutation_user",
    "extract_mutated_prompt",
    "extract_python_block",
    "DATA_MUTATION_SYSTEM",
    "DATA_MUTATION_INFORMATION",
    "DATA_MUTATION_FORMAT",
    "extract_data_entry",
]
