"""
generate_samples.py
Thesis — Prompt Engineering for LLM Code Generation
Author    : Samyak Diwan (z5611048)
Edited by : Gurdiraj Bal (z5386590)   # added the `katana` (vLLM OpenAI-compatible) backend;
                                      # added optional system_prompt/temperature/max_tokens
                                      # overrides to generate_raw_completion for the Prochemy baseline

Pipeline
    build_prompt()
    generate_raw_completion()
    post_process()
    write_samples_jsonl()
    save_run_log()

Evaluation (run separately after this script)
    # 1. Raw evaluation
    docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
      evalplus.evaluate --dataset humaneval \
      --samples /app/samples/<output_file>.jsonl

    # 2. Sanitized evaluation
    evalplus.sanitize --samples samples/<output_file>.jsonl --dataset humaneval
    docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
      evalplus.evaluate --dataset humaneval \
      --samples /app/samples/<output_file>-sanitized.jsonl

Usage
    # Ollama (local)
    python generate_samples.py --model qwen2.5-coder:7b --backend ollama

    # Ollama with a different prompt strategy
    python generate_samples.py --model qwen2.5-coder:7b --backend ollama --prompt cot

    # Groq (API)
    export GROQ_API_KEY=<your_key>
    python generate_samples.py --model llama-3.3-70b-versatile --backend groq
"""

import json
import os
import re
import time
import random
import argparse
from datetime import date
from typing import Optional

from evalplus.data import get_human_eval_plus, get_mbpp_plus


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Fixed Parameters.
TEMPERATURE : float = 0.2
MAX_TOKENS  : int   = 512
MAX_RETRIES : int   = 5

SLEEP = {"ollama": 1.0, "groq": 2.0, "cerebras": 2.0, "openrouter": 6.0, "gemini": 5.0, "katana": 0.0}

# Conservative Python top-level stop markers. These truncate obvious
# post-function continuation (example usage, test blocks, second functions).
# \nif and \nassert are excluded because they appear legitimately inside bodies.
STOP_MARKERS = [
    "\ndef ",
    "\nclass ",
    "\nprint(",
    "\n# Example",
    "\n# Test",
    "\nif __name__",
]


def load_env_file(path: str = ENV_PATH) -> None:
    """
    Load simple KEY=value entries from project/.env without overriding values
    already exported in the shell.
    """
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate EvalPlus code-generation samples for thesis experiments."
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Model name. "
            "Ollama examples : qwen2.5-coder:7b, llama3.1:8b. "
            "Groq examples   : llama-3.3-70b-versatile, mixtral-8x7b-32768."
        ),
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=["ollama", "groq", "cerebras", "openrouter", "gemini", "katana"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--prompt",
        default="baseline",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="Prompt strategy to use. Selects from PROMPT_TEMPLATES. Default: baseline.",
    )
    parser.add_argument(
        "--dataset",
        default="humaneval",
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset to generate for. Default: humaneval.",
    )
    parser.add_argument(
        "--resume-missing",
        "--resume",
        dest="resume_missing",
        action="store_true",
        help="Keep existing samples in the output file and generate only missing task IDs.",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Run iterative self-repair after the normal Round 0 generation.",
    )
    parser.add_argument(
        "--repair_strategy",
        default="cot",
        choices=["minimal", "cot"],
        help="Repair prompt strategy to use when --repair is set. Default: cot.",
    )
    parser.add_argument(
        "--max_repair_rounds",
        type=int,
        default=2,
        help="Maximum repair rounds when --repair is set. Default: 2.",
    )
    parser.add_argument(
        "--feedback_mode",
        default="full",
        choices=["grounded+", "full", "error-type", "binary", "blind"],
        help=(
            "Feedback-content ablation rung (decisions.md D24): how much of the "
            "execution failure the repair prompt discloses. 'full' reproduces "
            "pre-ablation behaviour. Non-full modes add an _fb-<mode> suffix to "
            "every output path so arms cannot clobber each other. Default: full."
        ),
    )
    parser.add_argument(
        "--run_tag",
        default=None,
        help=(
            "Optional suffix appended to every output artifact (e.g. 'run2'), so "
            "repeat runs of the same configuration land side by side instead of "
            "overwriting (decisions.md D16)."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional task limit for smoke tests.",
    )
    parser.add_argument(
        "--round0_source_jsonl",
        default=None,
        help=(
            "Optional existing samples JSONL to reuse for Round 0 in repair mode. "
            "Tasks present here are not regenerated."
        ),
    )
    return parser.parse_args()


def model_to_slug(model: str) -> str:
    """
    Convert a model name to a safe filename slug.
    Examples
        qwen2.5-coder:7b        -> qwen25-coder_7b
        llama3.1:8b             -> llama31_8b
        llama-3.3-70b-versatile -> llama-33-70b-versatile
    """
    return (
        model
        .replace(".", "")
        .replace(":", "_")
        .replace("/", "-")
    )


def get_output_path(model: str, prompt_strategy: str, dataset: str) -> str:
    """
    Build a deterministic output path from model + temperature + strategy.

    Pattern:
        HumanEval: samples/{model_slug}_t{temp*10:02d}_{strategy}.jsonl
        MBPP+    : samples/mbpp_{model_slug}_t{temp*10:02d}_{strategy}.jsonl

    HumanEval keeps the historical filename pattern so existing runs are not
    invalidated by adding dataset support.

    Examples
        qwen25-coder_7b_t02_baseline.jsonl
        mbpp_qwen25-coder_7b_t02_baseline.jsonl
        llama31_8b_t02_cot.jsonl
    """
    slug     = model_to_slug(model)
    temp_tag = f"t{int(TEMPERATURE * 10):02d}"
    prefix   = "" if dataset == "humaneval" else f"{dataset}_"
    return os.path.join(BASE_DIR, "samples", f"{prefix}{slug}_{temp_tag}_{prompt_strategy}.jsonl")

def get_client(backend: str):
    """
    Initialise and return the appropriate API client.

    Ollama : OpenAI-compatible client pointing at localhost:11434. No API key required.
    Groq   : Official Groq client. Requires GROQ_API_KEY environment variable.
    """
    if backend == "ollama":
        from openai import OpenAI
        return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

    if backend == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Get a free key at https://console.groq.com and run:\n"
                "  export GROQ_API_KEY=<your_key>"
            )
        return Groq(api_key=api_key)
    
    if backend == "cerebras":
        from openai import OpenAI
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. "
                "Get a free key at https://cloud.cerebras.ai and run:\n"
                "  $env:CEREBRAS_API_KEY='your_key_here'"
           )
        return OpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=api_key,
        )
    
    if backend == "openrouter":
        from openai import OpenAI
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set. Sign up at openrouter.ai")
        return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    
    if backend == "gemini":
        from openai import OpenAI
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set.")
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )

    if backend == "katana":
        # UNSW Katana HPC: a vLLM (or any OpenAI-compatible) inference server
        # running on the allocated GPU node. The batch job talks to it over the
        # node-local endpoint, so no outbound internet / external API key is
        # needed. KATANA_BASE_URL and KATANA_API_KEY let the SLURM script point
        # at whatever host:port vLLM was launched on; the defaults match a
        # server started on the same node with `--port 8000`. vLLM ignores the
        # API key unless it was launched with --api-key, so "EMPTY" is the
        # conventional placeholder.
        from openai import OpenAI
        base_url = os.getenv("KATANA_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("KATANA_API_KEY", "EMPTY")
        return OpenAI(base_url=base_url, api_key=api_key)


PROMPT_TEMPLATES = {
    "baseline": lambda task_prompt: (
        "Complete the following Python function.\n\n"
        "Rules:\n"
        "- Output only the function body (the lines after the def line)\n"
        "- Do not repeat the function signature or docstring\n"
        "- Do not include markdown fences\n"
        "- Do not include example usage or test code\n"
        "- Do not include explanations\n\n"
        f"{task_prompt}"
    ),
    "zero_shot": lambda task_prompt: (
        f"Complete this Python function:\n\n{task_prompt}"
    ),
    "edge_case": lambda task_prompt: (
        "Complete the following Python function.\n"
        "Your implementation must correctly handle edge cases such as "
        "empty inputs, boundary values, and unusual but valid inputs.\n\n"
        f"{task_prompt}"
    ),
    "role_framing": lambda task_prompt: (
        "You are a senior Python engineer writing production quality code. "
        "Complete the following function with correct logic and careful handling "
        "of edge cases.\n\n"
        f"{task_prompt}"
    ),
    "cot": lambda task_prompt: (
        "Complete the following Python function.\n"
        "Think carefully about edge cases before writing the answer. "
        "Output only the final code.\n\n"
        f"{task_prompt}"
    ),
    "cop": lambda task_prompt: (
        "You are an expert Python programmer.\n\n"
        "Your goal: implement a correct Python function that satisfies "
        "all input/output requirements described below.\n\n"
        "Functional objectives:\n"
        "- Return the exact output type specified\n"
        "- Handle all valid inputs described in the docstring\n"
        "- Correctly handle boundary values and edge cases\n\n"
        "Output only the function body. No explanations. No markdown.\n\n"
        f"{task_prompt}"
    ),
    "io_spec": lambda task_prompt: (
        "You are an expert Python programmer.\n\n"
        "Complete the following Python function.\n\n"
        "Before writing code, identify:\n"
        "- Input types and valid ranges\n"
        "- Expected output type and format\n"
        "- Edge cases: empty input, zero, negative values, single elements\n\n"
        "Then output only the function body that handles all of these correctly.\n"
        "No markdown. No explanations. No repeated function signature.\n\n"
        f"{task_prompt}"
    ),
}


MBPP_PROMPT_TEMPLATES = {
    "baseline": lambda task_prompt: (
        "Write a complete Python solution for the following programming task.\n\n"
        "Rules:\n"
        "- Output only Python code\n"
        "- Include the required function definition and any imports it needs\n"
        "- Do not include markdown fences\n"
        "- Do not include example usage or test code\n"
        "- Do not include explanations\n\n"
        f"{task_prompt}"
    ),
    "zero_shot": lambda task_prompt: (
        f"Write Python code that solves this task:\n\n{task_prompt}"
    ),
    "edge_case": lambda task_prompt: (
        "Write a complete Python solution for the following programming task.\n"
        "Your implementation must correctly handle edge cases such as empty "
        "inputs, boundary values, repeated values, and unusual but valid inputs.\n\n"
        f"{task_prompt}"
    ),
    "role_framing": lambda task_prompt: (
        "You are a senior Python engineer writing production quality code. "
        "Write a complete, correct Python solution for the following task.\n\n"
        f"{task_prompt}"
    ),
    "cot": lambda task_prompt: (
        "Write a complete Python solution for the following programming task.\n"
        "Think carefully about edge cases before writing the answer. "
        "Output only the final Python code.\n\n"
        f"{task_prompt}"
    ),
    "cop": lambda task_prompt: (
        "You are an expert Python programmer.\n\n"
        "Your goal: implement a correct Python solution that satisfies "
        "all input/output requirements described below.\n\n"
        "Functional objectives:\n"
        "- Return the exact output type specified\n"
        "- Handle all valid inputs described in the task\n"
        "- Correctly handle boundary values and edge cases\n\n"
        "Output only Python code. Include the required function definition "
        "and any imports it needs. No explanations. No markdown.\n\n"
        f"{task_prompt}"
    ),
    "io_spec": lambda task_prompt: (
        "You are an expert Python programmer.\n\n"
        "Write a complete Python solution for the following programming task.\n\n"
        "Before writing code, identify:\n"
        "- Input types and valid ranges\n"
        "- Expected output type and format\n"
        "- Edge cases: empty input, zero, negative values, repeated values, single elements\n\n"
        "Then output only Python code that handles all of these correctly. "
        "Include the required function definition and any imports it needs. "
        "No markdown. No explanations.\n\n"
        f"{task_prompt}"
    ),
}


def build_prompt(task_prompt: str, prompt_strategy: str, dataset: str = "humaneval") -> str:
    """
    Return the full prompt string for the given strategy.

    task_prompt     : the 'prompt' field from get_human_eval_plus()
    prompt_strategy : key into PROMPT_TEMPLATES
    """
    templates = MBPP_PROMPT_TEMPLATES if dataset == "mbpp" else PROMPT_TEMPLATES
    return templates[prompt_strategy](task_prompt)


def strip_markdown_fences(text: str) -> str:
    """Remove ```python ... ``` or ``` ... ``` wrappers if present."""
    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    return text


def truncate_at_stop_markers(text: str, entry_point: Optional[str] = None) -> str:
    """Cut off content after the first top-level stop marker.

    If entry_point is provided, a ``\\ndef {entry_point}`` hit is not
    treated as a stop marker — the model returned the target function
    as part of a complete response and it should not be truncated.
    """
    earliest = len(text)
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            if (
                entry_point is not None
                and marker == "\ndef "
                and text[idx:].startswith(f"\ndef {entry_point}")
            ):
                continue
            earliest = idx
    return text[:earliest]

def strip_leading_def(text: str) -> str:
    """Remove a leading 'def ...:' line if the model echoed the signature back."""
    lines = text.splitlines(keepends=True)
    if lines and re.match(r"^def\s+\w+", lines[0]):
        lines = lines[1:]
    return "".join(lines)


def ensure_indented(text: str, entry_point: Optional[str] = None) -> str:
    """Add 4-space indent to all lines if the completion was returned unindented.

    HumanEval evaluation appends the completion directly after the docstring,
    so the body must be indented or the code runs at module level.

    If entry_point is provided and the text already contains a top-level
    definition of that function, the text is returned unchanged — the model
    echoed back a complete function and re-indenting would break it.
    """
    if entry_point is not None:
        for line in text.splitlines():
            if re.match(rf"^def\s+{re.escape(entry_point)}\s*\(", line):
                return text

    lines = text.splitlines(keepends=True)
    first_code = next((l for l in lines if l.strip()), "")
    if first_code and not first_code.startswith((" ", "\t")):
        lines = ["    " + l if l.strip() else l for l in lines]
    return "".join(lines)


def truncate_solution_noise(text: str) -> str:
    """Remove obvious generated examples/tests after a complete solution."""
    markers = [
        "\nif __name__",
        "\n# Example",
        "\n# Test",
        "\nprint(",
        "\nassert ",
    ]
    earliest = len(text)
    for marker in markers:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def post_process(raw_output: str, entry_point: Optional[str] = None) -> str:
    """
    Apply the full post-processing pipeline to raw model output.

    Steps:
        1. strip_markdown_fences    — remove ``` wrappers
        2. strip_leading_def        — drop echoed function signature
        3. ensure_indented          — add 4-space indent if model returned bare code
                                      (skipped if the text already defines entry_point)
        4. truncate_at_stop_markers — cut obvious trailing junk
        5. rstrip()                 — remove trailing whitespace only
                                      (never strip() — leading spaces = indentation)

    Returns the cleaned completion string, or "" if nothing remains.
    """
    code = strip_markdown_fences(raw_output)
    code = strip_leading_def(code)
    code = ensure_indented(code, entry_point)
    code = truncate_at_stop_markers(code, entry_point)
    return code.rstrip()


def post_process_solution(raw_output: str) -> str:
    """
    Post-process a self-contained solution.

    MBPP+ samples should normally use the `solution` field, so we preserve
    imports, helper functions, and the main function definition. Only obvious
    trailing examples/tests are removed.
    """
    code = strip_markdown_fences(raw_output)
    code = truncate_solution_noise(code)
    return code.strip()


def get_problems(dataset: str) -> dict:
    """Load the selected EvalPlus dataset."""
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def generate_raw_completion(
    client,
    model: str,
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """
    Call the model API and return the raw text output.

    Retries up to MAX_RETRIES times with exponential back-off on rate
    limit errors (HTTP 429 / RESOURCE_EXHAUSTED).

    Returns None if all retries fail so the caller can skip the task.

    Optional overrides (all default to the module-level behaviour when None,
    so existing callers are unaffected). Added for the Prochemy baseline,
    which needs the optimised prompt in the *system* role with the task in
    the user role, and temperature 1.0 for its prompt-mutation step:
        system_prompt : replaces the default "expert Python programmer" system message.
        temperature   : overrides TEMPERATURE (0.2).
        max_tokens    : overrides MAX_TOKENS (512).
    """
    system_content = (
        system_prompt
        if system_prompt is not None
        else (
            "You are an expert Python programmer. "
            "Output only raw Python code — "
            "no explanations, no markdown, no example usage."
        )
    )
    temp = TEMPERATURE if temperature is None else temperature
    max_tok = MAX_TOKENS if max_tokens is None else max_tokens

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=temp,
                max_tokens=max_tok,
            )
            return response.choices[0].message.content or ""

        except Exception as exc:
            err = str(exc)
            is_rate_limit = any(
                token in err
                for token in ["429", "rate_limit", "RESOURCE_EXHAUSTED", "RateLimitError"]
            )
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"    Rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    Error on attempt {attempt + 1}: {exc}")
                if attempt == MAX_RETRIES - 1:
                    return None

    return None

def write_samples_jsonl(path: str, samples: list) -> None:
    """Write samples to a JSONL file in EvalPlus format.

    Each line uses either:
        {"task_id": "HumanEval/N", "completion": "..."}
        {"task_id": "Mbpp/N", "solution": "..."}
    """
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def load_existing_samples(path: str) -> dict:
    """Load an existing samples JSONL file as task_id -> sample."""
    if not os.path.exists(path):
        return {}

    samples = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            task_id = sample.get("task_id")
            if not task_id:
                raise ValueError(f"Missing task_id in {path}:{line_no}")

            samples[task_id] = sample

    return samples


def apply_limit_to_path(path: str, limit: Optional[int]) -> str:
    """Add a smoke-test limit suffix to avoid clobbering full-run artifacts."""
    if limit is None:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}_limit{limit}{ext}"


def run_suffix(feedback_mode: str = "full", run_tag: Optional[str] = None) -> str:
    """Build the artifact-name suffix that keeps run variants from colliding.

    ``feedback_mode`` contributes ``_fb-<mode>`` for every rung except the
    default ``full`` - which is omitted deliberately, so the pre-ablation
    runs keep their historical filenames and the ablation's ``full`` arm can
    reuse them instead of being regenerated (decisions.md D24).
    ``run_tag`` contributes ``_<tag>`` for repeat runs (decisions.md D16).
    """
    suffix = ""
    if feedback_mode and feedback_mode != "full":
        suffix += f"_fb-{feedback_mode}"
    if run_tag:
        suffix += f"_{run_tag}"
    return suffix


def apply_run_suffix_to_path(path: str, suffix: str) -> str:
    """Insert a run suffix before a path's extension."""
    if not suffix:
        return path
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext}"


def get_repair_round_path(
    base_output_path: str,
    repair_strategy: str,
    round_num: int,
) -> str:
    """Build a cumulative per-round repair JSONL path."""
    if round_num == 0:
        return base_output_path
    root, ext = os.path.splitext(base_output_path)
    return f"{root}_repair-{repair_strategy}_round{round_num}{ext}"


def get_repair_trajectory_path(
    model: str,
    prompt_strategy: str,
    dataset: str,
    repair_strategy: str,
    limit: Optional[int] = None,
    suffix: str = "",
) -> str:
    """Build the full trajectory JSON path for repair analysis."""
    slug = model_to_slug(model)
    temp_tag = f"t{int(TEMPERATURE * 10):02d}"
    limit_tag = "" if limit is None else f"_limit{limit}"
    filename = (
        f"repair_trajectories_{slug}_{temp_tag}_{prompt_strategy}_"
        f"{dataset}_{repair_strategy}{suffix}{limit_tag}.json"
    )
    return os.path.join(BASE_DIR, "results", filename)


def load_existing_trajectories(path: str) -> dict:
    """Load existing repair trajectories as task_id -> trajectory."""
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        trajectories = json.load(f)

    return {
        result["task_id"]: result
        for result in trajectories
        if result.get("task_id")
    }


def sample_from_completion(task_id: str, completion: str, dataset: str) -> dict:
    """Build one EvalPlus sample row from a task completion."""
    sample_key = "solution" if dataset == "mbpp" else "completion"
    return {"task_id": task_id, sample_key: completion}


def completion_for_round(result: dict, target_round: int) -> str:
    """Return the latest available completion at or before target_round."""
    rounds = [
        round_entry
        for round_entry in result.get("rounds", [])
        if round_entry.get("round", 0) <= target_round
    ]
    if not rounds:
        raise ValueError(f"No rounds recorded for {result.get('task_id')}")
    return max(rounds, key=lambda round_entry: round_entry["round"])["completion"]


def write_repair_round_jsonls(
    round_paths: dict,
    trajectories_by_id: dict,
    task_ids: list,
    dataset: str,
) -> dict:
    """Write cumulative per-round JSONL files for EvalPlus compatibility."""
    saved_counts = {}
    for round_num, path in round_paths.items():
        samples = []
        for task_id in task_ids:
            result = trajectories_by_id.get(task_id)
            if result is None:
                continue
            completion = completion_for_round(result, round_num)
            samples.append(sample_from_completion(task_id, completion, dataset))

        write_samples_jsonl(path, samples)
        saved_counts[round_num] = len(samples)
        print(f"  Round {round_num} samples -> {path} ({len(samples)} rows)")
    return saved_counts


def solved_by_round(result: dict, round_num: int) -> bool:
    """Return True if a task solved at or before round_num."""
    solved_at = result.get("solved_at_round")
    return solved_at is not None and solved_at <= round_num


def summarize_repair_trajectories(results: list, repair_strategy: str) -> dict:
    """Compute repair-mode summary metrics."""
    from collections import Counter

    total = len(results)
    summary = {
        "total_tasks": total,
        "round_pass_counts": {},
        "round_pass_rates": {},
        "total_improvement_pp": 0.0,
        "stalled_count": sum(1 for result in results if result.get("stalled", False)),
        "round0_error_counts": {},
        "cot_extraction_method_counts": {},
    }

    for round_num in [0, 1, 2]:
        count = sum(1 for result in results if solved_by_round(result, round_num))
        summary["round_pass_counts"][f"round{round_num}"] = count
        summary["round_pass_rates"][f"round{round_num}"] = (
            round(count / total * 100, 1) if total else 0.0
        )

    summary["total_improvement_pp"] = round(
        summary["round_pass_rates"]["round2"]
        - summary["round_pass_rates"]["round0"],
        1,
    )

    error_counts = Counter()
    cot_counts = Counter()
    for result in results:
        if result.get("rounds"):
            r0 = result["rounds"][0]
            if not r0.get("passed", False):
                error_counts[r0.get("error_type")] += 1
        if repair_strategy == "cot":
            for round_entry in result.get("rounds", []):
                method = round_entry.get("cot_extraction_method")
                if round_entry.get("round", 0) > 0 and method:
                    cot_counts[method] += 1

    summary["round0_error_counts"] = dict(error_counts)
    if repair_strategy == "cot":
        summary["cot_extraction_method_counts"] = {
            method: cot_counts[method]
            for method in ["marker", "fence", "raw_fallback"]
        }

    return summary


def print_repair_summary(summary: dict, repair_strategy: str) -> None:
    """Print a compact repair-mode summary."""
    total = summary["total_tasks"]

    def fmt(round_key: str) -> str:
        count = summary["round_pass_counts"][round_key]
        rate = summary["round_pass_rates"][round_key]
        return f"{count}/{total} ({rate:.1f}%)"

    print("\n" + "=" * 60)
    print("REPAIR SUMMARY")
    print("=" * 60)
    print(f"  Total tasks                : {total}")
    print(f"  Round 0 pass@1             : {fmt('round0')}")
    print(f"  Cumulative Round 1 pass@1  : {fmt('round1')}")
    print(f"  Cumulative Round 2 pass@1  : {fmt('round2')}")
    print(f"  Total improvement          : {summary['total_improvement_pp']:+.1f} pp")
    print(f"  Stalled tasks              : {summary['stalled_count']}")

    if summary["round0_error_counts"]:
        print("\n  Round 0 error type breakdown:")
        for err_type, count in summary["round0_error_counts"].items():
            print(f"    {err_type}: {count}")

    if repair_strategy == "cot":
        print("\n  CoT extraction method breakdown:")
        for method, count in summary["cot_extraction_method_counts"].items():
            print(f"    {method}: {count}")


def save_repair_run_log(
    round_paths: dict,
    trajectory_path: str,
    round0_source_jsonl: Optional[str],
    model: str,
    backend: str,
    dataset: str,
    prompt_strategy: str,
    repair_strategy: str,
    feedback_mode: str,
    max_repair_rounds: int,
    total_api_calls: int,
    round0_generation_api_calls: int,
    round0_reused_count: int,
    repair_api_calls: int,
    skipped: list,
    summary: dict,
    limit: Optional[int],
) -> None:
    """Save a repair-mode JSON sidecar recording configuration and summary."""
    log = {
        "date": str(date.today()),
        "backend": backend,
        "dataset": dataset,
        "model": model,
        "prompt_strategy": prompt_strategy,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "max_retries": MAX_RETRIES,
        "repair": True,
        "repair_strategy": repair_strategy,
        "feedback_mode": feedback_mode,
        "max_repair_rounds": max_repair_rounds,
        "limit": limit,
        "round_output_files": {
            f"round{round_num}": path
            for round_num, path in round_paths.items()
        },
        "trajectory_file": trajectory_path,
        "round0_source_jsonl": round0_source_jsonl,
        "total_api_calls": total_api_calls,
        "round0_generation_api_calls": round0_generation_api_calls,
        "round0_reused_count": round0_reused_count,
        "repair_api_calls": repair_api_calls,
        "tasks_completed": summary["total_tasks"],
        "tasks_skipped": len(skipped),
        "skipped_ids": skipped,
        "summary": summary,
    }

    log_path = round_paths[0].replace(".jsonl", "_repair_run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"  Repair run log -> {log_path}")


def save_run_log(
    output_path: str,
    model: str,
    backend: str,
    dataset: str,
    prompt_strategy: str,
    num_saved: int,
    skipped: list,
) -> None:
    """
    Save a JSON sidecar file recording all run configuration.

    After running EvalPlus, manually fill in the four pass@1 fields.
    The log file is named identically to the JSONL but with _run_log.json.
    Example: samples/qwen25-coder_7b_t02_baseline_run_log.json
    """
    log = {
        "date"                     : str(date.today()),
        "backend"                  : backend,
        "dataset"                  : dataset,
        "model"                    : model,
        "prompt_strategy"          : prompt_strategy,
        "temperature"              : TEMPERATURE,
        "max_tokens"               : MAX_TOKENS,
        "max_retries"              : MAX_RETRIES,
        "output_file"              : output_path,
        "tasks_completed"          : num_saved,
        "tasks_skipped"            : len(skipped),
        "skipped_ids"              : skipped,
        "pass_at_1_base_raw"       : None,
        "pass_at_1_plus_raw"       : None,
        "pass_at_1_base_sanitized" : None,
        "pass_at_1_plus_sanitized" : None,
        "robustness_gap_raw"       : None,
        "robustness_gap_sanitized" : None,
    }

    log_path = output_path.replace(".jsonl", "_run_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    print(f"  Run log -> {log_path}")


def load_existing_task_ids(path: str) -> set:
    """Load task IDs that were already generated in an existing output file."""
    if not os.path.exists(path):
        return set()
    
    existing = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    existing.add(data.get("task_id"))
    except Exception as e:
        print(f"Warning: could not read existing file {path}: {e}")
    
    return existing


def run_repair_generation(args: argparse.Namespace) -> None:
    """Run normal Round 0 generation followed by iterative self-repair."""
    from self_repair import run_self_repair

    suffix = run_suffix(args.feedback_mode, args.run_tag)
    base_output_path = apply_run_suffix_to_path(
        apply_limit_to_path(
            get_output_path(args.model, args.prompt, args.dataset),
            args.limit,
        ),
        suffix,
    )
    round_paths = {
        round_num: get_repair_round_path(
            base_output_path,
            args.repair_strategy,
            round_num,
        )
        for round_num in range(args.max_repair_rounds + 1)
    }
    trajectory_path = get_repair_trajectory_path(
        model=args.model,
        prompt_strategy=args.prompt,
        dataset=args.dataset,
        repair_strategy=args.repair_strategy,
        limit=args.limit,
        suffix=suffix,
    )

    os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

    print(f"Backend  : {args.backend}")
    print(f"Dataset  : {args.dataset}")
    print(f"Model    : {args.model}")
    print(f"Strategy : {args.prompt}")
    print(f"Temp     : {TEMPERATURE}")
    print(f"Repair   : {args.repair_strategy}")
    print(f"Feedback : {args.feedback_mode}")
    print(f"Max rds  : {args.max_repair_rounds}")
    print(f"Limit    : {args.limit}")
    print(f"Round 0  : {round_paths[0]}")
    for round_num in range(1, args.max_repair_rounds + 1):
        print(f"Round {round_num}  : {round_paths[round_num]}")
    print(f"Traj     : {trajectory_path}")
    print(f"R0 source: {args.round0_source_jsonl or round_paths[0]}")
    print()

    client = get_client(args.backend)
    problems = get_problems(args.dataset)
    problem_items = list(problems.items())
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        problem_items = problem_items[:args.limit]

    total = len(problem_items)
    existing_trajectories = (
        load_existing_trajectories(trajectory_path)
        if args.resume_missing
        else {}
    )
    round0_source_path = args.round0_source_jsonl or round_paths[0]
    if args.round0_source_jsonl and not os.path.exists(round0_source_path):
        raise FileNotFoundError(f"Round 0 source JSONL not found: {round0_source_path}")
    round0_source_samples = load_existing_samples(round0_source_path)
    sample_key = "solution" if args.dataset == "mbpp" else "completion"

    trajectories_by_id = dict(existing_trajectories)
    skipped = []
    round0_generation_api_calls = 0
    round0_reused_count = 0
    repair_api_calls = 0
    sleep_s = SLEEP.get(args.backend, 1.0)

    if round0_source_samples:
        source_count = sum(
            1 for task_id, _ in problem_items if task_id in round0_source_samples
        )
        print(f"Round 0 source: found {source_count}/{total} selected task completions")
        print()

    if args.resume_missing:
        existing_count = sum(
            1 for task_id, _ in problem_items if task_id in existing_trajectories
        )
        print(f"Resume   : found {existing_count}/{total} existing trajectories")
        print()

    for index, (task_id, task) in enumerate(problem_items, start=1):
        if task_id in existing_trajectories:
            print(f"[{index:>3}/{total}] {task_id} ... RESUME")
            continue

        print(f"[{index:>3}/{total}] {task_id} ...", flush=True)

        source_sample = round0_source_samples.get(task_id)
        if source_sample is not None and sample_key in source_sample:
            completion = source_sample[sample_key]
            round0_reused_count += 1
            print("  Round 0 source: reused existing completion")
        else:
            prompt = build_prompt(task["prompt"], args.prompt, args.dataset)
            raw = generate_raw_completion(client, args.model, prompt)
            round0_generation_api_calls += 1

            if raw is None:
                print("  SKIPPED (Round 0 API error)")
                skipped.append(task_id)
                continue

            completion = (
                post_process_solution(raw)
                if args.dataset == "mbpp"
                else post_process(raw, entry_point=task["entry_point"])
            )
            print("  Round 0 source: generated via API")

        if not completion:
            print("  SKIPPED (empty after Round 0 post-process)")
            skipped.append(task_id)
            continue

        result = run_self_repair(
            task_id=task_id,
            initial_completion=completion,
            dataset=args.dataset,
            client=client,
            model=args.model,
            backend=args.backend,
            max_repair_rounds=args.max_repair_rounds,
            repair_strategy=args.repair_strategy,
            feedback_mode=args.feedback_mode,
            prompt_strategy=args.prompt,
        )
        repair_api_calls += max(0, len(result.get("rounds", [])) - 1)

        if result.get("api_failed"):
            # Do NOT persist: an unattempted repair is not a result. Storing it
            # would make --resume-missing skip the task on the next run (e.g.
            # after swapping to a fresh API key), silently locking in a round
            # JSONL that is just a copy of Round 0.
            print(
                f"[{index:>3}/{total}] {task_id} -> NOT SAVED (API/quota failure; "
                "will be retried by --resume-missing)"
            )
            skipped.append(task_id)
            continue

        trajectories_by_id[task_id] = result

        if result.get("solved"):
            print(f"[{index:>3}/{total}] {task_id} -> solved at round {result['solved_at_round']}")
        else:
            print(
                f"[{index:>3}/{total}] {task_id} -> "
                f"unsolved after {args.max_repair_rounds} rounds, "
                f"stalled={result.get('stalled', False)}"
            )

        ordered_partial = [
            trajectories_by_id[tid]
            for tid, _ in problem_items
            if tid in trajectories_by_id
        ]
        with open(trajectory_path, "w", encoding="utf-8") as f:
            json.dump(ordered_partial, f, indent=2)

        time.sleep(sleep_s)

    task_ids = [task_id for task_id, _ in problem_items]
    ordered_results = [
        trajectories_by_id[task_id]
        for task_id in task_ids
        if task_id in trajectories_by_id
    ]

    with open(trajectory_path, "w", encoding="utf-8") as f:
        json.dump(ordered_results, f, indent=2)
    print(f"\n  Trajectories -> {trajectory_path} ({len(ordered_results)} rows)")

    write_repair_round_jsonls(
        round_paths=round_paths,
        trajectories_by_id=trajectories_by_id,
        task_ids=task_ids,
        dataset=args.dataset,
    )

    if skipped:
        print(f"  Skipped {len(skipped)} tasks: {skipped}")
        print()
        print("  " + "!" * 68)
        print(f"  !! INCOMPLETE RUN - {len(skipped)}/{total} tasks have no trajectory.")
        print("  !! The round JSONLs above are PARTIAL and must NOT be graded or")
        print("  !! compared as if complete. Common cause: API/quota exhaustion.")
        print("  !! Re-run the same command (optionally with a fresh API key) and")
        print("  !! --resume-missing will retry exactly these tasks.")
        print("  " + "!" * 68)

    summary = summarize_repair_trajectories(
        ordered_results,
        args.repair_strategy,
    )
    total_api_calls = round0_generation_api_calls + repair_api_calls
    print_repair_summary(summary, args.repair_strategy)
    print()
    print(f"API calls: total={total_api_calls}, round0_generated={round0_generation_api_calls}, "
          f"round0_reused={round0_reused_count}, repair={repair_api_calls}")
    save_repair_run_log(
        round_paths=round_paths,
        trajectory_path=trajectory_path,
        round0_source_jsonl=round0_source_path,
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        prompt_strategy=args.prompt,
        repair_strategy=args.repair_strategy,
        feedback_mode=args.feedback_mode,
        max_repair_rounds=args.max_repair_rounds,
        total_api_calls=total_api_calls,
        round0_generation_api_calls=round0_generation_api_calls,
        round0_reused_count=round0_reused_count,
        repair_api_calls=repair_api_calls,
        skipped=skipped,
        summary=summary,
        limit=args.limit,
    )


def main() -> None:
    args = parse_args()
    load_env_file()

    if args.repair:
        run_repair_generation(args)
        return

    output_path = get_output_path(args.model, args.prompt, args.dataset)
    os.makedirs(os.path.join(BASE_DIR, "samples"), exist_ok=True)

    print(f"Backend  : {args.backend}")
    print(f"Dataset  : {args.dataset}")
    print(f"Model    : {args.model}")
    print(f"Strategy : {args.prompt}")
    print(f"Temp     : {TEMPERATURE}")
    print(f"Output   : {output_path}")
    print()

    client   = get_client(args.backend)
    problems = get_problems(args.dataset)
    total    = len(problems)

    sample_key = "solution" if args.dataset == "mbpp" else "completion"
    existing_samples = load_existing_samples(output_path) if args.resume_missing else {}
    generated_samples : dict = {}
    skipped : list = []
    sleep_s  = SLEEP.get(args.backend, 1.0)

    if args.resume_missing:
        existing_count = sum(1 for task_id in problems if task_id in existing_samples)
        print(f"Resume   : found {existing_count}/{total} existing samples")
        print()

    for index, (task_id, task) in enumerate(problems.items(), start=1):
        if task_id in existing_samples:
            continue

        print(f"[{index:>3}/{total}] {task_id} ...", end=" ", flush=True)

        prompt     = build_prompt(task["prompt"], args.prompt, args.dataset)
        raw        = generate_raw_completion(client, args.model, prompt)

        if raw is None:
            print("SKIPPED (API error)")
            skipped.append(task_id)
            continue

        completion = (
            post_process_solution(raw)
            if args.dataset == "mbpp"
            else post_process(raw, entry_point=task["entry_point"])
        )

        if not completion:
            print("SKIPPED (empty after post-process)")
            skipped.append(task_id)
            continue

        print("OK")
        generated_samples[task_id] = {"task_id": task_id, sample_key: completion}
        time.sleep(sleep_s)

    samples_by_id = {**existing_samples, **generated_samples}
    samples = []
    for task_id in problems:
        sample = samples_by_id.get(task_id)
        if sample is None:
            if task_id not in skipped:
                skipped.append(task_id)
            continue
        samples.append(sample)

    write_samples_jsonl(output_path, samples)
    print(f"\n  Saved {len(samples)} samples -> {output_path}")

    if skipped:
        print(f"  Skipped {len(skipped)} tasks: {skipped}")

    save_run_log(
        output_path     = output_path,
        model           = args.model,
        backend         = args.backend,
        dataset         = args.dataset,
        prompt_strategy = args.prompt,
        num_saved       = len(samples),
        skipped         = skipped,
    )

    rel_path       = os.path.relpath(output_path, BASE_DIR)
    sanitized_path = output_path.replace(".jsonl", "-sanitized.jsonl")
    rel_sanitized  = os.path.relpath(sanitized_path, BASE_DIR)
    print()
    print("Next steps:")
    print(f"  1. docker run --rm --pull=always -v \"${{PWD}}:/app\" ganler/evalplus:latest evalplus.evaluate --dataset {args.dataset} --samples /app/{rel_path}")
    print(f"  2. evalplus.sanitize --samples {rel_path} --dataset {args.dataset}")
    print(f"  3. docker run --rm --pull=always -v \"${{PWD}}:/app\" ganler/evalplus:latest evalplus.evaluate --dataset {args.dataset} --samples /app/{rel_sanitized}")
    print(f"  4. Fill in pass@1 scores in: {output_path.replace('.jsonl', '_run_log.json')}")


if __name__ == "__main__":
    main()
