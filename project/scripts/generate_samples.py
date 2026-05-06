"""
generate_samples.py
Thesis — Prompt Engineering for LLM Code Generation
Author : Samyak Diwan (z5611048)

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

SLEEP = {"ollama": 1.0, "groq": 2.0, "cerebras": 2.0, "openrouter": 6.0, "gemini": 5.0}

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
        choices=["ollama", "groq", "cerebras", "openrouter","gemini"],
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


def truncate_at_stop_markers(text: str) -> str:
    """Cut off content after the first top-level stop marker."""
    earliest = len(text)
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]

def strip_leading_def(text: str) -> str:
    """Remove a leading 'def ...:' line if the model echoed the signature back."""
    lines = text.splitlines(keepends=True)
    if lines and re.match(r"^def\s+\w+", lines[0]):
        lines = lines[1:]
    return "".join(lines)


def ensure_indented(text: str) -> str:
    """Add 4-space indent to all lines if the completion was returned unindented.

    HumanEval evaluation appends the completion directly after the docstring,
    so the body must be indented or the code runs at module level.
    """
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


def post_process(raw_output: str) -> str:
    """
    Apply the full post-processing pipeline to raw model output.

    Steps:
        1. strip_markdown_fences    — remove ``` wrappers
        2. strip_leading_def        — drop echoed function signature
        3. ensure_indented          — add 4-space indent if model returned bare code
        4. truncate_at_stop_markers — cut obvious trailing junk
        5. rstrip()                 — remove trailing whitespace only
                                      (never strip() — leading spaces = indentation)

    Returns the cleaned completion string, or "" if nothing remains.
    """
    code = strip_markdown_fences(raw_output)
    code = strip_leading_def(code)
    code = ensure_indented(code)
    code = truncate_at_stop_markers(code)
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
) -> Optional[str]:
    """
    Call the model API and return the raw text output.

    Retries up to MAX_RETRIES times with exponential back-off on rate
    limit errors (HTTP 429 / RESOURCE_EXHAUSTED).

    Returns None if all retries fail so the caller can skip the task.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Python programmer. "
                            "Output only raw Python code — "
                            "no explanations, no markdown, no example usage."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
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


def main() -> None:
    args = parse_args()
    load_env_file()

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
            else post_process(raw)
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
