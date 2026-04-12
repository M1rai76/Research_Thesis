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

from evalplus.data import get_human_eval_plus


# Do not change these between runs — they are part of your methodology.
TEMPERATURE : float = 0.2
MAX_TOKENS  : int   = 512
MAX_RETRIES : int   = 5

# Ollama is local so 1s is fine. Groq free tier allows ~30 RPM so 2s is safe.
SLEEP = {"ollama": 1.0, "groq": 2.0}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HumanEval completions for thesis experiments."
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
        choices=["ollama", "groq"],
        help="Inference backend.",
    )
    parser.add_argument(
        "--prompt",
        default="baseline",
        help=(
            "Prompt strategy label. Used in the output filename and run log. "
            "Examples: baseline, zero_shot, cot, role_framing, edge_case. "
            "Default: baseline."
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


def get_output_path(model: str, prompt_strategy: str) -> str:
    """
    Build a deterministic output path from model + temperature + strategy.

    Pattern: samples/{model_slug}_t{temp*10:02d}_{strategy}.jsonl
    Examples
        qwen25-coder_7b_t02_baseline.jsonl
        llama31_8b_t02_cot.jsonl
    """
    slug     = model_to_slug(model)
    temp_tag = f"t{int(TEMPERATURE * 10):02d}"
    return os.path.join("samples", f"{slug}_{temp_tag}_{prompt_strategy}.jsonl")


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

    raise ValueError(f"Unknown backend: '{backend}'. Choose 'ollama' or 'groq'.")


# THIS IS YOUR EXPERIMENTAL VARIABLE.
# Change build_prompt() between runs to test different prompt strategies.
# Do not change anything else between strategy runs.
# Current strategy: baseline (minimal instruction prompt)
def build_prompt(task_prompt: str) -> str:
    """
    Wrap the raw HumanEval task prompt in generation instructions.

    task_prompt : the 'prompt' field from get_human_eval_plus(), which
                  contains the function signature and docstring.

    Returns the full string sent to the model.
    """
    return (
        "Complete the following Python function.\n\n"
        "Rules:\n"
        "- Output only the function body (the lines after the def line)\n"
        "- Do not repeat the function signature or docstring\n"
        "- Do not include markdown fences\n"
        "- Do not include example usage or test code\n"
        "- Do not include explanations\n\n"
        f"{task_prompt}"
    )


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


def post_process(raw_output: str) -> str:
    """
    Apply the full post-processing pipeline to raw model output.

    Steps:
        1. strip_markdown_fences    — remove ``` wrappers
        2. truncate_at_stop_markers — cut obvious trailing junk
        3. rstrip()                 — remove trailing whitespace only
                                      (never strip() — leading spaces = indentation)

    Returns the cleaned completion string, or "" if nothing remains.
    """
    code = strip_markdown_fences(raw_output)
    code = truncate_at_stop_markers(code)
    return code.rstrip()


def generate_raw_completion(
    client,
    model: str,
    prompt: str,
    backend: str,
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
    """Write samples to a JSONL file in HumanEval format.

    Each line: {"task_id": "HumanEval/N", "completion": "..."}
    """
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def save_run_log(
    output_path: str,
    model: str,
    backend: str,
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


def main() -> None:
    args = parse_args()

    output_path = get_output_path(args.model, args.prompt)
    os.makedirs("samples", exist_ok=True)

    print(f"Backend  : {args.backend}")
    print(f"Model    : {args.model}")
    print(f"Strategy : {args.prompt}")
    print(f"Temp     : {TEMPERATURE}")
    print(f"Output   : {output_path}")
    print()

    client   = get_client(args.backend)
    problems = get_human_eval_plus()
    total    = len(problems)

    samples : list = []
    skipped : list = []
    sleep_s  = SLEEP.get(args.backend, 1.0)

    for index, (task_id, task) in enumerate(problems.items(), start=1):
        print(f"[{index:>3}/{total}] {task_id} ...", end=" ", flush=True)

        prompt     = build_prompt(task["prompt"])
        raw        = generate_raw_completion(client, args.model, prompt, args.backend)

        if raw is None:
            print("SKIPPED (API error)")
            skipped.append(task_id)
            continue

        completion = post_process(raw)

        if not completion:
            print("SKIPPED (empty after post-process)")
            skipped.append(task_id)
            continue

        print("OK")
        samples.append({"task_id": task_id, "completion": completion})
        time.sleep(sleep_s)

    write_samples_jsonl(output_path, samples)
    print(f"\n  Saved {len(samples)} samples -> {output_path}")

    if skipped:
        print(f"  Skipped {len(skipped)} tasks: {skipped}")

    save_run_log(
        output_path     = output_path,
        model           = args.model,
        backend         = args.backend,
        prompt_strategy = args.prompt,
        num_saved       = len(samples),
        skipped         = skipped,
    )

    sanitized_path = output_path.replace(".jsonl", "-sanitized.jsonl")
    print()
    print("Next steps:")
    print(f"  1. docker run --rm --pull=always -v \"${{PWD}}:/app\" ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/{output_path}")
    print(f"  2. evalplus.sanitize --samples {output_path} --dataset humaneval")
    print(f"  3. docker run --rm --pull=always -v \"${{PWD}}:/app\" ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/{sanitized_path}")
    print(f"  4. Fill in pass@1 scores in: {output_path.replace('.jsonl', '_run_log.json')}")


if __name__ == "__main__":
    main()
