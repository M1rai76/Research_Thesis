"""
llm_prompt_refine.py

Use an LLM to analyze an EvalPlus failure report and propose general prompt
refinements. This script consumes the JSON produced by analyze_failures.py.

Usage:
    python scripts/llm_prompt_refine.py \
      --failure-report samples/mbpp_llama-33-70b-versatile_t02_cot_failures.json \
      --backend groq \
      --model llama-3.3-70b-versatile \
      --prompt-name cot

Use --dry-run to write the diagnostic prompt without calling the LLM.
"""

import argparse
import json
import os
from datetime import datetime
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "experiments", "prompt_iterations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask an LLM to diagnose EvalPlus failures and refine a prompt."
    )
    parser.add_argument(
        "--failure-report",
        required=True,
        help="Path to the JSON report produced by analyze_failures.py.",
    )
    parser.add_argument(
        "--backend",
        default="groq",
        choices=["groq", "cerebras", "openrouter", "gemini"],
        help="LLM backend to use for diagnostic analysis.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name for the diagnostic/refinement LLM.",
    )
    parser.add_argument(
        "--prompt-name",
        required=True,
        help="Name of the prompt being refined, e.g. cgo_v1.",
    )
    parser.add_argument(
        "--current-prompt-file",
        default=None,
        help="Optional file containing the current prompt template text.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output markdown path. Defaults to experiments/prompt_iterations/...",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=40,
        help="Maximum failed tasks to include in the LLM context.",
    )
    parser.add_argument(
        "--max-code-chars",
        type=int,
        default=900,
        help="Maximum characters of generated solution to include per failure.",
    )
    parser.add_argument(
        "--max-test-chars",
        type=int,
        default=500,
        help="Maximum characters of failed test data to include per failure type.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Diagnostic LLM temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum output tokens for the diagnostic LLM.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the constructed diagnostic prompt without calling the LLM.",
    )
    return parser.parse_args()


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def load_env_file(path: str = ENV_PATH) -> None:
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


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(path: str | None) -> str | None:
    if not path:
        return None
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        return f.read().strip()


def truncate(text: Any, limit: int) -> str:
    rendered = json.dumps(text, ensure_ascii=True, indent=2) if not isinstance(text, str) else text
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit].rstrip() + "\n...[truncated]"


def failure_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    order = {
        "base_and_plus": 0,
        "base_only": 1,
        "plus_only": 2,
        "pass": 3,
    }
    return (order.get(record.get("failure_stage"), 9), record.get("task_id", ""))


def compact_failure(
    record: dict[str, Any],
    max_code_chars: int,
    max_test_chars: int,
) -> dict[str, Any]:
    return {
        "task_id": record.get("task_id"),
        "failure_stage": record.get("failure_stage"),
        "base_status": record.get("base_status"),
        "plus_status": record.get("plus_status"),
        "problem_prompt": truncate(record.get("prompt", ""), 900),
        "generated_solution": truncate(
            record.get("generated_solution", ""),
            max_code_chars,
        ),
        "base_fail_tests": truncate(record.get("base_fail_tests", []), max_test_chars),
        "plus_fail_tests": truncate(record.get("plus_fail_tests", []), max_test_chars),
    }


def select_failures(report: dict[str, Any], max_failures: int) -> list[dict[str, Any]]:
    failures = list(report.get("failures", []))
    failures.sort(key=failure_sort_key)
    return failures[:max_failures]


def build_diagnostic_prompt(
    report: dict[str, Any],
    prompt_name: str,
    current_prompt: str | None,
    max_failures: int,
    max_code_chars: int,
    max_test_chars: int,
) -> str:
    selected = [
        compact_failure(record, max_code_chars, max_test_chars)
        for record in select_failures(report, max_failures)
    ]

    summary = report.get("summary", {})
    current_prompt_block = (
        current_prompt
        if current_prompt
        else "Not provided. Infer likely weaknesses only from failure patterns."
    )

    context = {
        "dataset": report.get("dataset"),
        "prompt_name": prompt_name,
        "summary": summary,
        "failure_sample_count": len(selected),
        "failure_samples": selected,
    }

    return (
        "You are helping refine a prompt for an LLM code-generation benchmark.\n\n"
        "Use EvalPlus results as the source of truth. Your job is diagnostic: "
        "classify failure patterns and propose general prompt improvements.\n\n"
        "Important rules:\n"
        "- Do not propose task-specific fixes.\n"
        "- Do not mention task IDs in the revised prompt.\n"
        "- Do not encode benchmark answers or special-case examples.\n"
        "- Prefer concise, general prompt changes that could improve unseen tasks.\n"
        "- Preserve strict output-format requirements.\n"
        "- Separate observations from recommended prompt text.\n\n"
        f"Current prompt name: {prompt_name}\n\n"
        "Current prompt text:\n"
        "```text\n"
        f"{current_prompt_block}\n"
        "```\n\n"
        "Failure report context:\n"
        "```json\n"
        f"{json.dumps(context, indent=2, ensure_ascii=True)}\n"
        "```\n\n"
        "Return a markdown report with exactly these sections:\n"
        "1. Executive Summary\n"
        "2. Failure Pattern Counts\n"
        "3. Prompt Weaknesses\n"
        "4. Recommended General Changes\n"
        "5. Revised Prompt Candidate\n"
        "6. Risks And Anti-Overfitting Notes\n"
    )


def get_client(backend: str):
    if backend == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        return Groq(api_key=api_key)

    if backend == "cerebras":
        from openai import OpenAI

        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            raise RuntimeError("CEREBRAS_API_KEY is not set.")
        return OpenAI(base_url="https://api.cerebras.ai/v1", api_key=api_key)

    if backend == "openrouter":
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    if backend == "gemini":
        from openai import OpenAI

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        return OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=api_key,
        )

    raise ValueError(f"Unsupported backend: {backend}")


def call_llm(
    backend: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    client = get_client(backend)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a rigorous prompt-engineering researcher. "
                    "You analyze benchmark failures and propose general, "
                    "non-overfit prompt refinements."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def default_output_path(failure_report_path: str, prompt_name: str, dry_run: bool) -> str:
    stem = os.path.splitext(os.path.basename(failure_report_path))[0]
    suffix = "diagnostic_prompt" if dry_run else "refinement"
    filename = f"{stem}_{prompt_name}_{suffix}.md"
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def write_markdown(
    output_path: str,
    args: argparse.Namespace,
    failure_report_path: str,
    diagnostic_prompt: str,
    llm_response: str | None,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mode = "dry_run" if args.dry_run else "llm_refinement"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Prompt Refinement: {args.prompt_name}\n\n")
        f.write(f"- Mode: `{mode}`\n")
        f.write(f"- Created: `{datetime.now().isoformat(timespec='seconds')}`\n")
        f.write(f"- Failure report: `{failure_report_path}`\n")
        f.write(f"- Backend: `{args.backend}`\n")
        f.write(f"- Model: `{args.model}`\n")
        f.write(f"- Max failures included: `{args.max_failures}`\n\n")

        if llm_response is not None:
            f.write("## LLM Refinement Report\n\n")
            f.write(llm_response.strip())
            f.write("\n\n")

        f.write("## Diagnostic Prompt\n\n")
        f.write("```text\n")
        f.write(diagnostic_prompt)
        f.write("\n```\n")


def main() -> None:
    args = parse_args()
    load_env_file()

    failure_report_path = resolve_path(args.failure_report)
    failure_report = load_json(failure_report_path)
    current_prompt = load_text(args.current_prompt_file)

    diagnostic_prompt = build_diagnostic_prompt(
        report=failure_report,
        prompt_name=args.prompt_name,
        current_prompt=current_prompt,
        max_failures=args.max_failures,
        max_code_chars=args.max_code_chars,
        max_test_chars=args.max_test_chars,
    )

    output_path = (
        resolve_path(args.output)
        if args.output
        else default_output_path(failure_report_path, args.prompt_name, args.dry_run)
    )

    llm_response = None
    if not args.dry_run:
        llm_response = call_llm(
            backend=args.backend,
            model=args.model,
            prompt=diagnostic_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    write_markdown(
        output_path=output_path,
        args=args,
        failure_report_path=failure_report_path,
        diagnostic_prompt=diagnostic_prompt,
        llm_response=llm_response,
    )

    print(f"Wrote prompt refinement report -> {output_path}")
    if args.dry_run:
        print("Dry run only: no LLM call was made.")


if __name__ == "__main__":
    main()
