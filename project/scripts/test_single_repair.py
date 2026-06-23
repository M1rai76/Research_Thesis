"""One-off end-to-end test: single repair round for HumanEval/40."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evalplus.data import get_human_eval_plus
from code_executor import run_executor
from repair_prompt import build_repair_prompt, build_repair_context
from generate_samples import (
    load_env_file,
    get_client,
    post_process,
    ENV_PATH,
    TEMPERATURE,
    MAX_TOKENS,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(BASE_DIR, "samples", "llama-33-70b-versatile_t02_cgo.jsonl")
TASK_ID = "HumanEval/4"
DATASET = "humaneval"
MODEL = "llama-3.3-70b-versatile"
BACKEND = "groq"


def load_completion(jsonl_path: str, task_id: str) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("task_id") == task_id:
                return sample["completion"]
    raise KeyError(f"{task_id} not found in {jsonl_path}")


def main() -> None:
    load_env_file(ENV_PATH)

    # --- Round 0 ---
    print("=" * 60)
    print("ROUND 0: Execute original completion")
    print("=" * 60)

    r0_completion = load_completion(JSONL_PATH, TASK_ID)
    r0_result = run_executor(task_id=TASK_ID, completion=r0_completion, dataset=DATASET)

    print(f"  task_id       : {r0_result['task_id']}")
    print(f"  passed        : {r0_result['passed']}")
    print(f"  error_type    : {r0_result['error_type']}")
    print(f"  error_message : {r0_result['error_message']}")

    # --- Build repair prompt ---
    print(f"\n{'=' * 60}")
    print("REPAIR PROMPT (sent to LLM)")
    print("=" * 60)

    problems = get_human_eval_plus()
    task = problems[TASK_ID]
    task_prompt = task["prompt"]

    context = build_repair_context(r0_result, DATASET)
    repair_prompt_text = build_repair_prompt(
        task_prompt=task_prompt,
        broken_completion=r0_completion,
        error_type=context["error_type"],
        error_message=context["error_message"],
        dataset=DATASET,
    )

    print(repair_prompt_text)

    # --- Call LLM ---
    print(f"\n{'=' * 60}")
    print(f"CALLING LLM: {MODEL} via {BACKEND}")
    print("=" * 60)

    client = get_client(BACKEND)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert Python programmer. "
                    "Output only raw Python code — "
                    "no explanations, no markdown, no example usage."
                ),
            },
            {"role": "user", "content": repair_prompt_text},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    raw_response = response.choices[0].message.content or ""

    print(f"\n--- Raw model response ---")
    print(raw_response)

    # --- Post-process ---
    r1_completion = post_process(raw_response, entry_point=task["entry_point"])

    print(f"\n--- Post-processed Round 1 completion ---")
    print(r1_completion)

    # --- Round 1: Execute repaired completion ---
    print(f"\n{'=' * 60}")
    print("ROUND 1: Execute repaired completion")
    print("=" * 60)

    r1_result = run_executor(task_id=TASK_ID, completion=r1_completion, dataset=DATASET)

    print(f"  task_id       : {r1_result['task_id']}")
    print(f"  passed        : {r1_result['passed']}")
    print(f"  error_type    : {r1_result['error_type']}")
    print(f"  error_message : {r1_result['error_message']}")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"  Round 0: {'PASS' if r0_result['passed'] else 'FAIL'} "
          f"({r0_result['error_type']}: {r0_result['error_message']})")
    print(f"  Round 1: {'PASS' if r1_result['passed'] else 'FAIL'} "
          f"({r1_result['error_type']}: {r1_result['error_message']})")
    print(f"  Repair {'SUCCEEDED' if r1_result['passed'] else 'FAILED'}")


if __name__ == "__main__":
    main()
