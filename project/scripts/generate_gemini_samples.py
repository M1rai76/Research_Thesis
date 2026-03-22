import json
import os
import random
import re
import time
from typing import Any, Dict

from google import genai
from evalplus.data import get_human_eval_plus

MODEL_NAME = "gemini-2.0-flash"
SLEEP_SECONDS = 13.0

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_PATH = os.path.join(PROJECT_DIR, "samples", "gemini_samples.jsonl")


def build_prompt(task_prompt: str) -> str:
    return f"""Complete the following Python code.

Rules:
- Continue the code from where it stops
- Do not repeat the function signature
- Do not repeat the docstring
- Do not include markdown fences
- Do not include explanations
- Output only the remaining Python code

{task_prompt}
"""


def extract_code(text: str) -> str:
    text = text.strip()

    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return text


def generate_completion_with_retry(
    client: genai.Client,
    prompt: str,
    max_retries: int = 5,
) -> str:
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            if not response.text:
                return ""
            return extract_code(response.text)

        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"  Rate limited. Waiting {wait:.1f}s before retry {attempt+1}/{max_retries}..."
                )
                time.sleep(wait)
                continue

            print(f"  Non-rate-limit error: {exc}")
            return ""

    print("  Max retries exceeded, skipping.")
    return ""


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your environment.")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    client = genai.Client(api_key=api_key)

    problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()

    samples = []
    total = len(problems)

    for index, (task_id, task) in enumerate(problems.items(), start=1):
        print(f"[{index}/{total}] Generating for {task_id}...")

        prompt = build_prompt(task["prompt"])
        completion = generate_completion_with_retry(client, prompt)

        if not completion:
            print(f"  Warning: empty completion for {task_id}, skipping...")
            continue

        samples.append(
            {
                "task_id": task_id,
                "completion": completion,
            }
        )

        time.sleep(SLEEP_SECONDS)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()