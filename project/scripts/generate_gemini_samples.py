import json
import os
import re
import time
from typing import Dict, Any

from google import genai
from evalplus.data import get_human_eval_plus


MODEL_NAME = "gemini-2.5-flash"
OUTPUT_PATH = "samples/gemini_samples.jsonl"
SLEEP_SECONDS = 1.0


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

    # Remove ```python ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    return text


def generate_completion(client: genai.Client, prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        return ""

    return extract_code(response.text)


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in your environment.")

    os.makedirs("samples", exist_ok=True)

    client = genai.Client()
    problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()

    samples = []
    total = len(problems)

    for index, (task_id, task) in enumerate(problems.items(), start=1):
        print(f"[{index}/{total}] Generating for {task_id}...")

        prompt = build_prompt(task["prompt"])

        try:
            completion = generate_completion(client, prompt)
        except Exception as exc:
            print(f"Failed on {task_id}: {exc}")
            completion = ""

        samples.append({
            "task_id": task_id,
            "completion": completion,
        })

        time.sleep(SLEEP_SECONDS)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(samples)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()