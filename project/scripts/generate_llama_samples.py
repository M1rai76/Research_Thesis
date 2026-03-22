import json
import os
import re
import time
import random
from typing import Dict, Any, Literal

from evalplus.data import get_human_eval_plus

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BACKEND: Literal["groq", "ollama"] = "ollama"  # switch b/w "groq" or "ollama"

GROQ_MODEL  = "llama-3.3-70b-versatile"   # or "llama3-8b-8192", "mixtral-8x7b-32768"
OLLAMA_MODEL = "qwen2.5-coder:7b"         # or "llama3.1:8b"

OUTPUT_PATH = "samples/gemini_samples.jsonl"  # keeping same name for eval compatibility

# Rate limiting
SLEEP_SECONDS = 1.0
MAX_RETRIES   = 5
# ─────────────────────────────────────────────


def get_client():
    if BACKEND == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set. Get one free at console.groq.com")
        return Groq(api_key=api_key)

    elif BACKEND == "ollama":
        from openai import OpenAI
        return OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )

    else:
        raise ValueError(f"Unknown backend: {BACKEND}")


def get_model_name() -> str:
    return OLLAMA_MODEL if BACKEND == "ollama" else GROQ_MODEL


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
    fence_match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1).rstrip()

    return text.rstrip()

def generate_completion(client, prompt: str) -> str:
    """Works for both Groq and Ollama since both use OpenAI-compatible chat API."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=get_model_name(),
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Python programmer. "
                            "Output only raw Python code — no explanations, no markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,   # low temp = more deterministic, better for benchmarks
                max_tokens=512,
            )

            text = response.choices[0].message.content or ""
            return extract_code(text)

        except Exception as exc:
            err = str(exc)
            is_rate_limit = any(x in err for x in ["429", "rate_limit", "RESOURCE_EXHAUSTED"])

            if is_rate_limit and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"  Error on attempt {attempt + 1}: {exc}")
                if attempt == MAX_RETRIES - 1:
                    return ""

    return ""


def main() -> None:
    os.makedirs("samples", exist_ok=True)

    print(f"Backend : {BACKEND}")
    print(f"Model   : {get_model_name()}")
    print(f"Output  : {OUTPUT_PATH}\n")

    client = get_client()
    problems: Dict[str, Dict[str, Any]] = get_human_eval_plus()

    samples = []
    skipped = []
    total = len(problems)

    for index, (task_id, task) in enumerate(problems.items(), start=1):
        print(f"[{index}/{total}] Generating for {task_id}...", end=" ", flush=True)

        prompt = build_prompt(task["prompt"])
        completion = generate_completion(client, prompt)

        if not completion:
            print("SKIPPED (empty)")
            skipped.append(task_id)
            continue

        print("OK")
        samples.append({
            "task_id": task_id,
            "completion": completion,
        })

        time.sleep(SLEEP_SECONDS)

    # Save results
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row) + "\n")

    print(f"\n✓ Saved {len(samples)} samples to {OUTPUT_PATH}")
    if skipped:
        print(f"✗ Skipped {len(skipped)} tasks: {skipped}")


if __name__ == "__main__":
    main()