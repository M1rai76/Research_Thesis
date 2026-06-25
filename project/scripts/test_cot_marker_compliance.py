"""Diagnostic: measure CoT repair marker compliance on MBPP failures.

This intentionally does not modify the repair pipeline. It scans the existing
MBPP CGO samples for the first 8 real Round 0 failures, then runs exactly one
CoT self-repair round per failing task and reports whether the model followed
the expected "### Fixed Code" marker format.
"""

import inspect
import json
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import self_repair
from code_executor import run_executor
from generate_samples import ENV_PATH, get_client, load_env_file


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSONL_PATH = os.path.join(
    BASE_DIR,
    "samples",
    "mbpp_llama-33-70b-versatile_t02_cgo.jsonl",
)
DATASET = "mbpp"
BACKEND = "groq"
MODEL = "llama-3.3-70b-versatile"
FAILURE_LIMIT = 8
FIXED_CODE_MARKER = "### Fixed Code"


def load_samples(jsonl_path: str) -> list[dict[str, Any]]:
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "task_id" not in sample:
                raise KeyError(f"Line {line_num} has no task_id")
            samples.append(sample)
    return samples


def sample_completion(sample: dict[str, Any]) -> str:
    if "solution" in sample:
        return sample["solution"]
    if "completion" in sample:
        return sample["completion"]
    raise KeyError(
        f"Sample {sample.get('task_id')} has neither 'solution' nor 'completion'"
    )


def find_first_round0_failures(
    samples: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    failures = []

    print("=" * 80)
    print(f"Scanning {JSONL_PATH}")
    print(f"Looking for first {limit} Round 0 failures via run_executor()")
    print("=" * 80)

    for sample in samples:
        task_id = sample["task_id"]
        completion = sample_completion(sample)
        result = run_executor(task_id=task_id, completion=completion, dataset=DATASET)

        if result["passed"]:
            print(f"  {task_id}: PASS")
            continue

        failures.append(
            {
                "task_id": task_id,
                "completion": completion,
                "round0_result": result,
            }
        )
        print(
            f"  {task_id}: FAIL "
            f"({result.get('error_type')}: {result.get('error_message')})"
        )

        if len(failures) >= limit:
            break

    if len(failures) < limit:
        raise RuntimeError(
            f"Only found {len(failures)} Round 0 failures; expected {limit}"
        )

    print()
    print("Selected failing tasks:")
    for index, failure in enumerate(failures, start=1):
        result = failure["round0_result"]
        print(
            f"  {index}. {failure['task_id']} "
            f"({result.get('error_type')}: {result.get('error_message')})"
        )
    print()

    return failures


def round1_entry(repair_result: dict[str, Any]) -> Optional[dict[str, Any]]:
    for round_entry in repair_result.get("rounds", []):
        if round_entry.get("round") == 1:
            return round_entry
    return None


def first_present(mapping: Optional[dict[str, Any]], keys: list[str]) -> Any:
    if not mapping:
        return None
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def extract_fallback_flag(
    repair_result: dict[str, Any],
    round_entry: Optional[dict[str, Any]],
    raw_response: Optional[str],
) -> tuple[bool, str]:
    value = first_present(
        round_entry,
        [
            "cot_fallback_used",
            "fallback_used",
            "marker_fallback_used",
        ],
    )
    if value is None:
        value = first_present(
            repair_result,
            [
                "cot_fallback_used",
                "fallback_used",
                "marker_fallback_used",
            ],
        )
    if value is not None:
        return bool(value), "self_repair result"

    # Fallback for older result schemas: infer marker compliance directly from
    # the captured raw response. The exact self_repair field is preferred above.
    marker_followed = bool(
        raw_response
        and FIXED_CODE_MARKER in raw_response
        and raw_response.split(FIXED_CODE_MARKER, 1)[1].strip()
    )
    return (not marker_followed), "raw marker inference"


def extract_raw_response(
    round_entry: Optional[dict[str, Any]],
    captured_raw_responses: list[Optional[str]],
) -> Optional[str]:
    raw = first_present(
        round_entry,
        [
            "raw_llm_response",
            "raw_response",
            "llm_raw_response",
            "model_response",
        ],
    )
    if raw is not None:
        return raw
    if captured_raw_responses:
        return captured_raw_responses[-1]
    return None


def print_raw_response(task_id: str, raw_response: Optional[str]) -> None:
    print(f"\n--- RAW LLM RESPONSE for {task_id} ---")
    if raw_response is None:
        print("<no raw LLM response was available>")
    else:
        print(raw_response)
    print(f"--- END RAW LLM RESPONSE for {task_id} ---\n")


def ensure_cot_signature() -> None:
    signature = inspect.signature(self_repair.run_self_repair)
    if "repair_strategy" not in signature.parameters:
        raise RuntimeError(
            "self_repair.run_self_repair() in this checkout does not expose a "
            "'repair_strategy' parameter, so this script cannot run the CoT "
            "marker-compliance diagnostic without modifying self_repair.py."
        )


def run_one_round_cot_repairs(failures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ensure_cot_signature()

    load_env_file(ENV_PATH)
    client = get_client(BACKEND)

    original_generate_raw_completion = self_repair.generate_raw_completion
    records = []

    print("=" * 80)
    print(
        f"Running run_self_repair(..., repair_strategy='cot', "
        f"max_repair_rounds=1) via {BACKEND}/{MODEL}"
    )
    print("=" * 80)

    for index, failure in enumerate(failures, start=1):
        task_id = failure["task_id"]
        captured_raw_responses: list[Optional[str]] = []

        def capture_generate_raw_completion(client_arg, model_arg, prompt_arg):
            raw = original_generate_raw_completion(client_arg, model_arg, prompt_arg)
            captured_raw_responses.append(raw)
            return raw

        self_repair.generate_raw_completion = capture_generate_raw_completion
        try:
            print(f"\n[{index}/{len(failures)}] {task_id}")
            repair_result = self_repair.run_self_repair(
                task_id=task_id,
                initial_completion=failure["completion"],
                dataset=DATASET,
                client=client,
                model=MODEL,
                backend=BACKEND,
                repair_strategy="cot",
                max_repair_rounds=1,
            )
        finally:
            self_repair.generate_raw_completion = original_generate_raw_completion

        r1 = round1_entry(repair_result)
        raw_response = extract_raw_response(r1, captured_raw_responses)
        fallback_used, fallback_source = extract_fallback_flag(
            repair_result,
            r1,
            raw_response,
        )

        record = {
            "task_id": task_id,
            "cot_fallback_used": fallback_used,
            "fallback_source": fallback_source,
            "raw_response": raw_response,
        }
        records.append(record)

        print(f"  cot_fallback_used: {fallback_used} ({fallback_source})")
        if fallback_used:
            print_raw_response(task_id, raw_response)

    return records


def print_summary(records: list[dict[str, Any]]) -> None:
    fallback_records = [record for record in records if record["cot_fallback_used"]]
    fallback_count = len(fallback_records)
    followed_count = len(records) - fallback_count

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{followed_count}/{len(records)} tasks had the marker correctly followed")
    print(f"{fallback_count}/{len(records)} tasks triggered the fallback")

    if fallback_count >= 2:
        print()
        print("=" * 80)
        print("FALLBACK RAW RESPONSE APPENDIX")
        print("=" * 80)
        for record in fallback_records:
            print_raw_response(record["task_id"], record["raw_response"])


def main() -> int:
    samples = load_samples(JSONL_PATH)
    failures = find_first_round0_failures(samples, FAILURE_LIMIT)
    records = run_one_round_cot_repairs(failures)
    print_summary(records)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print()
        print("=" * 80)
        print("DIAGNOSTIC FAILED")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1)
