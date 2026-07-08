# Enhancing Code Generation via Prompt Engineering and Iterative Self-Repair

Thesis project evaluating the effect of prompt strategies and iterative self-repair on LLM code generation quality, benchmarked on HumanEval+ and MBPP+ using EvalPlus. Model: Llama 3.3 70B via Groq.

---

## Requirements

- Python 3.10 or newer
- Docker Desktop (running, for EvalPlus evaluation)
- Groq API key

---

## Installation

```bash
pip install evalplus openai groq bandit
```

---

## Supported Prompt Strategies (Seed Generation)

| Strategy | Description |
|----------|-------------|
| `cop` | Chain of Objectives Prompting — best overall strategy from Thesis B |
| `baseline` | Minimal instruction prompt |
| `zero_shot` | Bare instruction with no rules |
| `cot` | Chain-of-thought reasoning before code |
| `edge_case` | Instructs model to handle edge cases |
| `role_framing` | Frames model as a senior engineer |
| `io_spec` | Input/output specification prompt |

## Supported Repair Strategies (Iterative Self-Repair)

| Strategy | Description |
|----------|-------------|
| `cot` | Includes task description, broken code, error, and three-step reasoning before fix. Final code placed after `### Fixed Code` marker. |
| `minimal` | Broken code and error only. No reasoning instruction. |

---

## Usage

Run all commands from the `project/` directory.

### Step 1 — Seed Generation (Round 0)

```bash
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset humaneval

python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset mbpp
```

### Step 2 — Iterative Self-Repair (Rounds 1 and 2)

CoT repair, HumanEval:
```bash
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset humaneval --repair --repair_strategy cot --max_repair_rounds 2 --round0_source_jsonl samples/llama-33-70b-versatile_t02_cgo.jsonl --resume-missing
```

CoT repair, MBPP:
```bash
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset mbpp --repair --repair_strategy cot --max_repair_rounds 2 --round0_source_jsonl samples/mbpp_llama-33-70b-versatile_t02_cgo.jsonl --resume-missing
```

Minimal repair, HumanEval:
```bash
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset humaneval --repair --repair_strategy minimal --max_repair_rounds 2 --round0_source_jsonl samples/llama-33-70b-versatile_t02_cgo.jsonl --resume-missing
```

Minimal repair, MBPP:
```bash
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt cop --dataset mbpp --repair --repair_strategy minimal --max_repair_rounds 2 --round0_source_jsonl samples/mbpp_llama-33-70b-versatile_t02_cgo.jsonl --resume-missing
```

To run a single task through the repair loop for debugging:
```bash
python scripts/self_repair.py --task_id "HumanEval/4" --dataset humaneval --jsonl_path samples/llama-33-70b-versatile_t02_cgo.jsonl --backend groq --model llama-3.3-70b-versatile --max_rounds 2 --repair_strategy cot
```

If a run is cut off by Groq rate limits, re-run the exact same command. The `--resume-missing` flag skips already-completed tasks and continues from where it stopped.

### Step 3 — EvalPlus Evaluation

```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/samples/llama-33-70b-versatile_t02_cop.jsonl

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/samples/llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest evalplus.evaluate --dataset humaneval --samples /app/samples/llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl
```

Replace `humaneval` with `mbpp` and adjust filenames for MBPP. Replace `repair-cot` with `repair-minimal` for minimal strategy files.

### Step 4 — Safety Sweep

```bash
python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop.jsonl --dataset mbpp

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl --dataset mbpp

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl --dataset mbpp
```

---

## Key Results

CoT repair outperforms Minimal repair on correctness by more than 2x on both benchmarks, with no robustness cost. Robustness ratio degradation under repair is structural to single-assertion feedback loops regardless of repair strategy, confirmed consistently across both datasets.

| Metric | HumanEval CoT | HumanEval Minimal | MBPP CoT | MBPP Minimal |
|--------|--------------|-------------------|----------|--------------|
| R0 Base pass@1 | 0.805 | 0.805 | 0.865 | 0.865 |
| R2 Base pass@1 | 0.915 | 0.854 | 0.921 | 0.897 |
| Total gain | +11.0pp | +4.9pp | +5.6pp | +3.7pp |
| R0 Robustness Ratio | 0.917 | 0.917 | 0.838 | 0.838 |
| R2 Robustness Ratio | 0.906 | 0.906 | 0.836 | 0.832 |
| Safety Rate (all rounds) | ≥98.8% | — | 100.0% | — |

---

## Project Structure
project/
scripts/
generate_samples.py           — seed generation + repair mode (--repair flag)
self_repair.py                — N-round repair loop orchestrator (single task)
code_executor.py              — sandboxed execution + error extraction
repair_prompt.py              — minimal and CoT repair prompt builders
safety_check.py               — Bandit static analysis wrapper
verify_safety.py              — safety sweep across a full JSONL file
verify_executor.py            — HumanEval baseline execution sweep
verify_executor_mbpp.py       — MBPP baseline execution sweep
samples/
<model>t02<strategy>.jsonl
mbpp_<model>t02<strategy>.jsonl
<model>t02<strategy>repair-<repair_strategy>round<N>.jsonl
<model>t02<strategy>repair_run_log.json
results/
repair_trajectories<model>t02<strategy><dataset><repair_strategy>.json
README.md

---

## Notes

- All runs use temperature 0.2 and max_tokens 512 for reproducibility.
- Groq free tier limit is ~100k tokens/day. Repair mode uses far fewer tokens than seed generation since only failing tasks receive repair calls.
- Raw EvalPlus evaluation is the primary metric. Sanitized evaluation is secondary.
- Robustness Ratio = Plus pass@1 / Base pass@1. Robustness Gap = Base pass@1 minus Plus pass@1.
