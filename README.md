# Enhancing code generation via Prompt Engineering

Thesis project evaluating the effect of prompt strategies on code generation quality across multiple LLMs, benchmarked on **HumanEval+** and **MBPP+** using **EvalPlus**.

---

## Requirements

- Python 3.10 or newer
- Docker Desktop (running)
- Ollama (for local models)
- Groq or Cerebras API key (for cloud models)

---

## Installation

A dedicated venv lives at the repo root (`.venv/`), pinned via `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers `evalplus`, `openai`, `groq`, `google-genai`, `matplotlib`, `numpy` — everything imported across `scripts/` and `robustness/`.

---

## Supported Models

| Model | Backend | Size |
|-------|---------|------|
| `qwen2.5-coder:7b` | Ollama (local) | 7B |
| `llama3.1:8b` | Ollama (local) | 8B |
| `llama-3.3-70b-versatile` | Groq API | 70B |
| `llama-3.3-70b` | Cerebras API | 70B |

## Supported Prompt Strategies

| Strategy | Description |
|----------|-------------|
| `baseline` | Minimal instruction prompt with explicit rules |
| `zero_shot` | Bare instruction with no rules |
| `edge_case` | Instructs the model to handle edge cases |
| `role_framing` | Frames the model as a senior engineer |
| `cot` | Asks the model to think before writing code |
| `cgo` | Goal-oriented: lists functional objectives (return type, boundary values) |
| `io_spec` | Asks the model to identify input/output types and edge cases before coding |

---

## Usage

Run these commands from the `project/` directory.

**Ollama (local)**
```bash
python scripts/generate_samples.py --model qwen2.5-coder:7b --backend ollama --prompt baseline
```

**Groq (API)**
```bash
$env:GROQ_API_KEY = "your_key_here"
python scripts/generate_samples.py --model llama-3.3-70b-versatile --backend groq --prompt baseline
```

**Cerebras (API)**
```bash
$env:CEREBRAS_API_KEY = "your_key_here"
python scripts/generate_samples.py --model llama-3.3-70b --backend cerebras --prompt baseline
```

Output is saved with a sidecar `_run_log.json`.

- HumanEval+: `samples/{model_slug}_t02_{strategy}.jsonl`
- MBPP+: `samples/mbpp_{model_slug}_t02_{strategy}.jsonl`

**MBPP+**
```bash
python scripts/generate_samples.py --dataset mbpp --model llama-3.3-70b-versatile --backend groq --prompt baseline
```

---

## Evaluation

Run from the `project/` directory. EvalPlus executes generated code safely inside Docker.

**Raw evaluation**
```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/<output_file>.jsonl
```

For MBPP+, use `--dataset mbpp`:

```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset mbpp \
  --samples /app/samples/<output_file>.jsonl
```

**Sanitized evaluation** (secondary)
```bash
evalplus.sanitize --samples samples/<output_file>.jsonl --dataset humaneval

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/<output_file>-sanitized.jsonl
```

For MBPP+ sanitization, use `--dataset mbpp`.

---

## Robustness Analysis (Safety Oracle)

Run from the `project/` directory, after the Docker EvalPlus evaluation step above has produced an `_eval_results.json`.

**1. Scan generated code for defensive guard patterns**
```bash
python -m robustness.safety_oracle --dataset {humaneval,mbpp} --samples samples/<output_file>.jsonl
```
Writes `<output_file>_safety_oracle.json`. Detects four AST-pattern guard categories (None checks, empty/zero checks, range/boundary checks, `isinstance` checks), adapted from Li et al. (2025, arXiv:2503.20197) §2.3 — see `decisions.md` D8 for what was excluded and why.

**2. Test whether guard presence predicts Plus-test survival**
```bash
python -m robustness.safety_eval_correlation --safety-oracle samples/<output_file>_safety_oracle.json --results samples/<output_file>_eval_results.json
```
Joins the two files and compares Plus pass rate between guarded/unguarded code, **conditioned on the task already passing Base** (D9), reporting a two-proportion z-test per category (D10). Pooled across all 13 Llama 3.3 70B strategy runs (2,767 base-passing tasks, see `research_log.md` 2026-06-18): guard presence does **not** predict Plus survival — four of five categories are non-significant, and `has_type_check` is significantly *negative* (p=0.0035).

---

## Results (Baseline)

---

## Project Structure

```
COMP4952/
  .venv/                       — dedicated project venv (gitignored)
  requirements.txt
  decisions.md                 — design decisions and rationale (D1-D11)
  research_log.md              — dated session-by-session progress log
  README.md
  project/
    scripts/
      generate_samples.py       — unified generation script (all models, HumanEval+/MBPP+, all strategies)
      analyze_failures.py       — extracts failed EvalPlus tasks into a structured report
      llm_prompt_refine.py      — diagnose-then-revise prompt refinement loop (live LLM call untested — see D11)
      generate_gemini_samples.py
      test_gemini_setup.py
    robustness/
      safety_oracle.py           — AST guard-pattern scan (Safety Oracle)
      safety_eval_correlation.py — joins Safety Oracle output against eval_results.json
    results/
      results.py                 — matplotlib thesis figures
      fitness_oracle.py           — diagram of the evaluation framework
    samples/
      <model>_t02_<strategy>.jsonl
      mbpp_<model>_t02_<strategy>.jsonl
      <model>_t02_<strategy>_run_log.json
      <model>_t02_<strategy>_eval_results.json
      <model>_t02_<strategy>_safety_oracle.json
      <model>_t02_<strategy>_eval_results_safety_correlation.json
      pooled_safety_correlation.json
```

---

## Notes

- All runs use temperature 0.2 and max_tokens 512 for reproducibility.
- Raw evaluation is the primary metric. Sanitized evaluation is secondary.
- The `_run_log.json` sidecar records all run config and has fields for pass@1 scores to be filled in after evaluation.
- Groq and Cerebras free tiers have a 100k token/day limit. At ~640 tokens per problem, a full 164-problem run costs ~105k tokens, so the last few tasks sometimes get skipped.
- MBPP+ has more tasks than HumanEval+, so API quota pressure is higher. Prefer one prompt strategy/model at a time and expect a full run to take longer.
