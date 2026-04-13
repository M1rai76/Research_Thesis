# Prompt Engineering for LLM Code Generation

Thesis project evaluating the effect of prompt strategies on code generation quality across multiple LLMs, benchmarked on **HumanEval** and **HumanEval+** using **EvalPlus**.

---

## Requirements

- Python 3.10 or newer
- Docker Desktop (running)
- Ollama (for local models)
- Groq or Cerebras API key (for cloud models)

---

## Installation

```bash
pip install evalplus openai groq
```

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

---

## Usage

All commands can be run from the project root directory.

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

Output is saved to `samples/{model_slug}_t02_{strategy}.jsonl` with a sidecar `_run_log.json`.

---

## Evaluation

Run from the `project/` directory. EvalPlus executes generated code safely inside Docker.

**Raw evaluation**
```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/<output_file>.jsonl
```

**Sanitized evaluation** (secondary)
```bash
evalplus.sanitize --samples samples/<output_file>.jsonl --dataset humaneval

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/<output_file>-sanitized.jsonl
```

---

## Results (Baseline)

---

## Project Structure

```
project/
  scripts/
    generate_samples.py       — unified generation script (all models, all strategies)
    generate_gemini_samples.py
    test_gemini_setup.py
  samples/
    <model>_t02_<strategy>.jsonl
    <model>_t02_<strategy>_run_log.json
    <model>_t02_<strategy>_eval_results.json
  human-eval/                 — OpenAI HumanEval evaluation framework
  README.md
```

---

## Notes

- All runs use temperature 0.2 and max_tokens 512 for reproducibility.
- Raw evaluation is the primary metric. Sanitized evaluation is secondary.
- The `_run_log.json` sidecar records all run config and has fields for pass@1 scores to be filled in after evaluation.
- Groq and Cerebras free tiers have a 100k token/day limit. At ~640 tokens per problem, a full 164-problem run costs ~105k tokens, so the last few tasks sometimes get skipped.
