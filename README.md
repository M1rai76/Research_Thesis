# HumanEval + EvalPlus Pipeline (Qwen / Ollama)

This MVP for Enhanced code generation thesis evaluates code generation models on **HumanEval** and **HumanEval+** using **EvalPlus** inside Docker.
The workflow generates solutions using a local LLM (Qwen via Ollama) and measures correctness using standardized benchmark tests.

---

# Requirements

Install the following first:

* Python **3.10 or newer**
* Docker Desktop (running)
* Ollama (for local models)

---

# Step 1 — Install Dependencies

```bash
pip install evalplus
pip install ollama
```

---

# Step 2 — Pull the Model (Ollama)

Example:

```bash
ollama pull qwen
```

Verify:

```bash
ollama list
```

You should see:

```text
qwen
```

---

# Step 3 — Generate Model Outputs

Run the generation script:

```bash
python scripts/generate_qwen_samples.py
```

This will create:

```text
samples/qwen_samples.jsonl
```

That file contains the generated solutions for all HumanEval tasks.

---

# Step 4 — Run Evaluation (Docker)

Run:

```bash
docker run --rm --pull=always \
-v "${PWD}:/app" \
ganler/evalplus:latest \
evalplus.evaluate --dataset humaneval --samples /app/samples/qwen_samples.jsonl
```

This command:

* launches the EvalPlus container
* executes all generated code safely
* computes benchmark scores

---

# Output Example

```text
humaneval (base tests)
pass@1: 0.683

humaneval+ (base + extra tests)
pass@1: 0.646
```

These metrics represent:

* **HumanEval** — standard correctness
* **HumanEval+** — robustness testing

---

# Project Structure

```text
project/

scripts/
    generate_qwen_samples.py

samples/
    qwen_samples.jsonl

README.md
```

---

# Typical Workflow

```bash
python scripts/generate_qwen_samples.py

docker run --rm --pull=always \
-v "${PWD}:/app" \
ganler/evalplus:latest \
evalplus.evaluate --dataset humaneval \
--samples /app/samples/qwen_samples.jsonl
```

---

# Notes

* Docker is used to safely execute generated code.
* The evaluation is fully reproducible.
* You can replace the model (Qwen, Gemini, etc.) without changing the evaluation pipeline.
* Results are reported using **pass@1**, the standard HumanEval metric.

---

If you want later, I can also provide:

* a **slightly more formal academic README** for your thesis repo
* or a **reproducibility section** suitable for your dissertation

