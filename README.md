# Enhancing Code Generation via Prompt Engineering and Iterative Self-Repair

Thesis project evaluating the effect of prompt strategies and iterative self-repair on LLM code generation quality, benchmarked on HumanEval+ and MBPP+ using EvalPlus. Model: Llama 3.3 70B via Groq.

---

## Requirements

- Python 3.10 or newer
- Docker Desktop (running, for EvalPlus evaluation)
- Groq API key

---

## Installation

A dedicated venv lives at the repo root (`.venv/`), pinned via `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers `evalplus`, `openai`, `groq`, `google-genai`, `matplotlib`, `numpy`, `bandit` — everything imported across `scripts/` and `robustness/`.

---

## Supported Prompt Strategies (Seed Generation)

| Strategy | Description |
|----------|-------------|
| `cop` | Chain of Objectives Prompting — best overall strategy from Thesis B |
| `baseline` | Minimal instruction prompt with explicit rules |
| `zero_shot` | Bare instruction with no rules |
| `edge_case` | Instructs the model to handle edge cases |
| `role_framing` | Frames the model as a senior engineer |
| `cot` | Asks the model to think before writing code |
| `cgo` | Goal-oriented: lists functional objectives (return type, boundary values) |
| `io_spec` | Asks the model to identify input/output types and edge cases before coding |

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

Output is saved with a sidecar `_run_log.json`.

- HumanEval+: `samples/{model_slug}_t02_{strategy}.jsonl`
- MBPP+: `samples/mbpp_{model_slug}_t02_{strategy}.jsonl`

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

Run from the `project/` directory. EvalPlus executes generated code safely inside Docker.

**Raw evaluation**
```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/llama-33-70b-versatile_t02_cop.jsonl
```

Repair-round outputs are evaluated the same way, pointing at the round file:
```bash
docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl
```

Replace `humaneval` with `mbpp` and adjust filenames for MBPP+. Replace `repair-cot` with `repair-minimal` for minimal-strategy files.

**Sanitized evaluation** (secondary)
```bash
evalplus.sanitize --samples samples/<output_file>.jsonl --dataset humaneval

docker run --rm --pull=always -v "${PWD}:/app" ganler/evalplus:latest \
  evalplus.evaluate --dataset humaneval \
  --samples /app/samples/<output_file>-sanitized.jsonl
```

For MBPP+ sanitization, use `--dataset mbpp`.

### Step 4 — Safety Sweep (Bandit)

Static vulnerability scan of generated code via `bandit`, run per JSONL file:

```bash
python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl --dataset humaneval

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop.jsonl --dataset mbpp

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop_repair-cot_round1.jsonl --dataset mbpp

python scripts/verify_safety.py --jsonl_path samples/mbpp_llama-33-70b-versatile_t02_cop_repair-cot_round2.jsonl --dataset mbpp
```

Note: this Bandit-based sweep checks generated code for *vulnerability* patterns (e.g. `eval`, `exec`, shell injection). It is unrelated to the AST guard-presence Safety Oracle described below despite the shared "safety" name — see the naming-collision note under Robustness Analysis.

---

## Robustness Analysis (Safety Oracle)

Run from the `project/` directory, after the Docker EvalPlus evaluation step above has produced an `_eval_results.json`.

**1. Scan generated code for defensive guard patterns**
```bash
python -m robustness.safety_oracle --dataset {humaneval,mbpp} --samples samples/<output_file>.jsonl
```
Writes `<output_file>_safety_oracle.json`. Detects four AST-pattern guard categories (None checks, empty/zero checks, range/boundary checks, `isinstance` checks), adapted from Li et al. (2025, arXiv:2503.20197) §2.3.

**2. Test whether guard presence predicts Plus-test survival**
```bash
python -m robustness.safety_eval_correlation --safety-oracle samples/<output_file>_safety_oracle.json --results samples/<output_file>_eval_results.json
```
Joins the two files and compares Plus pass rate between guarded/unguarded code, **conditioned on the task already passing Base**, reporting a two-proportion z-test per category. Pooled across all 13 Llama 3.3 70B strategy runs (2,767 base-passing tasks): guard presence does **not** predict Plus survival — four of five categories are non-significant, and `has_type_check` is significantly *negative* (p=0.0035).

This module (`robustness/safety_oracle.py`, `robustness/safety_eval_correlation.py`) is distinct from the `scripts/safety_check.py` / `scripts/verify_safety.py` Bandit sweep in Step 4 above — one measures defensive coding patterns (guard presence), the other measures security vulnerabilities. Both are called "safety" for historical reasons; treat them as separate signals.

---

## Key Results

### Iterative self-repair (Thesis C)

CoT repair outperforms Minimal repair on correctness by more than 2x on both benchmarks, with no robustness cost. Robustness ratio degradation under repair is structural to single-assertion feedback loops regardless of repair strategy, confirmed consistently across both datasets.

| Metric | HumanEval CoT | HumanEval Minimal | MBPP CoT | MBPP Minimal |
|--------|--------------|-------------------|----------|--------------|
| R0 Base pass@1 | 0.805 | 0.805 | 0.865 | 0.865 |
| R2 Base pass@1 | 0.915 | 0.854 | 0.921 | 0.897 |
| Total gain | +11.0pp | +4.9pp | +5.6pp | +3.7pp |
| R0 Robustness Ratio | 0.917 | 0.917 | 0.838 | 0.838 |
| R2 Robustness Ratio | 0.906 | 0.906 | 0.836 | 0.832 |
| Safety Rate (all rounds) | ≥98.8% | — | 100.0% | — |

### Prompt strategies, Round 0 (Thesis B)

HumanEval+ (Llama 3.3 70B):

| Strategy | Base pass@1 | Plus pass@1 | Robustness Gap |
|----------|-------------|-------------|-----------------|
| cgo | 0.805 | 0.744 | 0.061 |
| io_spec | 0.787 | 0.713 | 0.073 |
| baseline | 0.726 | 0.677 | 0.049 |
| zero_shot | 0.677 | 0.634 | 0.043 |
| cot | 0.677 | 0.640 | 0.037 |
| role_framing | 0.665 | 0.616 | 0.049 |
| edge_case | 0.640 | 0.604 | 0.037 |

MBPP+ (Llama 3.3 70B):

| Strategy | Base pass@1 | Plus pass@1 | Robustness Gap |
|----------|-------------|-------------|-----------------|
| zero_shot | 0.876 | 0.722 | 0.153 |
| cot | 0.865 | 0.725 | 0.140 |
| cgo | 0.865 | 0.733 | 0.132 |
| edge_case | 0.860 | 0.696 | 0.164 |
| baseline | 0.839 | 0.675 | 0.164 |

---

## Baselines

External-method baselines live under `project/baselines/`, kept separate from the thesis's own `scripts/` pipeline — separately authored (Gurdiraj Bal, z5386590), does not modify `self_repair.py`/`code_executor.py`/`generate_samples.py`, only imports shared low-level utilities from them.

**Reflexion** (`project/baselines/reflexion/`) — a faithful re-implementation of Shinn et al. (2023), *"Reflexion: Language Agents with Verbal Reinforcement Learning"* (arXiv:2303.11366), §4.3 (Programming), run as a controlled comparison against the thesis's own `cot`/`minimal` repair strategies. Covers both HumanEval+ and MBPP+ via `--dataset`.

```bash
python baselines/reflexion/run_reflexion_batch.py --dataset {humaneval,mbpp} \
    --model llama-3.3-70b-versatile --backend groq \
    --max_repair_rounds 2 --reflexion_memory_size 1 --resume-missing
```

To run inference on UNSW Katana (local vLLM server, `--backend katana`) instead of Groq, see the shared scripts in `project/baselines/katana/` (used by all external baselines).

Results and analysis: `project/baselines/reflexion/results.md`.

**Prochemy** (`project/baselines/prochemy/`) — a faithful re-implementation of Ye et al. (2025), *"Prochemy: Automating Prompt Engineering by Prompt Alchemy"* (arXiv:2503.11085), an execution-driven automatic prompt-optimisation method. Where Reflexion competes with the repair layer, Prochemy competes with the prompt-strategy / seed-generation layer: it searches for one fixed system prompt `P*` (mutate → weighted-evaluate → select over a small training set), seeded from the authors' own verbatim prompts. `P*` is graded on Base/Plus/**Gap**/**Ratio**, so it also tests whether an accuracy-optimised prompt actually closes the robustness gap — something the paper (which reports Plus pass@1 only) does not measure.

```bash
python baselines/prochemy/run_prochemy.py --dataset {humaneval,mbpp} \
    --initial zero_shot --model llama-3.3-70b-versatile --backend katana \
    --k-max 10 --n-variants 10 --patience 3 --reoptimize --resume-missing
```

The prompt search is token-heavy (~1M+ tokens per full run), so it runs on Katana (local vLLM, no token cap) rather than Groq; see the shared scripts in `project/baselines/katana/`. To generate from the unoptimised seed `S(0)` as a comparison anchor, add `--seed-baseline`.

Results and analysis: `project/baselines/prochemy/results.md`.

Finding: on the confound-free same-stack comparison, the optimised `P*` **reproducibly fails to beat its own unoptimised seed `S(0)`** — below on both Base and Plus, and the search is deterministic (three independent searches converged to the identical `P*`). Accuracy-oriented prompt optimisation bought neither accuracy nor robustness in this setup.

---

## Multi-model generality (Qwen2.5-Coder-32B)

The thesis's own pipeline (goal-oriented `cop` seed → `cot`/`minimal` self-repair) — not an external baseline — run on a second model, **Qwen2.5-Coder-32B-Instruct**, on both HumanEval+ and MBPP+, to test whether the findings hold beyond the primary Llama-3.3-70B. Reported as within-model deltas (never subtracting across models, since the two differ in size and specialisation).

Result: self-repair reliably lifts Base and Plus on both models and both datasets, but the **robustness Gap never closes** — it widens in 7 of the 8 model×dataset×strategy cells and is flat in the 1 remaining. So *"accuracy-oriented repair buys accuracy, not robustness"* holds across a general-purpose 70B and a code-specialised 32B. Served at TP=1 on a single H200 via the shared `project/baselines/katana/` infra (`run_qwen_smoke.pbs` → `run_qwen_repair.pbs` / `run_qwen_mbpp.pbs`).

Results and analysis: `project/baselines/katana/qwen_results.md`.

---

## Project Structure

```text
COMP4952/
  .venv/                       — dedicated project venv (gitignored)
  requirements.txt
  README.md
  project/
    scripts/
      generate_samples.py       — seed generation + repair mode (--repair flag), all strategies
      self_repair.py            — N-round repair loop orchestrator (single task)
      code_executor.py          — sandboxed execution + error extraction
      repair_prompt.py          — minimal and CoT repair prompt builders
      run_repair_batch.py       — batch driver for the repair loop
      safety_check.py           — Bandit static analysis wrapper
      verify_safety.py          — safety sweep across a full JSONL file
      verify_executor.py        — HumanEval baseline execution sweep (diagnostic)
      verify_executor_mbpp.py   — MBPP baseline execution sweep (diagnostic)
      test_cot_marker_compliance.py — one-off diagnostic
      test_single_repair.py     — one-off diagnostic
      analyze_failures.py       — extracts failed EvalPlus tasks into a structured report
      llm_prompt_refine.py      — diagnose-then-revise prompt refinement loop (live LLM call untested)
    baselines/
      reflexion/
        reflexion_prompt.py      — test-gen/self-reflection/actor prompt builders
        reflexion_executor.py    — self-generated-test execution wrapper
        reflexion_repair.py      — per-task Reflexion repair loop orchestrator
        run_reflexion_batch.py   — CLI batch driver
        results.md               — results and analysis
      prochemy/
        prochemy_prompt.py       — verbatim seed/mutation prompts + parsers
        prochemy_trainset.py     — training-set load + candidate execution
        prochemy_optimize.py     — mutate→weighted-evaluate→select search loop
        run_prochemy.py          — CLI driver (optimise → freeze P* → generate; --seed-baseline anchor)
        prochemy_training_set.jsonl — authors' shipped 20-task training set
        results.md               — results and analysis
      katana/                    — shared vLLM run infra for all baselines
        serve_and_run.sh         — serve vLLM → wait → run baseline cmd → teardown
        setup_katana_env.sh      — build shared venv + stage bf16 weights
        run_reflexion_mbpp.pbs   — PBS wrapper: Reflexion MBPP+
        run_prochemy_humaneval.pbs      — PBS wrapper: Prochemy HumanEval+
        run_prochemy_humaneval_seed.pbs — PBS wrapper: Prochemy S(0) anchor
        run_prochemy_humaneval_stability.pbs — PBS wrapper: Prochemy P* stability re-runs
        run_qwen_smoke.pbs       — PBS wrapper: Qwen post-processing smoke test (gate)
        run_qwen_repair.pbs      — PBS wrapper: Qwen HumanEval+ seed + cot/minimal repair
        run_qwen_mbpp.pbs        — PBS wrapper: Qwen MBPP+ seed + cot/minimal repair
        qwen_results.md          — multi-model generality results (Qwen2.5-Coder-32B)
        probe_katana.sh          — read-only environment audit
    robustness/
      safety_oracle.py           — AST guard-pattern scan (Safety Oracle)
      safety_eval_correlation.py — joins Safety Oracle output against eval_results.json
    results/
      results.py                 — matplotlib thesis figures
      fitness_oracle.py           — diagram of the evaluation framework
      scatter_plot.py             — result visualization
      repair_trajectories_<model>_t02_<strategy>_<dataset>_<repair_strategy>.json
    samples/
      <model>_t02_<strategy>.jsonl
      mbpp_<model>_t02_<strategy>.jsonl
      <model>_t02_<strategy>_run_log.json
      <model>_t02_<strategy>_eval_results.json
      <model>_t02_<strategy>_safety_oracle.json
      <model>_t02_<strategy>_eval_results_safety_correlation.json
      <model>_t02_<strategy>_repair-<repair_strategy>_round<N>.jsonl
      <model>_t02_<strategy>_repair_run_log.json
      pooled_safety_correlation.json
```

---

## Notes

- All runs use temperature 0.2 and max_tokens 512 for reproducibility.
- Raw EvalPlus evaluation is the primary metric. Sanitized evaluation is secondary.
- Robustness Ratio = Plus pass@1 / Base pass@1. Robustness Gap = Base pass@1 minus Plus pass@1.
- The `_run_log.json` sidecar records all run config and has fields for pass@1 scores to be filled in after evaluation (not yet automated).
- Groq free tier limit is ~100k tokens/day. At ~640 tokens per problem, a full 164-problem seed generation run costs ~105k tokens, so the last few tasks sometimes get skipped — re-run with `--resume-missing`. Repair mode uses far fewer tokens than seed generation since only failing tasks receive repair calls.
- MBPP+ has more tasks than HumanEval+, so API quota pressure is higher during seed generation.
