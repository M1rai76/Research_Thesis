# Multi-Model Generality — Qwen2.5-Coder-32B (Results)

Author: Gurdiraj Bal (z5386590)

Runs the thesis's own pipeline — goal-oriented seed generation (the `cop` prompt strategy) followed by iterative self-repair (`cot` and `minimal`) — on a **second model**, Qwen2.5-Coder-32B-Instruct, to test whether the pipeline's improvement and, more importantly, the **robustness-gap findings** generalise beyond the primary model (Llama-3.3-70B).

This is a **generality demonstration, not a model comparison.** The unit of evidence is the **within-model delta** (repair − seed), computed separately for each model; absolute pass@1 is never subtracted across models. Qwen2.5-Coder-32B is code-specialised and smaller than the general-purpose Llama-3.3-70B, so the two are deliberately *not* placed head-to-head — the question is only whether the *sign and shape* of each pipeline effect hold within each model.

## Setup

- **Model:** `Qwen/Qwen2.5-Coder-32B-Instruct`, bf16, served with vLLM on a single UNSW Katana H200 (tensor-parallel = 1).
- **Benchmarks:** HumanEval+ (164 tasks) and MBPP+ (378 tasks), EvalPlus.
- **Pipeline:** goal-oriented `cop` seed, then self-repair for 2 rounds under two feedback styles — `cot` (reason-then-fix) and `minimal` (fix only) — against the frozen seed, matching the Llama pipeline exactly.
- **Grading:** native evalplus. Plus pass@1 counts a task as passing only when **both** base and plus statuses are `pass` (recomputed from `_eval_results.json`, verified against evalplus's printed summary).
- **Decoding:** temperature 0.2, single sample per task (no repeated sampling) — results are directional, not variance-bounded.

## Results — Qwen2.5-Coder-32B on HumanEval+

| Run | Base | Plus | Gap (Base−Plus) | Ratio (Plus/Base) |
|---|---|---|---|---|
| **R0** (`cop` seed) | 0.872 | 0.829 | 0.043 | 0.951 |
| `cot` round 1 | 0.890 | 0.848 | 0.043 | 0.952 |
| `cot` round 2 | **0.896** | **0.848** | 0.049 | 0.946 |
| `minimal` round 1 | 0.884 | 0.842 | 0.043 | 0.952 |
| `minimal` round 2 | 0.884 | 0.842 | 0.043 | 0.952 |

## Generality — within-model deltas vs Llama-3.3-70B

Each model's repair delta is measured against *its own* seed (R0). Llama's figures are from the existing HumanEval+ pipeline runs (goal-oriented seed + the same two repair strategies).

| Model | Strategy | ΔBase | ΔPlus | Gap behaviour (R0 → R2) |
|---|---|---|---|---|
| Llama-3.3-70B (general) | `cot` | +11.0pp | +9.1pp | widens 0.067 → 0.085 |
| Llama-3.3-70B | `minimal` | +4.9pp | +3.6pp | widens 0.067 → 0.079 |
| Qwen2.5-Coder-32B (code) | `cot` | +2.4pp | +1.8pp | widens 0.043 → 0.049 |
| Qwen2.5-Coder-32B | `minimal` | +1.2pp | +1.2pp | flat 0.043 → 0.043 |

## What generalises (HumanEval+)

1. **Direction of the pipeline effect.** Self-repair nets **positive on both Base and Plus for both models**, and `cot` outperforms `minimal` in both. The pipeline's improvement is not a Llama artefact.
2. **The robustness-gap finding (the thesis-relevant one).** The **robustness Gap never closes under repair** — in 3 of the 4 model×strategy cells it *widens*, in the 4th it is flat. Repair buys **accuracy, not robustness**, on both a general-purpose 70B and a code-specialised 32B. This is the central generality result: the accuracy-vs-robustness dissociation holds across models.
3. **Magnitude is model-dependent (a ceiling effect).** Qwen's deltas (+2.4pp Base under `cot`) are far smaller than Llama's (+11.0pp). This is expected: Qwen's *seed* already scores 0.829 Plus — level with Llama's fully `cot`-repaired result — so there is little headroom for repair to add. The generality claim therefore rests on the **sign and Gap-shape**, which are consistent across models, not on the magnitude, which is compressed by Qwen's stronger starting point.

## Notable detail

Under `cot`, Qwen's round 2 adds Base (0.890 → 0.896) but **not** Plus (0.848 → 0.848) — so the extra repair round fixes visible-test failures whose fixes do **not** survive the expanded stress tests, and the Gap widens as a result. This is the same mechanism seen on Llama: repair is drawn to base-visible failures, and the resulting fixes are not robustness-preserving.

## Results — Qwen2.5-Coder-32B on MBPP+ (second dataset)

| Run | Base | Plus | Gap (Base−Plus) | Ratio (Plus/Base) |
|---|---|---|---|---|
| **R0** (`cop` seed) | 0.899 | 0.775 | 0.124 | 0.862 |
| `cot` round 1 | 0.939 | 0.804 | 0.135 | 0.856 |
| `cot` round 2 | **0.942** | **0.804** | 0.138 | 0.854 |
| `minimal` round 1 | 0.937 | 0.802 | 0.135 | 0.856 |
| `minimal` round 2 | 0.937 | 0.799 | 0.138 | 0.853 |

## Cross-dataset synthesis

Within-model deltas (R2 − R0) across the full grid — 2 models × 2 datasets × 2 repair strategies. Llama figures are from the existing pipeline runs.

| Model | Dataset | Strategy | ΔBase | ΔPlus | Gap behaviour |
|---|---|---|---|---|---|
| Llama-3.3-70B | HumanEval+ | `cot` | +11.0pp | +9.1pp | widens 0.067 → 0.085 |
| Llama-3.3-70B | HumanEval+ | `minimal` | +4.9pp | +3.6pp | widens 0.067 → 0.079 |
| Qwen2.5-Coder-32B | HumanEval+ | `cot` | +2.4pp | +1.8pp | widens 0.043 → 0.049 |
| Qwen2.5-Coder-32B | HumanEval+ | `minimal` | +1.2pp | +1.2pp | flat 0.043 → 0.043 |
| Llama-3.3-70B | MBPP+ | `cot` | +5.6pp | +4.5pp | widens 0.140 → 0.151 |
| Llama-3.3-70B | MBPP+ | `minimal` | +3.2pp | +2.1pp | widens 0.140 → 0.151 |
| Qwen2.5-Coder-32B | MBPP+ | `cot` | +4.2pp | +2.9pp | widens 0.124 → 0.138 |
| Qwen2.5-Coder-32B | MBPP+ | `minimal` | +3.7pp | +2.4pp | widens 0.124 → 0.138 |

Across all eight cells:

1. **Repair always nets positive** on both Base and Plus, and `cot` ≥ `minimal` throughout — the pipeline's improvement is directionally consistent on both models and both datasets.
2. **The robustness Gap never closes.** It *widens* in 7 of 8 cells and is flat in the 1 remaining (Qwen HumanEval+ `minimal`). Repair buys **accuracy, not robustness** — on a general-purpose 70B and a code-specialised 32B, on both benchmarks. This is the central generality result, now confirmed cross-dataset.
3. **Magnitudes align better on MBPP+.** Qwen's MBPP+ seed (0.775 Plus) leaves real headroom, unlike its HumanEval+ seed (0.829, already at Llama's repaired level), so the ceiling effect that compressed Qwen's HumanEval+ deltas is much smaller here — Qwen's MBPP+ `cot` delta (+4.2pp Base) sits close to Llama's (+5.6pp).

## Cost

- Seed generation: ~8 min on one H200 (TP=1), model load included.
- Both HumanEval+ repair passes (`cot` + `minimal`, 2 rounds each, one model load): ~13 min on one H200.
- MBPP+ (seed + both repair passes, 378 tasks, one model load): ~20 min on one H200.
- Single-GPU serving (TP=1) means these run on **any** H200 — no tensor-parallel, no multi-node scheduling — so the whole model (both datasets) completed in one afternoon alongside other GPU work at zero contention.

## Caveats / limitations

- **Single run at temperature 0.2** — directional, not variance-bounded. Repeated sampling would tighten the deltas but is not expected to change the sign or the Gap behaviour.
- **Inference stack:** Qwen is served by vLLM; the Llama pipeline numbers it is compared against were produced on a different serving stack (Groq) using the same-precision weights. Because the comparison is of *within-model deltas* (not absolute cross-model numbers), this does not confound the generality claim, but exact token outputs are stack-dependent.
- **Ceiling effect** on Qwen (noted above) compresses the repair magnitude — read sign and shape, not size.
- **Seed strategy:** the goal-oriented **`cop`** prompt — it lists functional objectives (exact output type, valid-input handling, boundary/edge cases) — the **identical prompt** used to seed the Llama pipeline (verified byte-for-byte), so the comparison is exact. (`cop` was formerly named `cgo`, an identical-text rename; Llama's original *seed* files still carry the old `_cgo` filename, but it is the same prompt as `cop`.)

## Next

- **A third model** (e.g. DeepSeek-Coder) would strengthen the generality claim from N=2 toward N=3.
- Optionally, a repeated-sampling run to bound the single-run deltas — the sign and Gap-shape are stable across models and datasets, so this would be confirmatory rather than load-bearing.
