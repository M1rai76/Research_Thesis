# Multi-Model Generality — Qwen2.5-Coder-32B and DeepSeek-Coder-V2-Lite (Results)

Author: Gurdiraj Bal (z5386590)

Runs the thesis's own pipeline — goal-oriented seed generation (the `cop` prompt strategy) followed by iterative self-repair (`cot` and `minimal`) — on **two further models** — Qwen2.5-Coder-32B-Instruct and DeepSeek-Coder-V2-Lite-Instruct — to test whether the pipeline's improvement and, more importantly, the **robustness-gap findings** generalise beyond the primary model (Llama-3.3-70B).

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

## Status and remaining limitations

The generality matrix is **complete**: three models × two datasets × two repair strategies = 12 cells, all run and graded.

- **Single-run throughout.** Every cell is one sample at temperature 0.2, so the deltas are directional rather than variance-bounded. The sign and Gap-shape are consistent across all 12 cells, which is what the claim rests on — not any individual magnitude.
- **Two of three models are code-specialised.** Llama-3.3-70B is the only general-purpose model, so the "code specialisation halves brittleness" observation rests on a 1-vs-2 split and should be read as suggestive.
- **Prompt strategy held fixed** at `cop` for the seed across all models; the multi-strategy sweep was run on Llama only.

---

# Third model — DeepSeek-Coder-V2-Lite-Instruct (HumanEval+ and MBPP+)

Added as a third generality model. Its value is not another data point of the same kind but a **different architecture**: Llama-3.3-70B and Qwen2.5-Coder-32B are both *dense*, while V2-Lite is a **sparse Mixture-of-Experts** model (16B total parameters, ~2.4B active per token — 64 routed experts plus 2 shared, top-6 routing). Served bf16 at TP=1 on a single H200.

## Results — HumanEval+ (EvalPlus, Plus = base AND plus)

| Run | Base | Plus | Gap | Brittleness (1 − Plus/Base) |
|---|---|---|---|---|
| R0 (`cop` seed) | 0.750 | 0.713 | 0.037 | 4.9% |
| `cot` R1 | 0.780 | 0.738 | 0.043 | 5.5% |
| `cot` R2 | 0.793 | 0.750 | 0.043 | 5.4% |
| `minimal` R1 | 0.762 | 0.720 | 0.043 | 5.6% |
| `minimal` R2 | 0.762 | 0.720 | 0.043 | 5.6% |

## Results — MBPP+ (second dataset)

| Run | Base | Plus | Gap | Brittleness (1 − Plus/Base) |
|---|---|---|---|---|
| R0 (`cop` seed) | 0.825 | 0.698 | 0.127 | 15.4% |
| `cot` R2 | 0.857 | 0.720 | 0.137 | 16.0% |
| `minimal` R2 | 0.844 | 0.712 | 0.132 | 15.6% |

Within-model deltas: `cot` **+3.2pp Base / +2.2pp Plus**, `minimal` **+1.9pp / +1.4pp**. Both **widen** the Gap, and `cot` again beats `minimal`.

Note the contrast with this model's own HumanEval+ seed: brittleness 4.9% there against **15.4%** here. MBPP+ is markedly harder to be robust on for *every* model tested:

| Seed brittleness | HumanEval+ | MBPP+ |
|---|---|---|
| Llama-3.3-70B (general) | 8.3% | 16.2% |
| Qwen2.5-Coder-32B (code) | 4.9% | 13.8% |
| DeepSeek-V2-Lite (code) | 4.9% | 15.4% |

Two readings, and only one of them survives the second dataset. Whatever makes MBPP+ solutions brittle is a property of the **benchmark** — every model roughly doubles or triples its brittleness there — and that holds across all three. But the tempting second reading, that **code specialisation halves brittleness**, is **HumanEval+-specific**: the clean 4.9%-vs-8.3% split there collapses to a narrow 13.8–16.2% band on MBPP+, with the two code models straddling rather than beating the general one. Report the benchmark effect; do not generalise the specialisation effect.

## Within-model deltas — all three models side by side

Deltas only; absolute pass@1 is never subtracted across models.

| Model | Architecture | Strategy | ΔBase | ΔPlus | Gap R0 → R2 |
|---|---|---|---|---|---|
| Llama-3.3-70B | dense, general | cot | +11.0pp | +9.1pp | 0.067 → 0.086 |
| Llama-3.3-70B | dense, general | minimal | +4.9pp | +3.6pp | 0.067 → 0.080 |
| Qwen2.5-Coder-32B | dense, code | cot | +2.4pp | +1.9pp | 0.043 → 0.048 |
| Qwen2.5-Coder-32B | dense, code | minimal | +1.2pp | +1.3pp | 0.043 → 0.042 |
| DeepSeek-V2-Lite | **MoE**, code | cot | +4.3pp | +3.7pp | 0.037 → 0.043 |
| DeepSeek-V2-Lite | **MoE**, code | minimal | +1.2pp | +0.6pp | 0.037 → 0.043 |

And the same three models on MBPP+:

| Model | Architecture | Strategy | ΔBase | ΔPlus | Gap R0 → R2 |
|---|---|---|---|---|---|
| Llama-3.3-70B | dense, general | cot | +5.6pp | +4.5pp | 0.140 → 0.151 |
| Llama-3.3-70B | dense, general | minimal | +3.2pp | +2.1pp | 0.140 → 0.151 |
| Qwen2.5-Coder-32B | dense, code | cot | +4.2pp | +2.9pp | 0.124 → 0.138 |
| Qwen2.5-Coder-32B | dense, code | minimal | +3.7pp | +2.4pp | 0.124 → 0.138 |
| DeepSeek-V2-Lite | **MoE**, code | cot | +3.2pp | +2.2pp | 0.127 → 0.137 |
| DeepSeek-V2-Lite | **MoE**, code | minimal | +1.9pp | +1.4pp | 0.127 → 0.132 |

## What the third model establishes

1. **The pattern is not dense-model-specific.** Repair lifts both Base and Plus while the robustness Gap widens, on a sparse MoE just as on two dense models. Across the full repair matrix (three models × datasets × two strategies) the Gap now **widens in 11 of 12 cells** and is flat in the one remaining (Qwen / HumanEval+ / `minimal`, 0.043 → 0.042).
2. **`cot` outperforms `minimal` in every model on every dataset** — 6/6 model×dataset pairs.
3. **Delta magnitude tracks headroom, not model quality.** Qwen's `cot` delta was compressed to +2.4pp because its seed Plus (0.829) was already near ceiling; V2-Lite, starting at 0.713, moved +3.7pp on Plus. This is exactly why within-model deltas are the unit of evidence and cross-model subtraction is avoided.

## A pattern the Gap hides and brittleness reveals

Measured by Gap, V2-Lite's seed (0.037) looks *more robust* than Qwen's (0.043). That is an artefact: Gap ≤ Base by construction, so V2-Lite's lower Base mechanically caps its Gap.

Measured as brittleness — the share of correct-looking solutions that fail under stress, `1 − Plus/Base` — the two are **identical at 4.9%**, and the real pattern appears:

| Model | Specialisation | Seed brittleness (HumanEval+) |
|---|---|---|
| Llama-3.3-70B | general | **8.3%** |
| Qwen2.5-Coder-32B | code | **4.9%** |
| DeepSeek-V2-Lite | code | **4.9%** |

On HumanEval+, both code-specialised models are roughly **half as brittle** as the general-purpose model, and repair then adds ~0.5–1.1pp of brittleness regardless of which model it is applied to.

**But this specialisation effect does not replicate on MBPP+** — see the MBPP+ section above, where the clean split collapses to a narrow 13.8–16.2% band with the two code models straddling rather than beating the general one. What *does* hold on both datasets is the methodological point this section is really about: **Gap and brittleness rank the models differently, and Gap's ranking is the confounded one**, because `Gap ≤ Base` by construction. That argument stands independently of which model ends up ahead.

## Operational notes

- **A first DeepSeek attempt was abandoned.** `deepseek-coder-33b-instruct` (2023) declares `LlamaTokenizerFast`; under the serving venv's transformers 5.x the fast tokenizer fails to build and falls back to a slow `LlamaTokenizer` with no `tokenizer.model` to work from, producing a decoder that destroys all whitespace (`"def add(a, b)"` → `"defadd(a,b)"`). Its 164 generated samples were raw byte-level BPE. Verified as neither a post-processing, chat-template, nor download problem, and not fixable by a vLLM tokenizer flag — fast and slow modes fail identically. Other models on the same venv are unaffected (Qwen round-trips exactly).
- **Test a model's tokenizer before downloading its weights.** Fetching `tokenizer*` + `config.json` (four small files), round-tripping a whitespace-bearing string, and checking that vLLM's config loader parses the architecture takes under a minute and would have avoided a 63GB download and an overnight job.
- **Do not pass `--trust-remote-code` for V2-Lite.** Its bundled `configuration_deepseek.py` raises `AttributeError` on transformers 5.x; vLLM's native `DeepseekV2ForCausalLM` support reads `config.json` directly and works.
- Runtime: seed 5m48s, seed + `cot` + `minimal` repair 11m36s, both at TP=1 on one H200. The seed run landed on node k099 — which reliably crashes tensor-parallel jobs — and ran cleanly, confirming that single-GPU serving sidesteps that failure entirely.
