# Prochemy Baseline — Results

Author: Gurdiraj Bal (z5386590)

Faithful re-implementation of Ye et al. (2025), *"Prochemy: Automating Prompt Engineering by Prompt Alchemy"* (arXiv:2503.11085), run as a controlled comparison against the thesis's own prompt-strategy / seed-generation layer (Thesis B). Prochemy is an **execution-driven automatic prompt-optimisation** method: starting from a seed prompt `S(0)`, it searches for one fixed system prompt `P*` per (model, task-type) via a mutate → weighted-evaluate → select loop over a small training set.

Key design choices for this reproduction: it is seeded from the **authors' own verbatim prompts** (not this project's templates), so it reproduces the published method rather than a hybrid; all code generation runs at temperature 0.2 with the prompt-mutation step at 1.0 (so any comparison to the paper's reported gains is **directional**, not a numeric reproduction — the model and temperature both differ from theirs); and the search runs on **Katana / vLLM** rather than Groq, because a full search is ~1M+ tokens, far above Groq's free-tier daily cap.

**Scope:** **HumanEval+ (164 tasks)**, seeded from the authors' verbatim zero-shot `S(0)`, optimised on their shipped 20-task training set (`k_max=10`, `n_variants=10`, `patience=3`), then `P*` frozen and used to generate one sample per test task. Run on **Katana / vLLM** serving **bf16 `meta-llama/Llama-3.3-70B-Instruct`** under the served-name `llama-3.3-70b-versatile`. The headline comparison is `P*` against `S(0)` generated on the *same* stack (confound-free); a 3-run stability check was also completed (below). MBPP+ Prochemy not yet run.

> **Headline finding.** On the confound-free, same-stack comparison, Prochemy's optimised `P*` **does not beat its own unoptimised seed `S(0)`** — it lands *below* `S(0)` on both Base and Plus. A 3-run stability check showed the search is **deterministic**: all three independent searches converged to the byte-identical `P*`, so the regression is **reproducible, not an unlucky single draw**. Accuracy-oriented prompt optimisation did not buy accuracy *or* robustness in this setup.

## What `P*` converged to

The search departed from the authors' terse zero-shot seed —

> *"You are a code generation assistant. Your task is to generate Python code based on the given task description and complete the work described in the task."*

— and converged (prompt id 87) on a structured, multi-step instruction that explicitly asks for **error handling, input validation, and edge-case testing**, plus module selection, readability, and documentation. This matters for the robustness lens: the accuracy-optimised prompt *itself* now instructs the model to guard against exactly the boundary/invalid inputs that EvalPlus's Plus tests probe — so "does that instruction actually close the robustness gap?" is a live, non-trivial question, and one the paper (which reports Plus pass@1 only as a higher-rigor accuracy number, never as degradation-under-stress) structurally cannot answer. The answer here is **no**.

## The confound-free comparison — `P*` vs `S(0)` (same stack)

`S(0)` (unoptimised seed) and `P*` (optimised) were both generated on the **same Katana/vLLM stack, same bf16 weights, same temperature 0.2**, differing in exactly one variable — the system prompt. So this delta is attributable to Prochemy's optimisation alone. Plus pass@1 counts a task as passing only when **both** its base and plus statuses are `pass` (recomputed from `_eval_results.json`, verified against native evalplus's printed summary).

| Run | Base | Plus | Gap (Base−Plus) | Ratio (Plus/Base) |
|---|---|---|---|---|
| **`S(0)` seed** (unoptimised) | 0.805 | 0.768 | 0.037 | 0.955 |
| **`P*`** (optimised) | 0.774 | 0.726 | 0.048 | 0.937 |
| **Δ (`P*` − `S(0)`)** | **−3.1pp** | **−4.2pp** | **+0.011 (worse)** | **−0.018 (worse)** |

**`P*` regressed on every axis** — lower Base, lower Plus, a *wider* robustness Gap, and a worse Ratio. Prompt optimisation here did not merely fail to close the robustness gap; it made both accuracy and robustness worse than the prompt it started from.

## Stability check — the search is deterministic

To test whether that `P*` was an unlucky draw from a noisy temperature-0.2 search, two further **independent** full searches were run (fresh `--reoptimize`, identical settings). Result: **all three searches converged to the byte-identical `P*` prompt** (same text, same selected id 87, same training accuracy 0.55), and the two stability re-runs were byte-for-byte identical to each other (same `P*`, same test-set samples). Re-evaluating that same `P*` gave Plus **0.756** (vs the original run's 0.726) — a small test-set decoding difference across serving instances, but **still below `S(0)`'s 0.768**.

| `P*` evaluation | Base | Plus | Gap | vs `S(0)` Plus |
|---|---|---|---|---|
| original search | 0.774 | 0.726 | 0.048 | −4.2pp |
| stability searches ×2 (byte-identical) | 0.793 | 0.756 | 0.037 | −1.2pp |

So the regression is **reproducible, not a single-draw artefact**: the search deterministically produces this `P*`, and it underperforms `S(0)` on every evaluation. One honest nuance — the *original* run's `P*` also **widened** the Gap (0.048 vs 0.037), but the stability re-runs did **not** (0.037, level with `S(0)`); so *"widens the robustness gap"* is sensitive to test-set decoding, whereas *"does not beat `S(0)`"* is robust across all three evaluations.

## Optimisation trajectory

The search ran the **full `k_max=10`** iterations (early-stop `patience=3` never triggered — the top weighted score kept moving), ≈6h of generation on 2×H200. Notably, the final `P*` (id 87) has the **same** unweighted training accuracy as the seed (0.55) and *lower* than the mid-search peak (id 10 @ 0.65): the hill-climb's last-iteration winner is not the globally-best-scoring candidate. All three independent searches reproduced this same endpoint — so this is a stable property of the search on this setup, not noise.

## Verification performed

- **Plus pass@1 recomputed** locally from each `_eval_results.json` as `base_status==pass AND plus_status==pass`, matching native evalplus's printed `humaneval+` line: `S(0)` 126/164 = 0.768 (Base 132/164 = 0.805); `P*` 119/164 = 0.726 (Base 127/164 = 0.774).
- **Confound-free anchor**: `S(0)` and `P*` differ in exactly one variable (the system prompt) on an identical Katana/vLLM/bf16/temp-0.2 stack (`run_prochemy.py --seed-baseline` vs the `P*` run).
- **Determinism / stability**: the two stability searches' `P*` and test-set JSONLs are byte-identical (md5-matched) to each other; all three searches' `P*` prompt text is identical.
- **Faithful-seed check**: the `S(0)` used is the authors' verbatim seed text, not this project's own templates.

## Cost

- Each full `P*` search + test-set generation: ≈6h walltime on 2×H200 (bf16, TP=2), the search dominating. Three searches total (original + 2 stability). Run freely on Katana with no token cap — the same search aborted immediately on Groq (one iteration ≈230k tokens vs the 100k/day free-tier cap), which is why it was moved to Katana.

## Known limitations

- **Directional-only vs the paper's reported gains** — model *and* temperature both differ from Ye et al. (2025); no numeric reproduction is claimed. What *is* claimed is the within-setup `P*`-vs-`S(0)` delta, which is confound-free.
- **The stability check measures reproducibility, not search-seed sensitivity.** All three runs used search-`seed` 0, and the pipeline proved deterministic on a fixed vLLM instance (the two re-runs are byte-identical). So the check establishes that `P*` is *reproducible* and reproducibly below `S(0)` — it does not sample how `P*` might move under genuinely different random searches. A stricter variance study would force decoding non-determinism (e.g. per-call seeds); given the deterministic reproducibility and the consistent sub-`S(0)` performance, this was judged unnecessary for the "does not beat `S(0)`" conclusion.
- **Cross-method placement carries a vLLM-vs-Groq caveat**: comparing `P*`/`S(0)` (Katana/vLLM) against the Groq-served strategy runs would conflate the inference stack, so only the same-stack `P*`-vs-`S(0)` delta above is treated as confound-free.
- **MBPP+ Prochemy not yet run** (HumanEval+ first).
