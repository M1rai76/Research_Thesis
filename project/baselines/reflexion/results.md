# Reflexion Baseline — Results

Author: Gurdiraj Bal (z5386590)

Faithful re-implementation of Shinn et al. (2023), *"Reflexion: Language Agents with Verbal Reinforcement Learning"* (arXiv:2303.11366), §4.3 (Programming), run as a controlled comparison against the thesis's own `cot`/`minimal` repair strategies.

**Scope:** both **HumanEval+ (164 tasks)** and **MBPP+ (378 tasks)**, `max_repair_rounds=2`, each repairing its own shared `cgo` Round-0 seed (`[mbpp_]llama-33-70b-versatile_t02_cgo.jsonl`) — the same seed and round budget as the existing `cot`/`minimal` runs. Both datasets run on **Groq `llama-3.3-70b-versatile`**, i.e. the identical inference setup to `cot`/`minimal` (no inference-stack confound). Single run per dataset at `t02` (temperature 0.2) — not repeated to bound pass@1 variance (see Limitations).

## Headline finding — lead with the mechanism, not the point estimate

**The strongest, most trustworthy evidence is the self-test-vs-ground-truth confusion matrix**, aggregated across every repair round attempted — a large enough sample on each dataset to stand on its own regardless of how any individual task happened to resolve:

| Dataset (rounds) | TP (both pass) | FN (self-test fails, truth passes) | FP (self-test passes, truth fails) | TN (both fail) |
|---|---|---|---|---|
| HumanEval+ (253) | 23.7% | **52.2%** | 0.8% | 23.3% |
| MBPP+ (741) | 30.2% | **49.4%** | 1.8% | 18.6% |

On **both** datasets, roughly **half of all repair attempts were the agent "fixing" code that was already correct**, driven by its own unreliable self-generated tests. False positives stayed rare on both (0.8% / 1.8%), matching the pattern the paper itself reports. The ~50% FN rate is in the range of Shinn et al.'s own Table 2 (40% HumanEval / 59% MBPP, both on GPT-4) — this reads as a legitimate, model-specific finding about self-generated-test reliability on Llama-3.3-70B, plausibly connected to the paper's Appendix A framing of reliable self-correction as an emergent capability of stronger/larger models (their ablation shows a weak model, starchat-beta, getting zero benefit from Reflexion at all). **This mechanism — not the exact pass@1 deltas below — is the primary evidence for this baseline, and it reproduces almost identically across two independent benchmarks (n=253 and n=741).**

### Worked example: `HumanEval/4`

From an earlier 5-task smoke test — a clean illustration of the FN-driven pattern that played out at scale on both datasets:

- **Round 0**: both ground-truth and self-test fail (`IndentationError`). Reflection correctly diagnoses the indentation issue.
- **Round 1**: the fix actually works (ground-truth `PASS`), but the self-generated test suite has a bug of its own and reports `FAIL` (`UnknownError`) — a false negative. Since the loop only trusts the self-test signal (never sees ground truth, matching the paper's design), it believes it's still broken and continues.
- **Round 2**: the reflection invents a plausible-but-wrong diagnosis ("didn't handle empty list") based on the bad signal, "fixes" a solution that was already correct, and breaks it (`NameError`, ground-truth flips back to `FAIL`).

The MBPP+ smoke test showed the same shape (e.g. `Mbpp/6`: ground-truth `PASS` every round, self-test `FAIL` every round — a sustained false negative).

## Corroborating evidence — pass@1 (single-run point estimates, treat as directional)

All Plus pass@1 values computed as `base_status==pass AND plus_status==pass` directly from each run's `_eval_results.json`, applied identically — same function, same condition, no exceptions — to the `cot`/`minimal` rows as well as `reflexion` on both datasets, so every comparison is apples-to-apples.

### HumanEval+ (164 tasks)

| Run | Base pass@1 | Plus pass@1 | Gap | Ratio | vs R0 (Plus) |
|---|---|---|---|---|---|
| R0 (shared seed) | 0.805 | 0.738 | 0.067 | 0.917 | — |
| cot R2 | 0.915 | 0.829 | 0.085 | 0.907 | +9.1pp |
| minimal R2 | 0.854 | 0.774 | 0.079 | 0.907 | +3.6pp |
| reflexion R1 | 0.823 | 0.744 | 0.079 | 0.904 | +0.6pp |
| **reflexion R2** | **0.799** | **0.713** | 0.085 | 0.893 | **−2.5pp** |

On HumanEval+, **Reflexion is the only one of the three repair strategies whose Round-2 result lands *below* its own Round-0 starting point.** R1 nudges up (+0.6pp Plus), but R2 erases that and more (−2.5pp Plus vs R0), while `cot` and `minimal` both net-improve over the identical two rounds.

### MBPP+ (378 tasks)

| Run | Base pass@1 | Plus pass@1 | Gap | Ratio | vs R0 (Plus) |
|---|---|---|---|---|---|
| R0 (shared seed) | 0.865 | 0.725 | 0.140 | 0.838 | — |
| cot R2 | 0.921 | 0.770 | 0.151 | 0.836 | +4.5pp |
| minimal R2 | 0.897 | 0.746 | 0.151 | 0.832 | +2.1pp |
| reflexion R1 | 0.852 | 0.696 | 0.156 | 0.817 | −2.9pp |
| **reflexion R2** | **0.865** | **0.722** | 0.143 | 0.835 | **−0.3pp (flat)** |

On MBPP+, Reflexion **dips at R1 (−2.9pp Plus) and recovers only to ~R0 by R2 (−0.3pp, flat within noise)**, while `cot` (+4.5pp Plus) and `minimal` (+2.1pp Plus) both climb clearly. So Reflexion is again the sole strategy that fails to net-improve — here it breaks even rather than regressing.

## Cross-dataset synthesis

The robust, both-datasets claim is: **Reflexion is uniquely unable to benefit from iterative repair — while `cot` and `minimal` both raise Plus pass@1 over their Round-0 seed, Reflexion does not — and the cause is the same on both benchmarks: ~50% of its self-generated-test verdicts are false negatives, so roughly half of all repair rounds are spent "fixing" code that was already correct.**

One honest nuance to carry into the write-up, so it isn't overclaimed: the *endpoint* differs in degree between datasets. On **HumanEval+**, Reflexion R2 lands **clearly below** its R0 (−2.5pp Plus) — an outright regression. On **MBPP+**, R2 comes back to **flat** vs R0 (−0.3pp) after an R1 dip — no net benefit, but not a net regression. The strict "Reflexion regresses below its own R0" statement is therefore **HumanEval-specific**; the claim that holds on both is the "uniquely fails to benefit, driven by ~50% self-test FNs" framing above. The mechanism (the confusion matrix) is the invariant; the exact pass@1 endpoint is the dataset-dependent, single-run-sensitive part.

**Caveat — single run at t=0.2 per dataset, not a repeated-sampling estimate.** A handful of flipped tasks would move the exact R1→R2 deltas, so the pass@1 numbers are directional, not variance-bounded. The FN-rate mechanism (n=253 and n=741, the far larger samples) is the primary evidence and is what the deltas corroborate — not the other way around. Note also the superseded in-script REPAIR SUMMARY (narrow canonical-test check, not full EvalPlus): it reported R0→R2 of 80.5%→85.4% (HumanEval) and 85.7%→89.9% (MBPP), both **misleading and superseded** by the Docker-verified tables above.

## Verification performed before treating this as final

- **Formula consistency across strategies and datasets**: the `base_status==pass AND plus_status==pass` scorer was applied without exception to every row of both tables (`cot`/`minimal`/`reflexion`, R0–R2), read directly from each `_eval_results.json`, not from previously-reported numbers. The HumanEval R1/R2 evals were verified to match Docker's own printed summary exactly; the MBPP R1/R2 evals were produced by the same `ganler/evalplus:latest` image this session.
- **Data-integrity guard (MBPP run)**: an `api_failed` flag was added so a quota-exhaustion (`generate_raw_completion` returning `None`) aborts the batch *without persisting* the degraded task, rather than churning through the remaining tasks saving empty-self-test trajectories that `--resume-missing` would then skip. Verified post-run: **0 `api_failed` flags** across all 378 saved trajectories, despite the run spanning multiple quota windows.
- **`reflection_history` persistence**: present on all trajectories on both datasets — 89 reflection strings across 49 repair-triggering tasks (HumanEval), 363 across 202 (MBPP). Real, task-specific, first-person, code-free critique text; the artifact to draw on for later qualitative comparison against the thesis's own grounded-refinement method.
- **Resume/decoding-state independence**: `generate_raw_completion()` never passes a `seed` (only `model`/`messages`/`temperature`/`max_tokens`), so no run in this codebase has ever had decoding determinism — resuming across quota windows introduces no inconsistency beyond what any single uninterrupted run already has. Each task's loop state is a fresh local variable, never shared across tasks or persisted in-process; the only cross-resume state is the on-disk trajectory checkpoint.

## Cost

- **HumanEval+**: 404 LLM API calls across 164 tasks (1 test-gen + up to 2×(reflect+actor) per task). Hit Groq's daily token quota twice (at 107/164 and 132/164); resumed cleanly via `--resume-missing`.
- **MBPP+**: 1,104 LLM API calls across 378 tasks (202 tasks triggered ≥1 repair round). Reflexion's two-call-per-repair-round design (separate reflect + actor) is structurally ~2× the per-round call cost of `cot`/`minimal`'s single fix call — for a net-worse (HumanEval) or net-flat (MBPP) outcome. Hit the Groq token cap repeatedly; completed over several `--resume-missing` windows (78 → 150 → 227 → 305 → 378) plus a mid-run tier upgrade, guard-clean at every boundary with zero duplicate spend.

## Known limitations of this baseline

- **Single run per dataset at `t02`** — the pass@1 point estimates are directional, not variance-bounded (see caveat above). A second run would tighten the endpoint deltas (the FN mechanism needs no re-run).
- The confusion matrix is self-test vs. the *visible/canonical* test suite (what the repair loop had access to), not vs. the full EvalPlus Plus test suite — a stricter version of this diagnostic against Plus results is a possible follow-up.
- Both runs use Groq-served `llama-3.3-70b-versatile`; the Katana/vLLM bf16 route was scaffolded but not needed here, since running on Groq keeps the setup identical to the `cot`/`minimal` runs (a strength for comparability, not a limitation).
