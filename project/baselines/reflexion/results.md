# Reflexion Baseline — Results

Author: Gurdiraj Bal (z5386590)

Faithful re-implementation of Shinn et al. (2023), *"Reflexion: Language Agents with Verbal Reinforcement Learning"* (arXiv:2303.11366), §4.3 (Programming), run as a controlled comparison against the thesis's own `cot`/`minimal` repair strategies. See `decisions.md` D13 for the full design rationale and `research_log.md` 2026-07-09 (third and fourth entries) for the session-by-session build and run log.

Scope: HumanEval+ only, 164 tasks, `max_repair_rounds=2`, same shared Round-0 seed (`samples/llama-33-70b-versatile_t02_cgo.jsonl`) and `max_repair_rounds` as the existing `cot`/`minimal` runs. Single run at `t02` (temperature 0.2) — not repeated to quantify pass@1 variance under re-sampling (see Limitations).

## Headline finding — lead with the mechanism, not the point estimate

**The strongest, most trustworthy evidence from this run is the self-test-vs-ground-truth confusion matrix**, aggregated across all 253 repair rounds attempted — a large enough sample to stand on its own regardless of how any individual task happened to resolve:

| | Count | Rate |
|---|---|---|
| TP (both pass) | 60 | 23.7% |
| FN (self-test fails, ground truth passes) | 132 | **52.2%** |
| FP (self-test passes, ground truth fails) | 2 | 0.8% |
| TN (both fail) | 59 | 23.3% |

Over half of all repair attempts were the agent "fixing" code that was already correct, based on its own unreliable self-generated tests. FP stayed rare (2 tasks total), matching the pattern the paper itself reports — false positives are typically much less common than false negatives. A 52% FN rate is in a plausible range given Shinn et al.'s own Table 2 (40% for HumanEval / 59% for MBPP, both on GPT-4) — this reads as a legitimate finding about self-generated-test reliability specific to Llama-3.3-70B, plausibly connected to the paper's own Appendix A framing of reliable self-correction as an emergent capability of stronger/larger models (their own ablation shows a weak model, starchat-beta, getting zero benefit from Reflexion at all). **This mechanism, not the exact pass@1 delta below, is the primary evidence for this baseline's write-up.**

### Worked example: `HumanEval/4`

From an earlier 5-task smoke test, before the full run — a clean illustration of the FN-driven regression pattern that played out at scale in the full 164-task run:

- **Round 0**: both ground-truth and self-test fail (`IndentationError`). Reflection correctly diagnoses the indentation issue.
- **Round 1**: the fix actually works (ground-truth `PASS`), but the self-generated test suite has a bug of its own and reports `FAIL` (`UnknownError`) — a false negative. Since the loop only trusts the self-test signal (never sees ground truth, matching the paper's design), it believes it's still broken and continues.
- **Round 2**: the reflection invents a plausible-but-wrong diagnosis ("didn't handle empty list") based on the bad signal, "fixes" a solution that was already correct, and breaks it (`NameError`, ground-truth flips back to `FAIL`).

## Corroborating evidence — pass@1 (single-run point estimate, treat as directional)

All Plus pass@1 values computed as `base_status==pass AND plus_status==pass` directly from each run's `_eval_results.json` — verified to match Docker's own printed EvalPlus summary exactly on both fresh round1/round2 evaluations (see D14), and confirmed applied identically (same function, same condition, no exceptions) to the `cot`/`minimal` rows below as well as `reflexion`, not just reflexion — see Verification.

| Run | Base pass@1 | Plus pass@1 | Robustness Gap | Robustness Ratio |
|---|---|---|---|---|
| R0 (shared seed) | 0.805 | 0.738 | 0.067 | 0.917 |
| cot R2 | 0.915 | 0.829 | 0.085 | 0.907 |
| minimal R2 | 0.854 | 0.774 | 0.079 | 0.907 |
| reflexion R1 | 0.823 | 0.744 | 0.079 | 0.904 |
| **reflexion R2** | **0.799** | **0.713** | 0.085 | 0.893 |

**Reflexion is the only one of the three repair strategies whose Round-2 point estimate lands below its own Round-0 starting point.** R1 improves on R0 (+1.8pp base, +0.6pp plus), but R2 erases that gain and then some (−0.6pp base, −2.5pp plus relative to R0), while `cot` (+11.0pp base) and `minimal` (+4.9pp base) both net-improve over the identical two rounds.

**Caveat — this is a single 164-task run at t=0.2, not a repeated-sampling estimate.** A handful of flipped tasks (a few percentage points either way) would meaningfully move the exact R1→R2 delta, and Groq's daily quota was hit twice during this run (see Verification), so a second independent run to bound the variance was not attempted given the cost already incurred. The regression *direction* is corroborated by the FN-rate mechanism above (a large, robust sample) and the worked example, which is why the finding is reported with confidence — but the precise pass@1 numbers in this table should be read as one realization, not a tight point estimate. Re-running would be the natural next step if this baseline needs a tighter number for publication rather than a directional finding.

Note: an internal, in-script REPAIR SUMMARY computed during the run (against the narrower canonical HumanEval test used by `code_executor.run_executor` during the loop, not the full EvalPlus base-test definition) reported R0 80.5% → R2 85.4% (+4.9pp) — this number is superseded by the Docker-verified table above and should not be used for comparison; it's noted here only because it was reported provisionally mid-session before the Docker evaluation ran.

## Verification performed before treating this as final

- **Formula consistency across strategies**: re-ran the `base_status==pass AND plus_status==pass` scoring function against `cot` R2, `minimal` R2, and both `reflexion` rounds' `_eval_results.json` files directly (not against any previously-reported/older numbers) — same function, same condition, applied without exception to all five rows in the table above. The R2 comparisons are apples-to-apples.
- **`reflection_history` persistence**: confirmed present as a field on all 164 saved trajectories; populated with real, task-specific, first-person, code-free critique text on the 49 tasks that triggered at least one repair round (89 reflection strings total). This is the artifact to draw on later for comparing what Reflexion's unguided self-reflections focus on against what this thesis's own grounded-refinement method targets.
- **Resume/decoding-state independence**: confirmed `generate_raw_completion()` never passes a `seed` parameter to the API (only `model`, `messages`, `temperature`, `max_tokens`) — no run in this codebase has ever had decoding determinism, so resuming introduces no new inconsistency beyond what already exists in any single uninterrupted run. Also confirmed each task's `memory`/`reflection_history`/`current_completion` state is a fresh local variable scoped to that task's own `run_reflexion_repair()` call, never shared across tasks or persisted in-process — the only state that survives a resume is the trajectory JSON on disk, which is the intended checkpoint mechanism, not hidden decoding state.

## Cost

404 total LLM API calls across all 164 tasks (1 test-generation call + up to 2×(self-reflection + actor) per task) — roughly 23% more than `cot`'s ~328 calls for the same 164 tasks, for a net-worse Round-2 outcome. Hit Groq's daily token quota twice during the run (at 107/164 and 132/164 tasks); resumed cleanly both times via `--resume-missing` with zero wasted spend or duplicate work.

## Known limitations of this baseline

- HumanEval+ only — MBPP+ is not covered (tracked as a follow-up before this baseline is comparable across datasets to `cot`/`minimal`).
- Single run at `t02` — the pass@1 point estimates above are directional, not variance-bounded (see caveat above).
- The confusion matrix here is self-test vs. the *visible/canonical* test suite (what the repair loop itself had access to), not vs. the full EvalPlus Plus test suite — a stricter version of this diagnostic against Plus results is a possible follow-up.
