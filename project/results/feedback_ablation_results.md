# Feedback-Content Ablation — Results

Author: Gurdiraj Bal (z5386590)

An ablation on the iterative self-repair loop that isolates its *mechanism*: when repair improves a program, is the improvement bought by the **content of the failure feedback**, or merely by **getting another attempt**?

## Design

Five arms form a monotone information ladder. Only the failure signal shown to the model varies. Everything else is held fixed — same model, same Round 0 seed, same `minimal` repair substrate, two repair rounds, temperature 0.2, identical output constraints, and critically the **same ground-truth stopping oracle**, so every arm gets the same number of attempts and stops on the same condition.

| Arm | What the model is shown | Information |
|---|---|---|
| `grounded+` | the failing call, the value it returned, and the expected value | concrete failing case |
| `full` | the exception class, message, and failing assertion expression | exception + assertion |
| `error-type` | the exception class only | class only |
| `binary` | "This attempt failed." | one bit |
| `blind` | *nothing* — the Round 0 generation prompt is re-run | zero bits |

`blind` is a **pure resampling control**: the model is not told it failed, nor that a previous attempt exists. It simply writes the function again.

Two design notes that matter for interpreting the ladder:

- **`full` is richer than it first appears.** The executor rewrites a bare `AssertionError` into the failing assertion expression, so `full` already discloses the call *and* the expected value. The only genuinely new information `grounded+` adds is **the value actually produced**.
- **`grounded+` degrades to `full` on crashes.** When code fails with `NameError`, `IndentationError` and similar, no value is produced, so there is nothing extra to show. It is richer only on the assertion subset (57 of 81 observed HumanEval+ failures).

Runs on Groq, so that `blind` resampling is genuinely stochastic. (Under a fixed local vLLM instance, generation is deterministic given identical server state, which would make a zero-feedback resample return the same failing program every round and render the control meaningless.)

## HumanEval+ — 5 arms × 3 runs

Llama-3.3-70B, shared `cgo` Round 0 (Base 0.805 / Plus 0.738, Gap 0.067). Plus counts a task only when base and plus statuses both pass.

| Arm | Base mean | Base range | Plus mean | Gap mean | Gap vs R0 |
|---|---|---|---|---|---|
| `grounded+` | **0.870** | 0.866–0.872 | 0.783 | 0.087 | widens |
| `full` | 0.858 | 0.854–0.866 | 0.778 | 0.079 | widens |
| `error-type` | 0.850 | 0.848–0.854 | 0.774 | 0.075 | widens |
| `binary` | 0.841 | 0.835–0.848 | 0.762 | 0.079 | widens |
| **`blind`** | **0.870** | 0.860–0.878 | **0.787** | 0.083 | widens |

**The ladder is U-shaped.** Among arms that *give* feedback the ordering is perfectly monotone in signal quality (0.870 → 0.858 → 0.850 → 0.841). Then `blind`, with no feedback at all, returns to the top — tying `grounded+` on Base and leading on Plus.

Five pairs have **completely disjoint ranges across all three runs**, so extremes-beat-middle is a real separation rather than noise: `grounded+` and `blind` each above `error-type` and `binary`, and `full` above `binary`.

**Every arm widens the Gap in every run — 15 of 15.**

## MBPP+ — cross-dataset check

Same model, substrate and Round 0 lineage; 378 tasks; single seed.
Round 0: Base 0.865 / Plus 0.725, Gap 0.140.

| Arm | Base | Plus | Gap | ΔBase | ΔPlus | Gap move |
|---|---|---|---|---|---|---|
| `grounded+` | **0.902** | 0.754 | 0.148 | +3.7pp | +2.9pp | widens |
| `full` | **0.897** | 0.746 | 0.151 | +3.2pp | +2.1pp | widens |
| **`blind`** | **0.894** | 0.741 | 0.153 | +2.9pp | +1.6pp | widens |
| `error-type` | 0.873 | 0.735 | 0.138 | +0.8pp | +1.1pp | flat |
| `binary` | 0.868 | 0.725 | 0.143 | +0.3pp | +0.0pp | flat |

**The core structure replicates.** MBPP+ splits into a top cluster — `grounded+`, `full`, `blind`, all within 0.8pp — and a bottom cluster of `error-type` and `binary` some 2–3pp below. On **both** datasets `binary` is the worst arm, `grounded+` the best, and **zero feedback sits with the best**.

What differs: on HumanEval+ the shape is a strict U (`grounded+` = `blind` at the top, `full` mid); on MBPP+ `full` rises into the top cluster, so the pattern is better described as **rich-or-nothing ≫ partial**. The essential claim is unchanged, but the within-cluster ordering is not stable across datasets and should not be reported as though it were.

## Why the mechanism appears to be anchoring

Behavioural diagnostics from the HumanEval+ trajectories. All arms entered repair on the identical 32 tasks — confirming the shared seed and shared stopping oracle worked as a control.

| Arm | solved | stalled | repair rounds that changed the code |
|---|---|---|---|
| `grounded+` | 11 | 3 | **85.2%** |
| `full` | 8 | 5 | 78.9% |
| `error-type` | 8 | 4 | 82.5% |
| `binary` | 7 | 5 | 78.3% |
| `blind` | 10 | 12 | **47.4%** |

The feedback arms churn the code far harder — 78–85% of repair rounds produce a different program — while `blind` changes it under half the time, yet matches the best arm.

The reading this supports: **being told "this failed" anchors the model to its own broken program.** It patches rather than reconsiders. Degraded feedback anchors without informing — the worst of both, and `binary` is measurably the worst arm. Sufficiently concrete evidence overcomes the anchor. And no feedback sidesteps it entirely, because the model simply writes a fresh solution.

## What this says about robustness

The Gap result is the most heavily replicated finding here, and it survives the cross-dataset move in a refined form.

On HumanEval+ all five arms widened the Gap. On MBPP+ only three did — but the two that did not are exactly the two that **barely improved at all** (+0.8pp and +0.3pp Base). Every arm that meaningfully improved accuracy widened the Gap. So:

> **No arm, on either dataset, gained accuracy while closing the robustness Gap.** Across all ten arm×dataset cells there is not one counter-example. Accuracy gains arrive bundled with a wider Gap; where there is no accuracy gain, the Gap does not move.

This is a dose-response statement rather than a blanket one, and it is the better-supported form: it *explains* the two flat cells instead of treating them as exceptions.

**An honest wrinkle.** On HumanEval+, `grounded+` had the widest Gap — its extra correctness was the shallowest. On MBPP+ the reverse holds: `blind` has the widest Gap (0.153) while `grounded+` is narrowest among the improving arms (0.148). So *which* arm produces the shallowest gains does **not** replicate. Only the fact that all improving arms produce shallow gains does.

## Claim strengths

- **Robust (both datasets):** `binary` and `error-type` are the worst arms; `blind` performs at least as well as every degraded-feedback arm; no arm gains accuracy while closing the Gap.
- **HumanEval+-specific:** the strict U-shape with `blind` exactly tying `grounded+`; `grounded+` holding the widest Gap; all five arms widening.
- **Directional only:** MBPP+ is single-seed, so within-cluster ordering there is not variance-bounded. The HumanEval+ arm separations *are* — three runs with disjoint ranges.

## Limitations

- One model (Llama-3.3-70B) and one repair substrate (`minimal`), two rounds.
- `blind` resamples at temperature 0.2, deliberately a **lower bound** on what resampling can achieve — a hotter resample would explore more. That the lower bound already matches the best feedback arm is what makes the result strong; it also means the gap between resampling and feedback could only widen further in resampling's favour.
- MBPP+ is a single seed. The HumanEval+ ladder is variance-bounded at n=3; the MBPP+ ladder is not.
- The stopping oracle is ground truth throughout, so every arm is given the same *opportunity* to repair. This isolates feedback content but does not model a deployment setting where no ground truth exists.

## Reproducing

```bash
# one arm; --feedback_mode selects the rung, --run_tag separates repeat runs
python generate_samples.py --backend groq --model llama-3.3-70b-versatile \
    --dataset humaneval --prompt cop --repair --repair_strategy minimal \
    --feedback_mode grounded+ --max_repair_rounds 2 \
    --round0_source_jsonl ../samples/llama-33-70b-versatile_t02_cgo.jsonl

# collate all arms, with mean and range across repeat runs
python ablation_report.py --dataset humaneval --run_tags run2,run3
```

The `full` arm needs no run: the pre-existing `minimal` repair run *is* full feedback, and is reused unchanged.
