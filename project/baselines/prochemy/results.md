# Prochemy Baseline — Results

Author: Gurdiraj Bal (z5386590)

Faithful re-implementation of Ye et al. (2025), *"Prochemy: Automating Prompt Engineering by Prompt Alchemy"* (arXiv:2503.11085), run as a controlled comparison against the thesis's own prompt-strategy / seed-generation layer (Thesis B). Prochemy is an **execution-driven automatic prompt-optimisation** method: starting from a seed prompt `S(0)`, it searches for one fixed system prompt `P*` per (model, task-type) via a mutate → weighted-evaluate → select loop over a small training set.

Key design choices for this reproduction: it is seeded from the **authors' own verbatim prompts** (not this project's templates), so it reproduces the published method rather than a hybrid; all code generation runs at temperature 0.2 with the prompt-mutation step at 1.0 (so any comparison to the paper's reported gains is **directional**, not a numeric reproduction — the model and temperature both differ from theirs); and the search runs on **Katana / vLLM** rather than Groq, because a full search is ~1M+ tokens, far above Groq's free-tier daily cap.

**Scope:** **HumanEval+ (164 tasks)**, seeded from the authors' verbatim zero-shot `S(0)`, optimised on their shipped 20-task training set (`k_max=10`, `n_variants=10`, `patience=3`), then `P*` frozen and used to generate one sample per test task. Run on **Katana / vLLM** serving **bf16 `meta-llama/Llama-3.3-70B-Instruct`** under the served-name `llama-3.3-70b-versatile` (same weights as the Groq strategy runs; a *different inference stack* — see caveat). All code generation at temperature 0.2; single run, not repeated to bound variance.

> **Status (interim).** The `P*` run and its grading below are **final**. The like-for-like **`P*` vs `S(0)` internal comparison — the headline number — is PENDING**: it requires generating the unoptimised `S(0)` seed on the *same* Katana/vLLM stack (`run_prochemy.py --seed-baseline`), which is queued on Katana. Until it lands, the only comparison available is against the existing **Groq-served** strategy runs, which is **doubly confounded** (different inference stack *and* different seed prompt) and is reported below as provisional only. MBPP+ Prochemy is not yet run.

## What `P*` converged to

The search departed from the authors' terse zero-shot seed —

> *"You are a code generation assistant. Your task is to generate Python code based on the given task description and complete the work described in the task."*

— and converged (prompt id 87) on a structured 7-step instruction that explicitly asks for **error handling, input validation, and edge-case testing** (steps 5–6), plus module selection, readability, and documentation. This matters for the robustness lens below: the accuracy-optimised prompt *itself* now instructs the model to guard against exactly the boundary/invalid inputs that EvalPlus's Plus tests probe — so "does that instruction actually close the robustness gap?" is a live, non-trivial question, and one the paper (which reports Plus pass@1 only as a higher-rigor accuracy number, never as degradation-under-stress) structurally cannot answer.

## Results — `P*` on HumanEval+ (final)

Plus pass@1 computed by counting a task as passing only when **both** its base and plus statuses are `pass`, read directly from `_eval_results.json`, and verified to match native evalplus's printed summary (Base 0.774, Plus 0.726).

| Run | Base pass@1 | Plus pass@1 | Gap (Base−Plus) | Ratio (Plus/Base) |
|---|---|---|---|---|
| **Prochemy `P*`** (164 tasks) | **0.774** | **0.726** | **0.048** | **0.937** |

## Provisional placement vs the existing strategy runs (confounded — read with care)

⚠️ **Every row except `P*` is Groq-served and uses this project's own prompt templates; `P*` is Katana/vLLM-served from the authors' seed lineage. Differences below therefore conflate three things — the prompt, the seed lineage, and the inference stack — and cannot be attributed to Prochemy alone until the same-stack `S(0)` anchor is run.**

| Run (source) | Base | Plus | Gap | Ratio |
|---|---|---|---|---|
| zero_shot (Groq) | 0.677 | 0.628 | 0.049 | 0.928 |
| baseline (Groq) | 0.726 | 0.671 | 0.055 | 0.924 |
| cot (Groq) | 0.677 | 0.634 | 0.043 | 0.937 |
| cgo / R0 seed (Groq) | 0.805 | 0.738 | 0.067 | 0.917 |
| **Prochemy `P*`** (Katana/vLLM) | **0.774** | **0.726** | **0.048** | **0.937** |

**Provisional observation (to be confirmed against the clean `S(0)` anchor).** Against the Groq `zero_shot` template, `P*` is ~+9.7pp Base / ~+9.8pp Plus — but Base and Plus rise **in near-lockstep**, so the **Gap is essentially unchanged** (0.049 → 0.048) and the Ratio barely moves (0.928 → 0.937). On this (confounded) view, Prochemy's accuracy-optimised prompt buys accuracy **without closing the robustness gap** — the degradation-under-stress is carried along, not reduced. This is exactly the thesis-relevant question this baseline was built to probe; the pending `S(0)`-on-Katana run is what will let it be stated cleanly rather than provisionally.

## Optimisation trajectory

The search ran the **full `k_max=10`** iterations (early-stop `patience=3` never triggered — the top weighted score kept moving), ≈5h40m of generation on 2×H200.

| Iteration | Selected prompt id | Training acc (unweighted) | Top weighted score |
|---|---|---|---|
| 0 (seed `S(0)`) | 0 | 0.55 | 121.0 |
| 1–6 | 10 | 0.65 | 337.9 → 398.0 (non-monotonic) |
| 7 | 61 | 0.60 | 319.0 |
| 8 | 77 | 0.60 | 261.5 |
| 9 (`P*`) | **87** | 0.55 | 249.8 |

Two honest notes for the write-up:
- **The temp-0.2 search is noisy**: the weighted score is non-monotonic across iterations and the final `P*` (id 87) has the *same* unweighted training accuracy as the seed (0.55) and lower than the mid-search peak (id 10, 0.65). `P*` is the last iteration's selection — faithful to Prochemy's hill-climb design (each round's winner seeds the next), not the global-best-weighted candidate. `P*` should therefore be read as *a* sample of Prochemy's output, not *the* Prochemy prompt, which is what motivates the stability check below.
- Despite the flat training-accuracy endpoint, `P*` generalised well to the **test** set (Base 0.774 / Plus 0.726), i.e. training-set fitness at `n=20` is a weak proxy for test performance here — unsurprising at this training-set size.

## Verification performed

- **Plus pass@1 recomputed** locally from `_eval_results.json` as `base_status==pass AND plus_status==pass` — 119/164 = 0.726, matching native evalplus's own printed `humaneval+` line exactly; Base 127/164 = 0.774.
- **`P*` provenance**: the committed `_best_prompt.json` records `p_star_prompt_id=87`, `n_iterations=10`, seeded from `zero_shot`, model `llama-3.3-70b-versatile` — consistent with the trajectory in `_optimization.json`.
- **Faithful-seed check**: the `S(0)` used is the authors' verbatim `origin_prompt.jsonl` text, not this project's `zero_shot` template.

## Cost

- One full `P*` search + test-set generation: ≈6h walltime on 2×H200 (bf16, TP=2), the search dominating (~5h40m). Run freely on Katana with no token cap — the same search aborted immediately on Groq (one iteration ≈230k tokens vs the 100k/day free-tier cap), which is why it was moved to Katana.

## Known limitations / pending

- **Headline `P*` vs `S(0)` comparison is not yet clean** — pending the same-stack `S(0)` seed run (queued on Katana). The Groq placement above is provisional and confounded.
- **Comparability caveat**: vLLM is a different inference stack than the Groq-served strategy runs; exact token outputs can differ on the same weights. The `P*`-vs-`S(0)` internal delta (both Katana/vLLM) will be confound-free; only the cross-method placement carries this caveat.
- **Single run at t=0.2** — the search is non-deterministic; a re-run could converge to a different `P*`. A 2–3× stability check (using the batch driver's run-index option) to bound how much `P*` and its scores move is owed.
- **MBPP+ Prochemy not yet run** (HumanEval+ first, by request).
- Directional-only vs the paper's reported gains (model *and* temperature both differ from Ye et al.) — no numeric reproduction is claimed.
