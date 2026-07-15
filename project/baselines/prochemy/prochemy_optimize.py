"""
prochemy_optimize.py
Baseline - Prochemy (Ye et al., 2025, arXiv:2503.11085)
Author : Gurdiraj Bal (z5386590)

The Prochemy optimisation loop (paper Sec. III-C; authors'
2+3+4_reinfocement.py): iteratively Mutate -> Evaluate (weighted) -> Select a
system prompt against the training set, until early-stop or k_max.

    a) Mutation   - generate n=10 prompt variants from the current selected
                    pool (temperature 1.0, the paper's diversity mechanism).
    b) Evaluation - run each variant over every training task, execute the
                    generated code, score M_ij in {0,1} (pass@1). Solution
                    generation runs at this project's temperature 0.2 (the fixed
                    project-wide setting), overriding the paper's 0.
    c) Selection  - weighted score WS(P) = sum_j w_j * M_ij, where
                    w_j = total_correct / N_successful(T_j) (authors'
                    3_reinforcement_cal_score_and_select.py). Pick argmax.

Convergence: early-stop when the top weighted score is unchanged for `patience`
(=3) consecutive iterations, or k >= k_max (=10). Output: P* + a full trajectory
(per-iteration pool, scores, selected prompt) written incrementally so a
quota-interrupted run can be inspected/resumed.

Note: running the search at 0.2 rather than the paper's 0 makes the fitness
signal noisier and P* non-deterministic across reruns - accepted for
intra-project consistency; the --run-index stability check is the mitigation.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_samples import generate_raw_completion  # noqa: E402

from prochemy_prompt import (  # noqa: E402
    MUTATION_SYSTEM,
    build_mutation_user,
    extract_mutated_prompt,
    extract_python_block,
)
from prochemy_trainset import run_training_task  # noqa: E402

# Prochemy hyper-parameters (paper Sec. IV-D). n_variants and k_max are the
# method's own structural constants (like Reflexion's memory size) - kept at the
# paper's values, not this project's.
DEFAULT_K_MAX = 10
DEFAULT_N_VARIANTS = 10
DEFAULT_PATIENCE = 3

MUTATION_TEMPERATURE = 1.0      # paper's diversity mechanism - NOT overridden
MUTATION_MAX_TOKENS = 500       # authors' 1_prompt_mutate.py
SOLUTION_MAX_TOKENS = 512       # this project's fixed MAX_TOKENS ("match ours")
MUTATION_MAX_RETRIES = 5        # retries to obtain a {{...}}-wrapped mutation


@dataclass
class PromptCandidate:
    """A single prompt variant and its evaluation on the training set."""
    prompt_id: int
    text: str
    per_task: Dict[str, bool] = field(default_factory=dict)  # task_id -> passed
    original_score: float = 0.0      # unweighted pass@1 on the training set
    weighted_score: float = 0.0


# --------------------------------------------------------------------------- #
# a) Mutation                                                                  #
# --------------------------------------------------------------------------- #
def mutate_prompts(
    client,
    model: str,
    pool: List[str],
    n_variants: int,
    next_id: int,
    rng,
    sleep_s: float = 0.0,
) -> List[PromptCandidate]:
    """Generate `n_variants` new prompt variants, each mutated from a randomly
    chosen prompt in `pool` (authors' generate_new_prompts). Mutation runs at
    temperature 1.0. Returns candidates with sequential ids from next_id.
    """
    variants: List[PromptCandidate] = []
    for i in range(n_variants):
        seed_text = rng.choice(pool)
        user_msg = build_mutation_user(seed_text)
        mutated: Optional[str] = None
        for _ in range(MUTATION_MAX_RETRIES):
            raw = generate_raw_completion(
                client,
                model,
                prompt=user_msg,
                system_prompt=MUTATION_SYSTEM,
                temperature=MUTATION_TEMPERATURE,
                max_tokens=MUTATION_MAX_TOKENS,
            )
            if sleep_s:
                time.sleep(sleep_s)
            if raw is None:
                # API failure (e.g. quota) - propagate as None so the driver aborts
                # without persisting a degraded run (mirrors the reflexion api_failed guard).
                return None
            mutated = extract_mutated_prompt(raw)
            if mutated:
                break
        if not mutated:
            # Could not obtain a wrapped prompt; fall back to the seed unchanged.
            mutated = seed_text
        variants.append(PromptCandidate(prompt_id=next_id + i, text=mutated))
    return variants


# --------------------------------------------------------------------------- #
# b) Evaluation                                                                #
# --------------------------------------------------------------------------- #
def evaluate_candidate(
    client,
    model: str,
    candidate: PromptCandidate,
    trainset: List[Dict],
    solution_temperature: float,
    sleep_s: float = 0.0,
) -> Optional[PromptCandidate]:
    """Run one prompt variant over every training task and record pass/fail.

    The candidate prompt is the *system* message; each task's `prompt` (function
    signature + docstring) is the user message. Solution generation uses
    `solution_temperature` (0.2). Returns the candidate with per_task filled,
    or None if the API failed (so the driver can abort cleanly).
    """
    for task in trainset:
        raw = generate_raw_completion(
            client,
            model,
            prompt=task["prompt"],
            system_prompt=candidate.text,
            temperature=solution_temperature,
            max_tokens=SOLUTION_MAX_TOKENS,
        )
        if sleep_s:
            time.sleep(sleep_s)
        if raw is None:
            return None
        code = extract_python_block(raw)
        passed = run_training_task(code, task) if code else False
        candidate.per_task[task["task_id"]] = passed
    candidate.original_score = (
        sum(candidate.per_task.values()) / len(trainset) if trainset else 0.0
    )
    return candidate


# --------------------------------------------------------------------------- #
# c) Selection (weighted scoring)                                              #
# --------------------------------------------------------------------------- #
def compute_weighted_scores(candidates: List[PromptCandidate]) -> None:
    """Fill each candidate's weighted_score in place (authors'
    3_reinforcement_cal_score_and_select.py).

        task_correct_counts[t] = # candidates that solved t
        total_correct          = sum over t of task_correct_counts[t]
        w_t                    = total_correct / task_correct_counts[t]
        WS(P)                  = sum over solved-t of w_t

    The numerator (total_correct) is constant across tasks within an iteration,
    so it does not change the argmax; it matches the reference implementation
    exactly. Tasks solved by no candidate get zero weight (never contribute).
    """
    task_correct_counts: Dict[str, int] = {}
    for cand in candidates:
        for task_id, passed in cand.per_task.items():
            if passed:
                task_correct_counts[task_id] = task_correct_counts.get(task_id, 0) + 1

    total_correct = sum(task_correct_counts.values())
    task_weights = {
        task_id: (total_correct / count) if count else 0.0
        for task_id, count in task_correct_counts.items()
    }

    for cand in candidates:
        cand.weighted_score = sum(
            task_weights.get(task_id, 0.0)
            for task_id, passed in cand.per_task.items()
            if passed
        )


def select_best(candidates: List[PromptCandidate]) -> List[PromptCandidate]:
    """Return all candidates tied for the highest weighted score (authors keep
    ties and carry them all forward as the next mutation pool)."""
    if not candidates:
        return []
    top = max(c.weighted_score for c in candidates)
    return [c for c in candidates if c.weighted_score == top]


# --------------------------------------------------------------------------- #
# Full optimisation loop                                                       #
# --------------------------------------------------------------------------- #
def run_optimization(
    client,
    model: str,
    initial_prompt: str,
    trainset: List[Dict],
    solution_temperature: float = 0.2,
    k_max: int = DEFAULT_K_MAX,
    n_variants: int = DEFAULT_N_VARIANTS,
    patience: int = DEFAULT_PATIENCE,
    seed: int = 0,
    sleep_s: float = 0.0,
    trajectory_path: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Optional[Dict]:
    """Run the Prochemy search and return {p_star, trajectory, ...}.

    Iteration k:
        pool  -> mutate -> n_variants new candidates
        candidates = selected(pool from k-1) + new variants   (k>=1)
        evaluate all candidates on the training set (weighted)
        select argmax(weighted) -> selected pool for k+1
    Early-stop when the top weighted score is unchanged for `patience`
    consecutive iterations, else stop at k_max.

    Returns None if the API failed mid-search (driver should not persist a
    degraded P*). Writes the trajectory incrementally to trajectory_path.
    """
    import random

    rng = random.Random(seed)

    # Iteration 0 evaluates the seed prompt alone, then mutates from it.
    selected: List[PromptCandidate] = [PromptCandidate(prompt_id=0, text=initial_prompt)]
    next_id = 1
    prev_top: Optional[float] = None
    stable = 0
    trajectory: List[Dict] = []

    def _persist():
        if trajectory_path:
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"model": model, "solution_temperature": solution_temperature,
                     "k_max": k_max, "n_variants": n_variants, "patience": patience,
                     "seed": seed, "iterations": trajectory},
                    f, indent=2,
                )

    for k in range(k_max):
        t0 = time.time()
        # --- a) Mutation (skipped on the very first evaluation of the seed) ---
        if k == 0:
            candidates = list(selected)  # evaluate the seed by itself first
        else:
            new_variants = mutate_prompts(
                client, model, [c.text for c in selected], n_variants, next_id, rng,
                sleep_s=sleep_s,
            )
            if new_variants is None:
                log("  [abort] API failure during mutation - not persisting P*.")
                return None
            next_id += n_variants
            # Carry the selected pool forward alongside the fresh variants.
            candidates = list(selected) + new_variants

        # --- b) Evaluation ---
        for cand in candidates:
            # Re-evaluating a carried-over candidate is cheap insurance against
            # decoding drift, but to save quota we only evaluate those not yet scored.
            if cand.per_task:
                continue
            scored = evaluate_candidate(
                client, model, cand, trainset, solution_temperature, sleep_s=sleep_s
            )
            if scored is None:
                log("  [abort] API failure during evaluation - not persisting P*.")
                return None

        # --- c) Selection ---
        compute_weighted_scores(candidates)
        selected = select_best(candidates)
        top = selected[0].weighted_score

        trajectory.append({
            "iteration": k,
            "n_candidates": len(candidates),
            "top_weighted_score": top,
            "selected_prompt_ids": [c.prompt_id for c in selected],
            "selected_prompt": selected[0].text,
            "candidates": [
                {"prompt_id": c.prompt_id, "original_score": c.original_score,
                 "weighted_score": c.weighted_score, "text": c.text}
                for c in candidates
            ],
            "seconds": round(time.time() - t0, 1),
        })
        _persist()
        log(f"  iter {k}: {len(candidates)} candidates, "
            f"top weighted={top:.3f}, best original={max(c.original_score for c in candidates):.3f}")

        # --- convergence ---
        if prev_top is not None and top == prev_top:
            stable += 1
        else:
            stable = 1
        prev_top = top
        if stable >= patience and k >= patience - 1:
            log(f"  early stop: top score stable for {stable} iterations.")
            break

    # Tie-break a final multi-way selection at random (authors' behaviour).
    p_star = rng.choice(selected) if len(selected) > 1 else selected[0]
    return {
        "p_star": p_star.text,
        "p_star_prompt_id": p_star.prompt_id,
        "p_star_weighted_score": p_star.weighted_score,
        "p_star_original_score": p_star.original_score,
        "n_iterations": len(trajectory),
        "trajectory": trajectory,
    }


__all__ = [
    "PromptCandidate",
    "mutate_prompts",
    "evaluate_candidate",
    "compute_weighted_scores",
    "select_best",
    "run_optimization",
]
