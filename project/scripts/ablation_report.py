"""
ablation_report.py
Thesis - Prompt Engineering for LLM Code Generation
Author : Gurdiraj Bal (z5386590)

Collate the feedback-content ablation (decisions.md D24) into one table.

The ablation varies only the failure signal shown to the model across a
monotone information ladder - full (exception + message), error-type
(class only), binary (one bit), blind (nothing; the Round 0 prompt is
re-run) - holding the model, the Round 0 seed, the repair substrate, the
round budget, the temperature and the ground-truth stopping oracle fixed.

This script reads each arm's per-round `_eval_results.json` and reports
Base / Plus / Gap / Ratio, so the question the ablation exists to answer -
does feedback *content* buy robustness, or only accuracy? - is read off
the Gap column rather than the Base column.

Plus pass@1 follows decisions.md D14: a task counts as passing only when
its base and plus statuses are BOTH `pass`.

Usage
    python ablation_report.py --dataset humaneval \
        --model llama-3.3-70b-versatile --prompt cop --repair_strategy minimal
"""

import argparse
import json
import os
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# Ladder order: most informative feedback first, so the table reads as a
# monotone descent in information.
ARMS = ("grounded+", "full", "error-type", "binary", "blind")


def model_slug(model: str) -> str:
    """Mirror generate_samples.model_to_slug for filename reconstruction."""
    return model.replace(".", "").replace(":", "_").replace("/", "-")


def score_eval_results(path: str) -> Optional[dict]:
    """Compute Base/Plus/Gap/Ratio from an EvalPlus results file (D14)."""
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        evaluations = json.load(f)["eval"]

    total = len(evaluations)
    if total == 0:
        return None

    base_passed = 0
    plus_passed = 0
    for record in evaluations.values():
        entry = record[0] if isinstance(record, list) else record
        base_ok = entry.get("base_status") == "pass"
        if base_ok:
            base_passed += 1
            if entry.get("plus_status") == "pass":
                plus_passed += 1

    base = base_passed / total
    plus = plus_passed / total
    return {
        "n": total,
        "base": base,
        "plus": plus,
        "base_n": base_passed,
        "plus_n": plus_passed,
        "gap": base - plus,
        "ratio": (plus / base) if base else float("nan"),
    }


def arm_round_path(
    model: str,
    prompt: str,
    dataset: str,
    repair_strategy: str,
    arm: str,
    round_num: int,
    round0_source: Optional[str],
    run_tag: str = "",
) -> str:
    """Reconstruct the eval-results path for one arm at one round.

    ``run_tag`` selects a repeat run ("" is the original, "run2"/"run3" are
    the variance seeds). It must be appended *after* the feedback suffix to
    match generate_samples.run_suffix()'s ordering.
    """
    if round_num == 0 and round0_source:
        # Round 0 is the shared seed - identical across every arm AND every
        # repeat run, so it is graded once and carries no variance. This is
        # deliberate: it means the spread measured here is purely REPAIR
        # variance, which is the only thing the ablation varies.
        return os.path.join(SAMPLES_DIR, f"{round0_source}_eval_results.json")

    prefix = "" if dataset == "humaneval" else f"{dataset}_"
    suffix = "" if arm == "full" else f"_fb-{arm}"
    if run_tag:
        suffix += f"_{run_tag}"
    stem = f"{prefix}{model_slug(model)}_t02_{prompt}{suffix}"
    if round_num == 0:
        return os.path.join(SAMPLES_DIR, f"{stem}_eval_results.json")
    return os.path.join(
        SAMPLES_DIR,
        f"{stem}_repair-{repair_strategy}_round{round_num}_eval_results.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collate the D24 feedback-content ablation into one table."
    )
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--prompt", default="cop")
    parser.add_argument("--repair_strategy", default="minimal", choices=["minimal", "cot"])
    parser.add_argument("--max_repair_rounds", type=int, default=2)
    parser.add_argument(
        "--run_tags",
        default="",
        help=(
            "Comma-separated repeat-run tags to aggregate for variance bounds, "
            "e.g. 'run2,run3'. The original untagged run is always included."
        ),
    )
    parser.add_argument(
        "--round0_source",
        default="llama-33-70b-versatile_t02_cgo",
        help=(
            "Basename (no extension) of the shared Round 0 seed's eval results. "
            "Round 0 is identical across arms, so it is reported once."
        ),
    )
    args = parser.parse_args()

    rounds = list(range(args.max_repair_rounds + 1))

    print(f"Feedback-content ablation (D24) - {args.dataset}, {args.model}")
    print(f"Substrate: {args.repair_strategy} repair on the {args.prompt} seed\n")
    print("| Arm | Round | Base | Plus | Gap | Ratio | n |")
    print("|---|---|---|---|---|---|---|")

    scores: dict = {}
    for arm in ARMS:
        for round_num in rounds:
            path = arm_round_path(
                args.model, args.prompt, args.dataset,
                args.repair_strategy, arm, round_num, args.round0_source,
            )
            result = score_eval_results(path)
            scores[(arm, round_num)] = result
            if result is None:
                print(f"| {arm} | R{round_num} | — | — | — | — | *not graded* |")
                continue
            print(
                f"| {arm} | R{round_num} | {result['base']:.3f} | {result['plus']:.3f} "
                f"| {result['gap']:.3f} | {result['ratio']:.3f} | {result['n']} |"
            )

    # The ablation's actual question lives in the deltas, not the levels.
    final = args.max_repair_rounds
    print(f"\nWithin-arm delta, R0 -> R{final} (the unit of evidence):\n")
    print("| Arm | ΔBase | ΔPlus | Gap R0 → Rn | Gap movement |")
    print("|---|---|---|---|---|")
    for arm in ARMS:
        start, end = scores.get((arm, 0)), scores.get((arm, final))
        if not start or not end:
            print(f"| {arm} | — | — | — | *incomplete* |")
            continue
        gap_delta = end["gap"] - start["gap"]
        if abs(gap_delta) < 0.005:
            movement = "flat"
        elif gap_delta > 0:
            movement = "widens"
        else:
            movement = "closes"
        print(
            f"| {arm} | {(end['base'] - start['base']) * 100:+.1f}pp "
            f"| {(end['plus'] - start['plus']) * 100:+.1f}pp "
            f"| {start['gap']:.3f} → {end['gap']:.3f} | **{movement}** |"
        )

    if args.run_tags:
        _report_variance(args, final)

    print(
        "\nReading guide: if Base scales with feedback richness while the Gap "
        "column stays flat, repair buys accuracy rather than robustness. If "
        "blind matches full, the mechanism is resampling, not feedback content."
    )


def _report_variance(args: argparse.Namespace, final_round: int) -> None:
    """Aggregate repeat runs into mean +/- range per arm.

    Answers the question a single run cannot: is the difference between two
    arms larger than the wobble of the same arm re-run? Round 0 is a fixed
    seed file, so the spread reported here is repair variance only.
    """
    tags = [""] + [t.strip() for t in args.run_tags.split(",") if t.strip()]

    print(f"\n\nVariance across {len(tags)} runs (final round R{final_round}):\n")
    print("| Arm | Base mean | Base range | Plus mean | Plus range | Gap mean | Gap range | runs |")
    print("|---|---|---|---|---|---|---|---|")

    collected: dict = {}
    for arm in ARMS:
        rows = []
        for tag in tags:
            path = arm_round_path(
                args.model, args.prompt, args.dataset, args.repair_strategy,
                arm, final_round, args.round0_source, tag,
            )
            scored = score_eval_results(path)
            if scored:
                rows.append(scored)
        collected[arm] = rows
        if not rows:
            print(f"| {arm} | — | — | — | — | — | — | 0 |")
            continue

        def stats(key: str) -> tuple:
            values = [r[key] for r in rows]
            return (sum(values) / len(values), min(values), max(values))

        base_m, base_lo, base_hi = stats("base")
        plus_m, plus_lo, plus_hi = stats("plus")
        gap_m, gap_lo, gap_hi = stats("gap")
        print(
            f"| {arm} | {base_m:.3f} | {base_lo:.3f}–{base_hi:.3f} "
            f"| {plus_m:.3f} | {plus_lo:.3f}–{plus_hi:.3f} "
            f"| {gap_m:.3f} | {gap_lo:.3f}–{gap_hi:.3f} | {len(rows)} |"
        )

    # The decisive check: does the headline ordering survive re-running?
    complete = {a: r for a, r in collected.items() if len(r) == len(tags) and r}
    if len(complete) >= 2 and len(tags) > 1:
        print("\n**Does the ordering survive re-running?**\n")
        for metric, better_is_higher in (("base", True), ("plus", True), ("gap", False)):
            best = max(
                complete,
                key=lambda a: (
                    sum(r[metric] for r in complete[a]) / len(complete[a])
                ) * (1 if better_is_higher else -1),
            )
            best_lo = min(r[metric] for r in complete[best])
            best_hi = max(r[metric] for r in complete[best])
            overlapping = [
                a for a in complete
                if a != best
                and min(r[metric] for r in complete[a]) <= best_hi
                and max(r[metric] for r in complete[a]) >= best_lo
            ]
            direction = "highest" if better_is_higher else "narrowest"
            verdict = (
                f"separates cleanly from every other arm"
                if not overlapping
                else f"OVERLAPS with: {', '.join(overlapping)} -> not distinguishable"
            )
            print(f"- **{metric}**: {direction} = `{best}`, {verdict}")


if __name__ == "__main__":
    main()
