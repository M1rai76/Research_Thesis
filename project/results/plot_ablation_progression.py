"""
plot_ablation_progression.py
Thesis - Prompt Engineering for LLM Code Generation
Author : Gurdiraj Bal (z5386590)

Per-round progression figure for the feedback-content ablation.

Plots Base and Plus pass@1 across repair rounds (R0 -> R1 -> R2) for each rung
of the information ladder, on both benchmarks. Because every arm starts from the
identical Round 0 seed, all lines share a starting point and fan out - so the
spread at R2 is attributable to the feedback content alone.

Layout: 2x2 small multiples. Rows are the measure (Base, Plus) with a shared
y-scale per row; columns are the benchmark. One line per arm, five arms.

Deliberately NOT a dual-axis chart: Base and Plus are different measures and get
their own panel rather than two scales on one plot.

Plus pass@1 follows the project convention: a task counts only when its base and
plus statuses are BOTH `pass`.

Usage
    python plot_ablation_progression.py
    python plot_ablation_progression.py --outfile ../results/figures/ablation.png
"""

import argparse
import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(BASE_DIR, "samples")

# Ladder order, richest feedback to none. Hues are assigned in this fixed order
# and never cycled - a validated categorical palette (CVD-checked).
ARMS = [
    ("grounded+",  "#2a78d6"),
    ("full",       "#eb6834"),
    ("error-type", "#1baf7a"),
    ("binary",     "#eda100"),
    ("blind",      "#e87ba4"),
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"

MODEL_SLUG = "llama-33-70b-versatile"
# HumanEval+ has three independent runs per arm; MBPP+ has one.
HE_RUN_TAGS = ["", "_run2", "_run3"]
MBPP_RUN_TAGS = [""]


def score(path: str) -> Optional[tuple]:
    """Return (base, plus) pass@1 from an EvalPlus results file, or None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        evaluations = json.load(f)["eval"]
    total = len(evaluations)
    if not total:
        return None
    base = plus = 0
    for record in evaluations.values():
        entry = record[0] if isinstance(record, list) else record
        if entry.get("base_status") == "pass":
            base += 1
            if entry.get("plus_status") == "pass":
                plus += 1
    return base / total, plus / total


def arm_path(dataset: str, arm: str, round_num: int, run_tag: str) -> str:
    """Reconstruct an eval-results path, mirroring generate_samples' naming."""
    prefix = "" if dataset == "humaneval" else f"{dataset}_"
    fb = "" if arm == "full" else f"_fb-{arm}"
    stem = f"{prefix}{MODEL_SLUG}_t02_cop{fb}{run_tag}"
    if round_num == 0:
        # Round 0 is the shared cgo seed - identical for every arm and run.
        return os.path.join(SAMPLES, f"{prefix}{MODEL_SLUG}_t02_cgo_eval_results.json")
    return os.path.join(
        SAMPLES, f"{stem}_repair-minimal_round{round_num}_eval_results.json"
    )


def series(dataset: str, arm: str, tags: list) -> tuple:
    """Mean (base, plus) per round across available runs; None where ungraded."""
    bases, pluses = [], []
    for round_num in (0, 1, 2):
        vals = [score(arm_path(dataset, arm, round_num, t)) for t in tags]
        vals = [v for v in vals if v]
        if not vals:
            bases.append(None)
            pluses.append(None)
            continue
        bases.append(sum(v[0] for v in vals) / len(vals))
        pluses.append(sum(v[1] for v in vals) / len(vals))
    return bases, pluses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-round progression for the feedback-content ablation."
    )
    parser.add_argument(
        "--outfile",
        default=os.path.join(BASE_DIR, "results", "figures", "ablation_progression.png"),
    )
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    panels = [("humaneval", "HumanEval+", HE_RUN_TAGS), ("mbpp", "MBPP+", MBPP_RUN_TAGS)]
    data = {
        ds: {arm: series(ds, arm, tags) for arm, _ in ARMS}
        for ds, _, tags in panels
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    rounds = [0, 1, 2]

    for row, measure in enumerate(("Base pass@1", "Plus pass@1")):
        # Shared y-scale across the row so the two benchmarks are comparable.
        vals = [
            v
            for ds, _, _ in panels
            for arm, _ in ARMS
            for v in data[ds][arm][row]
            if v is not None
        ]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.18 or 0.02

        for col, (ds, label, tags) in enumerate(panels):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(GRID)

            endpoints = []
            for arm, colour in ARMS:
                ys = data[ds][arm][row]
                pts = [(r, y) for r, y in zip(rounds, ys) if y is not None]
                if not pts:
                    continue
                xs, yy = zip(*pts)
                ax.plot(
                    xs, yy, color=colour, linewidth=2.0, marker="o",
                    markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.4,
                    label=arm if (row == 0 and col == 0) else None, zorder=3,
                )
                endpoints.append([xs[-1], yy[-1], arm, colour])

            ax.set_ylim(lo - pad, hi + pad)

            # Direct-label the endpoints: relief for the low-contrast hues, and
            # it identifies a line without a trip to the legend. Arms often end
            # within a hair of each other (that IS the finding), so nudge labels
            # apart vertically rather than letting them overprint.
            span = (hi + pad) - (lo - pad)
            min_sep = span * 0.055
            endpoints.sort(key=lambda e: e[1])
            for i in range(1, len(endpoints)):
                if endpoints[i][1] - endpoints[i - 1][1] < min_sep:
                    endpoints[i][1] = endpoints[i - 1][1] + min_sep
            for x, y_label, arm, _colour in endpoints:
                ax.annotate(
                    arm, xy=(x, y_label), xytext=(7, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, color=INK_SOFT, annotation_clip=False,
                )
            ax.set_xticks(rounds)
            ax.set_xlim(-0.15, 2.75)
            ax.tick_params(colors=INK_SOFT, labelsize=8.5, length=0)
            if row == 0:
                ax.set_title(
                    f"{label}  ({'mean of 3 runs' if len(tags) > 1 else 'single run'})",
                    fontsize=10, color=INK, pad=10,
                )
            if row == 1:
                ax.set_xlabel("repair round", fontsize=9, color=INK_SOFT)
            if col == 0:
                ax.set_ylabel(measure, fontsize=9.5, color=INK)

    fig.suptitle(
        "Feedback content changes how far repair goes — not whether it stays robust",
        fontsize=12.5, color=INK, x=0.055, ha="left", y=0.975,
    )
    fig.text(
        0.055, 0.925,
        "All arms share one Round 0 seed, so the fan-out is attributable to feedback content alone. "
        "Llama-3.3-70B, minimal substrate.",
        fontsize=8.5, color=INK_SOFT, ha="left",
    )
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=5, frameon=False,
        fontsize=9, labelcolor=INK_SOFT, bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.90])
    fig.savefig(args.outfile, dpi=200, facecolor=SURFACE)
    fig.savefig(args.outfile.replace(".png", ".pdf"), facecolor=SURFACE)
    print(f"wrote {args.outfile}")
    print(f"wrote {args.outfile.replace('.png', '.pdf')}")


if __name__ == "__main__":
    main()
