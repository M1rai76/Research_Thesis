"""
run_prochemy.py
Baseline - Prochemy (Ye et al., 2025, arXiv:2503.11085)
Author : Gurdiraj Bal (z5386590)

CLI driver for the Prochemy baseline. Two phases:

    1. Optimise - run the Prochemy search (prochemy_optimize.run_optimization) on
       the training set to obtain the final fixed system prompt P*.
    2. Generate - freeze P* and generate one sample per HumanEval+/MBPP+ test task
       (P* as the system prompt, the task as the user message), post-process with
       this project's post_process / post_process_solution, and write a JSONL that
       sits beside the other strategy runs for the standard Docker EvalPlus command.

Comparability: same Groq/Katana model as the rest of the project (default
llama-3.3-70b-versatile), the paper's own seed + mutation prompts, all code
generation at this project's temperature 0.2, mutation at the paper's 1.0.
Prochemy competes with the *prompt-strategy* layer, so P* is graded on
Base/Plus/Gap/Ratio, not pass@1 alone (rationale recorded in results.md once run).

Examples
    # Smoke test (tiny, real API - gate on go-ahead):
    python run_prochemy.py --dataset humaneval --backend groq \
        --k-max 2 --limit-trainset 3 --limit-tasks 5

    # Full HumanEval+ run from the zero-shot seed:
    python run_prochemy.py --dataset humaneval --initial zero_shot --backend groq

    # Full MBPP+ run:
    python run_prochemy.py --dataset mbpp --initial zero_shot --backend groq

    # Grade afterwards (unchanged pattern):
    docker run --rm -v "${PWD}:/app" ganler/evalplus:latest \
        evalplus.evaluate --dataset mbpp --samples /app/samples/<output>.jsonl
"""

import argparse
import json
import os
import sys
import time

_SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_samples import (  # noqa: E402
    get_client,
    get_problems,
    get_output_path,
    post_process,
    post_process_solution,
    save_run_log,
    generate_raw_completion,
    TEMPERATURE,
    MAX_TOKENS,
    SLEEP,
)

from prochemy_prompt import INITIAL_PROMPTS  # noqa: E402
from prochemy_trainset import load_training_set  # noqa: E402
from prochemy_optimize import run_optimization  # noqa: E402


def _sample_key(dataset: str) -> str:
    """MBPP seeds/samples store code under 'solution', HumanEval under 'completion'."""
    return "solution" if dataset == "mbpp" else "completion"


def optimize_phase(client, args, trajectory_path, best_prompt_path):
    """Run (or reload) the Prochemy search and return P* (the frozen system prompt)."""
    if os.path.exists(best_prompt_path) and not args.reoptimize:
        with open(best_prompt_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        print(f"Reusing existing P* from {best_prompt_path} "
              f"(pass --reoptimize to search again).")
        return saved["p_star"]

    trainset = load_training_set(args.trainset)
    if args.limit_trainset:
        trainset = trainset[: args.limit_trainset]
    initial_prompt = INITIAL_PROMPTS[args.initial]

    print(f"Optimising: initial={args.initial}, trainset={len(trainset)} tasks, "
          f"k_max={args.k_max}, n_variants={args.n_variants}, patience={args.patience}, "
          f"seed={args.seed}, solution_temp={TEMPERATURE}, mutation_temp=1.0")

    result = run_optimization(
        client,
        model=args.model,
        initial_prompt=initial_prompt,
        trainset=trainset,
        solution_temperature=TEMPERATURE,
        k_max=args.k_max,
        n_variants=args.n_variants,
        patience=args.patience,
        seed=args.seed,
        sleep_s=SLEEP.get(args.backend, 0.0),
        trajectory_path=trajectory_path,
    )
    if result is None:
        print("Optimisation aborted (API failure) - no P* written. "
              "Re-run with --resume-missing once quota is back.")
        sys.exit(1)

    with open(best_prompt_path, "w", encoding="utf-8") as f:
        json.dump(
            {"p_star": result["p_star"],
             "p_star_prompt_id": result["p_star_prompt_id"],
             "p_star_weighted_score": result["p_star_weighted_score"],
             "p_star_original_score": result["p_star_original_score"],
             "n_iterations": result["n_iterations"],
             "initial": args.initial, "model": args.model, "dataset": args.dataset,
             "seed": args.seed},
            f, indent=2,
        )
    print(f"P* saved -> {best_prompt_path}  "
          f"(original_score={result['p_star_original_score']:.3f}, "
          f"{result['n_iterations']} iterations)")
    print(f"Trajectory -> {trajectory_path}")
    return result["p_star"]


def generate_phase(client, args, p_star, output_path):
    """Generate one sample per test task with the frozen P*, write EvalPlus JSONL."""
    problems = get_problems(args.dataset)
    task_ids = list(problems.keys())
    if args.limit_tasks:
        task_ids = task_ids[: args.limit_tasks]

    key = _sample_key(args.dataset)

    # Resume-missing: keep already-generated samples.
    existing = {}
    if args.resume_missing and os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    s = json.loads(line)
                    existing[s["task_id"]] = s
        print(f"Resume: {len(existing)} samples already present in {output_path}")

    samples = dict(existing)
    skipped = []
    sleep_s = SLEEP.get(args.backend, 0.0)

    for i, task_id in enumerate(task_ids, start=1):
        if task_id in samples and samples[task_id].get(key):
            continue
        task_prompt = problems[task_id]["prompt"]
        raw = generate_raw_completion(
            client,
            args.model,
            prompt=task_prompt,
            system_prompt=p_star,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        if raw is None:
            print(f"  [{i}/{len(task_ids)}] {task_id}: API failure - skipping (resume later).")
            skipped.append(task_id)
            continue

        if args.dataset == "mbpp":
            code = post_process_solution(raw)
        else:
            entry_point = problems[task_id].get("entry_point")
            code = post_process(raw, entry_point)

        samples[task_id] = {"task_id": task_id, key: code}
        if not code:
            print(f"  [{i}/{len(task_ids)}] {task_id}: empty after post-processing.")

        # Persist incrementally so a quota interruption loses nothing.
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples.values():
                f.write(json.dumps(s) + "\n")
        if sleep_s:
            time.sleep(sleep_s)

    print(f"Wrote {len(samples)} samples -> {output_path}  (skipped {len(skipped)})")
    return len(samples), skipped


def main():
    parser = argparse.ArgumentParser(description="Prochemy baseline: optimise a prompt, then generate.")
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    parser.add_argument("--initial", choices=["zero_shot", "cot"], default="zero_shot",
                        help="Seed prompt S(0). zero_shot is the authors' shipped origin prompt.")
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--backend", choices=["groq", "katana", "ollama", "cerebras", "openrouter", "gemini"],
                        default="groq")
    parser.add_argument("--trainset", default=None,
                        help="Training set JSONL (default: shipped prochemy_training_set.jsonl).")
    parser.add_argument("--k-max", type=int, default=10)
    parser.add_argument("--n-variants", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-index", type=int, default=1,
                        help="Stability-check index for repeat runs. >1 appends _run<N> to output names.")
    parser.add_argument("--limit-trainset", type=int, default=None, help="Cap training tasks (smoke).")
    parser.add_argument("--limit-tasks", type=int, default=None, help="Cap test tasks (smoke).")
    parser.add_argument("--reoptimize", action="store_true", help="Re-run search even if P* exists.")
    parser.add_argument("--resume-missing", action="store_true", help="Keep existing samples, fill gaps.")
    parser.add_argument("--optimize-only", action="store_true", help="Stop after producing P*.")
    parser.add_argument("--seed-baseline", action="store_true",
                        help="Skip the search entirely and generate with the unoptimised seed "
                             "prompt S(0) as the system prompt. Produces the confound-free "
                             "P*-vs-S(0) comparison anchor; writes to a distinct '-seed' output.")
    args = parser.parse_args()

    strategy = f"prochemy-{args.initial}"
    if args.seed_baseline:
        strategy = f"{strategy}-seed"
    if args.run_index and args.run_index != 1:
        strategy = f"{strategy}_run{args.run_index}"

    output_path = get_output_path(args.model, strategy, args.dataset)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    best_prompt_path = output_path.replace(".jsonl", "_best_prompt.json")
    trajectory_path = output_path.replace(".jsonl", "_optimization.json")

    client = get_client(args.backend)

    if args.seed_baseline:
        # Confound-free anchor: generate with the unoptimised seed S(0) itself as the
        # system prompt, using the identical generation path/temperature/post-processing
        # as the P* run. The only difference from the P* run is the system prompt.
        p_star = INITIAL_PROMPTS[args.initial]
        print(f"Seed-baseline mode: generating with the unoptimised S(0) seed "
              f"('{args.initial}') as the system prompt -- no search.")
    else:
        p_star = optimize_phase(client, args, trajectory_path, best_prompt_path)
        if args.optimize_only:
            return

    num_saved, skipped = generate_phase(client, args, p_star, output_path)
    save_run_log(
        output_path=output_path,
        model=args.model,
        backend=args.backend,
        dataset=args.dataset,
        prompt_strategy=strategy,
        num_saved=num_saved,
        skipped=skipped,
    )


if __name__ == "__main__":
    main()
