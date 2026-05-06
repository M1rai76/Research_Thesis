import argparse

from evalplus.data import get_human_eval_plus, get_mbpp_plus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an EvalPlus dataset.")
    parser.add_argument(
        "--dataset",
        choices=["humaneval", "mbpp"],
        default="humaneval",
        help="Dataset to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    problems = get_mbpp_plus() if args.dataset == "mbpp" else get_human_eval_plus()

    print("Dataset:", args.dataset)
    print("Number of tasks:", len(problems))

    first_key = list(problems.keys())[0]
    print("First task:", first_key)

    task = problems[first_key]
    print("Task keys:", task.keys())

    print("Prompt preview:")
    print(task["prompt"][:500])


if __name__ == "__main__":
    main()
