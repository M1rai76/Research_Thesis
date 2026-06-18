"""
safety_oracle.py

Static AST-based Safety Oracle. Scans generated code for the presence of
defensive guard patterns adapted from Li et al. (2025), "A Preliminary Study
on the Robustness of Code Generation by Large Language Models"
(arXiv:2503.20197), Section 2.3 -- restricted to the four guard categories
that transfer from CoderEval's Java/enterprise setting to standalone Python
functions (HumanEval+/MBPP+): None checks, specific-value checks, range
checks, and type checks. Error handling (try/except) is tracked as a
secondary signal. See decisions.md D8 for why the paper's Boolean-state,
Assertion, and Error categories are out of scope.

Usage:
    python -m robustness.safety_oracle \
      --dataset mbpp \
      --samples samples/mbpp_llama-33-70b-versatile_t02_cgo.jsonl
"""

import argparse
import ast
import json
import os
from typing import Any

from evalplus.data import get_human_eval_plus, get_mbpp_plus


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GUARD_KEYS = (
    "has_none_check",
    "has_specific_value_check",
    "has_range_check",
    "has_type_check",
)


def _is_none_constant(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _is_empty_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value in (0, "", b"")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) == 0
    if isinstance(node, ast.Dict):
        return len(node.keys) == 0
    return False


class GuardVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_none_check = False
        self.has_specific_value_check = False
        self.has_range_check = False
        self.has_type_check = False
        self.has_error_handling = False

    def visit_Try(self, node: ast.Try) -> None:
        self.has_error_handling = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "isinstance":
            self.has_type_check = True
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        operands = [node.left, *node.comparators]
        for op in node.ops:
            if isinstance(op, (ast.Is, ast.IsNot)):
                if any(_is_none_constant(o) for o in operands):
                    self.has_none_check = True
            elif isinstance(op, (ast.Eq, ast.NotEq)):
                if any(_is_empty_literal(o) for o in operands):
                    self.has_specific_value_check = True
            elif isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                self.has_range_check = True
        self.generic_visit(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if isinstance(node.op, ast.Not) and isinstance(node.operand, ast.Name):
            self.has_specific_value_check = True
        self.generic_visit(node)

    def _check_truthiness_test(self, test: ast.expr) -> None:
        if isinstance(test, ast.Name):
            self.has_specific_value_check = True

    def visit_If(self, node: ast.If) -> None:
        self._check_truthiness_test(node.test)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self._check_truthiness_test(node.test)
        self.generic_visit(node)


def extract_guards(source: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        result = {key: False for key in GUARD_KEYS}
        result.update(
            parse_ok=False,
            has_error_handling=False,
            guard_count=0,
            any_guard_present=False,
        )
        return result

    visitor = GuardVisitor()
    visitor.visit(tree)

    flags = {key: getattr(visitor, key) for key in GUARD_KEYS}
    return {
        "parse_ok": True,
        **flags,
        "has_error_handling": visitor.has_error_handling,
        "guard_count": sum(flags.values()),
        "any_guard_present": any(flags.values()),
    }


def assemble_source(dataset: str, problem: dict[str, Any], sample: dict[str, Any]) -> str:
    if dataset == "humaneval":
        return problem.get("prompt", "") + sample.get("completion", "")
    return sample.get("solution", "")


def load_problems(dataset: str) -> dict[str, dict[str, Any]]:
    if dataset == "humaneval":
        return get_human_eval_plus()
    if dataset == "mbpp":
        return get_mbpp_plus()
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_samples(path: str) -> dict[str, dict[str, Any]]:
    samples = {}
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            task_id = sample.get("task_id")
            if not task_id:
                raise ValueError(f"Missing task_id in {path}:{line_no}")
            samples[task_id] = sample
    return samples


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total_tasks": 0}

    def rate(key: str) -> float:
        return sum(record[key] for record in records) / total

    return {
        "total_tasks": total,
        "parse_ok_count": sum(record["parse_ok"] for record in records),
        "none_check_rate": rate("has_none_check"),
        "specific_value_check_rate": rate("has_specific_value_check"),
        "range_check_rate": rate("has_range_check"),
        "type_check_rate": rate("has_type_check"),
        "error_handling_rate": rate("has_error_handling"),
        "any_guard_present_rate": rate("any_guard_present"),
        "mean_guard_count": sum(record["guard_count"] for record in records) / total,
    }


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(BASE_DIR, path)


def default_output_path(samples_path: str) -> str:
    if samples_path.endswith(".jsonl"):
        return samples_path[: -len(".jsonl")] + "_safety_oracle.json"
    return samples_path + "_safety_oracle.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safety Oracle: AST guard-pattern scan for generated code."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["humaneval", "mbpp"],
        help="EvalPlus dataset used for the run.",
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="Path to the generated samples JSONL file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <samples>_safety_oracle.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples_path = resolve_path(args.samples)
    output_path = (
        resolve_path(args.output) if args.output else default_output_path(samples_path)
    )

    problems = load_problems(args.dataset)
    samples = load_samples(samples_path)

    records = []
    for task_id, sample in sorted(samples.items()):
        problem = problems.get(task_id, {})
        source = assemble_source(args.dataset, problem, sample)
        guards = extract_guards(source)
        records.append({"task_id": task_id, **guards})

    report = {
        "dataset": args.dataset,
        "samples": samples_path,
        "summary": summarize(records),
        "results": records,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    summary = report["summary"]
    print(f"Wrote {len(records)} records -> {output_path}")
    if summary["total_tasks"]:
        print(f"Any guard present: {summary['any_guard_present_rate']:.3f}")
        print(
            f"None: {summary['none_check_rate']:.3f}  "
            f"Specific value: {summary['specific_value_check_rate']:.3f}  "
            f"Range: {summary['range_check_rate']:.3f}  "
            f"Type: {summary['type_check_rate']:.3f}  "
            f"Error handling: {summary['error_handling_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
