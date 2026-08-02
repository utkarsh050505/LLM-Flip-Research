"""
Script 14 — Evaluate Difficulty Generations (Stage 4)

Re-evaluates difficulty_generations.jsonl, computing per-prefix accuracy
and producing parsed_responses_difficulty.jsonl with summary statistics.

Produces:
  - parsed_responses_difficulty.jsonl  (per-prefix correct/incorrect)

Usage:
  python scripts/14_evaluate_difficulty.py --benchmark gsm8k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.pipeline_config import (
    RESULTS_DIR,
    EVAL_EXPERIMENT_NAME,
    EVAL_SEED,
    PATH_BUDGET_FORCING_LABEL,
)
from configs.model_config import ACTIVE_MODEL_KEY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 4: Evaluate Difficulty Generations")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--model", type=str, default=ACTIVE_MODEL_KEY, help="Model key")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=EVAL_EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--input_file", type=str, default=None, help="Path to difficulty_generations.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Find input
    if args.input_file:
        diff_path = Path(args.input_file)
    else:
        diff_path = (
            RESULTS_DIR
            / args.experiment_name
            / args.model
            / args.benchmark
            / f"seed_{args.seed}"
            / f"budget_prompt_{PATH_BUDGET_FORCING_LABEL}"
            / "difficulty_generations.jsonl"
        )

    if not diff_path.exists():
        print(f"ERROR: difficulty_generations.jsonl not found at {diff_path}")
        print("Run script 13_difficulty_analysis.py first.")
        sys.exit(1)

    output_path = diff_path.parent / "parsed_responses_difficulty.jsonl"

    print(f"{'=' * 60}")
    print(f"  EVALUATE DIFFICULTY GENERATIONS")
    print(f"{'=' * 60}")
    print(f"  Input:  {diff_path}")
    print(f"  Output: {output_path}")
    print(f"{'=' * 60}")

    # Load difficulty results
    records = []
    with open(diff_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"\n  Loaded {len(records)} difficulty records")

    # Aggregate statistics
    total_samples = len(records)
    total_flipped = sum(1 for r in records if r.get("flipped", False))
    total_prefixes = sum(r.get("num_prefixes", 0) for r in records)

    # Per-prefix accuracy aggregation
    prefix_accuracy: dict[int, dict[str, int]] = {}
    for record in records:
        for pr in record.get("prefix_results", []):
            idx = pr["prefix_idx"]
            if idx not in prefix_accuracy:
                prefix_accuracy[idx] = {"correct": 0, "total": 0}
            prefix_accuracy[idx]["total"] += 1
            if pr.get("correct"):
                prefix_accuracy[idx]["correct"] += 1

    # Write parsed responses
    if output_path.exists():
        output_path.unlink()

    parsed_records = []
    for record in records:
        parsed = {
            "pid": record["pid"],
            "ground_truth": record["ground_truth"],
            "original_correct": record.get("original_correct"),
            "flipped": record.get("flipped", False),
            "last_correct_prefix_idx": record.get("last_correct_prefix_idx"),
            "first_incorrect_prefix_idx": record.get("first_incorrect_prefix_idx"),
            "num_prefixes": record.get("num_prefixes", 0),
            "prefix_correctness": [
                pr.get("correct") for pr in record.get("prefix_results", [])
            ],
        }
        parsed_records.append(parsed)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(parsed, ensure_ascii=False) + "\n")

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  DIFFICULTY EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total samples:    {total_samples}")
    print(f"  Flipped samples:  {total_flipped} ({total_flipped / max(total_samples, 1):.1%})")
    print(f"  Total prefixes:   {total_prefixes}")
    print(f"\n  Accuracy by prefix index:")
    for idx in sorted(prefix_accuracy.keys())[:20]:
        pa = prefix_accuracy[idx]
        acc = pa["correct"] / pa["total"] if pa["total"] > 0 else 0
        print(f"    prefix[{idx:3d}]: {acc:.1%} ({pa['correct']}/{pa['total']})")
    if len(prefix_accuracy) > 20:
        print(f"    ... ({len(prefix_accuracy) - 20} more prefix levels)")
    print(f"\n  Output: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
