"""
Run Full Pipeline — execute all 5 stages of the thinking-past-the-answer pipeline.

Stages:
  1. Benchmark Evaluation    (12_eval_benchmark.py)
  3. Difficulty Analysis      (13_difficulty_analysis.py)
  4. Evaluate Difficulty      (14_evaluate_difficulty.py)
  5. Taxonomy Classification  (15_taxonomy.py)

Usage:
  python scripts/run_full_pipeline.py --benchmark gsm8k --limit 20
  python scripts/run_full_pipeline.py --benchmark math500 --limit 50
  python scripts/run_full_pipeline.py --benchmark local_problems
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full thinking-past-the-answer pipeline"
    )
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--model", type=str, default="deepseek_qwen_1.5b", help="Model key")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per stage")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="main", help="Experiment name")
    parser.add_argument("--difficulty_level", type=str, default="utterance",
                        help="Splitting: 'utterance' or 'token'")
    parser.add_argument("--granularity", type=int, default=1, help="Segments per split unit")
    parser.add_argument("--skip_eval", action="store_true", help="Skip Stage 1 (eval)")
    parser.add_argument("--skip_difficulty", action="store_true", help="Skip Stage 3 (difficulty)")
    parser.add_argument("--skip_taxonomy", action="store_true", help="Skip Stage 5 (taxonomy)")
    parser.add_argument("--no_llm_judge", action="store_true", help="Heuristic-only taxonomy")
    return parser.parse_args()


def run_stage(name: str, script: str, extra_args: list[str]) -> bool:
    """Run a pipeline stage as a subprocess."""
    print(f"\n{'#' * 70}")
    print(f"  STAGE: {name}")
    print(f"{'#' * 70}\n")

    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + extra_args
    print(f"  CMD: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print(f"\n  ERROR: Stage '{name}' failed with return code {result.returncode}")
        return False
    return True


def main() -> None:
    args = parse_args()

    common_args = [
        "--benchmark", args.benchmark,
        "--model", args.model,
        "--seed", str(args.seed),
        "--experiment_name", args.experiment_name,
    ]
    limit_args = ["--limit", str(args.limit)] if args.limit else []

    print(f"{'=' * 70}")
    print(f"  THINKING-PAST-THE-ANSWER: FULL PIPELINE")
    print(f"{'=' * 70}")
    print(f"  Model:      {args.model}")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Limit:      {args.limit or 'ALL'}")
    print(f"  Seed:       {args.seed}")
    print(f"{'=' * 70}")

    start = time.time()
    stages_run = 0
    stages_ok = 0

    # Stage 1: Benchmark Evaluation
    if not args.skip_eval:
        stages_run += 1
        ok = run_stage(
            "Benchmark Evaluation",
            "12_eval_benchmark.py",
            common_args + limit_args,
        )
        if not ok:
            print("\nPipeline aborted at Stage 1.")
            sys.exit(1)
        stages_ok += 1
    else:
        print("\n  SKIPPED: Stage 1 (Benchmark Evaluation)")

    # Stage 3: Difficulty Analysis
    if not args.skip_difficulty:
        stages_run += 1
        diff_args = [
            "--difficulty_level", args.difficulty_level,
            "--granularity", str(args.granularity),
        ]
        ok = run_stage(
            "Difficulty Prefix-Continuation Analysis",
            "13_difficulty_analysis.py",
            common_args + limit_args + diff_args,
        )
        if not ok:
            print("\nPipeline aborted at Stage 3.")
            sys.exit(1)
        stages_ok += 1

        # Stage 4: Evaluate Difficulty (always runs after Stage 3)
        stages_run += 1
        ok = run_stage(
            "Evaluate Difficulty Generations",
            "14_evaluate_difficulty.py",
            common_args,
        )
        if not ok:
            print("\nPipeline aborted at Stage 4.")
            sys.exit(1)
        stages_ok += 1
    else:
        print("\n  SKIPPED: Stage 3 & 4 (Difficulty Analysis)")

    # Stage 5: Taxonomy Classification
    if not args.skip_taxonomy:
        stages_run += 1
        tax_args = ["--no_llm_judge"] if args.no_llm_judge else []
        ok = run_stage(
            "Taxonomy Classification",
            "15_taxonomy.py",
            common_args + tax_args,
        )
        if not ok:
            print("\nPipeline aborted at Stage 5.")
            sys.exit(1)
        stages_ok += 1
    else:
        print("\n  SKIPPED: Stage 5 (Taxonomy)")

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Stages run:  {stages_ok}/{stages_run}")
    print(f"  Total time:  {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
