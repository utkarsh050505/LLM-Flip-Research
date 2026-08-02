"""
Script 12 — Benchmark Evaluation (Stage 1)

Load a model and benchmark, generate answers for each problem,
evaluate correctness, and save results.

Produces:
  - generations.jsonl  (per-sample: query, model_answer, extracted_answer, correct)
  - results.json       (aggregate accuracy metrics)

Usage:
  python scripts/12_eval_benchmark.py --benchmark gsm8k --limit 50
  python scripts/12_eval_benchmark.py --benchmark math500 --limit 100
  python scripts/12_eval_benchmark.py --benchmark local_problems
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.pipeline_config import (
    RESULTS_DIR,
    EVAL_MAX_TOKENS,
    EVAL_TEMPERATURE,
    EVAL_EXPERIMENT_NAME,
    EVAL_SEED,
    BUDGET_FORCING_PROMPT,
    PATH_BUDGET_FORCING_LABEL,
)
from configs.model_config import MODEL_REGISTRY, ACTIVE_MODEL_KEY, HF_CACHE_DIR
from backends import TransformersBackend, TransformersBackendConfig
from benchmarking import load_benchmark, list_benchmarks
from evaluation.parsing import ParsingHelper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Benchmark Evaluation")
    parser.add_argument(
        "--benchmark", type=str, default="gsm8k",
        help=f"Benchmark name. Available: {', '.join(list_benchmarks())}",
    )
    parser.add_argument(
        "--model", type=str, default=ACTIVE_MODEL_KEY,
        help=f"Model key from registry. Available: {', '.join(MODEL_REGISTRY.keys())}",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for debugging)")
    parser.add_argument("--max_tokens", type=int, default=EVAL_MAX_TOKENS, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=EVAL_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=EVAL_EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    parser.add_argument(
        "--show_benchmarks", action="store_true",
        help="Print available benchmarks and exit",
    )
    return parser.parse_args()


def build_output_dir(args: argparse.Namespace) -> Path:
    """Build the output directory path following the paper's convention."""
    if args.output:
        return Path(args.output)
    return (
        RESULTS_DIR
        / args.experiment_name
        / args.model
        / args.benchmark
        / f"seed_{args.seed}"
        / f"budget_prompt_{PATH_BUDGET_FORCING_LABEL}"
    )


def main() -> None:
    args = parse_args()

    if args.show_benchmarks:
        print("Available benchmarks:", ", ".join(list_benchmarks()))
        return

    # Reproducibility
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    # Setup output
    output_dir = build_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"
    results_path = output_dir / "results.json"

    print(f"{'=' * 60}")
    print(f"  BENCHMARK EVALUATION")
    print(f"{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Seed:       {args.seed}")
    print(f"  Output:     {output_dir}")
    print(f"{'=' * 60}")

    # Load benchmark
    print(f"\n[1/3] Loading benchmark: {args.benchmark}")
    bench = load_benchmark(args.benchmark, debug_limit=args.limit)
    samples = bench.get_samples()
    print(f"  Loaded {len(samples)} samples")

    # Load model
    print(f"\n[2/3] Loading model: {args.model}")
    model_config = MODEL_REGISTRY[args.model]
    backend_config = TransformersBackendConfig(
        model_id=model_config["hf_path"],
        quantization="4bit" if model_config.get("load_in_4bit") else None,
    )
    backend = TransformersBackend(backend_config)
    backend.load()
    print(f"  Model loaded: {model_config['hf_path']}")

    # Generate and evaluate
    print(f"\n[3/3] Generating answers for {len(samples)} samples...")
    correct_count = 0
    total_count = 0

    # Clear previous generations file
    if generations_path.exists():
        generations_path.unlink()

    start_time = time.time()

    for idx, sample in enumerate(samples):
        query = sample["query"]
        ground_truth = sample["answer"]

        # Generate model response
        model_answer = backend.generate_text(
            prompt=query,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        # Extract and evaluate answer
        extracted = ParsingHelper.extract_answer(model_answer)
        is_correct = (
            ParsingHelper.compare_answers(extracted, ground_truth)
            if extracted else False
        )

        if is_correct:
            correct_count += 1
        total_count += 1

        # Write generation record
        record = {
            "pid": sample["pid"],
            "query": query,
            "ground_truth": ground_truth,
            "model_answer": model_answer,
            "extracted_answer": extracted,
            "correct": is_correct,
            "subject": sample.get("subject", ""),
            "level": sample.get("level", ""),
        }
        with open(generations_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Progress
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(
            f"  [{idx + 1}/{len(samples)}] "
            f"acc={accuracy:.1%} "
            f"({correct_count}/{total_count}) "
            f"[{rate:.1f} samples/s] "
            f"pid={sample['pid']}"
        )

    # Write results summary
    elapsed = time.time() - start_time
    accuracy = correct_count / total_count if total_count > 0 else 0

    results = {
        "model": args.model,
        "model_hf_path": model_config["hf_path"],
        "benchmark": args.benchmark,
        "total_samples": total_count,
        "correct": correct_count,
        "accuracy": round(accuracy, 4),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed, 1),
    }
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Accuracy:    {accuracy:.1%} ({correct_count}/{total_count})")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Generations: {generations_path}")
    print(f"  Results:     {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
