"""
Script 13 — Difficulty Prefix-Continuation Analysis (Stage 3)

Loads generations.jsonl from Stage 1, runs the prefix-continuation
difficulty experiment: splits each reasoning trace into progressively
longer prefixes, forces an early answer at each, and tracks where
the model flips from correct to incorrect.

Produces:
  - difficulty_generations.jsonl  (per-prefix accuracy data)

Usage:
  python scripts/13_difficulty_analysis.py --benchmark gsm8k --limit 20
  python scripts/13_difficulty_analysis.py --benchmark gsm8k --difficulty_level token --granularity 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.pipeline_config import (
    RESULTS_DIR,
    EVAL_EXPERIMENT_NAME,
    EVAL_SEED,
    BUDGET_FORCING_PROMPT,
    PATH_BUDGET_FORCING_LABEL,
    DIFFICULTY_LEVEL,
    DIFFICULTY_GRANULARITY,
    DIFFICULTY_MAX_CONTINUATION,
    DIFFICULTY_TEMPERATURE,
)
from configs.model_config import MODEL_REGISTRY, ACTIVE_MODEL_KEY
from backends import TransformersBackend, TransformersBackendConfig
from difficulty import DifficultyConfig, DifficultyExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: Difficulty Prefix-Continuation Analysis")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--model", type=str, default=ACTIVE_MODEL_KEY, help="Model key")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=EVAL_EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--difficulty_level", type=str, default=DIFFICULTY_LEVEL,
                        help="Splitting strategy: 'utterance' or 'token'")
    parser.add_argument("--granularity", type=int, default=DIFFICULTY_GRANULARITY,
                        help="Number of utterances/tokens per segment")
    parser.add_argument("--max_continuation_tokens", type=int, default=512,
                        help="Max tokens per prefix continuation (default: 512)")
    parser.add_argument("--temperature", type=float, default=DIFFICULTY_TEMPERATURE,
                        help="Continuation temperature")
    parser.add_argument("--max_prefixes", type=int, default=15,
                        help="Max number of prefixes to evaluate per sample (default: 15)")
    parser.add_argument("--max_prefix_chars", type=int, default=3000,
                        help="Skip prefixes longer than this many chars (default: 3000)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to generations.jsonl (auto-detected if not set)")
    parser.add_argument("--output", type=str, default=None, help="Output directory override")
    return parser.parse_args()


def find_generations_file(args: argparse.Namespace) -> Path:
    """Find the generations.jsonl from Stage 1."""
    if args.input_file:
        return Path(args.input_file)
    return (
        RESULTS_DIR
        / args.experiment_name
        / args.model
        / args.benchmark
        / f"seed_{args.seed}"
        / f"budget_prompt_{PATH_BUDGET_FORCING_LABEL}"
        / "generations.jsonl"
    )


def load_generations(path: Path) -> list[dict]:
    """Load generation records from JSONL."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()

    # Reproducibility
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass

    # Find input file
    gen_path = find_generations_file(args)
    if not gen_path.exists():
        print(f"ERROR: generations.jsonl not found at {gen_path}")
        print("Run script 12_eval_benchmark.py first.")
        sys.exit(1)

    # Setup output
    output_dir = gen_path.parent if args.output is None else Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    difficulty_path = output_dir / "difficulty_generations.jsonl"

    print(f"{'=' * 60}")
    print(f"  DIFFICULTY PREFIX-CONTINUATION ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Model:            {args.model}")
    print(f"  Benchmark:        {args.benchmark}")
    print(f"  Difficulty level: {args.difficulty_level}")
    print(f"  Granularity:      {args.granularity}")
    print(f"  Max continuation: {args.max_continuation_tokens} tokens")
    print(f"  Input:            {gen_path}")
    print(f"  Output:           {difficulty_path}")
    print(f"{'=' * 60}")

    # Load generations
    print(f"\n[1/3] Loading generations from {gen_path.name}")
    generations = load_generations(gen_path)
    if args.limit:
        generations = generations[:args.limit]
    print(f"  Loaded {len(generations)} generation records")

    # Load model
    print(f"\n[2/3] Loading model: {args.model}")
    model_config = MODEL_REGISTRY[args.model]
    backend_config = TransformersBackendConfig(
        model_id=model_config["hf_path"],
        quantization="4bit" if model_config.get("load_in_4bit") else None,
    )
    backend = TransformersBackend(backend_config)
    backend.load()

    # Setup difficulty experiment
    diff_config = DifficultyConfig(
        difficulty_level=args.difficulty_level,
        granularity=args.granularity,
        budget_forcing_prompt=BUDGET_FORCING_PROMPT,
        max_continuation_tokens=args.max_continuation_tokens,
        continuation_temperature=args.temperature,
        max_prefixes=args.max_prefixes,
        max_prefix_chars=args.max_prefix_chars,
        seed=args.seed,
        experiment_name=args.experiment_name,
    )
    experiment = DifficultyExperiment(backend, diff_config)

    # Run difficulty analysis
    print(f"\n[3/3] Running difficulty analysis on {len(generations)} samples...")

    # Clear previous output
    if difficulty_path.exists():
        difficulty_path.unlink()

    start_time = time.time()
    flip_count = 0

    for idx, gen in enumerate(generations):
        sample = {
            "pid": gen["pid"],
            "query": gen["query"],
            "answer": gen["ground_truth"],
        }
        reasoning_trace = gen.get("model_answer", "")

        if not reasoning_trace.strip():
            print(f"  [{idx + 1}/{len(generations)}] SKIP (empty trace) pid={gen['pid']}")
            continue

        result = experiment.run_single(sample, reasoning_trace)

        if result.flipped:
            flip_count += 1

        # Write result
        record = {
            "pid": result.pid,
            "query": result.query,
            "ground_truth": result.ground_truth,
            "original_answer": result.original_answer,
            "original_correct": result.original_correct,
            "num_prefixes": result.num_prefixes,
            "last_correct_prefix_idx": result.last_correct_prefix_idx,
            "first_incorrect_prefix_idx": result.first_incorrect_prefix_idx,
            "flipped": result.flipped,
            "prefix_results": [
                {
                    "prefix_idx": pr.prefix_idx,
                    "prefix_length_chars": pr.prefix_length_chars,
                    "prefix_length_lines": pr.prefix_length_lines,
                    "extracted_answer": pr.extracted_answer,
                    "correct": pr.correct,
                    "continuation_preview": pr.continuation[:200],
                }
                for pr in result.prefix_results
            ],
        }
        with open(difficulty_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        elapsed = time.time() - start_time
        flip_pct = flip_count / (idx + 1) * 100
        print(
            f"  [{idx + 1}/{len(generations)}] "
            f"prefixes={result.num_prefixes} "
            f"flipped={'YES' if result.flipped else 'no'} "
            f"flip_rate={flip_pct:.0f}% "
            f"pid={gen['pid']}"
        )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  DIFFICULTY ANALYSIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Samples processed: {len(generations)}")
    print(f"  Flipped samples:   {flip_count}")
    print(f"  Flip rate:         {flip_count / max(len(generations), 1):.1%}")
    print(f"  Time:              {elapsed:.1f}s")
    print(f"  Output:            {difficulty_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
