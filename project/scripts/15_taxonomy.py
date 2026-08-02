"""
Script 15 — Taxonomy Classification (Stage 5)

Classifies failure modes for samples that flipped from correct to
incorrect during difficulty analysis. Uses the model as an LLM judge
(or heuristic fallback) to categorize each failure.

Produces:
  - failure_taxonomy.jsonl          (per-sample failure labels)
  - failure_taxonomy_summary.json   (aggregate statistics)
  - failure_taxonomy_rows.csv       (tabular results)

Usage:
  python scripts/15_taxonomy.py --benchmark gsm8k
  python scripts/15_taxonomy.py --benchmark gsm8k --no_llm_judge  (heuristic only)
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
    PATH_BUDGET_FORCING_LABEL,
)
from configs.model_config import MODEL_REGISTRY, ACTIVE_MODEL_KEY
from backends import TransformersBackend, TransformersBackendConfig
from taxonomy.classifier import TaxonomyClassifier, TaxonomyResult, TAXONOMY_LABELS
from taxonomy.summarizer import summarize_taxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5: Taxonomy Classification")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name")
    parser.add_argument("--model", type=str, default=ACTIVE_MODEL_KEY, help="Model key")
    parser.add_argument("--seed", type=int, default=EVAL_SEED, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default=EVAL_EXPERIMENT_NAME, help="Experiment name")
    parser.add_argument("--input_file", type=str, default=None, help="Path to difficulty_generations.jsonl")
    parser.add_argument("--no_llm_judge", action="store_true", help="Use heuristic-only classification")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Find input
    base_dir = (
        RESULTS_DIR
        / args.experiment_name
        / args.model
        / args.benchmark
        / f"seed_{args.seed}"
        / f"budget_prompt_{PATH_BUDGET_FORCING_LABEL}"
    )

    if args.input_file:
        diff_path = Path(args.input_file)
    else:
        diff_path = base_dir / "difficulty_generations.jsonl"

    if not diff_path.exists():
        print(f"ERROR: difficulty_generations.jsonl not found at {diff_path}")
        print("Run script 13_difficulty_analysis.py first.")
        sys.exit(1)

    output_dir = diff_path.parent
    taxonomy_path = output_dir / "failure_taxonomy.jsonl"
    summary_json = output_dir / "failure_taxonomy_summary.json"
    summary_csv = output_dir / "failure_taxonomy_rows.csv"

    print(f"{'=' * 60}")
    print(f"  TAXONOMY CLASSIFICATION")
    print(f"{'=' * 60}")
    print(f"  Model:        {args.model}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  LLM Judge:    {'NO (heuristic only)' if args.no_llm_judge else 'YES'}")
    print(f"  Input:        {diff_path}")
    print(f"{'=' * 60}")

    # Load difficulty records
    records = []
    with open(diff_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Filter to flipped samples only
    flipped = [r for r in records if r.get("flipped", False)]
    print(f"\n  Total records: {len(records)}")
    print(f"  Flipped records (to classify): {len(flipped)}")

    if not flipped:
        print("\n  No flipped samples found. Nothing to classify.")
        # Write empty results
        summary = summarize_taxonomy([], output_json=summary_json, output_csv=summary_csv)
        print(f"  Summary written to: {summary_json}")
        return

    # Setup classifier
    backend = None
    if not args.no_llm_judge:
        print(f"\n  Loading model for LLM judge: {args.model}")
        model_config = MODEL_REGISTRY[args.model]
        backend_config = TransformersBackendConfig(
            model_id=model_config["hf_path"],
            quantization="4bit" if model_config.get("load_in_4bit") else None,
        )
        backend = TransformersBackend(backend_config)
        backend.load()

    classifier = TaxonomyClassifier(backend=backend)

    # Classify each flipped sample
    print(f"\n  Classifying {len(flipped)} flipped samples...")

    if taxonomy_path.exists():
        taxonomy_path.unlink()

    start_time = time.time()
    taxonomy_results: list[TaxonomyResult] = []

    for idx, record in enumerate(flipped):
        prefix_results = record.get("prefix_results", [])
        last_correct_idx = record.get("last_correct_prefix_idx")

        if last_correct_idx is None or not prefix_results:
            continue

        # Get the last correct and final prefix traces
        last_correct_pr = prefix_results[last_correct_idx] if last_correct_idx < len(prefix_results) else None
        final_pr = prefix_results[-1] if prefix_results else None

        if not last_correct_pr or not final_pr:
            continue

        result = classifier.classify(
            pid=record["pid"],
            query=record.get("query", ""),
            ground_truth=record.get("ground_truth", ""),
            last_correct_trace=last_correct_pr.get("continuation_preview", ""),
            last_correct_answer=last_correct_pr.get("extracted_answer"),
            final_trace=final_pr.get("continuation_preview", ""),
            final_answer=final_pr.get("extracted_answer"),
            last_correct_idx=last_correct_idx,
        )
        taxonomy_results.append(result)

        # Write incrementally
        tax_record = {
            "pid": result.pid,
            "category": result.category,
            "severity": result.severity,
            "went_wrong": result.went_wrong,
            "evidence": result.evidence,
            "ground_truth": result.ground_truth,
            "last_correct_answer": result.last_correct_answer,
            "final_answer": result.final_answer,
            "last_correct_prefix_idx": result.last_correct_prefix_idx,
        }
        with open(taxonomy_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(tax_record, ensure_ascii=False) + "\n")

        print(
            f"  [{idx + 1}/{len(flipped)}] "
            f"category={result.category} "
            f"severity={result.severity} "
            f"pid={record['pid']}"
        )

    # Summarize
    elapsed = time.time() - start_time
    summary = summarize_taxonomy(
        taxonomy_results,
        output_json=summary_json,
        output_csv=summary_csv,
    )

    print(f"\n{'=' * 60}")
    print(f"  TAXONOMY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Classified: {summary['total_classified']}")
    print(f"  Mean severity: {summary['severity_mean']}")
    print(f"\n  Category breakdown:")
    for label, stats in summary.get("categories", {}).items():
        print(f"    {label:30s}: {stats['count']:3d} ({stats['percentage']:5.1f}%)")
    print(f"\n  Time: {elapsed:.1f}s")
    print(f"  Taxonomy:     {taxonomy_path}")
    print(f"  Summary JSON: {summary_json}")
    print(f"  Summary CSV:  {summary_csv}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
