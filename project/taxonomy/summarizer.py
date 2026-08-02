"""
Summarizer — aggregate taxonomy labels into summary statistics.

Produces JSON summary and CSV rows from taxonomy JSONL outputs,
matching the paper's summarize_overthinking_categories.py.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from taxonomy.classifier import TaxonomyResult, TAXONOMY_LABELS


def summarize_taxonomy(
    results: list[TaxonomyResult],
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, Any]:
    """
    Aggregate taxonomy results into summary statistics.

    Args:
        results: List of TaxonomyResult from classification.
        output_json: Optional path to write JSON summary.
        output_csv: Optional path to write CSV rows.

    Returns:
        Summary dict with counts, percentages, and per-category breakdowns.
    """
    if not results:
        summary = {
            "total_classified": 0,
            "categories": {},
            "severity_mean": 0.0,
            "severity_distribution": {},
        }
        if output_json:
            _write_json(summary, output_json)
        return summary

    # Count categories
    category_counts = Counter(r.category for r in results)
    total = len(results)

    # Severity statistics
    severities = [r.severity for r in results]
    severity_mean = sum(severities) / total

    severity_bins = {"0-25": 0, "26-50": 0, "51-75": 0, "76-100": 0}
    for s in severities:
        if s <= 25:
            severity_bins["0-25"] += 1
        elif s <= 50:
            severity_bins["26-50"] += 1
        elif s <= 75:
            severity_bins["51-75"] += 1
        else:
            severity_bins["76-100"] += 1

    # Per-category breakdown
    per_category = {}
    for label in TAXONOMY_LABELS:
        count = category_counts.get(label, 0)
        category_results = [r for r in results if r.category == label]
        cat_severities = [r.severity for r in category_results]

        per_category[label] = {
            "count": count,
            "percentage": round(100 * count / total, 1) if total > 0 else 0.0,
            "mean_severity": round(sum(cat_severities) / len(cat_severities), 1) if cat_severities else 0.0,
        }

    summary = {
        "total_classified": total,
        "categories": per_category,
        "severity_mean": round(severity_mean, 1),
        "severity_distribution": severity_bins,
    }

    # Write outputs
    if output_json:
        _write_json(summary, output_json)

    if output_csv:
        _write_csv(results, output_csv)

    return summary


def _write_json(data: dict[str, Any], path: Path) -> None:
    """Write summary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(results: list[TaxonomyResult], path: Path) -> None:
    """Write per-sample taxonomy results to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "pid", "category", "severity", "went_wrong", "evidence",
        "ground_truth", "last_correct_answer", "final_answer",
        "last_correct_prefix_idx",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "pid": r.pid,
                "category": r.category,
                "severity": r.severity,
                "went_wrong": r.went_wrong,
                "evidence": r.evidence[:200],
                "ground_truth": r.ground_truth,
                "last_correct_answer": r.last_correct_answer or "",
                "final_answer": r.final_answer or "",
                "last_correct_prefix_idx": r.last_correct_prefix_idx,
            })
