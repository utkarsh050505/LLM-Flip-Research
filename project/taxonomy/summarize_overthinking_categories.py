import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize last-true-vs-final overthinking category JSONL outputs."
    )
    parser.add_argument("--input_file", type=str, required=True, help="Path to *_last_true_failure_categories.jsonl.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path for aggregate stats JSON.")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path for per-row compact CSV.")
    parser.add_argument("--top_k", type=int, default=15, help="Number of examples to show in preview lists.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_parse_error(row: dict) -> bool:
    parsed = row.get("judge_parsed_output")
    return isinstance(parsed, dict) and bool(parsed.get("parse_error"))


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return 100.0 * count / total


def build_stats(rows: list[dict]) -> dict:
    total = len(rows)
    parse_error_count = sum(1 for row in rows if is_parse_error(row))
    valid_rows = [row for row in rows if not is_parse_error(row)]

    category_counts = Counter(row.get("judge_category", "missing") for row in rows)
    valid_category_counts = Counter(row.get("judge_category", "missing") for row in valid_rows)
    secondary_counts = Counter()
    for row in valid_rows:
        for category in row.get("judge_secondary_categories", []) or []:
            secondary_counts[category] += 1

    severity_by_category = defaultdict(list)
    confidence_by_category = defaultdict(list)
    drift_by_category = defaultdict(list)
    for row in valid_rows:
        category = row.get("judge_category", "missing")
        severity = row.get("judge_severity")
        confidence = row.get("judge_confidence")
        if isinstance(severity, (int, float)):
            severity_by_category[category].append(float(severity))
        if isinstance(confidence, (int, float)):
            confidence_by_category[category].append(float(confidence))
        drift_by_category[category].append(
            int(row["last_difficulty_idx"]) - int(row["last_true_difficulty_idx"])
        )

    stats = {
        "total_rows": total,
        "valid_json_rows": len(valid_rows),
        "parse_error_rows": parse_error_count,
        "parse_error_rate": pct(parse_error_count, total),
        "category_counts": dict(category_counts.most_common()),
        "valid_category_counts": dict(valid_category_counts.most_common()),
        "secondary_category_counts": dict(secondary_counts.most_common()),
        "mean_severity_by_category": {
            category: safe_mean(values) for category, values in sorted(severity_by_category.items())
        },
        "mean_confidence_by_category": {
            category: safe_mean(values) for category, values in sorted(confidence_by_category.items())
        },
        "mean_added_steps_by_category": {
            category: safe_mean(values) for category, values in sorted(drift_by_category.items())
        },
        "mean_added_steps": safe_mean(
            [
                int(row["last_difficulty_idx"]) - int(row["last_true_difficulty_idx"])
                for row in rows
            ]
        ),
    }
    return stats


def print_stats(stats: dict, rows: list[dict], top_k: int):
    total = stats["total_rows"]
    print(f"Rows: {total}")
    print(
        f"Valid JSON rows: {stats['valid_json_rows']} "
        f"({pct(stats['valid_json_rows'], total):.1f}%)"
    )
    print(
        f"Parse-error rows: {stats['parse_error_rows']} "
        f"({stats['parse_error_rate']:.1f}%)"
    )
    print(f"Mean added difficulty steps: {stats['mean_added_steps']:.2f}")

    print("\nCategories including parse-error fallbacks:")
    for category, count in stats["category_counts"].items():
        print(f"  {category}: {count} ({pct(count, total):.1f}%)")

    if stats["valid_json_rows"]:
        print("\nCategories on valid JSON rows only:")
        for category, count in stats["valid_category_counts"].items():
            print(f"  {category}: {count} ({pct(count, stats['valid_json_rows']):.1f}%)")

    print("\nPer-category means on valid JSON rows:")
    for category in sorted(stats["mean_added_steps_by_category"]):
        sev = stats["mean_severity_by_category"].get(category)
        conf = stats["mean_confidence_by_category"].get(category)
        drift = stats["mean_added_steps_by_category"].get(category)
        sev_text = "NA" if sev is None else f"{sev:.2f}"
        conf_text = "NA" if conf is None else f"{conf:.2f}"
        drift_text = "NA" if drift is None else f"{drift:.2f}"
        print(f"  {category}: severity={sev_text}, confidence={conf_text}, added_steps={drift_text}")

    parse_errors = [row for row in rows if is_parse_error(row)]
    if parse_errors:
        print(f"\nParse-error examples, first {min(top_k, len(parse_errors))}:")
        for row in parse_errors[:top_k]:
            drift = int(row["last_difficulty_idx"]) - int(row["last_true_difficulty_idx"])
            preview = (row.get("judge_raw_output") or "").replace("\n", " ")[:180]
            print(f"  idx={row['idx']} drift={drift} raw={preview}")


def write_csv(rows: list[dict], output_csv: Path):
    fieldnames = [
        "idx",
        "judge_category",
        "judge_severity",
        "judge_confidence",
        "parse_error",
        "last_true_difficulty_idx",
        "last_difficulty_idx",
        "added_difficulty_steps",
        "last_true_prediction",
        "last_prediction",
        "ground_truth",
        "judge_went_wrong",
        "judge_evidence",
        "judge_example",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "idx": row.get("idx"),
                    "judge_category": row.get("judge_category"),
                    "judge_severity": row.get("judge_severity"),
                    "judge_confidence": row.get("judge_confidence"),
                    "parse_error": is_parse_error(row),
                    "last_true_difficulty_idx": row.get("last_true_difficulty_idx"),
                    "last_difficulty_idx": row.get("last_difficulty_idx"),
                    "added_difficulty_steps": int(row["last_difficulty_idx"])
                    - int(row["last_true_difficulty_idx"]),
                    "last_true_prediction": row.get("last_true_prediction"),
                    "last_prediction": row.get("last_prediction"),
                    "ground_truth": row.get("ground_truth"),
                    "judge_went_wrong": row.get("judge_went_wrong"),
                    "judge_evidence": row.get("judge_evidence"),
                    "judge_example": row.get("judge_example"),
                }
            )


def main():
    args = parse_args()
    input_path = Path(args.input_file).resolve()
    rows = load_jsonl(input_path)
    stats = build_stats(rows)
    print_stats(stats, rows, args.top_k)

    if args.output_json is not None:
        output_json = Path(args.output_json).resolve()
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"\nWrote aggregate JSON: {output_json}")

    if args.output_csv is not None:
        output_csv = Path(args.output_csv).resolve()
        write_csv(rows, output_csv)
        print(f"Wrote compact CSV: {output_csv}")


if __name__ == "__main__":
    main()
