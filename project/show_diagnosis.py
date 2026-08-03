#!/usr/bin/env python3
"""
CLI tool to inspect and print LLM Judge diagnoses and Overthinking Taxonomy results.
"""
import os
import sys
import glob
import json
import argparse


def find_latest_category_file(results_dir="results"):
    pattern = os.path.join(results_dir, "**", "*_failure_categories.jsonl")
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def print_diagnoses(file_path):
    print("=" * 80)
    print(f"  OVERTHINKING TAXONOMY DIAGNOSIS REPORT")
    print(f"  File: {file_path}")
    print("=" * 80)

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        print("\n  [INFO] No overthinking failures diagnosed in this file.")
        print("  (Either all samples were correct in the final step, or no failures occurred).\n")
        return

    print(f"\nTotal Diagnosed Overthinking Failures: {len(records)}\n")

    for i, r in enumerate(records, 1):
        idx = r.get("idx", "N/A")
        cat = r.get("judge_category", "N/A")
        sub_cats = r.get("judge_secondary_categories", [])
        went_wrong = r.get("judge_went_wrong", "N/A")
        evidence = r.get("judge_evidence", "N/A")
        example = r.get("judge_example", "N/A")
        
        last_true_step = r.get("last_true_difficulty_idx", "N/A")
        last_step = r.get("last_difficulty_idx", "N/A")
        last_true_pred = r.get("last_true_prediction", "N/A")
        last_pred = r.get("last_prediction", "N/A")
        gt = r.get("ground_truth", "N/A")
        question = r.get("original_question", "").strip()

        print("-" * 80)
        print(f"  [{i}/{len(records)}] Sample #{idx} (Question Index {idx})")
        print("-" * 80)
        if question:
            # truncate question if too long
            q_display = question if len(question) < 200 else question[:200] + "..."
            print(f"  Question:            {q_display}")
        print(f"  Ground Truth:        {gt}")
        print(f"  Earlier (Step {last_true_step}):     Answer was CORRECT -> {last_true_pred}")
        print(f"  Final   (Step {last_step}):     Answer FLIPPED TO WRONG -> {last_pred}")
        print(f"\n  [CATEGORY] Primary:      {cat.upper()}")
        if sub_cats:
            print(f"  [SECONDARY]         {', '.join(sub_cats)}")
        print(f"  [WHY IT FAILED]     {went_wrong}")
        if evidence and evidence != "N/A":
            print(f"  [KEY EVIDENCE]      \"{evidence}\"")
        if example and example != "N/A":
            print(f"  [QUOTE FROM TRACE]  \"{example}\"")
        print()


def main():
    parser = argparse.ArgumentParser(description="View LLM Judge Overthinking Diagnoses")
    parser.add_argument("file", nargs="?", default=None, help="Path to *_failure_categories.jsonl file")
    args = parser.parse_args()

    target_file = args.file
    if not target_file:
        target_file = find_latest_category_file()
        if not target_file:
            print("No failure categories file found in results/ folder.")
            sys.exit(1)

    print_diagnoses(target_file)


if __name__ == "__main__":
    main()
