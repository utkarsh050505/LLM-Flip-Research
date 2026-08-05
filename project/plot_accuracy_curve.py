"""
Plot Accuracy Curve & Flip Analysis -- Thinking Past the Answer.

Generates a publication-quality plot visualizing:
1. Accuracy Progression across reasoning / prefix steps.
2. Baseline full-thinking accuracy reference line.
3. Optimal reasoning budget point.
4. Flip detection summary card (cases where model was correct initially but flipped to wrong).

Usage:
  python plot_accuracy_curve.py                                             # Auto-detects latest run
  python plot_accuracy_curve.py --input results/main/r1_distill_qwen1_5b/gsm8k/seed_42
  python plot_accuracy_curve.py --input results/.../flip_analysis.json
  python plot_accuracy_curve.py --model r1_distill_qwen1_5b --benchmark gsm8k
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

# Fix Windows console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def find_latest_experiment_file(base_dir: str = "results") -> Optional[str]:
    """Auto-detect the most recently modified flip_analysis.json or difficulty file."""
    candidates = glob.glob(os.path.join(base_dir, "**", "flip_analysis.json"), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join(base_dir, "**", "difficulty_*.jsonl"), recursive=True)
    
    if not candidates:
        return None
    # Sort by modification time descending
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def load_analysis_data(input_path: str) -> Dict[str, Any]:
    """Load or compute flip analysis and accuracy curve data."""
    path = Path(input_path)
    
    if path.is_dir():
        # Look for flip_analysis.json in dir or subdirs
        fa = list(path.glob("**/flip_analysis.json"))
        if fa:
            path = fa[0]
        else:
            diff_files = list(path.glob("**/difficulty_*.jsonl"))
            if diff_files:
                path = diff_files[0]
            else:
                raise FileNotFoundError(f"No flip_analysis.json or difficulty_*.jsonl found in {input_path}")

    if path.name == "flip_analysis.json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["source_file"] = str(path)
        data["output_dir"] = str(path.parent)
        
        # If all_trajectories not present, reconstruct from parsed_responses file
        if "all_trajectories" not in data:
            parsed_candidates = list(path.parent.glob("parsed_responses_*.jsonl"))
            if parsed_candidates:
                from collections import defaultdict
                records = defaultdict(list)
                with open(parsed_candidates[0], "r", encoding="utf-8") as pf:
                    for line in pf:
                        if line.strip():
                            r = json.loads(line.strip())
                            records[r["idx"]].append(r)
                all_traj = {}
                for idx, plist in records.items():
                    plist.sort(key=lambda x: x.get("difficulty_idx", 0))
                    c_list = [str(pr.get("prediction", "")).strip().lower() == str(pr.get("ground_truth", "")).strip().lower() and bool(pr.get("prediction")) for pr in plist]
                    is_flip = any(fe["idx"] == idx for fe in data.get("flips", []))
                    all_traj[str(idx)] = {
                        "idx": idx,
                        "ground_truth": plist[0].get("ground_truth", ""),
                        "final_correct": c_list[-1] if c_list else False,
                        "is_flip": is_flip,
                        "correctness": c_list,
                    }
                data["all_trajectories"] = all_traj
        return data

    if path.suffix == ".jsonl":
        # Compute flip analysis directly from JSONL
        from collections import defaultdict
        records_by_idx = defaultdict(list)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    records_by_idx[r["idx"]].append(r)

        prefix_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
        flip_events = []
        total_samples = len(records_by_idx)

        for idx, plist in records_by_idx.items():
            plist.sort(key=lambda x: x.get("difficulty_idx", 0))
            correctness = []
            for pr in plist:
                pred = str(pr.get("model_parsed_answer", "")).strip().lower()
                gt = str(pr.get("ground_truth", "")).strip().lower()
                is_correct = (pred == gt) if pred and gt else False
                correctness.append(is_correct)
                d_idx = pr.get("difficulty_idx", 0)
                prefix_accuracy[d_idx]["total"] += 1
                if is_correct:
                    prefix_accuracy[d_idx]["correct"] += 1

            if len(correctness) > 1 and any(correctness[:-1]) and not correctness[-1]:
                last_v = max(i for i, c in enumerate(correctness[:-1]) if c)
                first_x = next(i for i in range(last_v + 1, len(correctness)) if not correctness[i])
                flip_events.append({
                    "idx": idx,
                    "ground_truth": plist[0].get("ground_truth", ""),
                    "final_prediction": plist[-1].get("model_parsed_answer", ""),
                    "last_correct_prefix_idx": last_v,
                    "first_incorrect_prefix_idx": first_x,
                    "total_prefixes": len(plist),
                    "prefix_trajectory": "".join(["[V]" if c else "[X]" for c in correctness]),
                })

        curve_data = []
        for d_idx in sorted(prefix_accuracy.keys()):
            pa = prefix_accuracy[d_idx]
            acc = (pa["correct"] / pa["total"] * 100) if pa["total"] > 0 else 0.0
            curve_data.append({"prefix_idx": d_idx, "accuracy": acc, "correct": pa["correct"], "total": pa["total"]})

        flip_rate = (len(flip_events) / total_samples * 100) if total_samples > 0 else 0.0
        return {
            "total_samples": total_samples,
            "flipped_count": len(flip_events),
            "flip_rate_pct": flip_rate,
            "flips": flip_events,
            "accuracy_curve": curve_data,
            "source_file": str(path),
            "output_dir": str(path.parent),
        }

    raise ValueError(f"Unsupported input file format: {input_path}")


def find_baseline_accuracy(output_dir: str) -> Optional[float]:
    """Find baseline accuracy from results.json if available."""
    # Look in parent or current directory
    current = Path(output_dir)
    for check_dir in [current, current.parent, current.parent.parent]:
        res_file = check_dir / "results.json"
        if res_file.exists():
            try:
                with open(res_file, "r", encoding="utf-8") as f:
                    res = json.load(f)
                    if "accuracy" in res:
                        return float(res["accuracy"])
            except Exception:
                pass
    return None


def plot_curve(
    data: Dict[str, Any],
    save_path: Optional[str] = None,
    title_suffix: str = "",
    dark_mode: bool = False,
    granularity: int = 10,
) -> str:
    """Generate and save the accuracy curve plot."""
    curve = data.get("accuracy_curve", [])
    if not curve:
        print("Error: No accuracy curve data found to plot.")
        return ""

    x_steps = [item["prefix_idx"] for item in curve]
    y_acc = [item["accuracy"] for item in curve]
    totals = [item.get("total", 0) for item in curve]

    total_samples = data.get("total_samples", len(totals))
    flipped_count = data.get("flipped_count", 0)
    flip_rate = data.get("flip_rate_pct", 0.0)
    flips = data.get("flips", [])
    baseline_acc = find_baseline_accuracy(data.get("output_dir", "."))

    # Styling
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"
    primary_color = "#38bdf8" if dark_mode else "#0284c7"
    fill_color = "#0284c7" if dark_mode else "#38bdf8"
    peak_color = "#10b981"
    flip_color = "#ef4444"
    baseline_color = "#f59e0b"

    # Trajectories data check
    trajectories = data.get("all_trajectories", {})
    has_trajectories = bool(trajectories) and len(trajectories) <= 50

    if has_trajectories:
        fig, (ax, ax_traj) = plt.subplots(
            2, 1, 
            figsize=(12, 11), 
            dpi=300, 
            facecolor=bg_color,
            gridspec_kw={"height_ratios": [1.1, 1.3], "hspace": 0.35}
        )
    else:
        fig, ax = plt.subplots(figsize=(11, 6.8), dpi=300, facecolor=bg_color)
        ax_traj = None

    ax.set_facecolor(bg_color)

    # Plot primary curve
    ax.plot(
        x_steps,
        y_acc,
        color=primary_color,
        linewidth=2.8,
        marker="o",
        markersize=6.5,
        markerfacecolor="white",
        markeredgecolor=primary_color,
        markeredgewidth=2.0,
        label="Prefix Budget Accuracy",
        zorder=5,
    )

    # Shaded area under curve
    ax.fill_between(x_steps, y_acc, alpha=0.15, color=fill_color, zorder=2)

    # Find peak accuracy
    max_acc = max(y_acc) if y_acc else 0.0
    max_indices = [i for i, a in enumerate(y_acc) if a == max_acc]
    first_peak_idx = max_indices[0] if max_indices else 0
    peak_step = x_steps[first_peak_idx] if x_steps else 0

    # Peak marker
    if y_acc:
        ax.scatter(
            [peak_step],
            [max_acc],
            color=peak_color,
            s=160,
            zorder=7,
            edgecolors="white",
            linewidth=2,
            label=f"Peak Accuracy ({max_acc:.1f}% @ Step {peak_step})",
        )
        ax.annotate(
            f"Peak: {max_acc:.1f}%\n(Step {peak_step})",
            xy=(peak_step, max_acc),
            xytext=(peak_step, min(100, max_acc + 6)),
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color=peak_color,
            arrowprops=dict(arrowstyle="->", color=peak_color, lw=1.5),
            zorder=8,
        )

    # Baseline reference line
    if baseline_acc is not None:
        ax.axhline(
            y=baseline_acc,
            color=baseline_color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.85,
            label=f"Full-Thinking Baseline ({baseline_acc:.1f}%)",
            zorder=3,
        )

    # Grid & Axes formatting
    ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7, zorder=1)
    ax.set_ylim(-2, 108)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold", color=text_color, labelpad=8)
    ax.set_xlabel(f"Reasoning Prefix Step (Granularity = {granularity} utterances)", fontsize=11, fontweight="bold", color=text_color, labelpad=8)

    # Title & Subtitle
    title_text = "Panel A: Accuracy Progression vs. Thinking Step"
    if title_suffix:
        title_text += f" ({title_suffix})"
    ax.set_title(title_text, fontsize=13, fontweight="bold", color=text_color, pad=12, loc="left")

    for spine in ax.spines.values():
        spine.set_color(grid_color)
        spine.set_linewidth(1.0)

    ax.tick_params(colors=subtext_color, labelsize=9.5)

    # Categorize all trajectories into rich archetypes
    recovered_count = 0
    stable_count = 0
    discovery_count = 0
    never_count = 0

    archetype_map = {}
    if trajectories:
        for k, v in trajectories.items():
            c_list = v.get("correctness", [])
            final_c = v.get("final_correct", False)
            any_c = any(c_list)
            is_flip = v.get("is_flip", False)

            if is_flip:
                archetype_map[k] = {"type": "FLIP", "label": "⚠️ [HARMFUL FLIP]", "color": "#f87171"}
            elif final_c and any(not c for c in c_list[:-1]) and any(c for c in c_list[:-1]):
                archetype_map[k] = {"type": "RECOVERED", "label": "⚡ [RECOVERED]", "color": "#fbbf24"}
                recovered_count += 1
            elif final_c and all(c_list):
                archetype_map[k] = {"type": "STABLE", "label": "✓ [STABLE CORRECT]", "color": "#34d399"}
                stable_count += 1
            elif final_c:
                archetype_map[k] = {"type": "DISCOVERY", "label": "💡 [DISCOVERY]", "color": "#60a5fa"}
                discovery_count += 1
            else:
                archetype_map[k] = {"type": "NEVER", "label": "✗ [NEVER CORRECT]", "color": "#94a3b8"}
                never_count += 1

    # Info Badge / Flip Summary Box
    flip_status_text = (
        f"Harmful Flips (V➔X)     : {flipped_count} ({flip_rate:.1f}%)\n"
        f"Hesitation/Recovered    : {recovered_count}\n"
        f"Total Samples           : {total_samples}\n"
        f"Max Accuracy            : {max_acc:.1f}%\n"
        f"Initial Acc             : {y_acc[0]:.1f}%"
    )
    if baseline_acc is not None:
        flip_status_text += f"\nFull Baseline           : {baseline_acc:.1f}%"

    badge_box_color = flip_color if flipped_count > 0 else peak_color
    badge_title = "⚠️ OVERTHINKING DETECTED" if flipped_count > 0 else "✓ NO HARMFUL FLIPS"
    box_text = f"{badge_title}\n" + "—" * 28 + f"\n{flip_status_text}"

    ax.text(
        0.98,
        0.05,
        box_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(
            boxstyle="round,pad=0.6,rounding_size=0.3",
            facecolor=card_bg,
            edgecolor=badge_box_color,
            linewidth=1.5,
            alpha=0.95,
        ),
        color=text_color,
        family="monospace",
        zorder=10,
    )

    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=card_bg,
        edgecolor=grid_color,
        fontsize=8.5,
        labelcolor=text_color,
    )

    # Panel B: Problem Trajectory Matrix (Heatmap)
    if ax_traj is not None and has_trajectories:
        ax_traj.set_facecolor(bg_color)
        sorted_indices = sorted(trajectories.keys(), key=lambda k: int(k))
        
        # Sort by priority: Flips first, then Recovered, then Discovery, then Stable, then Never
        type_priority = {"FLIP": 0, "RECOVERED": 1, "DISCOVERY": 2, "STABLE": 3, "NEVER": 4}
        ordered_keys = sorted(sorted_indices, key=lambda k: (type_priority.get(archetype_map.get(k, {}).get("type", "NEVER"), 5), int(k)))

        max_steps = max(len(trajectories[k]["correctness"]) for k in ordered_keys) if ordered_keys else 1
        num_rows = len(ordered_keys)

        matrix = np.full((num_rows, max_steps), np.nan)
        for r_idx, k in enumerate(ordered_keys):
            c_list = trajectories[k]["correctness"]
            for step_idx, is_c in enumerate(c_list):
                matrix[r_idx, step_idx] = 1.0 if is_c else 0.0

        # Custom colormap: 0 = Red (Incorrect), 1 = Green (Correct)
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(["#ef4444", "#10b981"])
        cmap.set_bad(color="#1e293b" if dark_mode else "#e2e8f0")

        im = ax_traj.imshow(matrix, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)

        # Highlight transition regions
        for r_idx, k in enumerate(ordered_keys):
            arch = archetype_map.get(k, {})
            c_list = trajectories[k]["correctness"]
            
            if arch.get("type") == "FLIP":
                last_v = max(i for i, c in enumerate(c_list[:-1]) if c)
                first_x = next(i for i in range(last_v + 1, len(c_list)) if not c_list[i])
                
                rect = plt.Rectangle(
                    (last_v - 0.45, r_idx - 0.45),
                    (first_x - last_v + 1.9),
                    0.9,
                    fill=False,
                    edgecolor="#ef4444",
                    linewidth=2.2,
                    linestyle="--",
                    zorder=15,
                )
                ax_traj.add_patch(rect)
                ax_traj.annotate(
                    " ⚠️ FLIP (Correct ➔ Wrong) ",
                    xy=(first_x + 0.5, r_idx),
                    xytext=(first_x + 1.2, r_idx),
                    fontsize=8.5,
                    fontweight="bold",
                    color="#f87171",
                    va="center",
                    zorder=20,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=card_bg, edgecolor="#ef4444", lw=1.0)
                )
            elif arch.get("type") == "RECOVERED":
                # Find the red dip regions that happened between first correct and final correct
                c_true_indices = [i for i, c in enumerate(c_list) if c]
                if len(c_true_indices) >= 2:
                    start_v = c_true_indices[0]
                    end_v = c_true_indices[-1]
                    dips = [i for i in range(start_v, end_v + 1) if not c_list[i]]
                    if dips:
                        first_dip = dips[0]
                        last_dip = dips[-1]
                        rect = plt.Rectangle(
                            (first_dip - 0.45, r_idx - 0.45),
                            (last_dip - first_dip + 0.9),
                            0.9,
                            fill=False,
                            edgecolor="#fbbf24",
                            linewidth=1.6,
                            linestyle=":",
                            zorder=15,
                        )
                        ax_traj.add_patch(rect)

        # Y-ticks
        y_labels = []
        for k in ordered_keys:
            tag = archetype_map.get(k, {}).get("label", "")
            y_labels.append(f"Sample #{k:<2} {tag}")

        ax_traj.set_yticks(range(num_rows))
        ax_traj.set_yticklabels(y_labels, fontsize=8.5, family="monospace")
        
        for r_idx, k in enumerate(ordered_keys):
            c_color = archetype_map.get(k, {}).get("color", text_color)
            ax_traj.get_yticklabels()[r_idx].set_color(c_color)
            if archetype_map.get(k, {}).get("type") in ["FLIP", "RECOVERED"]:
                ax_traj.get_yticklabels()[r_idx].set_fontweight("bold")

        ax_traj.set_xticks(range(max_steps))
        ax_traj.set_xticklabels([f"Step {s}" for s in range(max_steps)], fontsize=8.5, color=subtext_color)
        ax_traj.set_xlabel("Reasoning Prefix Step (Thinking Depth)", fontsize=11, fontweight="bold", color=text_color, labelpad=8)
        ax_traj.set_title("Panel B: Reasoning Trajectories by Archetype (🟩 Correct | 🟥 Incorrect | ⬛ Completed)", fontsize=12, fontweight="bold", color=text_color, pad=10, loc="left")

        for spine in ax_traj.spines.values():
            spine.set_color(grid_color)
            spine.set_linewidth(1.0)

    plt.tight_layout()

    # Determine save path
    if not save_path:
        out_dir = data.get("output_dir", ".")
        save_path = os.path.join(out_dir, "accuracy_curve.png")

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    
    # Also save PDF version for paper / reports
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()

    return save_path


def print_summary_table(data: Dict[str, Any], output_img: str) -> None:
    """Print a clean CLI summary table."""
    curve = data.get("accuracy_curve", [])
    total_samples = data.get("total_samples", 0)
    flips = data.get("flips", [])
    flip_rate = data.get("flip_rate_pct", 0.0)

    print("\n" + "=" * 70)
    print("  THINKING PAST THE ANSWER: ACCURACY CURVE & FLIP REPORT")
    print("=" * 70)
    print(f"  Total Samples Evaluated : {total_samples}")
    print(f"  Harmful Overthinking    : {len(flips)} flipped cases ({flip_rate:.1f}%)")
    print(f"  High-Res Plot Saved To  : {output_img}")
    print("=" * 70)

    if flips:
        print("\n  [FLIPPED CASES] (Initially Correct -> Flipped to Incorrect):")
        print(f"  {'Idx':<6} | {'Last V Step':<12} | {'First X Step':<12} | {'Trajectory'}")
        print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*25}")
        for f in flips:
            print(f"  {f.get('idx', '-'):<6} | {f.get('last_correct_prefix_idx', '-'):<12} | {f.get('first_incorrect_prefix_idx', '-'):<12} | {f.get('prefix_trajectory', '')}")

    print("\n  [ACCURACY PROGRESSION]:")
    print(f"  {'Prefix Step':<15} | {'Accuracy':<12} | {'Correct / Total'}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*18}")
    for item in curve:
        p_idx = item["prefix_idx"]
        acc = item["accuracy"]
        cor = item.get("correct", 0)
        tot = item.get("total", 0)
        bar = "#" * int(acc / 5)
        print(f"  Step {p_idx:<10} | {acc:6.1f}%      | ({cor}/{tot}) {bar}")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Accuracy Curve and Overthinking Flips")
    parser.add_argument(
        "--input", "-i", type=str, default=None,
        help="Path to flip_analysis.json, difficulty_*.jsonl, or experiment directory"
    )
    parser.add_argument("--model", type=str, default=None, help="Model name (e.g. r1_distill_qwen1_5b)")
    parser.add_argument("--benchmark", type=str, default=None, help="Benchmark name (e.g. gsm8k)")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for experiment")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image file path (.png)")
    parser.add_argument("--dark", action="store_true", help="Render modern dark-mode theme")
    parser.add_argument("--granularity", type=int, default=10, help="Step granularity in utterances")

    args = parser.parse_args()

    input_path = args.input
    if not input_path:
        if args.model and args.benchmark:
            # Construct standard path
            pattern = os.path.join("results", "*", args.model, args.benchmark, f"seed_{args.seed}")
            matches = glob.glob(pattern)
            if matches:
                input_path = matches[0]
        if not input_path:
            input_path = find_latest_experiment_file()

    if not input_path:
        print("Error: No experiment results found. Please specify --input path.")
        sys.exit(1)

    print(f"Loading data from: {input_path}")
    data = load_analysis_data(input_path)

    title_suffix = ""
    if args.model or args.benchmark:
        title_suffix = f"{args.model or ''} on {args.benchmark or ''}".strip()

    img_path = plot_curve(
        data=data,
        save_path=args.output,
        title_suffix=title_suffix,
        dark_mode=args.dark,
        granularity=args.granularity,
    )

    print_summary_table(data, img_path)


if __name__ == "__main__":
    main()
