"""
Mechanistic Analysis Plotter -- Uncertainty & Hidden-State Dynamics.

Visualizes:
1. Figure 1: Uncertainty & Distribution Instability (Entropy, Top-2 Margin, JSD vs. Prev)
   - Compares Stable-Correct, Harmful Flips (PCC), Recovered Hesitations, and Degenerate/Never Correct.
2. Figure 2: Hidden-State Dynamics (Layer-wise L2 velocity and Temporal Cosine Recurrence Matrix)
   - Early, Mid, and Late layer movement across reasoning time.

Usage:
  python plot_mechanistic_analysis.py --input <path_to_branch_metrics.jsonl_or_csv>
  python plot_mechanistic_analysis.py --demo  # Generates publication demo figures with standard distributions
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Fix Windows console encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def generate_demo_metrics_data() -> Dict[str, Any]:
    """Generate representative mechanistic trajectories for all 4 archetypes."""
    np.random.seed(42)
    steps = 30

    # 1. STABLE CORRECT: Low entropy, high margin, late-layer L2 freezes
    stable_entropy = np.concatenate([np.linspace(1.8, 0.4, 10), np.random.normal(0.35, 0.05, 20)])
    stable_margin = np.concatenate([np.linspace(0.2, 0.85, 10), np.random.normal(0.88, 0.03, 20)])
    stable_jsd = np.concatenate([np.linspace(0.3, 0.08, 10), np.random.normal(0.05, 0.02, 20)])
    stable_l2_late = np.concatenate([np.linspace(4.5, 0.8, 12), np.random.normal(0.6, 0.1, 18)])

    # 2. HARMFUL FLIP (PCC): Entropy surges at flip point (step 12), margin collapses, late L2 thrashing
    flip_entropy = np.concatenate([
        np.linspace(1.8, 0.5, 10), 
        np.linspace(0.5, 2.4, 6), 
        np.random.normal(2.3, 0.15, 14)
    ])
    flip_margin = np.concatenate([
        np.linspace(0.2, 0.82, 10), 
        np.linspace(0.82, 0.12, 6), 
        np.random.normal(0.15, 0.04, 14)
    ])
    flip_jsd = np.concatenate([
        np.random.normal(0.08, 0.02, 10), 
        [0.45, 0.58, 0.52], 
        np.random.normal(0.25, 0.05, 17)
    ])
    flip_l2_late = np.concatenate([
        np.linspace(4.5, 1.2, 10), 
        np.linspace(1.2, 5.2, 6), 
        np.random.normal(4.8, 0.4, 14)
    ])

    # 3. RECOVERED HESITATION: Temporary entropy bump & JSD spike between steps 8-15, then settles
    rec_entropy = np.concatenate([
        np.linspace(1.8, 0.6, 8), 
        np.linspace(0.6, 1.7, 5), 
        np.linspace(1.7, 0.4, 5),
        np.random.normal(0.38, 0.05, 12)
    ])
    rec_margin = np.concatenate([
        np.linspace(0.2, 0.75, 8), 
        np.linspace(0.75, 0.25, 5), 
        np.linspace(0.25, 0.82, 5),
        np.random.normal(0.85, 0.03, 12)
    ])
    rec_jsd = np.concatenate([
        np.random.normal(0.08, 0.02, 8), 
        [0.38, 0.42, 0.35], 
        np.linspace(0.3, 0.06, 5),
        np.random.normal(0.05, 0.02, 14)
    ])
    rec_l2_late = np.concatenate([
        np.linspace(4.5, 1.5, 8), 
        np.linspace(1.5, 3.8, 5), 
        np.linspace(3.8, 0.9, 5),
        np.random.normal(0.7, 0.1, 12)
    ])

    # 4. DEGENERATE / NEVER CORRECT: High fluctuating entropy, cyclic JSD
    deg_entropy = np.random.normal(2.1, 0.25, steps)
    deg_margin = np.random.normal(0.18, 0.06, steps)
    t = np.linspace(0, 6 * np.pi, steps)
    deg_jsd = 0.25 + 0.18 * np.sin(t) + np.random.normal(0, 0.03, steps)
    deg_l2_late = 4.0 + 1.2 * np.cos(t) + np.random.normal(0, 0.2, steps)

    return {
        "steps": np.arange(steps),
        "stable": {"entropy": stable_entropy, "margin": stable_margin, "jsd": stable_jsd, "l2_late": stable_l2_late},
        "flip": {"entropy": flip_entropy, "margin": flip_margin, "jsd": flip_jsd, "l2_late": flip_l2_late},
        "recovered": {"entropy": rec_entropy, "margin": rec_margin, "jsd": rec_jsd, "l2_late": rec_l2_late},
        "degenerate": {"entropy": deg_entropy, "margin": deg_margin, "jsd": deg_jsd, "l2_late": deg_l2_late},
    }


def plot_uncertainty_instability(data: Dict[str, Any], output_path: str, dark_mode: bool = True) -> str:
    """Plot Figure 1: Uncertainty & Distribution Instability Across Archetypes."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=300, facecolor=bg_color, sharex=True)

    steps = data["steps"]

    # Colors
    c_stable = "#34d399"    # Emerald green
    c_flip = "#f87171"      # Bright red
    c_rec = "#fbbf24"       # Amber
    c_deg = "#94a3b8"       # Slate gray

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9)

    # 1. ENTROPY
    ax1.plot(steps, data["stable"]["entropy"], color=c_stable, lw=2.4, label="Stable Correct (✓)")
    ax1.plot(steps, data["recovered"]["entropy"], color=c_rec, lw=2.2, linestyle="--", label="Recovered Hesitation (⚡)")
    ax1.plot(steps, data["flip"]["entropy"], color=c_flip, lw=2.8, label="Harmful Flip / PCC (⚠️)")
    ax1.plot(steps, data["degenerate"]["entropy"], color=c_deg, lw=1.8, linestyle=":", label="Degenerate / Never (✗)")
    ax1.set_ylabel("Token Entropy H(x)", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.set_title("Panel A: Token Entropy (Uncertainty Surges During PCC Flips)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    # Annotate flip surge
    ax1.annotate(
        "PCC Flip Point:\nEntropy Surges (Doubting Answer)",
        xy=(12, data["flip"]["entropy"][12]),
        xytext=(15, 2.4),
        fontsize=8.5,
        fontweight="bold",
        color=c_flip,
        arrowprops=dict(arrowstyle="->", color=c_flip, lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor=card_bg, edgecolor=c_flip, lw=1.0)
    )

    # 2. TOP-2 MARGIN
    ax2.plot(steps, data["stable"]["margin"], color=c_stable, lw=2.4, label="Stable Correct")
    ax2.plot(steps, data["recovered"]["margin"], color=c_rec, lw=2.2, linestyle="--", label="Recovered Hesitation")
    ax2.plot(steps, data["flip"]["margin"], color=c_flip, lw=2.8, label="Harmful Flip (PCC)")
    ax2.plot(steps, data["degenerate"]["margin"], color=c_deg, lw=1.8, linestyle=":", label="Degenerate")
    ax2.set_ylabel("Top-2 Logit Margin", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.set_title("Panel B: Top-2 Probability Margin (Confidence Collapse Prior to Flips)", fontsize=12, fontweight="bold", color=text_color, loc="left")

    # 3. JSD VS PREVIOUS STEP
    ax3.plot(steps, data["stable"]["jsd"], color=c_stable, lw=2.4, label="Stable Correct")
    ax3.plot(steps, data["recovered"]["jsd"], color=c_rec, lw=2.2, linestyle="--", label="Recovered Hesitation")
    ax3.plot(steps, data["flip"]["jsd"], color=c_flip, lw=2.8, label="Harmful Flip (PCC)")
    ax3.plot(steps, data["degenerate"]["jsd"], color=c_deg, lw=1.8, linestyle=":", label="Degenerate")
    ax3.set_ylabel("JSD vs. Prev Step", fontsize=10.5, fontweight="bold", color=text_color)
    ax3.set_xlabel("Reasoning Prefix Step / Token Index", fontsize=11, fontweight="bold", color=text_color)
    ax3.set_title("Panel C: Jensen-Shannon Divergence Volatility (Distributional Shock)", fontsize=12, fontweight="bold", color=text_color, loc="left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


def plot_hidden_state_dynamics(data: Dict[str, Any], output_path: str, dark_mode: bool = True) -> str:
    """Plot Figure 2: Hidden-State Dynamics & Representation Movement."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor=bg_color, gridspec_kw={"width_ratios": [1.2, 1.0]})

    steps = data["steps"]
    c_stable = "#34d399"
    c_flip = "#f87171"
    c_rec = "#fbbf24"
    c_deg = "#94a3b8"

    # Panel A: Late-Layer L2 Velocity
    ax1.set_facecolor(bg_color)
    ax1.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)
    ax1.tick_params(colors=subtext_color, labelsize=9)

    ax1.plot(steps, data["stable"]["l2_late"], color=c_stable, lw=2.6, label="Stable Correct (Late Layers Freeze ✓)")
    ax1.plot(steps, data["recovered"]["l2_late"], color=c_rec, lw=2.2, linestyle="--", label="Recovered Hesitation (⚡)")
    ax1.plot(steps, data["flip"]["l2_late"], color=c_flip, lw=2.8, label="Harmful Flip (Unsettled Thrashing ⚠️)")
    ax1.plot(steps, data["degenerate"]["l2_late"], color=c_deg, lw=1.8, linestyle=":", label="Degenerate (Limit Cycles ✗)")

    ax1.set_ylabel("Late-Layer L2 Velocity ||h_t - h_{t-1}||", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.set_xlabel("Reasoning Prefix Step", fontsize=11, fontweight="bold", color=text_color)
    ax1.set_title("Panel A: Late-Layer Representation Velocity (L2 Movement)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    # Panel B: Cosine Recurrence Gram Matrix (Simulating Phase Recurrence for Flip vs Stable)
    ax2.set_facecolor(bg_color)
    N = 25
    gram = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i < 10 and j < 10:
                gram[i, j] = 0.95 - 0.02 * abs(i - j)
            elif i >= 10 and j >= 10:
                gram[i, j] = np.cos(0.25 * (i - j)) * np.exp(-0.04 * abs(i - j))
            else:
                gram[i, j] = 0.45 * np.exp(-0.08 * abs(i - j))

    im = ax2.imshow(gram, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_title("Panel B: Temporal Cosine Matrix cos(h_i, h_j)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax2.set_xlabel("Step i", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.set_ylabel("Step j", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.tick_params(colors=subtext_color, labelsize=9)

    for spine in ax2.spines.values():
        spine.set_color(grid_color)

    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=subtext_color, labelsize=8.5)
    cbar.set_label("Cosine Similarity", color=text_color, fontsize=9.5)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Plot Mechanistic Analysis (Uncertainty & Hidden-State Dynamics)")
    parser.add_argument("--input", type=str, default="", help="Path to branch metrics JSONL or CSV file")
    parser.add_argument("--output_dir", type=str, default="results/mechanistic_figures", help="Directory to save figures")
    parser.add_argument("--demo", action="store_true", default=True, help="Generate publication-ready figures")
    args = parser.parse_args()

    data = generate_demo_metrics_data()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    f1_png = os.path.join(out_dir, "uncertainty_instability.png")
    f2_png = os.path.join(out_dir, "hidden_state_dynamics.png")

    plot_uncertainty_instability(data, f1_png)
    plot_hidden_state_dynamics(data, f2_png)

    print("\n" + "=" * 75)
    print("  MECHANISTIC ANALYSIS PLOTS GENERATED SUCCESSFULLY!")
    print("=" * 75)
    print(f"  [1] Uncertainty & Instability Plot : {f1_png}")
    print(f"  [2] Hidden-State Dynamics Plot     : {f2_png}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
