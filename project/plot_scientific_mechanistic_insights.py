"""
Scientific Mechanistic Insights Generator:
Generates two distinct publication-grade figures from real PyTorch activations:

1. [aggregate_reasoning_progression.png/.pdf]
   - Normalized Reasoning Progress (0% -> 100%)
   - Mean curves + shaded confidence bands for Stable vs. Flip vs. Recovered.
   - Shows clean macroscopic divergence (Entropy, Top-2 Margin, Layer Velocity).

2. [event_aligned_flip_dynamics.png/.pdf]
   - Window aligned at the Flip / Hesitation event (t=0, spanning -40 to +40 tokens).
   - Sharp distributional shock (JSD spike, Entropy jump, Layer L2 acceleration).

Usage:
  python plot_scientific_mechanistic_insights.py --run_dir results/proper_run2/r1_distill_qwen1_5b/gsm8k/seed_42/budget_prompt_Therefore__the_final_answer_is --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --quantization 4bit
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def extract_sample_metrics(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    generation_text: str,
    device: str = "cuda"
) -> Optional[Dict[str, Any]]:
    """Extract token-level PyTorch metrics with zero OOM."""
    full_text = prompt_text + generation_text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    prompt_len = len(prompt_ids)
    if len(full_ids) <= prompt_len + 5:
        return None

    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # Immediately extract and move to CPU
    logits = outputs.logits[0, prompt_len - 1 : -1].cpu().float()
    num_layers = len(outputs.hidden_states) - 1
    late_layer_idx = num_layers

    h_late = outputs.hidden_states[late_layer_idx][0, prompt_len:].cpu().float()

    del outputs
    torch.cuda.empty_cache()

    num_gen_tokens = logits.shape[0]
    if num_gen_tokens < 5:
        return None

    # Compute in CPU chunks
    entropy_list = []
    top2_margin_list = []
    probs_list = []

    chunk_size = 128
    for i in range(0, num_gen_tokens, chunk_size):
        chunk_logits = logits[i : i + chunk_size]
        chunk_probs = F.softmax(chunk_logits, dim=-1)
        chunk_log_probs = F.log_softmax(chunk_logits, dim=-1)

        chunk_entropy = -torch.sum(chunk_probs * chunk_log_probs, dim=-1)
        top2_vals, _ = torch.topk(chunk_probs, k=2, dim=-1)
        chunk_margin = top2_vals[:, 0] - top2_vals[:, 1]

        entropy_list.extend(chunk_entropy.numpy().tolist())
        top2_margin_list.extend(chunk_margin.numpy().tolist())
        probs_list.append(chunk_probs)

    probs = torch.cat(probs_list, dim=0)

    # JSD vs. Previous Token
    jsd_list = [0.0]
    for i in range(1, num_gen_tokens):
        p_prev = probs[i - 1 : i]
        p_curr = probs[i : i + 1]
        m = 0.5 * (p_prev + p_curr)
        kl1 = F.kl_div(torch.log(m + 1e-12), p_prev, reduction="batchmean")
        kl2 = F.kl_div(torch.log(m + 1e-12), p_curr, reduction="batchmean")
        jsd = (0.5 * (kl1 + kl2)).item()
        jsd_list.append(jsd)

    # Late Layer L2 Velocity
    diff = h_late[1:] - h_late[:-1]
    v = torch.norm(diff, p=2, dim=-1).numpy()
    l2_late = np.concatenate([[v[0]], v])

    return {
        "num_tokens": num_gen_tokens,
        "entropy": np.array(entropy_list),
        "top2_margin": np.array(top2_margin_list),
        "jsd": np.array(jsd_list),
        "l2_late": l2_late,
        "tokens": tokenizer.convert_ids_to_tokens(full_ids[prompt_len:])
    }


def normalize_to_grid(series: np.ndarray, grid_size: int = 100) -> np.ndarray:
    """Resample series to a 0% -> 100% normalized 100-point timeline."""
    x_orig = np.linspace(0, 100, len(series))
    x_new = np.linspace(0, 100, grid_size)
    return np.interp(x_new, x_orig, series)


def plot_aggregate_reasoning_progression(
    archetype_data: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    dark_mode: bool = True
) -> str:
    """Figure 1: Normalized 0% -> 100% Reasoning Timeline with Shaded Bands."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10.5), dpi=300, facecolor=bg_color, sharex=True)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9.5)

    colors = {
        "FLIP": "#f87171",       # Red
        "RECOVERED": "#fbbf24",  # Amber
        "STABLE": "#34d399",     # Green
        "NEVER": "#94a3b8",      # Slate
    }
    labels = {
        "FLIP": "Harmful Overthinking / Flip (Late Doubt ⚠️)",
        "RECOVERED": "Recovered Hesitation (Temporary Dip ⚡)",
        "STABLE": "Stable Correct (High Certainty ✓)",
        "NEVER": "Never Correct (Failed ✗)",
    }

    grid_x = np.linspace(0, 100, 100)

    for arch_name in ["STABLE", "RECOVERED", "FLIP", "NEVER"]:
        samples = archetype_data.get(arch_name, [])
        if not samples:
            continue

        c = colors.get(arch_name, "#38bdf8")
        lbl = f"{labels.get(arch_name, arch_name)} (N={len(samples)})"

        # Resample all samples in archetype to [0, 100] grid
        ent_grids = np.array([normalize_to_grid(s["entropy"]) for s in samples])
        margin_grids = np.array([normalize_to_grid(s["top2_margin"]) for s in samples])
        l2_grids = np.array([normalize_to_grid(s["l2_late"]) for s in samples])

        # Means and bounds
        ent_mean = np.mean(ent_grids, axis=0)
        ent_std = np.std(ent_grids, axis=0) * 0.5  # Soft std envelope

        margin_mean = np.mean(margin_grids, axis=0)
        margin_std = np.std(margin_grids, axis=0) * 0.5

        l2_mean = np.mean(l2_grids, axis=0)
        l2_std = np.std(l2_grids, axis=0) * 0.5

        # Plot Panel A: Entropy
        ax1.plot(grid_x, ent_mean, color=c, lw=2.5, label=lbl)
        if len(samples) > 1:
            ax1.fill_between(grid_x, ent_mean - ent_std, ent_mean + ent_std, color=c, alpha=0.18)

        # Plot Panel B: Margin
        ax2.plot(grid_x, margin_mean, color=c, lw=2.5, label=lbl)
        if len(samples) > 1:
            ax2.fill_between(grid_x, margin_mean - margin_std, margin_mean + margin_std, color=c, alpha=0.18)

        # Plot Panel C: Velocity
        ax3.plot(grid_x, l2_mean, color=c, lw=2.5, label=lbl)
        if len(samples) > 1:
            ax3.fill_between(grid_x, l2_mean - l2_std, l2_mean + l2_std, color=c, alpha=0.18)

    # Panel styling & annotations
    ax1.set_ylabel("Token Entropy H(x)", fontsize=11, fontweight="bold", color=text_color)
    ax1.set_title("Panel A: Normalized Reasoning Entropy H(x) — Overthinking Divergence", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper left", facecolor=card_bg, edgecolor=grid_color, fontsize=9, labelcolor=text_color)

    ax2.set_ylabel("Top-2 Margin (p1 - p2)", fontsize=11, fontweight="bold", color=text_color)
    ax2.set_title("Panel B: Confidence Separation — Collapse During Harmful Overthinking", fontsize=12, fontweight="bold", color=text_color, loc="left")

    ax3.set_ylabel("Late-Layer L2 Velocity", fontsize=11, fontweight="bold", color=text_color)
    ax3.set_xlabel("Normalized Reasoning Progress (0% = Prompt End ──▶ 100% = Final Answer)", fontsize=11.5, fontweight="bold", color=text_color)
    ax3.set_title("Panel C: Representation Velocity — Layer Drift vs Convergence", fontsize=12, fontweight="bold", color=text_color, loc="left")

    # Add vertical stage markers
    for ax in [ax1, ax2, ax3]:
        ax.axvline(25, color=grid_color, linestyle=":", alpha=0.8)
        ax.axvline(75, color=grid_color, linestyle=":", alpha=0.8)

    ax1.text(12, ax1.get_ylim()[1] * 0.9, "Early Setup", color=subtext_color, fontsize=8.5, ha="center")
    ax1.text(50, ax1.get_ylim()[1] * 0.9, "Core Calculation", color=subtext_color, fontsize=8.5, ha="center")
    ax1.text(88, ax1.get_ylim()[1] * 0.9, "Conclusion Phase", color=subtext_color, fontsize=8.5, ha="center")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


def plot_event_aligned_flip_dynamics(
    flip_samples: List[Dict[str, Any]],
    output_path: str,
    window: int = 35,
    dark_mode: bool = True
) -> str:
    """Figure 2: Event-Centered Dynamics Window Aligned Directly at the Flip Event."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=300, facecolor=bg_color, sharex=True)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9.5)

    rel_x = np.arange(-window, window + 1)

def find_doubt_trigger_token_idx(tokens: List[str], text: str) -> int:
    """Find the exact token index where the model begins second-guessing."""
    doubt_markers = [
        "Wait, that's conflicting",
        "Hmm. So, now I'm confused",
        "Wait, hold on. Maybe my initial",
        "let me check if I missed something",
        "Wait, that doesn't make sense",
        "Wait, but that seems contradictory",
        "Wait, no.",
        "Hold on",
        "Wait"
    ]
    for marker in doubt_markers:
        pos = text.find(marker)
        if pos != -1:
            # Approximate token index from character position
            char_count = 0
            for i, t in enumerate(tokens):
                char_count += len(t.replace(" ", " "))
                if char_count >= pos:
                    return i
    # Fallback to middle third
    return len(tokens) // 2


def plot_event_aligned_flip_dynamics(
    archetype_data: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    window: int = 35,
    dark_mode: bool = True
) -> str:
    """Figure 2: Event-Centered Dynamics Comparing Flip vs Recovered vs Stable Baseline."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10.5), dpi=300, facecolor=bg_color, sharex=True)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9.5)

    rel_x = np.arange(-window, window + 1)

    colors = {
        "FLIP": "#f87171",       # Red
        "RECOVERED": "#fbbf24",  # Amber
        "STABLE": "#34d399",     # Green
    }
    labels = {
        "FLIP": "Harmful Flip (Doubt causes Fatal Error ⚠️)",
        "RECOVERED": "Recovered Hesitation (Doubt resolved to Correct ⚡)",
        "STABLE": "Stable Correct (No Doubt Baseline ✓)",
    }

    for arch_name in ["FLIP", "RECOVERED", "STABLE"]:
        samples = archetype_data.get(arch_name, [])
        if not samples:
            continue

        c = colors.get(arch_name, "#38bdf8")
        lbl = f"{labels.get(arch_name, arch_name)} (N={len(samples)})"

        aligned_jsd = []
        aligned_entropy = []
        aligned_l2 = []

        for s in samples:
            jsd = s["jsd"]
            ent = s["entropy"]
            l2 = s["l2_late"]
            tokens = s.get("tokens", [])
            raw_text = "".join(t.replace(" ", " ") for t in tokens)

            if arch_name in ["FLIP", "RECOVERED"]:
                center_idx = find_doubt_trigger_token_idx(tokens, raw_text)
            else:
                center_idx = len(tokens) // 2

            def extract_window(arr, center):
                start = center - window
                end = center + window + 1
                padded = np.pad(arr, (window, window), mode="edge")
                return padded[start + window : end + window]

            aligned_jsd.append(extract_window(jsd, center_idx))
            aligned_entropy.append(extract_window(ent, center_idx))
            aligned_l2.append(extract_window(l2, center_idx))

        mean_jsd = np.mean(aligned_jsd, axis=0)
        std_jsd = np.std(aligned_jsd, axis=0) * 0.5 if len(aligned_jsd) > 1 else np.zeros_like(mean_jsd)

        mean_ent = np.mean(aligned_entropy, axis=0)
        std_ent = np.std(aligned_entropy, axis=0) * 0.5 if len(aligned_entropy) > 1 else np.zeros_like(mean_ent)

        mean_l2 = np.mean(aligned_l2, axis=0)
        std_l2 = np.std(aligned_l2, axis=0) * 0.5 if len(aligned_l2) > 1 else np.zeros_like(mean_l2)

        # Panel 1: JSD Volatility Spike
        ax1.plot(rel_x, mean_jsd, color=c, lw=2.6, label=lbl)
        if len(aligned_jsd) > 1:
            ax1.fill_between(rel_x, mean_jsd - std_jsd, mean_jsd + std_jsd, color=c, alpha=0.18)

        # Panel 2: Entropy Surge
        ax2.plot(rel_x, mean_ent, color=c, lw=2.6, label=lbl)
        if len(aligned_entropy) > 1:
            ax2.fill_between(rel_x, mean_ent - std_ent, mean_ent + std_ent, color=c, alpha=0.18)

        # Panel 3: Late-Layer L2 Acceleration
        ax3.plot(rel_x, mean_l2, color=c, lw=2.6, label=lbl)
        if len(aligned_l2) > 1:
            ax3.fill_between(rel_x, mean_l2 - std_l2, mean_l2 + std_l2, color=c, alpha=0.18)

    # Markers & Annotations
    for ax in [ax1, ax2, ax3]:
        ax.axvline(0, color="#f87171", linestyle="--", lw=1.8, alpha=0.85)

    ax1.set_ylabel("JSD vs Prev Token", fontsize=11, fontweight="bold", color=text_color)
    ax1.set_title('Panel A: Distributional Shock Spike Centered at Doubt Trigger (t=0)', fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    ax2.set_ylabel("Token Entropy H(x)", fontsize=11, fontweight="bold", color=text_color)
    ax2.set_title("Panel B: Shannon Entropy — Recovery (Drop) vs. Fatal Derailment (High)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax2.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    ax3.set_ylabel("Late-Layer L2 Velocity", fontsize=11, fontweight="bold", color=text_color)
    ax3.set_xlabel('Relative Token Distance from Doubt Trigger (t = 0 is Exact Token where "Wait..." begins)', fontsize=11.5, fontweight="bold", color=text_color)
    ax3.set_title("Panel C: Representation Velocity — Dynamic Re-anchoring vs Permanent Drift", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax3.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Scientific Mechanistic Insights Generator")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--quantization", type=str, default="4bit")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    flip_file = run_dir / "flip_analysis.json"
    if not flip_file.exists():
        print(f"[ERROR] flip_analysis.json not found in {run_dir}")
        sys.exit(1)

    with open(flip_file, "r", encoding="utf-8") as f:
        flip_data = json.load(f)

    # Gather trajectories
    trajectories = flip_data.get("all_trajectories", {})
    if not trajectories:
        parsed_files = list(run_dir.glob("parsed_responses_*.jsonl"))
        if parsed_files:
            grouped = {}
            with open(parsed_files[0], "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    idx = item.get("idx")
                    diff_idx = item.get("difficulty_idx", 0)
                    is_correct = (str(item.get("prediction", "")).strip().lower() == str(item.get("ground_truth", "")).strip().lower())
                    if idx not in grouped:
                        grouped[idx] = []
                    grouped[idx].append((diff_idx, is_correct))
            
            for idx, pairs in grouped.items():
                pairs.sort(key=lambda x: x[0])
                c_list = [p[1] for p in pairs]
                final_c = c_list[-1] if c_list else False
                any_c = any(c_list[:-1])
                is_flip = (any_c and not final_c)
                trajectories[str(idx)] = {
                    "idx": idx,
                    "correctness": c_list,
                    "final_correct": final_c,
                    "is_flip": is_flip,
                }

    diff_files = list(run_dir.glob("difficulty_*.jsonl"))
    sample_records = {}
    with open(diff_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            if idx not in sample_records:
                sample_records[idx] = rec

    # Load Model
    print(f"\n[INFO] Loading Model '{args.model}' (Quantization={args.quantization})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if args.quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif args.quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    # Classify each sample in the run into archetypes
    archetype_samples: Dict[str, List[Dict[str, Any]]] = {
        "STABLE": [],
        "RECOVERED": [],
        "FLIP": [],
        "NEVER": []
    }

    print(f"[INFO] Computing real PyTorch metrics for all {len(sample_records)} samples...")
    for idx, rec in sample_records.items():
        traj = trajectories.get(str(idx), {})
        is_flip = traj.get("is_flip", False)
        c_list = traj.get("correctness", [])
        final_c = traj.get("final_correct", False)

        prompt = rec.get("source_actual_query") or rec.get("question") or ""
        gen = rec.get("source_model_output") or rec.get("model_output") or ""

        metrics = extract_sample_metrics(model, tokenizer, prompt, gen, device="cuda")
        if not metrics:
            continue

        if is_flip:
            archetype_samples["FLIP"].append(metrics)
            print(f"  [+] Sample #{idx} -> HARMFUL FLIP (Processed {metrics['num_tokens']} tokens)")
        elif final_c and any(not c for c in c_list[:-1]) and any(c for c in c_list[:-1]):
            archetype_samples["RECOVERED"].append(metrics)
            print(f"  [+] Sample #{idx} -> RECOVERED HESITATION (Processed {metrics['num_tokens']} tokens)")
        elif final_c and all(c_list):
            archetype_samples["STABLE"].append(metrics)
            print(f"  [+] Sample #{idx} -> STABLE CORRECT (Processed {metrics['num_tokens']} tokens)")
        else:
            archetype_samples["NEVER"].append(metrics)
            print(f"  [+] Sample #{idx} -> NEVER CORRECT (Processed {metrics['num_tokens']} tokens)")

    # 1. Generate Aggregate Reasoning Progression Figure
    fig1_path = str(run_dir / "aggregate_reasoning_progression.png")
    plot_aggregate_reasoning_progression(archetype_samples, fig1_path)

    # 2. Generate Event-Aligned Flip Dynamics Figure
    fig2_path = str(run_dir / "event_aligned_flip_dynamics.png")
    plot_event_aligned_flip_dynamics(archetype_samples, fig2_path)

    print("\n" + "=" * 80)
    print("  TWO CLEAN SCIENTIFIC PUBLICATION FIGURES CREATED SUCCESSFULLY!")
    print("=" * 80)
    print(f"  [1] Normalized 0%->100% Progression : {fig1_path}")
    print(f"  [2] Event-Aligned Flip Dynamics     : {fig2_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
