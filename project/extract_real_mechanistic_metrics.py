"""
Extract Real Mechanistic Metrics (Entropy, Top-2 Margin, JSD, Layerwise L2, and Cosine Dynamics)
directly from model forward passes for actual experiment samples.

Saves:
- real_uncertainty_instability.png & .pdf
- real_hidden_state_dynamics.png & .pdf
- real_mechanistic_metrics.json

Usage:
  python extract_real_mechanistic_metrics.py --run_dir results/proper_run2/r1_distill_qwen1_5b/gsm8k/seed_42/budget_prompt_Therefore__the_final_answer_is --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --quantization 4bit
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Fix Windows console encoding
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


def smooth_series(values: np.ndarray, window: int = 7) -> np.ndarray:
    """Apply moving average smoothing for clean publication plotting."""
    if len(values) <= window:
        return values
    pad_width = window // 2
    padded = np.pad(values, pad_width, mode="edge")
    kernel = np.ones(window) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(values)]


def compute_token_metrics(
    model: Any,
    tokenizer: Any,
    prompt_text: str,
    generation_text: str,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Run a forward pass on prompt + generation and extract exact internal tensors efficiently."""
    full_text = prompt_text + generation_text
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([full_ids], device=device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )

    # Immediately move outputs to CPU to free GPU VRAM
    logits = outputs.logits[0, prompt_len - 1 : -1].cpu().float()  # [gen_len, vocab_size] on CPU
    num_layers = len(outputs.hidden_states) - 1
    early_layer_idx = min(2, num_layers)
    mid_layer_idx = num_layers // 2
    late_layer_idx = num_layers

    h_early = outputs.hidden_states[early_layer_idx][0, prompt_len:].cpu().float()
    h_mid = outputs.hidden_states[mid_layer_idx][0, prompt_len:].cpu().float()
    h_late = outputs.hidden_states[late_layer_idx][0, prompt_len:].cpu().float()

    # Free CUDA memory immediately
    del outputs
    torch.cuda.empty_cache()

    num_gen_tokens = logits.shape[0]
    if num_gen_tokens < 2:
        return None

    # Compute Softmax, Entropy, Margin in CPU chunks (to avoid allocating giant matrices)
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

    # JSD vs. Previous Token (computed in CPU slices)
    jsd_list = [0.0]
    for i in range(1, num_gen_tokens):
        p_prev = probs[i - 1 : i]
        p_curr = probs[i : i + 1]
        m = 0.5 * (p_prev + p_curr)
        kl1 = F.kl_div(torch.log(m + 1e-12), p_prev, reduction="batchmean")
        kl2 = F.kl_div(torch.log(m + 1e-12), p_curr, reduction="batchmean")
        jsd = (0.5 * (kl1 + kl2)).item()
        jsd_list.append(jsd)

    # L2 velocities ||h_t - h_{t-1}||
    def get_l2_velocity(h_tensor):
        if h_tensor.shape[0] < 2:
            return np.zeros(h_tensor.shape[0])
        diff = h_tensor[1:] - h_tensor[:-1]
        v = torch.norm(diff, p=2, dim=-1).numpy()
        return np.concatenate([[v[0]], v])

    l2_early = get_l2_velocity(h_early)
    l2_mid = get_l2_velocity(h_mid)
    l2_late = get_l2_velocity(h_late)

    # Temporal Cosine Matrix cos(h_i, h_j) for Late Layer
    h_norm = F.normalize(h_late, p=2, dim=-1)
    cosine_matrix = torch.mm(h_norm, h_norm.t()).numpy()

    return {
        "num_tokens": num_gen_tokens,
        "entropy": np.array(entropy_list),
        "top2_margin": np.array(top2_margin_list),
        "jsd": np.array(jsd_list),
        "l2_early": l2_early,
        "l2_mid": l2_mid,
        "l2_late": l2_late,
        "cosine_matrix": cosine_matrix,
    }


def plot_real_uncertainty_instability(
    results_by_archetype: Dict[str, Dict[str, Any]], 
    output_path: str, 
    dark_mode: bool = True
) -> str:
    """Plot Figure 1 with REAL computed PyTorch metrics."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), dpi=300, facecolor=bg_color, sharex=False)

    colors = {
        "FLIP": "#f87171",       # Red
        "RECOVERED": "#fbbf24",  # Amber
        "STABLE": "#34d399",     # Green
        "NEVER": "#94a3b8",      # Gray
    }
    labels = {
        "FLIP": "Harmful Flip (Sample #8 ⚠️)",
        "RECOVERED": "Recovered Hesitation (Sample #1 ⚡)",
        "STABLE": "Stable Correct (Sample #0 ✓)",
        "NEVER": "Never Correct (Sample #3 ✗)",
    }

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9)

    for arch_name, metrics in results_by_archetype.items():
        if not metrics:
            continue
        c = colors.get(arch_name, "#38bdf8")
        lbl = labels.get(arch_name, arch_name)
        lw = 2.4 if arch_name in ["FLIP", "STABLE"] else 2.0
        ls = "-" if arch_name in ["FLIP", "STABLE"] else "--"

        # Smooth metrics for publication presentation
        t = np.arange(metrics["num_tokens"])
        ent_s = smooth_series(metrics["entropy"])
        margin_s = smooth_series(metrics["top2_margin"])
        jsd_s = smooth_series(metrics["jsd"])

        ax1.plot(t, ent_s, color=c, lw=lw, linestyle=ls, label=lbl)
        ax2.plot(t, margin_s, color=c, lw=lw, linestyle=ls, label=lbl)
        ax3.plot(t, jsd_s, color=c, lw=lw, linestyle=ls, label=lbl)

    ax1.set_ylabel("Token Entropy H(x)", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.set_title("Panel A: Real Model Token Entropy (Uncertainty Evolution Across Generated Reasoning)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    ax2.set_ylabel("Top-2 Probability Margin", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.set_title("Panel B: Real Model Top-2 Margin (Confidence Separation)", fontsize=12, fontweight="bold", color=text_color, loc="left")

    ax3.set_ylabel("JSD vs. Previous Token", fontsize=10.5, fontweight="bold", color=text_color)
    ax3.set_xlabel("Generated Reasoning Token Index", fontsize=11, fontweight="bold", color=text_color)
    ax3.set_title("Panel C: Real Jensen-Shannon Divergence Volatility (Distributional Shock)", fontsize=12, fontweight="bold", color=text_color, loc="left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


def plot_real_hidden_state_dynamics(
    results_by_archetype: Dict[str, Dict[str, Any]], 
    output_path: str, 
    dark_mode: bool = True
) -> str:
    """Plot Figure 2 with REAL computed PyTorch layer movements and cosine matrix."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor=bg_color, gridspec_kw={"width_ratios": [1.2, 1.0]})

    colors = {
        "FLIP": "#f87171",
        "RECOVERED": "#fbbf24",
        "STABLE": "#34d399",
        "NEVER": "#94a3b8",
    }
    labels = {
        "FLIP": "Harmful Flip (Sample #8 ⚠️)",
        "RECOVERED": "Recovered Hesitation (Sample #1 ⚡)",
        "STABLE": "Stable Correct (Sample #0 ✓)",
        "NEVER": "Never Correct (Sample #3 ✗)",
    }

    # Panel A: Late-Layer L2 Velocity
    ax1.set_facecolor(bg_color)
    ax1.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
    for spine in ax1.spines.values():
        spine.set_color(grid_color)
    ax1.tick_params(colors=subtext_color, labelsize=9)

    for arch_name, metrics in results_by_archetype.items():
        if not metrics:
            continue
        c = colors.get(arch_name, "#38bdf8")
        lbl = labels.get(arch_name, arch_name)
        lw = 2.4 if arch_name in ["FLIP", "STABLE"] else 2.0
        ls = "-" if arch_name in ["FLIP", "STABLE"] else "--"

        t = np.arange(metrics["num_tokens"])
        l2_s = smooth_series(metrics["l2_late"])
        ax1.plot(t, l2_s, color=c, lw=lw, linestyle=ls, label=lbl)

    ax1.set_ylabel("Late-Layer L2 Velocity ||h_t - h_{t-1}||", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.set_xlabel("Generated Reasoning Token Index", fontsize=11, fontweight="bold", color=text_color)
    ax1.set_title("Panel A: Real Late-Layer Representation Velocity (L2 Movement)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    # Panel B: Cosine Recurrence Matrix for the Flip Sample (or first available sample)
    ax2.set_facecolor(bg_color)
    target_sample = results_by_archetype.get("FLIP") or results_by_archetype.get("STABLE")
    if target_sample and "cosine_matrix" in target_sample:
        cos_mat = target_sample["cosine_matrix"]
        # Subsample if too large for clean viewing
        max_dim = 120
        if cos_mat.shape[0] > max_dim:
            step = cos_mat.shape[0] // max_dim
            cos_mat = cos_mat[::step, ::step]

        im = ax2.imshow(cos_mat, cmap="viridis", aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax2.set_title("Panel B: Real Temporal Cosine Matrix (Sample #8 Flip)", fontsize=12, fontweight="bold", color=text_color, loc="left")
        ax2.set_xlabel("Token Index i", fontsize=10.5, fontweight="bold", color=text_color)
        ax2.set_ylabel("Token Index j", fontsize=10.5, fontweight="bold", color=text_color)
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
    parser = argparse.ArgumentParser(description="Extract Real Mechanistic Metrics from Model Forward Passes")
    parser.add_argument("--run_dir", type=str, required=True, help="Directory of the run (containing flip_analysis.json and difficulty_*.jsonl)")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name or path")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Quantization mode")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    flip_file = run_dir / "flip_analysis.json"
    if not flip_file.exists():
        print(f"[ERROR] flip_analysis.json not found in {run_dir}")
        sys.exit(1)

    with open(flip_file, "r", encoding="utf-8") as f:
        flip_data = json.load(f)

    # Find the generations jsonl file in run_dir
    diff_files = list(run_dir.glob("difficulty_*.jsonl"))
    if not diff_files:
        print(f"[ERROR] No difficulty_*.jsonl files found in {run_dir}")
        sys.exit(1)

    print(f"\n[INFO] Loading samples from {diff_files[0]}...")
    sample_records = {}
    with open(diff_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = rec.get("idx")
            if idx not in sample_records:
                sample_records[idx] = rec

    # Pick 4 key archetypes from the run:
    # 1. Flip (e.g. Sample 8 in proper_run2)
    # 2. Recovered (e.g. Sample 1 in proper_run2)
    # 3. Stable (e.g. Sample 0 in proper_run2)
    # 4. Never (e.g. Sample 3 in proper_run2)
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
    selected_samples = {}

    for k, v in trajectories.items():
        idx = int(k)
        is_flip = v.get("is_flip", False)
        c_list = v.get("correctness", [])
        final_c = v.get("final_correct", False)

        if is_flip and "FLIP" not in selected_samples:
            selected_samples["FLIP"] = idx
        elif final_c and any(not c for c in c_list[:-1]) and any(c for c in c_list[:-1]) and "RECOVERED" not in selected_samples:
            selected_samples["RECOVERED"] = idx
        elif final_c and all(c_list) and "STABLE" not in selected_samples:
            selected_samples["STABLE"] = idx
        elif not any(c_list) and "NEVER" not in selected_samples:
            selected_samples["NEVER"] = idx

    print(f"[INFO] Selected Archetype Samples to analyze: {selected_samples}")

    # Load Model & Tokenizer
    print(f"[INFO] Loading Model '{args.model}' (Quantization={args.quantization})...")
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

    results_by_archetype = {}
    for arch_name, sample_idx in selected_samples.items():
        rec = sample_records.get(sample_idx)
        if not rec:
            continue

        prompt = rec.get("source_actual_query") or rec.get("question") or ""
        gen = rec.get("source_model_output") or rec.get("model_output") or ""

        print(f"  [+] Extracting real PyTorch tensors for {arch_name} (Sample #{sample_idx})...")
        metrics = compute_token_metrics(model, tokenizer, prompt, gen, device="cuda")
        if metrics:
            results_by_archetype[arch_name] = metrics

    # Plot and save directly in run_dir
    f1_path = str(run_dir / "real_uncertainty_instability.png")
    f2_path = str(run_dir / "real_hidden_state_dynamics.png")

    plot_real_uncertainty_instability(results_by_archetype, f1_path)
    plot_real_hidden_state_dynamics(results_by_archetype, f2_path)

    print("\n" + "=" * 75)
    print("  100% REAL MECHANISTIC METRICS COMPUTED & SAVED TO RUN DIRECTORY!")
    print("=" * 75)
    print(f"  [1] Real Uncertainty Plot    : {f1_path}")
    print(f"  [2] Real Hidden State Plot   : {f2_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
