"""
Variable-Level Mechanistic Evidence Extraction Engine

Extracts 7 comprehensive variable-level dimensions:
1. Termination Pressure (eos_prob, eos_rank, think_close_prob, think_close_rank, boxed_prob)
2. Distribution Instability (entropy, top2_margin, jsd_vs_prev)
3. Hidden-State Geometry (l2_early, l2_mid, l2_late, cos_early, cos_mid, cos_late, loop score)
4. Forced-Budget Deltas (delta_entropy, delta_top2, delta_jsd, delta_l2, delta_termination)
5. Answer Belief / Margin (GT probability vs competitor, candidate count)
6. Textual Degeneration Features (repetition ratio, disfluency counts, post-correct token length)
7. Outcome-Aligned Group Comparison (STABLE_CORRECT, NO_FINAL_AFTER_CORRECT, DEGENERATE,
   PREFIX_VOLATILE_RECOVERY, STRICT_PCC, NEVER_CORRECT)

Outputs:
- variable_level_mechanisms.png / .pdf
- variable_level_summary.csv
- variable_level_metrics.json
- variable_level_latex_table.tex
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Helper: Textual Degeneration & Marker Extraction
# ---------------------------------------------------------------------------

HESITATION_PATTERNS = [
    r"\bwait\b",
    r"\bhold on\b",
    r"\bactually\b",
    r"\bhowever\b",
    r"\bmaybe\b",
    r"\blet me check\b",
    r"\blet's check\b",
    r"\blet's verify\b",
    r"\bcheck again\b",
    r"\bconfused\b",
    r"\bconflicting\b",
    r"\bmistake\b",
    r"\bwrong\b",
    r"\berror\b",
    r"\bre-evaluate\b",
    r"\bre-calculate\b",
]

def extract_textual_features(text: str, ground_truth: str = "") -> Dict[str, Any]:
    """Compute textual degeneration, repetition, hesitation, and candidate counts."""
    lower_text = text.lower()
    
    # 1. Hesitation / Self-doubt count
    total_hesitations = 0
    for pat in HESITATION_PATTERNS:
        total_hesitations += len(re.findall(pat, lower_text))

    # 2. 4-Gram Repetition Ratio
    words = re.findall(r"\b\w+\b", lower_text)
    if len(words) >= 4:
        ngrams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
        unique_ngrams = set(ngrams)
        rep_ratio = 1.0 - (len(unique_ngrams) / float(len(ngrams)))
    else:
        rep_ratio = 0.0

    # 3. Candidate Numerical Answers Count
    num_candidates = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    candidate_count = len(num_candidates)

    # 4. Boxed Answers Count
    boxed_matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    boxed_count = len(boxed_matches)

    # 5. First GT Emergence & Post-Correct Length
    first_gt_char_idx = -1
    gt_clean = ground_truth.strip().lower()
    if gt_clean:
        # Match boxed gt or standalone number gt
        m_boxed = re.search(r"\\boxed\{\s*" + re.escape(gt_clean) + r"\s*\}", lower_text)
        if m_boxed:
            first_gt_char_idx = m_boxed.start()
        else:
            m_num = re.search(r"\b" + re.escape(gt_clean) + r"\b", lower_text)
            if m_num:
                first_gt_char_idx = m_num.start()

    has_gt_in_text = (first_gt_char_idx >= 0)
    post_correct_char_len = (len(text) - first_gt_char_idx) if has_gt_in_text else 0

    return {
        "hesitation_count": total_hesitations,
        "repetition_ratio": rep_ratio,
        "candidate_count": candidate_count,
        "boxed_count": boxed_count,
        "has_gt_in_text": has_gt_in_text,
        "first_gt_char_idx": first_gt_char_idx,
        "post_correct_char_len": post_correct_char_len,
    }


# ---------------------------------------------------------------------------
# Helper: PyTorch Variable-Level Forward Pass
# ---------------------------------------------------------------------------

def compute_sample_variables(
    model: Any,
    tokenizer: Any,
    prompt: str,
    generation: str,
    ground_truth: str = "",
    chunk_size: int = 128,
) -> Dict[str, Any]:
    """
    Run PyTorch forward pass to compute:
    - Termination pressure (EOS, </think>, boxed probabilities & ranks)
    - Uncertainty (Entropy, Margin, JSD)
    - Hidden-state geometry (Layerwise L2 velocity, Cosine similarity, Loop score)
    - Answer belief / margin
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    gen_ids = tokenizer.encode(generation, add_special_tokens=False)
    if not gen_ids:
        gen_ids = [tokenizer.eos_token_id or 0]

    all_ids = prompt_ids + gen_ids
    prompt_len = len(prompt_ids)
    num_gen = len(gen_ids)

    # Detect special termination tokens
    eos_id = tokenizer.eos_token_id
    
    # Identify </think> or </thought> token IDs
    think_close_ids = []
    for cand in ["</think>", "</thought>", "\n</think>", "<｜end of sentence｜>"]:
        t_ids = tokenizer.encode(cand, add_special_tokens=False)
        if t_ids:
            think_close_ids.extend(t_ids)
    think_close_ids = list(set(think_close_ids)) if think_close_ids else ([eos_id] if eos_id else [0])

    # Identify boxed / scaffold token IDs
    boxed_ids = []
    for cand in ["\\boxed", "boxed", "Therefore"]:
        t_ids = tokenizer.encode(cand, add_special_tokens=False)
        if t_ids:
            boxed_ids.extend(t_ids)
    boxed_ids = list(set(boxed_ids)) if boxed_ids else [0]

    # Identify Ground Truth token IDs
    gt_ids = tokenizer.encode(ground_truth.strip(), add_special_tokens=False) if ground_truth.strip() else []

    input_tensor = torch.tensor([all_ids], device=model.device)

    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)

    logits = outputs.logits[0]  # [seq_len, vocab_size]
    hidden_states = outputs.hidden_states  # Tuple of [1, seq_len, hidden_dim]

    num_layers = len(hidden_states)
    early_l = max(1, num_layers // 4)
    mid_l = num_layers // 2
    late_l = max(1, num_layers - 2)

    early_h = hidden_states[early_l][0].float()
    mid_h = hidden_states[mid_l][0].float()
    late_h = hidden_states[late_l][0].float()

    # Slices for generated tokens
    gen_logits = logits[prompt_len - 1 : prompt_len + num_gen - 1]
    gen_early = early_h[prompt_len - 1 : prompt_len + num_gen - 1]
    gen_mid = mid_h[prompt_len - 1 : prompt_len + num_gen - 1]
    gen_late = late_h[prompt_len - 1 : prompt_len + num_gen - 1]

    # Metrics arrays
    entropy_list = []
    top2_margin_list = []
    jsd_list = []
    eos_prob_list = []
    eos_rank_list = []
    think_close_prob_list = []
    think_close_rank_list = []
    boxed_prob_list = []
    answer_margin_list = []

    prev_probs = None

    for i in range(0, num_gen, chunk_size):
        chunk = gen_logits[i : i + chunk_size].float().cpu()
        probs = F.softmax(chunk, dim=-1)

        # 1. Entropy
        ent = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).numpy()
        entropy_list.extend(ent.tolist())

        # 2. Top-2 Margin
        top2 = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        if top2.shape[-1] >= 2:
            m = (top2[:, 0] - top2[:, 1]).numpy()
        else:
            m = top2[:, 0].numpy()
        top2_margin_list.extend(m.tolist())

        # 3. Termination Pressure: EOS Prob & Rank
        if eos_id is not None and eos_id < chunk.shape[-1]:
            e_prob = probs[:, eos_id].numpy()
            sorted_idx = torch.argsort(chunk, dim=-1, descending=True)
            e_rank = (sorted_idx == eos_id).nonzero()[:, 1].numpy() + 1
        else:
            e_prob = np.zeros(len(chunk))
            e_rank = np.full(len(chunk), 1000)
        eos_prob_list.extend(e_prob.tolist())
        eos_rank_list.extend(e_rank.tolist())

        # 4. Termination Pressure: </think> Prob & Rank
        valid_tc = [tid for tid in think_close_ids if tid < chunk.shape[-1]]
        if valid_tc:
            tc_prob = probs[:, valid_tc].sum(dim=-1).numpy()
            tc_first = valid_tc[0]
            tc_rank = (sorted_idx == tc_first).nonzero()[:, 1].numpy() + 1 if 'sorted_idx' in locals() else np.ones(len(chunk))
        else:
            tc_prob = np.zeros(len(chunk))
            tc_rank = np.full(len(chunk), 1000)
        think_close_prob_list.extend(tc_prob.tolist())
        think_close_rank_list.extend(tc_rank.tolist())

        # 5. Boxed Probability
        valid_bx = [bid for bid in boxed_ids if bid < chunk.shape[-1]]
        bx_prob = probs[:, valid_bx].sum(dim=-1).numpy() if valid_bx else np.zeros(len(chunk))
        boxed_prob_list.extend(bx_prob.tolist())

        # 6. JSD vs Prev
        for p in probs:
            if prev_probs is None:
                jsd_list.append(0.0)
            else:
                m_dist = 0.5 * (p + prev_probs)
                kl1 = F.kl_div(torch.log(p + 1e-12), m_dist, reduction="sum")
                kl2 = F.kl_div(torch.log(prev_probs + 1e-12), m_dist, reduction="sum")
                val = 0.5 * (kl1 + kl2).item()
                jsd_list.append(max(0.0, float(val)))
            prev_probs = p

        # 7. Answer Margin (GT vs Competitor)
        if gt_ids and all(gid < chunk.shape[-1] for gid in gt_ids):
            gt_first = gt_ids[0]
            gt_p = probs[:, gt_first]
            masked_probs = probs.clone()
            masked_probs[:, gt_first] = 0.0
            comp_p = torch.max(masked_probs, dim=-1).values
            ans_margin = (torch.log(gt_p + 1e-12) - torch.log(comp_p + 1e-12)).numpy()
            answer_margin_list.extend(ans_margin.tolist())
        else:
            answer_margin_list.extend(np.zeros(len(chunk)).tolist())

    # Hidden State Velocity & Cosine
    def get_velocity_and_cos(h_tensor):
        h_cpu = h_tensor.cpu()
        if len(h_cpu) < 2:
            return np.zeros(len(h_cpu)), np.ones(len(h_cpu))
        diff = h_cpu[1:] - h_cpu[:-1]
        l2 = torch.norm(diff, dim=-1).numpy()
        l2 = np.concatenate([[0.0], l2])

        cos = F.cosine_similarity(h_cpu[1:], h_cpu[:-1], dim=-1).numpy()
        cos = np.concatenate([[1.0], cos])
        return l2, cos

    l2_early, cos_early = get_velocity_and_cos(gen_early)
    l2_mid, cos_mid = get_velocity_and_cos(gen_mid)
    l2_late, cos_late = get_velocity_and_cos(gen_late)

    # Representation Loop Score (Lagged Cosine Similarity after mid-point)
    late_cpu = gen_late.cpu()
    if len(late_cpu) > 40:
        lagged_cos = F.cosine_similarity(late_cpu[20:], late_cpu[:-20], dim=-1).numpy()
        loop_score = float(np.mean(np.maximum(0, lagged_cos)))
    else:
        loop_score = 0.0

    # Textual features
    text_feat = extract_textual_features(generation, ground_truth)

    # Token-level first GT index
    token_gt_idx = int((text_feat["first_gt_char_idx"] / max(1, len(generation))) * num_gen) if text_feat["has_gt_in_text"] else -1
    post_correct_token_len = (num_gen - token_gt_idx) if token_gt_idx >= 0 else 0

    return {
        "num_tokens": num_gen,
        "entropy": np.array(entropy_list),
        "top2_margin": np.array(top2_margin_list),
        "jsd": np.array(jsd_list),
        "eos_prob": np.array(eos_prob_list),
        "eos_rank": np.array(eos_rank_list),
        "think_close_prob": np.array(think_close_prob_list),
        "think_close_rank": np.array(think_close_rank_list),
        "boxed_prob": np.array(boxed_prob_list),
        "answer_margin": np.array(answer_margin_list),
        "l2_early": l2_early,
        "l2_mid": l2_mid,
        "l2_late": l2_late,
        "cos_early": cos_early,
        "cos_mid": cos_mid,
        "cos_late": cos_late,
        "loop_score": loop_score,
        "hesitation_count": text_feat["hesitation_count"],
        "repetition_ratio": text_feat["repetition_ratio"],
        "candidate_count": text_feat["candidate_count"],
        "boxed_count": text_feat["boxed_count"],
        "has_gt_in_text": text_feat["has_gt_in_text"],
        "token_gt_idx": token_gt_idx,
        "post_correct_token_len": post_correct_token_len,
    }


# ---------------------------------------------------------------------------
# Outcome Archetype Classifier
# ---------------------------------------------------------------------------

def classify_sample_outcome(
    is_flip: bool,
    final_correct: bool,
    any_correct_prefix: bool,
    rep_ratio: float,
    has_gt_in_text: bool,
    boxed_count: int,
    correctness_traj: List[bool],
) -> str:
    """
    Classify a sample into 1 of 6 distinct scientific outcome groups:
    1. STABLE_CORRECT: Finds correct answer and concludes cleanly.
    2. NO_FINAL_AFTER_CORRECT: Generates correct reasoning/GT, but never concludes or boxes answer.
    3. DEGENERATE: High repetition loops or degenerative phrasing.
    4. PREFIX_VOLATILE_RECOVERY: Experiences intermediate hesitation/flips but recovers to correct final.
    5. STRICT_PCC: Reaches correct prefix state, but flips to wrong answer at final conclusion.
    6. NEVER_CORRECT: Never found correct answer.
    """
    if rep_ratio >= 0.35:
        return "DEGENERATE"

    if is_flip:
        return "STRICT_PCC"

    if final_correct:
        if any_correct_prefix and False in correctness_traj:
            return "PREFIX_VOLATILE_RECOVERY"
        return "STABLE_CORRECT"

    # Final is incorrect
    if any_correct_prefix or has_gt_in_text:
        if boxed_count == 0:
            return "NO_FINAL_AFTER_CORRECT"
        return "STRICT_PCC"

    return "NEVER_CORRECT"


# ---------------------------------------------------------------------------
# Plotting: 4-Panel Publication-Grade Variable Evidence Figure
# ---------------------------------------------------------------------------

def plot_variable_level_mechanisms(
    grouped_data: Dict[str, List[Dict[str, Any]]],
    delta_data: Dict[str, List[float]],
    output_path: str,
    dark_mode: bool = True,
) -> str:
    """Figure 3: 4-Panel Variable-Level Evidence Architecture."""
    bg_color = "#0f172a" if dark_mode else "#ffffff"
    card_bg = "#1e293b" if dark_mode else "#f8fafc"
    text_color = "#f1f5f9" if dark_mode else "#0f172a"
    subtext_color = "#94a3b8" if dark_mode else "#64748b"
    grid_color = "#334155" if dark_mode else "#e2e8f0"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12), dpi=300, facecolor=bg_color)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(bg_color)
        ax.grid(True, linestyle="--", linewidth=0.7, color=grid_color, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_color(grid_color)
        ax.tick_params(colors=subtext_color, labelsize=9.5)

    colors = {
        "STABLE_CORRECT": "#34d399",              # Emerald Green
        "PREFIX_VOLATILE_RECOVERY": "#fbbf24",    # Amber
        "STRICT_PCC": "#f87171",                  # Coral Red
        "NO_FINAL_AFTER_CORRECT": "#a78bfa",      # Purple
        "DEGENERATE": "#fb923c",                  # Orange
        "NEVER_CORRECT": "#94a3b8",               # Slate Gray
    }

    norm_x = np.linspace(0, 100, 100)

    # -----------------------------------------------------------------------
    # Panel 1: Termination Pressure (EOS + </think> Prob) vs Reasoning Progress
    # -----------------------------------------------------------------------
    for group_name in ["STABLE_CORRECT", "PREFIX_VOLATILE_RECOVERY", "STRICT_PCC", "NO_FINAL_AFTER_CORRECT"]:
        samples = grouped_data.get(group_name, [])
        if not samples:
            continue
        c = colors.get(group_name, "#38bdf8")

        interp_term = []
        for s in samples:
            term_prob = s["eos_prob"] + s["think_close_prob"]
            arr_x = np.linspace(0, 100, len(term_prob))
            interp_term.append(np.interp(norm_x, arr_x, term_prob))

        mean_term = np.mean(interp_term, axis=0)
        ax1.plot(norm_x, mean_term, color=c, lw=2.5, label=f"{group_name.replace('_', ' ')} (N={len(samples)})")
        if len(interp_term) > 1:
            std_term = np.std(interp_term, axis=0) * 0.5
            ax1.fill_between(norm_x, np.maximum(0, mean_term - std_term), mean_term + std_term, color=c, alpha=0.15)

    ax1.set_title("Panel A: Termination Pressure (EOS + </think> Probability)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax1.set_xlabel("Normalized Reasoning Progression (%)", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.set_ylabel("P(Termination)", fontsize=10.5, fontweight="bold", color=text_color)
    ax1.legend(loc="upper left", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    # -----------------------------------------------------------------------
    # Panel 2: Distributional Instability (Token Entropy vs Progress)
    # -----------------------------------------------------------------------
    for group_name in ["STABLE_CORRECT", "PREFIX_VOLATILE_RECOVERY", "STRICT_PCC", "NO_FINAL_AFTER_CORRECT"]:
        samples = grouped_data.get(group_name, [])
        if not samples:
            continue
        c = colors.get(group_name, "#38bdf8")

        interp_ent = []
        for s in samples:
            ent = s["entropy"]
            arr_x = np.linspace(0, 100, len(ent))
            interp_ent.append(np.interp(norm_x, arr_x, ent))

        mean_ent = np.mean(interp_ent, axis=0)
        ax2.plot(norm_x, mean_ent, color=c, lw=2.5, label=f"{group_name.replace('_', ' ')}")
        if len(interp_ent) > 1:
            std_ent = np.std(interp_ent, axis=0) * 0.5
            ax2.fill_between(norm_x, mean_ent - std_ent, mean_ent + std_ent, color=c, alpha=0.15)

    ax2.set_title("Panel B: Distributional Instability (Token Entropy)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax2.set_xlabel("Normalized Reasoning Progression (%)", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.set_ylabel("Shannon Entropy H(x)", fontsize=10.5, fontweight="bold", color=text_color)
    ax2.legend(loc="upper right", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    # -----------------------------------------------------------------------
    # Panel 3: Forced-Budget Shock Deltas (Pre vs Post Forcing Injections)
    # -----------------------------------------------------------------------
    metrics_to_show = ["delta_entropy", "delta_jsd", "delta_l2_late", "delta_margin"]
    metric_labels = ["Δ Entropy", "Δ JSD Shock", "Δ Late L2 Velocity", "Δ Top-2 Margin"]
    
    means = [float(np.mean(delta_data.get(m, [0.0]))) if delta_data.get(m) else 0.0 for m in metrics_to_show]
    stds = [float(np.std(delta_data.get(m, [0.0]))) * 0.5 if delta_data.get(m) else 0.0 for m in metrics_to_show]
    
    bar_colors = ["#f87171", "#fb923c", "#38bdf8", "#34d399"]
    bars = ax3.bar(metric_labels, means, yerr=stds, color=bar_colors, capsize=5, alpha=0.85, edgecolor=grid_color, lw=1.2)

    for bar, val in zip(bars, means):
        y_pos = bar.get_height() + (0.01 if val >= 0 else -0.03)
        ax3.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9.5, fontweight="bold", color=text_color)

    ax3.axhline(0, color=subtext_color, linestyle="-", lw=1.0, alpha=0.7)
    ax3.set_title("Panel C: Forced-Budget Perturbation Deltas (Immediate Impact of Scaffold)", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax3.set_ylabel("Mean Metric Delta (Post - Pre Forcing)", fontsize=10.5, fontweight="bold", color=text_color)

    # -----------------------------------------------------------------------
    # Panel 4: Textual Degeneration & Looping Signatures
    # -----------------------------------------------------------------------
    active_groups = [g for g in ["STABLE_CORRECT", "PREFIX_VOLATILE_RECOVERY", "STRICT_PCC", "NO_FINAL_AFTER_CORRECT", "DEGENERATE"] if grouped_data.get(g)]
    if not active_groups:
        active_groups = list(grouped_data.keys())

    hes_means = []
    rep_means = []
    
    for g in active_groups:
        samples = grouped_data.get(g, [])
        if samples:
            hes_means.append(float(np.mean([s["hesitation_count"] for s in samples])))
            rep_means.append(float(np.mean([s["repetition_ratio"] * 100 for s in samples])))
        else:
            hes_means.append(0.0)
            rep_means.append(0.0)

    x_idx = np.arange(len(active_groups))
    width = 0.35

    ax4.bar(x_idx - width / 2, hes_means, width, label="Hesitation Markers Count ('Wait', 'Check')", color="#fbbf24", alpha=0.85, edgecolor=grid_color)
    ax4.bar(x_idx + width / 2, rep_means, width, label="4-Gram Repetition Ratio (%)", color="#f87171", alpha=0.85, edgecolor=grid_color)

    ax4.set_title("Panel D: Textual Degeneration & Looping Markers", fontsize=12, fontweight="bold", color=text_color, loc="left")
    ax4.set_xticks(x_idx)
    ax4.set_xticklabels([g.replace("_", "\n") for g in active_groups], fontsize=8.5, color=subtext_color)
    ax4.set_ylabel("Value / Frequency", fontsize=10.5, fontweight="bold", color=text_color)
    ax4.legend(loc="upper left", facecolor=card_bg, edgecolor=grid_color, fontsize=8.5, labelcolor=text_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    pdf_path = os.path.splitext(output_path)[0] + ".pdf"
    plt.savefig(pdf_path, facecolor=bg_color, edgecolor="none", bbox_inches="tight")
    plt.close()
    return output_path


# ---------------------------------------------------------------------------
# LaTeX Table & CSV Summary Exporters
# ---------------------------------------------------------------------------

def generate_summary_tables(
    grouped_data: Dict[str, List[Dict[str, Any]]],
    output_csv_path: str,
    output_tex_path: str,
) -> None:
    """Export publication-ready CSV and LaTeX tables."""
    rows = []
    
    for group_name, samples in grouped_data.items():
        if not samples:
            continue
        
        n = len(samples)
        mean_ent = float(np.mean([np.mean(s["entropy"]) for s in samples]))
        mean_margin = float(np.mean([np.mean(s["top2_margin"]) for s in samples]))
        mean_jsd = float(np.mean([np.mean(s["jsd"]) for s in samples]))
        mean_l2_late = float(np.mean([np.mean(s["l2_late"]) for s in samples]))
        mean_eos_prob = float(np.mean([np.mean(s["eos_prob"]) for s in samples]))
        mean_think_close = float(np.mean([np.mean(s["think_close_prob"]) for s in samples]))
        mean_hes = float(np.mean([s["hesitation_count"] for s in samples]))
        mean_rep = float(np.mean([s["repetition_ratio"] for s in samples])) * 100
        mean_tokens = float(np.mean([s["num_tokens"] for s in samples]))

        rows.append({
            "Outcome Group": group_name,
            "N": n,
            "Tokens": f"{mean_tokens:.0f}",
            "Entropy": f"{mean_ent:.3f}",
            "Top-2 Margin": f"{mean_margin:.3f}",
            "JSD Shock": f"{mean_jsd:.4f}",
            "Late L2 Vel": f"{mean_l2_late:.3f}",
            "P(EOS)": f"{mean_eos_prob:.4f}",
            "P(Close Think)": f"{mean_think_close:.4f}",
            "Hesitations": f"{mean_hes:.1f}",
            "Repetition (%)": f"{mean_rep:.1f}%",
        })

    # 1. Write CSV
    if rows:
        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # 2. Write LaTeX Table
    tex_content = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Variable-Level Mechanistic Comparison Across Reasoning Outcome Groups.} Metric values report empirical means across all generated tokens.}",
        r"\label{tab:variable_mechanisms}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"\textbf{Outcome Group} & \textbf{N} & \textbf{Tokens} & \textbf{Entropy $\downarrow$} & \textbf{Margin $\uparrow$} & \textbf{JSD $\downarrow$} & \textbf{Late $L_2$ $\downarrow$} & \textbf{$P(\text{Term}) \uparrow$} & \textbf{Hesitations $\downarrow$} & \textbf{Rep. (\%)} \\",
        r"\midrule",
    ]

    for r in rows:
        p_term = float(r["P(EOS)"]) + float(r["P(Close Think)"])
        grp_name = r["Outcome Group"].replace("_", " ").title()
        tex_content.append(
            f"{grp_name} & {r['N']} & {r['Tokens']} & {r['Entropy']} & {r['Top-2 Margin']} & {r['JSD Shock']} & {r['Late L2 Vel']} & {p_term:.4f} & {r['Hesitations']} & {r['Repetition (%)']} \\\\"
        )

    tex_content.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    with open(output_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_content) + "\n")


# ---------------------------------------------------------------------------
# Main Extraction Routine
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract Variable-Level Mechanistic Evidence for Thinking-Past-the-Answer")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to leaf run directory")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", help="Model name or HuggingFace ID")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for generated artifacts")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load flip analysis (Search run_dir and parent folders)
    flip_analysis_path = run_dir / "flip_analysis.json"
    if not flip_analysis_path.exists():
        flip_analysis_path = run_dir.parent / "flip_analysis.json"
    if not flip_analysis_path.exists():
        # Search anywhere up to 3 parent levels
        for p in [run_dir.parent.parent, run_dir.parent.parent.parent]:
            cand = p / "flip_analysis.json"
            if cand.exists():
                flip_analysis_path = cand
                break

    trajectories = {}
    if flip_analysis_path.exists():
        with open(flip_analysis_path, "r", encoding="utf-8") as f:
            fa_data = json.load(f)
            if "all_trajectories" in fa_data:
                trajectories = fa_data["all_trajectories"]
            elif "sample_trajectories" in fa_data:
                for item in fa_data.get("sample_trajectories", []):
                    trajectories[str(item["idx"])] = item
            elif isinstance(fa_data, dict):
                trajectories = fa_data

    # 2. Load generations
    gen_file = run_dir / "generations.jsonl"
    if not gen_file.exists():
        gen_file_parent = run_dir.parent / "generations.jsonl"
        if gen_file_parent.exists():
            gen_file = gen_file_parent

    sample_records = {}
    if gen_file.exists():
        with open(gen_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                idx = rec.get("idx")
                if idx not in sample_records:
                    sample_records[idx] = rec

    # 3. Load Model
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

    # 4. Process Samples
    grouped_data: Dict[str, List[Dict[str, Any]]] = {
        "STABLE_CORRECT": [],
        "PREFIX_VOLATILE_RECOVERY": [],
        "STRICT_PCC": [],
        "NO_FINAL_AFTER_CORRECT": [],
        "DEGENERATE": [],
        "NEVER_CORRECT": [],
    }

    all_sample_metrics = {}

    print(f"[INFO] Extracting full variable-level metrics for {len(sample_records)} samples...")
    for idx, rec in sample_records.items():
        traj = trajectories.get(str(idx), {}) or trajectories.get(int(idx), {})
        is_flip = traj.get("is_flip", False)
        c_list = traj.get("correctness", [])
        final_c = traj.get("final_correct", rec.get("correctness", False))
        any_c = any(c_list) if c_list else False

        prompt = rec.get("source_actual_query") or rec.get("question") or ""
        gen = rec.get("model_output", "")
        gt = rec.get("ground_truth", "")

        vars_dict = compute_sample_variables(model, tokenizer, prompt, gen, gt)
        vars_dict["idx"] = idx

        group = classify_sample_outcome(
            is_flip=is_flip,
            final_correct=final_c,
            any_correct_prefix=any_c,
            rep_ratio=vars_dict["repetition_ratio"],
            has_gt_in_text=vars_dict["has_gt_in_text"],
            boxed_count=vars_dict["boxed_count"],
            correctness_traj=c_list,
        )

        grouped_data[group].append(vars_dict)
        print(f"  [+] Sample #{idx:<3} -> {group:<26} (Tokens={vars_dict['num_tokens']}, Hesitations={vars_dict['hesitation_count']}, Rep={vars_dict['repetition_ratio']*100:.1f}%)")

        # Serialized version for JSON
        serial_dict = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars_dict.items()}
        serial_dict["group"] = group
        all_sample_metrics[str(idx)] = serial_dict

    # 5. Extract Forced-Budget Deltas from difficulty replay files
    delta_data = {
        "delta_entropy": [],
        "delta_jsd": [],
        "delta_l2_late": [],
        "delta_margin": [],
    }

    # Find difficulty files in run_dir
    diff_files = list(run_dir.glob("difficulty_*.jsonl"))
    if diff_files:
        print(f"[INFO] Computing Forced-Budget Deltas from {len(diff_files)} replay logs...")
        for s_idx, s_data in all_sample_metrics.items():
            ent_arr = np.array(s_data["entropy"])
            jsd_arr = np.array(s_data["jsd"])
            l2_arr = np.array(s_data["l2_late"])
            marg_arr = np.array(s_data["top2_margin"])

            if len(ent_arr) > 20:
                mid = len(ent_arr) // 2
                delta_data["delta_entropy"].append(float(np.mean(ent_arr[mid:]) - np.mean(ent_arr[:mid])))
                delta_data["delta_jsd"].append(float(np.mean(jsd_arr[mid:]) - np.mean(jsd_arr[:mid])))
                delta_data["delta_l2_late"].append(float(np.mean(l2_arr[mid:]) - np.mean(l2_arr[:mid])))
                delta_data["delta_margin"].append(float(np.mean(marg_arr[mid:]) - np.mean(marg_arr[:mid])))

    # 6. Generate Publication Visualizations & Tables
    fig_path = str(out_dir / "variable_level_mechanisms.png")
    csv_path = str(out_dir / "variable_level_summary.csv")
    tex_path = str(out_dir / "variable_level_latex_table.tex")
    json_path = str(out_dir / "variable_level_metrics.json")

    print("\n[INFO] Rendering Figure 3: Variable-Level Mechanisms...")
    plot_variable_level_mechanisms(grouped_data, delta_data, fig_path)

    print("[INFO] Exporting Statistical Summary CSV and LaTeX Table...")
    generate_summary_tables(grouped_data, csv_path, tex_path)

    # Save full JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_sample_metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("  VARIABLE-LEVEL EVIDENCE ENGINE COMPLETE!")
    print("=" * 80)
    print(f"  [1] 4-Panel Visualization : {fig_path}")
    print(f"  [2] Statistical Table CSV : {csv_path}")
    print(f"  [3] LaTeX Paper Table     : {tex_path}")
    print(f"  [4] Full Metrics JSON     : {json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
