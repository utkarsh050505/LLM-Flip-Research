"""
Run Full Pipeline -- End-to-end Thinking Past the Answer Pipeline with Model Detection & Flip Analysis.

Features:
  - Scans local HuggingFace cache (HF_HOME / A:\\LLMResearch\\hf_cache) for downloaded models.
  - Interactive selection menu of DOWNLOADED models so you choose and confirm before execution.
  - Interactive Quantization selection (4-bit NF4, 8-bit int8, or None/BF16 full precision).
  - Native 4-bit / 8-bit quantization support via bitsandbytes to run smoothly on local GPUs.
  - Built-in Flip Detection & Accuracy Curve report across difficulty steps.
  - Failure taxonomy categorization.

Usage Examples:
  python run_pipeline.py                                    # Interactive model & quantization picker -> runs pipeline
  python run_pipeline.py --benchmark gsm8k                  # Interactive model & quantization picker with benchmark
  python run_pipeline.py --model r1_distill_llama8b --benchmark gsm8k --quantization 4bit
  python run_pipeline.py --model r1_distill_qwen1_5b --benchmark gsm8k --quantization none
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform == "win32":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure default cache is A:\LLMResearch\hf_cache
DEFAULT_CACHE = os.environ.get("HF_HOME", r"A:\LLMResearch\hf_cache")
os.environ["HF_HOME"] = DEFAULT_CACHE
os.environ["TRANSFORMERS_CACHE"] = DEFAULT_CACHE

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.parsing import ParsingHelper
from modeling import MODEL_REGISTER

# Preferred clean shortcuts for display
CANONICAL_SHORTCUTS = {
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": "r1_distill_qwen1_5b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "r1_distill_qwen7b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "r1_distill_llama8b",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2_5_1_5b",
    "Qwen/Qwen2.5-3B-Instruct": "qwen2_5_3b",
    "Qwen/Qwen2.5-7B-Instruct": "qwen2_5_7b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2_5_vl",
    "Qwen/Qwen3-8B": "qwen3",
    "Qwen/Qwen3-VL-8B-Thinking": "qwen3_vl",
    "Qwen/Qwen3.5-9B": "qwen3_5",
}


def get_cached_models(cache_dir: str = DEFAULT_CACHE) -> list[dict]:
    """Scan local HuggingFace cache for downloaded models."""
    cached = []
    try:
        from huggingface_hub import scan_cache_dir
        info = scan_cache_dir(cache_dir=cache_dir)
        for repo in info.repos:
            if repo.repo_type == "model":
                size_gb = repo.size_on_disk / (1024 ** 3)
                has_weights = any(
                    f.file_name.endswith((".safetensors", ".bin", ".pt"))
                    for rev in repo.revisions
                    for f in rev.files
                )
                cached.append({
                    "repo_id": repo.repo_id,
                    "size_gb": size_gb,
                    "has_weights": has_weights,
                })
    except Exception:
        hub_dir = Path(cache_dir) / "hub"
        if not hub_dir.exists():
            hub_dir = Path(cache_dir)
        if hub_dir.exists():
            for d in hub_dir.iterdir():
                if d.is_dir() and d.name.startswith("models--"):
                    parts = d.name.replace("models--", "").split("--")
                    cached.append({
                        "repo_id": "/".join(parts),
                        "size_gb": sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / (1024 ** 3),
                        "has_weights": True,
                    })
    return cached


def map_cached_to_registered(cached_models: list[dict]) -> list[dict]:
    """Map cached HF repos to registered model keys in modeling/."""
    available = []
    seen_classes = set()

    for cm in cached_models:
        if not cm.get("has_weights") or cm.get("size_gb", 0) <= 0.1:
            continue
        repo_lower = cm["repo_id"].lower()

        matched_key = None
        matched_cls = None

        # Check canonical shortcuts first
        for hf_repo, shortcut in CANONICAL_SHORTCUTS.items():
            if hf_repo.lower() == repo_lower:
                matched_key = shortcut
                matched_cls = MODEL_REGISTER.get(shortcut)
                break

        if not matched_cls:
            for k, cls in MODEL_REGISTER.items():
                hf_name = getattr(cls, "hf_name", "").lower()
                if hf_name and (hf_name == repo_lower or repo_lower.endswith(hf_name)):
                    matched_key = k
                    matched_cls = cls
                    break

        if matched_cls and matched_cls not in seen_classes:
            seen_classes.add(matched_cls)
            available.append({
                "model_key": matched_key,
                "hf_name": getattr(matched_cls, "hf_name", cm["repo_id"]),
                "size_gb": cm["size_gb"],
                "class": matched_cls.__name__,
            })
    return available


def select_model_interactive(cached_available: list[dict]) -> str:
    """Prompt user to select from locally downloaded models."""
    print(f"\n{'=' * 75}")
    print(f"  [DETECTED DOWNLOADED MODELS IN CACHE] ({DEFAULT_CACHE})")
    print(f"{'=' * 75}")

    if cached_available:
        print(f"  {'#':<3} | {'Model Key':<24} | {'Hugging Face Repo ID':<40} | {'Disk Size'}")
        print(f"  {'-'*3}-+-{'-'*24}-+-{'-'*40}-+-{'-'*10}")
        for i, m in enumerate(cached_available, 1):
            print(f"  {i:<3} | {m['model_key']:<24} | {m['hf_name']:<40} | {m['size_gb']:5.2f} GB")
        print(f"{'=' * 75}")
        print(f"  Select a downloaded model [1-{len(cached_available)}] or [M] to see all registered models:")

        if sys.stdin.isatty():
            try:
                choice = input(f"Enter choice [default: 1 ({cached_available[0]['model_key']})]: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(cached_available):
                    selected = cached_available[int(choice) - 1]["model_key"]
                    print(f"  [SELECTED] {selected}")
                    return selected
                elif choice.lower() == "m":
                    pass
                elif not choice:
                    return cached_available[0]["model_key"]
            except (EOFError, KeyboardInterrupt):
                return cached_available[0]["model_key"]
        else:
            return cached_available[0]["model_key"]

    print(f"\n  ALL REGISTERED MODELS (Some may require downloading):")
    all_keys = list(CANONICAL_SHORTCUTS.values())
    for i, k in enumerate(all_keys, 1):
        cls = MODEL_REGISTER.get(k)
        hf_name = getattr(cls, "hf_name", "") if cls else ""
        print(f"  [{i:2d}] {k:<22} -> {hf_name}")
    print(f"{'=' * 75}")

    if sys.stdin.isatty():
        try:
            choice = input(f"Enter model number [1-{len(all_keys)}] or model name: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(all_keys):
                return all_keys[int(choice) - 1]
            elif choice in MODEL_REGISTER:
                return choice
        except Exception:
            pass

    return cached_available[0]["model_key"] if cached_available else all_keys[0]


def select_quantization_interactive(selected_model: str, default_choice: str = "4bit") -> str:
    """Prompt user to choose quantization option."""
    # Recommend full precision for small 1.5B/3B models, and 4-bit for 7B/8B+ models
    is_small = any(sz in selected_model.lower() for sz in ["1_5b", "1.5b", "3b", "1b"])
    suggested = "none" if is_small else "4bit"

    print(f"\n{'=' * 75}")
    print(f"  [SELECT QUANTIZATION FOR: {selected_model}]")
    print(f"{'=' * 75}")
    print(f"  [1] 4-bit  (NF4 via bitsandbytes — ~1.5GB-5GB VRAM, lowest memory footprint)")
    print(f"  [2] 8-bit  (int8 via bitsandbytes — ~3GB-9GB VRAM)")
    print(f"  [3] None   (Full Precision BF16/FP16 — faster if model fits completely in GPU)")
    print(f"{'=' * 75}")

    if sys.stdin.isatty():
        default_num = "3" if suggested == "none" else "1"
        try:
            choice = input(f"Enter choice [1-3, default: {default_num} ({suggested})]: ").strip()
            if choice == "1":
                return "4bit"
            elif choice == "2":
                return "8bit"
            elif choice == "3":
                return "none"
            elif not choice:
                return suggested
        except (EOFError, KeyboardInterrupt):
            return suggested
    return suggested


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run thinking-past-the-answer pipeline with model selection and flip detection"
    )
    parser.add_argument("--model", type=str, default=None, help="Model name registered in modeling (omit to pick from downloaded)")
    parser.add_argument("--benchmark", type=str, default="gsm8k", help="Benchmark name registered in benchmarking")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "sglang", "ddp"], help="Backend inference engine")
    parser.add_argument("--quantization", type=str, default=None, choices=["4bit", "8bit", "bf16", "fp16", "none"], help="Model quantization (omit for prompt)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="main", help="Experiment name / folder name (e.g. --experiment_name run1)")
    parser.add_argument("--timestamp_experiment", action="store_true", help="Auto-append timestamp to experiment name to avoid overwriting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--difficulty_level", type=str, default="utterance", choices=["utterance", "token"], help="Splitting level")
    parser.add_argument("--granularity", type=int, default=1, help="Splitting granularity step")
    parser.add_argument("--budget_forcing_prompt", type=str, default="Therefore, the final answer is:", help="Budget forcing prompt prefix")
    parser.add_argument("--limit", "--max_samples", dest="limit", type=int, default=None, help="Maximum number of dataset samples to process (e.g. --limit 10)")
    parser.add_argument("--skip_eval", action="store_true", help="Skip Stage 1 (eval)")
    parser.add_argument("--skip_difficulty", action="store_true", help="Skip Stage 2 (difficulty replay)")
    parser.add_argument("--skip_standalone_eval", action="store_true", help="Skip Stage 3 (standalone eval)")
    parser.add_argument("--skip_flips", action="store_true", help="Skip Stage 4 (flip detection analysis)")
    parser.add_argument("--skip_taxonomy", action="store_true", help="Skip Stage 5 (taxonomy classification)")
    parser.add_argument("--skip_mechanistic", action="store_true", help="Skip Stage 6 (mechanistic insights & dynamics plots)")
    parser.add_argument("--judge_provider", type=str, default="groq", choices=["groq", "openai", "openrouter", "custom"], help="Judge LLM provider (default: groq)")
    parser.add_argument("--judge_model", type=str, default="llama-3.3-70b-versatile", help="Judge model name (default: llama-3.3-70b-versatile)")
    parser.add_argument("--judge_api_key", type=str, default=None, help="API key for judge LLM (or set GROQ_API_KEY)")
    return parser.parse_args()


def run_stage(cmd: list[str], stage_name: str) -> bool:
    print(f"\n{'#' * 75}")
    print(f"  STAGE: {stage_name}")
    print(f"{'#' * 75}")
    print(f"  CMD: {' '.join(cmd)}\n")

    start_t = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start_t

    if res.returncode != 0:
        print(f"\n  [ERROR] Stage '{stage_name}' failed with return code {res.returncode}")
        return False
    print(f"\n  [SUCCESS] Stage '{stage_name}' completed in {elapsed:.1f}s")
    return True


def analyze_flips_and_difficulty(diff_file: str, run_path: str) -> dict:
    """Analyze difficulty records to detect flips and compute accuracy curves."""
    if not os.path.exists(diff_file):
        print(f"Difficulty file not found: {diff_file}")
        return {}

    records_by_idx = defaultdict(list)
    with open(diff_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                records_by_idx[r["idx"]].append(r)

    flip_events = []
    prefix_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})
    total_samples = len(records_by_idx)

    for idx, prefix_list in records_by_idx.items():
        prefix_list.sort(key=lambda x: x.get("difficulty_idx", 0))

        correctness_list = []
        preds = []
        for pr in prefix_list:
            pred = pr.get("prediction")
            if pred is None:
                pred = pr.get("model_parsed_answer")
            if pred is None:
                raw_out = pr.get("model_output", "")
                pred = ParsingHelper.extract_last_boxed(raw_out)
                if not pred:
                    match = re.findall(r"[-+]?\d*\.\d+|\d+", raw_out)
                    pred = match[-1] if match else ""
            pred = ParsingHelper.clean(str(pred))
            gt = ParsingHelper.clean(str(pr.get("ground_truth", "")))

            is_correct = (pred == gt and pred != "")
            correctness_list.append(is_correct)
            preds.append(pred)

            d_idx = pr.get("difficulty_idx", 0)
            prefix_accuracy[d_idx]["total"] += 1
            if is_correct:
                prefix_accuracy[d_idx]["correct"] += 1

        final_correct = correctness_list[-1] if correctness_list else False
        any_correct_before = any(correctness_list[:-1]) if len(correctness_list) > 1 else False

        if any_correct_before and not final_correct:
            last_correct_idx = max(i for i, c in enumerate(correctness_list[:-1]) if c)
            first_incorrect_idx = next(i for i in range(last_correct_idx + 1, len(correctness_list)) if not correctness_list[i])

            flip_events.append({
                "idx": idx,
                "pid": prefix_list[0].get("pid", idx),
                "ground_truth": prefix_list[0].get("ground_truth", ""),
                "final_prediction": preds[-1],
                "last_correct_prefix_idx": last_correct_idx,
                "first_incorrect_prefix_idx": first_incorrect_idx,
                "total_prefixes": len(prefix_list),
                "prefix_trajectory": "".join(["[V]" if c else "[X]" for c in correctness_list]),
            })

    flip_rate = (len(flip_events) / total_samples * 100) if total_samples > 0 else 0.0

    print(f"\n{'=' * 75}")
    print(f"  [FLIP DETECTION & OVERTHINKING SUMMARY]")
    print(f"{'=' * 75}")
    print(f"  Total Problems Evaluated: {total_samples}")
    print(f"  Flipped Cases (Harmful Overthinking): {len(flip_events)} ({flip_rate:.1f}%)")
    print(f"{'=' * 75}")

    if flip_events:
        print(f"\n  TABLE OF FLIPPED PROBLEMS (Correct Initially -> Incorrect Final):")
        print(f"  {'Idx':<6} | {'Last V':<8} | {'First X':<8} | {'Ground Truth':<15} | {'Final Pred':<15} | {'Trajectory'}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*15}-+-{'-'*15}-+-{'-'*20}")
        for fe in flip_events[:25]:
            print(f"  {fe['idx']:<6} | {fe['last_correct_prefix_idx']:<8} | {fe['first_incorrect_prefix_idx']:<8} | {str(fe['ground_truth'])[:14]:<15} | {str(fe['final_prediction'])[:14]:<15} | {fe['prefix_trajectory']}")
        if len(flip_events) > 25:
            print(f"  ... and {len(flip_events) - 25} more flipped cases")

    print(f"\n  [PREFIX ACCURACY CURVE]:")
    print(f"  {'Prefix Step':<15} | {'Accuracy':<12} | {'Correct/Total'}")
    print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*15}")
    curve_data = []
    for d_idx in sorted(prefix_accuracy.keys()):
        pa = prefix_accuracy[d_idx]
        acc = (pa["correct"] / pa["total"] * 100) if pa["total"] > 0 else 0.0
        curve_data.append({"prefix_idx": d_idx, "accuracy": acc, "correct": pa["correct"], "total": pa["total"]})
        if d_idx <= 15 or d_idx == max(prefix_accuracy.keys()):
            print(f"  Prefix {d_idx:<8} | {acc:6.1f}%      | ({pa['correct']}/{pa['total']})")

    all_trajectories = {}
    for idx, prefix_list in records_by_idx.items():
        prefix_list.sort(key=lambda x: x.get("difficulty_idx", 0))
        c_list = []
        p_list = []
        for pr in prefix_list:
            pred = pr.get("prediction", pr.get("model_parsed_answer"))
            if pred is None:
                raw_out = pr.get("model_output", "")
                pred = ParsingHelper.extract_last_boxed(raw_out)
                if not pred:
                    match = re.findall(r"[-+]?\d*\.\d+|\d+", raw_out)
                    pred = match[-1] if match else ""
            pred = ParsingHelper.clean(str(pred))
            gt = ParsingHelper.clean(str(pr.get("ground_truth", "")))
            c_list.append(pred == gt and pred != "")
            p_list.append(pred)
        
        is_flip = any(fe["idx"] == idx for fe in flip_events)
        all_trajectories[idx] = {
            "idx": idx,
            "ground_truth": prefix_list[0].get("ground_truth", ""),
            "final_correct": c_list[-1] if c_list else False,
            "is_flip": is_flip,
            "correctness": c_list,
            "predictions": p_list,
            "trajectory_str": "".join(["[V]" if c else "[X]" for c in c_list]),
        }

    os.makedirs(os.path.join(PROJECT_ROOT, run_path), exist_ok=True)
    flip_analysis_path = os.path.join(PROJECT_ROOT, run_path, "flip_analysis.json")
    flip_csv_path = os.path.join(PROJECT_ROOT, run_path, "flip_curve.csv")

    with open(flip_analysis_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": total_samples,
            "flipped_count": len(flip_events),
            "flip_rate_pct": flip_rate,
            "flips": flip_events,
            "accuracy_curve": curve_data,
            "all_trajectories": all_trajectories,
        }, f, indent=2)

    with open(flip_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["prefix_idx", "accuracy", "correct", "total"])
        writer.writeheader()
        writer.writerows(curve_data)

    print(f"\n  Saved flip analysis to: {flip_analysis_path}")
    print(f"  Saved flip curve CSV to: {flip_csv_path}")
    print(f"{'=' * 75}\n")

    return {"flips": flip_events, "flip_rate": flip_rate}


def main() -> None:
    args = parse_args()
    py = sys.executable

    # 1. Scan downloaded models
    cached = get_cached_models(DEFAULT_CACHE)
    cached_available = map_cached_to_registered(cached)

    # 2. Resolve model
    selected_model = args.model
    if not selected_model:
        selected_model = select_model_interactive(cached_available)
    else:
        is_cached = any(ca["model_key"] == selected_model for ca in cached_available)
        if is_cached:
            print(f"\n  [OK] Found '{selected_model}' in local HF cache! Proceeding...")
        else:
            print(f"\n  [INFO] Selected model: '{selected_model}'")

    # Normalize benchmark name and handle typos
    BENCHMARK_ALIASES = {"gms8k": "gsm8k", "math-500": "math500", "math_500": "math500"}
    args.benchmark = BENCHMARK_ALIASES.get(args.benchmark.lower().strip(), args.benchmark)

    # 3. Resolve Quantization
    quantization = args.quantization
    if not quantization:
        quantization = select_quantization_interactive(selected_model)

    print(f"\n{'=' * 75}")
    print(f"  THINKING-PAST-THE-ANSWER: RUNNING PIPELINE")
    print(f"{'=' * 75}")
    print(f"  Model:        {selected_model}")
    print(f"  Benchmark:    {args.benchmark}")
    print(f"  Backend:      {args.backend}")
    print(f"  Quantization: {quantization}")
    print(f"  Seed:         {args.seed}")
    print(f"  Limit:        {args.limit if args.limit is not None else 'All'}")
    if getattr(args, "timestamp_experiment", False):
        import datetime
        args.experiment_name = f"{args.experiment_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"  Experiment:   {args.experiment_name}")
    print(f"{'=' * 75}")

    total_start = time.time()

    # Paths
    run_path = os.path.join(args.output, args.experiment_name, selected_model, args.benchmark, f"seed_{args.seed}")
    os.makedirs(run_path, exist_ok=True)

    generations_file = os.path.join(run_path, "generations.jsonl")
    diff_filename = f"difficulty_difficulty_{args.difficulty_level}_granularity_{args.granularity}.jsonl"
    
    # difficulty.py puts output inside a subfolder matching the budget prompt
    clean_bf = re.sub(r"[^a-zA-Z0-9_-]", "_", args.budget_forcing_prompt)[:30]
    budget_subfolder = os.path.join(run_path, f"budget_prompt_{clean_bf}")
    
    difficulty_file = os.path.join(budget_subfolder, diff_filename)
    if not os.path.exists(difficulty_file):
        difficulty_file_alt = os.path.join(run_path, diff_filename)
        if os.path.exists(difficulty_file_alt):
            difficulty_file = difficulty_file_alt

    parsed_diff_file = os.path.join(os.path.dirname(difficulty_file), f"parsed_responses_{diff_filename}")

    additional_model_args = f"quantization={quantization}"

    # Stage 1: Benchmark Evaluation
    if not args.skip_eval:
        cmd = [
            py, "eval.py",
            "--model", selected_model,
            "--benchmark", args.benchmark,
            "--backend", args.backend,
            "--additional_model_args", additional_model_args,
            "--output", args.output,
            "--experiment_name", args.experiment_name,
            "--seed", str(args.seed),
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if not run_stage(cmd, "Stage 1: Benchmark Evaluation"):
            sys.exit(1)

    # Stage 2: Difficulty Prefix Replay
    if not args.skip_difficulty:
        cmd = [
            py, "difficulty.py",
            "--model", selected_model,
            "--benchmark", args.benchmark,
            "--backend", args.backend,
            "--additional_model_args", additional_model_args,
            "--output", args.output,
            "--experiment_name", args.experiment_name,
            "--seed", str(args.seed),
            "--difficulty_level", args.difficulty_level,
            "--granularity", str(args.granularity),
            "--budget_forcing_prompt", args.budget_forcing_prompt,
            "--max_tokens", "256",
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if not run_stage(cmd, "Stage 2: Difficulty Prefix Replay"):
            sys.exit(1)

    # Refresh difficulty path in case stage 2 just created it
    # Refresh difficulty path in case stage 2 just created it
    if os.path.exists(os.path.join(budget_subfolder, diff_filename)):
        difficulty_file = os.path.join(budget_subfolder, diff_filename)
        parsed_diff_file = os.path.join(budget_subfolder, f"parsed_responses_{diff_filename}")

    # Stage 3: Standalone Answers Evaluation
    if not args.skip_standalone_eval and os.path.exists(difficulty_file):
        target_eval_file = difficulty_file
        cmd = [
            py, "evaluate_answers_standalone.py",
            "--benchmark", args.benchmark,
            "--input_file", target_eval_file,
            "--parsed_responses_filename", f"parsed_responses_{diff_filename}",
            "--include_difficulty",
        ]
        if not run_stage(cmd, "Stage 3: Standalone Answer Accuracy Evaluation"):
            sys.exit(1)

    # Stage 4: Flip Detection & Overthinking Report
    if not args.skip_flips and os.path.exists(difficulty_file):
        print(f"\n{'#' * 75}")
        print(f"  STAGE: Stage 4: Flip Detection & Accuracy Curve Analysis")
        print(f"{'#' * 75}")
        analysis_input = parsed_diff_file if os.path.exists(parsed_diff_file) else difficulty_file
        analyze_flips_and_difficulty(analysis_input, os.path.dirname(difficulty_file))
        
        # Automatically generate high-res plot
        try:
            from plot_accuracy_curve import load_analysis_data, plot_curve
            fa_path = os.path.join(os.path.dirname(difficulty_file), "flip_analysis.json")
            if os.path.exists(fa_path):
                data = load_analysis_data(fa_path)
                saved_plot = plot_curve(data, title_suffix=f"{selected_model} on {args.benchmark}", granularity=args.granularity)
                print(f"  [PLOT] Saved high-res accuracy curve to: {saved_plot}")
        except Exception as e:
            print(f"  [PLOT] Could not generate plot automatically: {e}")

    # Stage 5: Taxonomy Classification
    if not args.skip_taxonomy and os.path.exists(difficulty_file):
        target_parsed = parsed_diff_file if os.path.exists(parsed_diff_file) else difficulty_file
        
        # Base URL resolution
        base_url = "https://api.groq.com/openai/v1"
        judge_model = args.judge_model or "llama-3.3-70b-versatile"
        if args.judge_provider == "openai":
            base_url = "https://api.openai.com/v1"
            if not args.judge_model:
                judge_model = "gpt-4o-mini"
        elif args.judge_provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
            if not args.judge_model:
                judge_model = "anthropic/claude-3.5-sonnet"

        api_key = args.judge_api_key or os.environ.get("GROQ_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

        if not api_key:
            print("  [INFO] No judge API key provided (use --judge_api_key or set GROQ_API_KEY). Skipping Stage 5.")
        else:
            cmd = [
                py, "taxonomy/categorize_overthinking_from_last_true.py",
                "--input_file", difficulty_file,
                "--parsed_responses_file", target_parsed,
                "--benchmark", args.benchmark,
                "--backend", "openai",
                "--openai_base_url", base_url,
                "--openai_api_key", api_key,
                "--openai_model", judge_model,
            ]
            if not run_stage(cmd, "Stage 5: Failure Taxonomy Classification"):
                print("Note: Stage 5 completed with warnings.")

    # Stage 6: Mechanistic Insights & Dynamics Generation
    if not args.skip_mechanistic and os.path.exists(difficulty_file):
        experiment_leaf_dir = os.path.dirname(difficulty_file)
        # Determine actual HuggingFace model path from register
        hf_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        if selected_model in MODEL_REGISTER:
            hf_model_id = getattr(MODEL_REGISTER[selected_model], "hf_name", selected_model)

        cmd_mech1 = [
            py, "extract_real_mechanistic_metrics.py",
            "--run_dir", experiment_leaf_dir,
            "--model", hf_model_id,
            "--quantization", quantization,
        ]
        run_stage(cmd_mech1, "Stage 6A: Extract Real Mechanistic Metrics & Hidden-State Dynamics")

        cmd_mech2 = [
            py, "plot_scientific_mechanistic_insights.py",
            "--run_dir", experiment_leaf_dir,
            "--model", hf_model_id,
            "--quantization", quantization,
        ]
        run_stage(cmd_mech2, "Stage 6B: Generate Scientific Progression & Event-Aligned Dynamics")

        cmd_mech3 = [
            py, "extract_variable_level_evidence.py",
            "--run_dir", experiment_leaf_dir,
            "--model", hf_model_id,
            "--quantization", quantization,
        ]
        run_stage(cmd_mech3, "Stage 6C: Extract Variable-Level Mechanistic Evidence (Termination, Instability, Geometry)")

        # Copy all plots to root experiment folder for instant access
        try:
            import glob
            import shutil
            root_exp_dir = os.path.join(args.output, args.experiment_name)
            for ext in ("*.png", "*.pdf", "*.json", "*.csv", "*.tex"):
                for src_file in glob.glob(os.path.join(experiment_leaf_dir, ext)):
                    dst_file = os.path.join(root_exp_dir, os.path.basename(src_file))
                    shutil.copy2(src_file, dst_file)
            print(f"  [SYNC] All publication plots copied to top-level: {root_exp_dir}")
        except Exception as e:
            print(f"  [SYNC WARNING] Could not copy files to top-level: {e}")

    elapsed = time.time() - total_start
    print(f"\n{'=' * 75}")
    print(f"  [PIPELINE COMPLETE] Time: {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"{'=' * 75}\n")


if __name__ == "__main__":
    main()
