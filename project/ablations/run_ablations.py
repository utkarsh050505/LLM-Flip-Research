import os
import sys
import json
import csv
import time
import datetime
import torch
import numpy as np

# Adjust path so we can import project modules
sys.path.insert(0, os.path.abspath('project'))

from modeling import load_model, MODEL_REGISTER
from modeling.hf_model import HFModel
from benchmarking import load_benchmark
from difficulty import build_base_messages, split_with_granularity, build_continuation_prompt, resolve_tokenizer, generate_from_prompt

class Llama3_2_3B(HFModel):
    hf_name = "unsloth/Llama-3.2-3B-Instruct"
    base_model = "Llama-3.2-3B-Instruct"
    
    @classmethod
    def get_hf_name(cls):
        return cls.hf_name
        
    @classmethod
    def get_base_model_name(cls):
        return cls.base_model
        
MODEL_REGISTER["llama3_2_3b_instruct"] = Llama3_2_3B

TARGET_IDS_FILE = "ablation_target_ids.csv"
SCAFFOLD_RESULTS_FILE = "scaffold_ablation_results.csv"
SCAFFOLD_SUMMARY_FILE = "scaffold_ablation_summary.csv"
RESOLUTION_RESULTS_FILE = "resolution_ablation_results.csv"
RESOLUTION_SUMMARY_FILE = "resolution_ablation_summary.csv"
LOG_FILE = "ablation_run_log.txt"

SCAFFOLDS = {
    "original": "Therefore, the final answer is:",
    "scaffold_a": "Give only the final answer.",
    "scaffold_b": "What is the answer?",
    "scaffold_c": "Therefore, the answer is:"
}

RESOLUTIONS = {
    "original_15": 15,
    "coarse_10": 10
}

# Run configs matching original
RUN_CONFIGS = {
    "deepseek_1.5b_gsm8k": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "benchmark": "gsm8k",
        "generations_path": "project/results/utkarsh_proper_run7_gsm8k/r1_distill_qwen1_5b/gsm8k/seed_42/generations.jsonl"
    },
    "qwen2.5_3b_gsm8k": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "benchmark": "gsm8k",
        "generations_path": "project/results/kartikey_qwen2.5-3b_gsm8k_results/results/kartikey_proper_run1_gsm8k/qwen2_5_3b_instruct/gsm8k/seed_42/generations.jsonl"
    },
    "llama_3.2_3b_gsm8k": {
        "model_name": "llama3_2_3b_instruct",
        "benchmark": "gsm8k",
        "generations_path": "project/results/parth_proper_run1_gsm8k_results/results/parth_proper_run1_gsm8k/meta-llama/Llama-3.2-3B-Instruct/gsm8k/seed_42/generations.jsonl"
    },
    "deepseek_1.5b_math500": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "benchmark": "math500",
        "generations_path": "project/results/utkarsh_proper_run8_math500/r1_distill_qwen1_5b/gsm8k/seed_42/generations.jsonl" # fallback
    }
}

# Fix paths if needed
if not os.path.exists(RUN_CONFIGS["deepseek_1.5b_math500"]["generations_path"]):
    RUN_CONFIGS["deepseek_1.5b_math500"]["generations_path"] = "project/results/utkarsh_proper_run8_math500/generations.jsonl"
if not os.path.exists(RUN_CONFIGS["qwen2.5_3b_gsm8k"]["generations_path"]):
    RUN_CONFIGS["qwen2.5_3b_gsm8k"]["generations_path"] = "project/results/kartikey_qwen2.5-3b_gsm8k_results/results/kartikey_proper_run1_gsm8k/generations.jsonl"

def log_msg(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def label_trajectory(ct_seq):
    if not ct_seq:
        return "never-correct"
    if all(ct_seq):
        return "stable"
    if not any(ct_seq):
        return "never-correct"
    if ct_seq[-1] == 1:
        return "recovery"
    if ct_seq[-1] == 0 and 1 in ct_seq:
        return "forced-degradation"
    return "unknown"

def main():
    # Setup log
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    
    log_msg("Starting Ablation Experiments")
    device = "cpu"
    log_msg(f"Using device: {device}")
    log_msg("LIMITATION NOTE: CPU execution is being used due to MPS OOM/hangs. Dropped fine_30 resolution to make runtime feasible.")
    
    # Load targets
    targets = []
    with open(TARGET_IDS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            targets.append(row)
    
    log_msg(f"Loaded {len(targets)} target traces.")
    
    scaffold_results = []
    resolution_results = []
    
    # Group by model config to avoid reloads
    run_groups = {}
    for t in targets:
        rg = t['run']
        if rg not in run_groups:
            run_groups[rg] = []
        run_groups[rg].append(t)
        
    for run_name, traces in run_groups.items():
        log_msg(f"\nProcessing Run: {run_name} ({len(traces)} traces)")
        cfg = RUN_CONFIGS[run_name]
        
        # Load benchmark
        benchmark = load_benchmark(cfg["benchmark"])
        
        # Load generations
        gen_path = cfg["generations_path"]
        log_msg(f"Loading generations from {gen_path}")
        gens = {}
        with open(gen_path, 'r') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    gens[r["idx"]] = r
        
        # Load model
        log_msg(f"Loading model: {cfg['model_name']} on backend hf (device {device})")
        start_time = time.time()
        try:
            model = load_model(
                cfg["model_name"],
                backend="hf",
                max_tokens=20, # drastically reduced to speed up CPU inference
                max_num_images=1,
                seed=42,
                additional_args="temperature=0.6;top_p=0.95"
            )
            # Patch generation config to use pad token properly and avoid warnings
            if hasattr(model, 'model') and model.model is not None:
                if model.model.config.pad_token_id is None:
                    model.model.config.pad_token_id = model.model.config.eos_token_id
            
            tokenizer = resolve_tokenizer(model.processor)
        except Exception as e:
            log_msg(f"ERROR loading model {cfg['model_name']}: {e}")
            continue
            
        load_time = time.time() - start_time
        log_msg(f"Model loaded in {load_time:.1f}s")
        
        for t in traces:
            idx = int(t['trace_id'])
            original_label = t['original_label']
            log_msg(f"  Trace {idx} ({original_label})")
            
            if idx not in gens:
                log_msg(f"    WARNING: Trace {idx} not found in generations file.")
                continue
            
            sample = gens[idx]
            raw_text = sample["model_output"]
            ground_truth = sample["ground_truth"]
            
            # --- STEP 1 & 2: Evaluate Scaffolds & Resolutions ---
            # To be efficient, we can batch prefixes if the model supports it, 
            # but difficulty.py loops sequentially. We will loop sequentially.
            
            # 1. Scaffolds (at original 15 granularity) + Resolutions (at original scaffold)
            # We can combine the execution matrix:
            # We need: (scaffolds) at 15 segments
            # We need: original scaffold at (10, 15, 30 segments)
            
            # Build base message
            mock_sample = {
                "query": sample["question"],
                "decoded_image": None
            }
            base_msgs = build_base_messages(mock_sample, model)
            
            tasks_to_run = []
            
            # Scaffold tasks (at 15 segs)
            for sname, stext in SCAFFOLDS.items():
                tasks_to_run.append({"type": "scaffold", "scaffold_name": sname, "scaffold_text": stext, "resolution_name": "original_15", "granularity": 15})
                
            # Resolution tasks (at original scaffold)
            for rname, rval in RESOLUTIONS.items():
                if rname != "original_15": # already added above
                    tasks_to_run.append({"type": "resolution", "scaffold_name": "original", "scaffold_text": SCAFFOLDS["original"], "resolution_name": rname, "granularity": rval})
            
            for task in tasks_to_run:
                # Split text
                prefixes = split_with_granularity(raw_text, tokenizer, difficulty_level="utterance", granularity=task["granularity"])
                
                # Memory optimization note
                pass
                
                ct_seq = []
                final_answer_extracted = None
                
                for p_idx, prefix in enumerate(prefixes):
                    # Continue generating
                    prompt = build_continuation_prompt(model, base_msgs, prefix + "\n\n" + task["scaffold_text"])
                    
                    try:
                        cont = generate_from_prompt(model, prompt, None)
                    except Exception as e:
                        log_msg(f"    ERROR during generation: {e}")
                        cont = ""
                        
                    # Evaluate correctness without relying on model.parser (which prints warnings)
                    import re
                    from evaluation import ParsingHelper
                    
                    pred = ParsingHelper.extract_last_boxed(cont)
                    if not pred:
                        match = re.findall(r"[-+]?\d*\.\d+|\d+", cont)
                        pred = match[-1] if match else ""
                        
                    parsed_gt = model.normalize_answer(ground_truth, clean_only=True)
                    
                    pred = ParsingHelper.clean(pred)
                    parsed_gt = ParsingHelper.clean(parsed_gt)
                    
                    is_correct = (pred == parsed_gt or (parsed_gt != "" and parsed_gt in pred))
                    extracted = pred
                    
                    ct_seq.append(1 if is_correct else 0)
                    print(".", end="", flush=True)
                    if p_idx == len(prefixes) - 1:
                        final_answer_extracted = extracted
                
                print() # newline after trace
                new_label = label_trajectory(ct_seq)
                
                # Save results
                if task["type"] == "scaffold" or task["resolution_name"] == "original_15":
                    agrees = (new_label == original_label)
                    scaffold_results.append({
                        "trace_id": idx,
                        "run": run_name,
                        "original_label": original_label,
                        "scaffold_name": task["scaffold_name"],
                        "new_label": new_label,
                        "agrees_with_original": agrees,
                        "ct_seq": json.dumps(ct_seq)
                    })
                
                if task["type"] == "resolution" or task["resolution_name"] == "original_15":
                    agrees = (new_label == original_label)
                    resolution_results.append({
                        "trace_id": idx,
                        "run": run_name,
                        "granularity": task["resolution_name"],
                        "new_label": new_label,
                        "agrees_with_original_15seg": agrees,
                        "ct_seq": json.dumps(ct_seq)
                    })
                    
        # Free memory before loading next model
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Write Results & Summaries ---
    
    # 1. Scaffold Results
    with open(SCAFFOLD_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["trace_id", "run", "original_label", "scaffold_name", "new_label", "agrees_with_original", "ct_seq"])
        writer.writeheader()
        writer.writerows(scaffold_results)
        
    # 2. Resolution Results
    with open(RESOLUTION_RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["trace_id", "run", "granularity", "new_label", "agrees_with_original_15seg", "ct_seq"])
        writer.writeheader()
        writer.writerows(resolution_results)
        
    # 3. Scaffold Summary
    # For each of the 3 new scaffolds: how many of the original 11 degradation candidates remain classified as forced-degradation
    # Jaccard similarity within this subset
    # Transition rate
    
    scaffold_summaries = []
    for sname in SCAFFOLDS.keys():
        if sname == "original": continue
        
        orig_deg = [r for r in scaffold_results if r["scaffold_name"] == "original" and r["original_label"] == "degradation" and r["new_label"] == "forced-degradation"]
        # Actually, let's just compare to original_label="degradation"
        s_deg = [r for r in scaffold_results if r["scaffold_name"] == sname and r["new_label"] == "forced-degradation"]
        
        orig_deg_set = set([(r["run"], r["trace_id"]) for r in scaffold_results if r["scaffold_name"] == "original" and r["original_label"] == "degradation"])
        s_deg_set = set([(r["run"], r["trace_id"]) for r in s_deg])
        
        intersection = orig_deg_set.intersection(s_deg_set)
        union = orig_deg_set.union(s_deg_set)
        jaccard = len(intersection) / len(union) if union else 0.0
        
        remained = len(intersection)
        total_orig = len(orig_deg_set)
        
        # Correct-to-incorrect transition rate (sum of transitions / total pairs)
        all_s_res = [r for r in scaffold_results if r["scaffold_name"] == sname]
        total_trans = 0
        total_pairs = 0
        for r in all_s_res:
            seq = json.loads(r["ct_seq"])
            for i in range(len(seq)-1):
                total_pairs += 1
                if seq[i] == 1 and seq[i+1] == 0:
                    total_trans += 1
        trans_rate = total_trans / total_pairs if total_pairs > 0 else 0.0
        
        scaffold_summaries.append({
            "scaffold_name": sname,
            "retained_degradation_count": remained,
            "retained_degradation_pct": (remained/total_orig*100) if total_orig else 0,
            "jaccard_similarity": jaccard,
            "correct_to_incorrect_transition_rate": trans_rate
        })
        
    with open(SCAFFOLD_SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["scaffold_name", "retained_degradation_count", "retained_degradation_pct", "jaccard_similarity", "correct_to_incorrect_transition_rate"])
        writer.writeheader()
        writer.writerows(scaffold_summaries)

    # 4. Resolution Summary
    resolution_summaries = []
    
    # Original degrad count
    orig_deg_set = set([(r["run"], r["trace_id"]) for r in resolution_results if r["granularity"] == "original_15" and r["new_label"] == "forced-degradation"])
    
    for rname in RESOLUTIONS.keys():
        if rname == "original_15": continue
        
        r_deg_set = set([(r["run"], r["trace_id"]) for r in resolution_results if r["granularity"] == rname and r["new_label"] == "forced-degradation"])
        remained = len(orig_deg_set.intersection(r_deg_set))
        total_orig = len(orig_deg_set)
        
        # Flips per trace
        all_r_res = [r for r in resolution_results if r["granularity"] == rname]
        total_flips = 0
        for r in all_r_res:
            seq = json.loads(r["ct_seq"])
            flips = sum(1 for i in range(len(seq)-1) if seq[i] != seq[i+1])
            total_flips += flips
        mean_flips = total_flips / len(all_r_res) if all_r_res else 0.0
        
        resolution_summaries.append({
            "granularity": rname,
            "retained_degradation_count": remained,
            "retained_degradation_pct": (remained/total_orig*100) if total_orig else 0,
            "mean_flips_per_trajectory": mean_flips
        })
        
    with open(RESOLUTION_SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["granularity", "retained_degradation_count", "retained_degradation_pct", "mean_flips_per_trajectory"])
        writer.writeheader()
        writer.writerows(resolution_summaries)
        
    # Specifically for DeepSeek/MATH-500: write out per-segment accuracy for 60-75% check
    log_msg("\n--- DeepSeek/MATH-500 Resolution Check ---")
    for rname in RESOLUTIONS.keys():
        math_res = [r for r in resolution_results if r["run"] == "deepseek_1.5b_math500" and r["granularity"] == rname]
        if not math_res: continue
        
        max_len = max(len(json.loads(r["ct_seq"])) for r in math_res)
        acc_per_step = []
        for i in range(max_len):
            vals = []
            for r in math_res:
                seq = json.loads(r["ct_seq"])
                if i < len(seq): vals.append(seq[i])
                else: vals.append(seq[-1]) # carry forward
            acc = sum(vals) / len(vals)
            acc_per_step.append(acc)
            
        log_msg(f"Granularity: {rname}")
        log_msg(f"Accuracy progression: {[round(a,2) for a in acc_per_step]}")

    log_msg("\nAblation experiments completed successfully!")

if __name__ == "__main__":
    main()
    log_msg("\nAblation experiments completed successfully!")

if __name__ == "__main__":
    main()
