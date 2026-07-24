import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from project.configs import pipeline_config as cfg

def main():
    print("Stage 3: Flip Detection")
    
    trace_dirs = list(cfg.TRACES_DIR.glob("*"))
    if not trace_dirs:
        print("No traces found. Run Stage 1 & 2 first.")
        return
        
    all_traces = []
    
    for trace_dir in trace_dirs:
        checkpoint_files = list(trace_dir.glob("checkpoints_*.json"))
        for cp_file in checkpoint_files:
            with open(cp_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_traces.append(data)
                
    if not all_traces:
        print("No checkpoint files found.")
        return
        
    print(f"Loaded {len(all_traces)} traces.")
    
    # 1. Per-trace flip analysis
    trace_records = []
    max_budget = 0
    
    for trace in all_traces:
        problem_file = trace["problem_file"]
        trace_file = trace["trace_file"]
        checkpoints = trace["checkpoints"]
        
        # Sort by position just in case
        checkpoints.sort(key=lambda x: x["token_position"])
        
        valid_states = []
        for cp in checkpoints:
            if cp["correct"] is not None:
                valid_states.append({
                    "pos": cp["token_position"],
                    "correct": cp["correct"]
                })
            if cp["token_position"] > max_budget:
                max_budget = cp["token_position"]
                
        # Find transitions
        transitions = []
        c2i_flips = 0
        i2c_flips = 0
        
        for i in range(1, len(valid_states)):
            prev = valid_states[i-1]
            curr = valid_states[i]
            if prev["correct"] and not curr["correct"]:
                transitions.append(f"C2I at {curr['pos']}")
                c2i_flips += 1
            elif not prev["correct"] and curr["correct"]:
                transitions.append(f"I2C at {curr['pos']}")
                i2c_flips += 1
                
        trace_records.append({
            "problem": problem_file,
            "trace": trace_file,
            "c2i_flips": c2i_flips,
            "i2c_flips": i2c_flips,
            "transitions": transitions,
            "final_correct": trace["final_correct"]
        })
        
    # Save trace records
    analysis_file = cfg.RESULTS_DIR / "flip_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(trace_records, f, indent=4)
        
    # 2. Flip-ratio-by-fraction curve
    # For each relative budget fraction, what is the state of each trace?
    fractions = [0.25, 0.50, 0.70, 0.85, 0.95, 1.0] # 1.0 represents the final answer
    
    curve_rows = []
    cumulative_c2i = 0
    
    for frac in fractions:
        total_valid = 0
        total_correct = 0
        flips_c2i_at_b = 0
        flips_i2c_at_b = 0
        total_transitions_at_b = 0 # How many valid (prev, curr) pairs exist at this budget
        
        for trace in all_traces:
            curr_state = None
            if frac == 1.0:
                curr_state = trace.get("final_correct")
            else:
                cps_at_frac = [cp for cp in trace["checkpoints"] if cp.get("fraction") == frac]
                if cps_at_frac and cps_at_frac[0]["correct"] is not None:
                    curr_state = cps_at_frac[0]["correct"]
                    
            if curr_state is not None:
                total_valid += 1
                if curr_state:
                    total_correct += 1
                    
            # Check for flips against the most recent prior valid state
            if curr_state is not None:
                prior_state = None
                if frac == 1.0:
                    prior_cps = [cp for cp in trace["checkpoints"] if cp["correct"] is not None]
                    if prior_cps:
                        prior_state = prior_cps[-1]["correct"]
                else:
                    prior_cps = [cp for cp in trace["checkpoints"] if cp.get("fraction", 0) < frac and cp["correct"] is not None]
                    if prior_cps:
                        prior_state = prior_cps[-1]["correct"]
                        
                if prior_state is not None:
                    total_transitions_at_b += 1
                    if prior_state and not curr_state:
                        flips_c2i_at_b += 1
                        cumulative_c2i += 1
                    elif not prior_state and curr_state:
                        flips_i2c_at_b += 1
                            
        acc = (total_correct / total_valid) if total_valid > 0 else 0
        c2i_rate = (flips_c2i_at_b / total_transitions_at_b) if total_transitions_at_b > 0 else 0
        i2c_rate = (flips_i2c_at_b / total_transitions_at_b) if total_transitions_at_b > 0 else 0
        
        curve_rows.append({
            "budget_fraction": f"{frac*100:.0f}%",
            "valid_traces": total_valid,
            "accuracy": acc,
            "c2i_rate": c2i_rate,
            "i2c_rate": i2c_rate,
            "cumulative_c2i": cumulative_c2i
        })
        
    csv_file = cfg.RESULTS_DIR / "flip_curve.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["budget_fraction", "valid_traces", "accuracy", "c2i_rate", "i2c_rate", "cumulative_c2i"])
        writer.writeheader()
        writer.writerows(curve_rows)
        
    # 3. Summary statistics
    # Which problems exhibit the most overthinking?
    problem_c2i_counts = defaultdict(int)
    for r in trace_records:
        if r["c2i_flips"] > 0:
            problem_c2i_counts[r["problem"]] += 1
            
    summary_file = cfg.RESULTS_DIR / "flip_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("=== Flip Detection Summary ===\n\n")
        f.write(f"Total traces analyzed: {len(all_traces)}\n")
        f.write(f"Total C2I flips across all traces: {sum(r['c2i_flips'] for r in trace_records)}\n")
        f.write(f"Total I2C flips across all traces: {sum(r['i2c_flips'] for r in trace_records)}\n\n")
        
        f.write("Problems with most C2I traces (traces exhibiting at least one C2I flip):\n")
        sorted_probs = sorted(problem_c2i_counts.items(), key=lambda x: x[1], reverse=True)
        for p, count in sorted_probs:
            f.write(f"  {p}: {count} traces\n")
            
    print(f"Saved analysis to {analysis_file}")
    print(f"Saved curve to {csv_file}")
    print(f"Saved summary to {summary_file}")
    print("\nStage 3 Complete.")

if __name__ == "__main__":
    main()
