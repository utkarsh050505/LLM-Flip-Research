import copy
import json
import re
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).parent.parent.parent))
from project.configs import pipeline_config as cfg


def normalize_answer(ans):
    if ans is None:
        return None
    ans = str(ans)
    # Strip \text{...}
    ans = re.sub(r"\\text\{[^}]*\}", "", ans)
    # Normalize \frac{a}{b} -> a/b
    ans = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", ans)
    return re.sub(r"[^\dA-Za-z\.\-]", "", ans)

def extract_boxed_answer(text: str):
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + 7
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


def find_natural_think_close(token_ids, tokenizer, prompt_length):
    """Finds the token position just before the natural </think> tag."""
    # Decode progressively to find the boundary
    # This is a bit brute force but completely robust against tokenization quirks
    full_text = tokenizer.decode(token_ids[prompt_length:], skip_special_tokens=False)
    
    # If there's no </think> in the whole trace, just return the end
    if "</think>" not in full_text:
        return len(token_ids)
        
    # Find exact token position by scanning
    for i in range(prompt_length, len(token_ids)):
        text_so_far = tokenizer.decode(token_ids[prompt_length:i], skip_special_tokens=False)
        if "</think>" in text_so_far:
            # The </think> was just completed at token i-1
            # We want the position BEFORE </think> closes
            # Actually, to be safe, we return i
            # Let's refine: find the position where it appears
            return i
            
    return len(token_ids)


def get_final_answer_from_trace(token_ids, tokenizer, prompt_length):
    text = tokenizer.decode(token_ids[prompt_length:], skip_special_tokens=True)
    return extract_boxed_answer(text)


def main():
    print("Stage 2: Checkpoint Probes")
    
    trace_dirs = list(cfg.TRACES_DIR.glob("*"))
    
    target_problem = sys.argv[1] if len(sys.argv) > 1 else None
    if target_problem:
        if target_problem.endswith(".json"):
            target_problem = target_problem[:-5]
        trace_dirs = [d for d in trace_dirs if d.name == target_problem]
        
    if not trace_dirs:
        print(f"No traces found. (Target: {target_problem if target_problem else 'All'})")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, padding_side="left")
    
    probe_prefix_str = "</think>\n\n**Final Answer:**\n\\boxed{"
    probe_prefix_ids = tokenizer(probe_prefix_str, add_special_tokens=False, return_tensors="pt").input_ids.to(cfg.DEVICE)
    
    model = None

    for trace_dir in trace_dirs:
        problem_name = trace_dir.name
        
        # Load ground truth from problem file
        prob_file = cfg.PROBLEMS_DIR / f"{problem_name}.json"
        if not prob_file.exists():
            print(f"Warning: {prob_file} not found. Skipping trace dir.")
            continue
            
        with open(prob_file, "r", encoding="utf-8") as f:
            prob_data = json.load(f)
            
        gt_answer = normalize_answer(prob_data["answer"])
        
        trace_files = list(trace_dir.glob("trace_*.pt"))
        if not trace_files:
            continue
            
        print(f"\nProcessing {problem_name} ({len(trace_files)} traces)...")
        
        if model is None:
            print("Loading model...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_NAME, 
                    torch_dtype=torch.bfloat16, 
                    device_map=cfg.DEVICE,
                    attn_implementation="sdpa"
                )
            except (ValueError, ImportError):
                model = AutoModelForCausalLM.from_pretrained(
                    cfg.MODEL_NAME, 
                    torch_dtype=torch.bfloat16, 
                    device_map=cfg.DEVICE
                )
            model.eval()

        for trace_file in trace_files:
            checkpoint_file = trace_dir / f"checkpoints_{trace_file.stem.split('_')[-1]}.json"
            if checkpoint_file.exists():
                continue
                
            print(f"  {trace_file.name}...")
            trace_data = torch.load(trace_file)
            token_ids = trace_data["token_ids"].unsqueeze(0).to(cfg.DEVICE) # (1, seq_len)
            prompt_length = trace_data["prompt_length"]
            
            # Extract natural final answer for record
            final_ans = get_final_answer_from_trace(token_ids[0], tokenizer, prompt_length)
            final_ans_norm = normalize_answer(final_ans)
            
            # Find natural </think> boundary
            think_close_pos = find_natural_think_close(token_ids[0], tokenizer, prompt_length)
            closed_naturally = (think_close_pos < token_ids.shape[1])
            
            checkpoints_result = {
                "problem_file": prob_data.get("problem", ""), 
                "trace_file": trace_file.name,
                "ground_truth": gt_answer,
                "natural_think_close_pos": think_close_pos,
                "checkpoints": [],
                "final_answer": final_ans_norm,
                "final_correct": (final_ans_norm == gt_answer) if final_ans_norm is not None else None
            }
            
            running_cache = None
            last_cp = 0
            
            # Determine checkpoints using relative fractions of the reasoning phase
            fractions = [0.25, 0.50, 0.70, 0.85, 0.95]
            checkpoints_to_eval = []
            for frac in fractions:
                cp = prompt_length + int((think_close_pos - prompt_length) * frac)
                # Ensure we don't accidentally step backwards if tokens are too few
                if not checkpoints_to_eval or cp > checkpoints_to_eval[-1][1]:
                    checkpoints_to_eval.append((frac, cp))
            
            # Step 1: Seed the cache with the prompt
            with torch.no_grad():
                out = model(
                    input_ids=token_ids[:, :prompt_length],
                    use_cache=True
                )
            running_cache = out.past_key_values
            last_cp = prompt_length
            
            # Step 2: Walk the fractional checkpoints
            for frac, cp in checkpoints_to_eval:
                # Extend the cache incrementally
                new_chunk = token_ids[:, last_cp:cp]
                with torch.no_grad():
                    out = model(
                        input_ids=new_chunk,
                        past_key_values=running_cache,
                        use_cache=True
                    )
                running_cache = out.past_key_values
                last_cp = cp
                
                # Clone for probe
                probe_cache = copy.deepcopy(running_cache)
                
                # Construct FULL input sequence: History + Prefix
                # generate() will see past_key_values has length `cp`, automatically slice the
                # first `cp` tokens off this tensor, and only process the prefix efficiently.
                probe_input_ids = torch.cat([token_ids[:, :cp], probe_prefix_ids], dim=1)
                
                # Mask must match the full sequence length
                probe_attention_mask = torch.ones_like(probe_input_ids)
                
                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=probe_input_ids,
                        attention_mask=probe_attention_mask,
                        past_key_values=probe_cache,
                        max_new_tokens=cfg.PROBE_MAX_TOKENS,
                        do_sample=True,
                        temperature=cfg.PROBE_TEMPERATURE,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                
                # Decode only the newly generated tokens (gen_out.sequences contains the full sequence)
                new_tokens = gen_out.sequences[0, probe_input_ids.shape[1]:]
                probe_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                # Use robust extraction by pretending the \boxed{ was part of the output
                raw_probe_ans = extract_boxed_answer(f"\\boxed{{{probe_text}")
                norm_probe_ans = normalize_answer(raw_probe_ans)
                
                is_correct = (norm_probe_ans == gt_answer) if norm_probe_ans is not None else None
                
                checkpoints_result["checkpoints"].append({
                    "fraction": frac,
                    "token_position": cp,
                    "probe_answer": norm_probe_ans,
                    "correct": is_correct
                })
                
                print(f"    CP {cp:05d} ({frac*100:02.0f}%) | ans: {norm_probe_ans!r:<10} | correct: {is_correct}")

            # --- Recovery Probe for Truncated Traces ---
            if checkpoints_result["final_answer"] is None:
                if last_cp < think_close_pos:
                    new_chunk = token_ids[:, last_cp:think_close_pos]
                    with torch.no_grad():
                        out = model(
                            input_ids=new_chunk,
                            past_key_values=running_cache,
                            use_cache=True
                        )
                    running_cache = out.past_key_values
                    
                if closed_naturally:
                    final_probe_str = "\n\n**Final Answer:**\n\\boxed{"
                else:
                    final_probe_str = "</think>\n\n**Final Answer:**\n\\boxed{"
                    
                final_probe_ids = tokenizer(final_probe_str, add_special_tokens=False, return_tensors="pt").input_ids.to(cfg.DEVICE)
                probe_input_ids = torch.cat([token_ids[:, :think_close_pos], final_probe_ids], dim=1)
                probe_attention_mask = torch.ones_like(probe_input_ids)
                
                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=probe_input_ids,
                        attention_mask=probe_attention_mask,
                        past_key_values=running_cache,
                        max_new_tokens=cfg.PROBE_MAX_TOKENS,
                        do_sample=True,
                        temperature=cfg.PROBE_TEMPERATURE,
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )
                new_tokens = gen_out.sequences[0, probe_input_ids.shape[1]:]
                probe_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                raw_probe_ans = extract_boxed_answer(f"\\boxed{{{probe_text}")
                recovered_ans = normalize_answer(raw_probe_ans)
                
                checkpoints_result["final_answer"] = recovered_ans
                checkpoints_result["final_correct"] = (recovered_ans == gt_answer) if recovered_ans is not None else None
                print(f"    [Recovered Final Ans] ans: {recovered_ans!r:<10} | correct: {checkpoints_result['final_correct']}")

            # Save the checkpoint results
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoints_result, f, indent=4)
                
    print("\nStage 2 Complete.")

if __name__ == "__main__":
    main()