import json
import os
from pathlib import Path
import sys

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

# Adjust sys.path to allow importing from project.configs
sys.path.append(str(Path(__file__).parent.parent.parent))
from project.configs import pipeline_config as cfg

def main():
    print("Stage 1: Natural Traces")
    print(f"Model: {cfg.MODEL_NAME}")
    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, padding_side="left")
    
    # Check if there are any problems to process
    problem_files = list(cfg.PROBLEMS_DIR.glob("*.json"))
    
    target_problem = sys.argv[1] if len(sys.argv) > 1 else None
    if target_problem:
        if target_problem.endswith(".json"):
            target_problem = target_problem[:-5]
        problem_files = [p for p in problem_files if p.stem == target_problem]
        
    if not problem_files:
        print(f"No problem files found. (Target: {target_problem if target_problem else 'All'})")
        return
    print(f"Found {len(problem_files)} problems.")

    model = None # Lazy load

    for prob_file in problem_files:
        problem_name = prob_file.stem
        problem_traces_dir = cfg.TRACES_DIR / problem_name
        
        # Check if already fully processed
        last_trace_file = problem_traces_dir / f"trace_{cfg.N_TRACES_PER_PROBLEM-1:03d}.pt"
        if last_trace_file.exists():
            print(f"Skipping {problem_name}, already processed.")
            continue
            
        print(f"\nProcessing {problem_name}...")
        problem_traces_dir.mkdir(parents=True, exist_ok=True)
        
        with open(prob_file, "r", encoding="utf-8") as f:
            problem_data = json.load(f)
            
        problem_text = problem_data["problem"]
        messages = [{"role": "user", "content": f"Solve step by step and put your final answer in \\boxed{{}}: {problem_text}"}]
        
        # Load model only when we actually have work to do
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

        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        prompt_length = inputs["input_ids"].shape[1]
        
        # Define a safe batch size for 8GB VRAM (2 is usually safe for 1.5B models with high context)
        SAFE_BATCH_SIZE = 2 
        
        print(f"Generating {cfg.N_TRACES_PER_PROBLEM} traces in chunks of {SAFE_BATCH_SIZE} "
              f"(prompt len: {prompt_length}, max tokens: {cfg.TRACE_MAX_TOKENS})...")
        
        torch.manual_seed(cfg.RANDOM_SEED)
        
        all_outputs = []
        
        # Process in smaller chunks to prevent VRAM overflow and system-RAM swapping
        for chunk_start in range(0, cfg.N_TRACES_PER_PROBLEM, SAFE_BATCH_SIZE):
            current_batch_size = min(SAFE_BATCH_SIZE, cfg.N_TRACES_PER_PROBLEM - chunk_start)
            print(f"  -> Generating traces {chunk_start + 1} to {chunk_start + current_batch_size}...")
            
            batched_input_ids = inputs["input_ids"].repeat(current_batch_size, 1)
            batched_attention_mask = torch.ones_like(batched_input_ids)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batched_input_ids,
                    attention_mask=batched_attention_mask,
                    max_new_tokens=cfg.TRACE_MAX_TOKENS,
                    do_sample=True,
                    temperature=cfg.TRACE_TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            all_outputs.extend(outputs)
            
            # Force clear the VRAM cache before the next chunk
            del batched_input_ids, batched_attention_mask, outputs
            torch.cuda.empty_cache()
            
        print("Saving traces...")
        for i in range(cfg.N_TRACES_PER_PROBLEM):
            sequence = all_outputs[i]
            # Strip padding if any (eos_token_id padding might happen in batching)
            # Find the first EOS token after the prompt and slice there
            eos_positions = (sequence[prompt_length:] == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                first_eos = eos_positions[0].item() + prompt_length
                sequence = sequence[:first_eos + 1]
                
            trace_data = {
                "problem_file": prob_file.name,
                "token_ids": sequence.cpu(),
                "prompt_length": prompt_length,
                "total_tokens": sequence.shape[0]
            }
            
            out_file = problem_traces_dir / f"trace_{i:03d}.pt"
            torch.save(trace_data, out_file)
            
        print(f"Saved {cfg.N_TRACES_PER_PROBLEM} traces to {problem_traces_dir}")

    print("\nStage 1 Complete.")

if __name__ == "__main__":
    main()