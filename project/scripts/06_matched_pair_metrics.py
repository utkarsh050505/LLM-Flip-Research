"""
Step 6 — Matched Pair Metrics (Stable Correct vs PCC)

Combines the metric gathering of Step 4 with the cache-cloned branching
of Step 5. Runs until it finds a set of branches containing at least one
Stable Correct branch and at least one PCC branch, then saves their
step-by-step metrics side-by-side to CSVs for comparison.
"""

import copy
import csv
import re
import sys
from pathlib import Path

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.metrics import (
    token_entropy,
    top2_margin,
    jensen_shannon_divergence,
    l2_transition,
    cosine_progress,
)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FCS_SEARCH_CEILING = 16000
BRANCH_CEILING = 16000
N_BRANCHES = 6
BRANCH_TEMPERATURE = 1.1
TARGET_BUDGET = 2500
BRANCH_CEILING = TARGET_BUDGET + 500  # Budget + room to output the answer
MIN_TOKENS_BEFORE_FCS = 15
CHECK_EVERY = 4

GROUND_TRUTH_ANSWER = "60"
PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "Let $x, y, z,$ and $w$ be real numbers greater than $1$ such that "
    "$\\log_x w = 24$, $\\log_y w = 40$, and $\\log_{xyz} w = 12$. Find $\\log_z w$."
)

def get_layer_indices(model):
    n = model.config.num_hidden_layers
    return {"early": 1, "mid": n // 2, "late": n}

def answer_appears(text: str) -> bool:
    # A smart regex to catch tentative answers without grabbing intermediate calculations
    pattern = r"(?i)(?:final\s+answer\s+is|the\s+answer\s+is|so\s+the\s+answer\s+is|value\s+of\s+\\log_z\s+w\s*is|\\log_z\s*w\s*=)\s*[-+]?\d+|\\boxed\{[-+]?\d+"
    return re.search(pattern, text) is not None

class StopOnPattern(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len, check_fn, check_every=4, min_tokens=0):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_fn = check_fn
        self.check_every = check_every
        self.min_tokens = min_tokens
        self.stopped_at = None

    def __call__(self, input_ids, scores, **kwargs):
        new_len = input_ids.shape[1] - self.prompt_len
        if new_len < self.min_tokens or new_len % self.check_every != 0:
            return False
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_len:], skip_special_tokens=True
        )
        if self.check_fn(text, input_ids[0, self.prompt_len:]):
            self.stopped_at = input_ids.shape[1]
            return True
        return False

# pyrefly: ignore [missing-import]
from transformers import LogitsProcessor, LogitsProcessorList

class BudgetForcingLogitsProcessor(LogitsProcessor):
    """
    Implements the core 'Budget Forcing' methodology from 'When More Thinking Hurts'.
    If the model tries to end its thought process before TARGET_BUDGET is reached,
    we ban the </think> token (ID 151649) and EOS token, forcing it to keep reasoning.
    """
    def __init__(self, target_budget: int, prompt_length: int, eos_token_id: int):
        self.target_budget = target_budget
        self.prompt_length = prompt_length
        self.banned_tokens = [151649, eos_token_id]
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        generated_len = input_ids.shape[1] - self.prompt_length
        if generated_len < self.target_budget:
            for t in self.banned_tokens:
                scores[:, t] = -float("inf")
        elif generated_len >= self.target_budget + 400:
            # Force exit from think block before we hit ceiling, if not already exited
            if 151649 not in input_ids[0, self.prompt_length:]:
                scores[:] = -float("inf")
                scores[:, 151649] = 0.0
        return scores

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    layer_idx = get_layer_indices(model)
    print(f"Instrumenting layers: {layer_idx}")

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    fcs_criteria = StopOnPattern(
        tokenizer, prompt_len,
        check_fn=lambda text, ids: answer_appears(text),
        check_every=CHECK_EVERY,
        min_tokens=MIN_TOKENS_BEFORE_FCS,
    )

    print("Searching for FCS boundary...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=FCS_SEARCH_CEILING,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([fcs_criteria]),
        )
    sequences = outputs.sequences[0]

    if fcs_criteria.stopped_at is None:
        print("Answer never appeared.")
        return

    prefix_len = fcs_criteria.stopped_at
    prefix_ids = sequences[:prefix_len].unsqueeze(0)
    print(f"FCS found after {prefix_len - prompt_len} tokens generated.")

    print("\nBuilding KV cache...")
    with torch.no_grad():
        cache_out = model(
            input_ids=prefix_ids[:, :-1],
            attention_mask=torch.ones_like(prefix_ids[:, :-1]),
            use_cache=True,
        )
    base_cache = cache_out.past_key_values

    print(f"\nBranching {N_BRANCHES} times...")
    branches_data = []

    for i in range(N_BRANCHES):
        branch_cache = copy.deepcopy(base_cache)
        branch_criteria = StopOnPattern(
            tokenizer, prefix_len,
            check_fn=lambda text, ids: re.search(r"\\boxed\{[^}]*\}", text) is not None and 151649 in ids,
            check_every=CHECK_EVERY,
        )
        budget_processor = BudgetForcingLogitsProcessor(
            target_budget=TARGET_BUDGET, 
            prompt_length=prefix_ids.shape[1], 
            eos_token_id=tokenizer.eos_token_id
        )
        
        with torch.no_grad():
            branch_out = model.generate(
                input_ids=prefix_ids,
                past_key_values=branch_cache,
                attention_mask=torch.ones_like(prefix_ids),
                max_new_tokens=BRANCH_CEILING,
                do_sample=True,
                temperature=BRANCH_TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                logits_processor=LogitsProcessorList([budget_processor]),
                stopping_criteria=StoppingCriteriaList([branch_criteria]),
            )
        
        num_steps = len(branch_out.scores)
        continuation = tokenizer.decode(
            branch_out.sequences[0][prefix_ids.shape[1]:], skip_special_tokens=True
        )
        boxed = re.findall(r"\\boxed\{([^}]*)\}", continuation)
        final_answer = boxed[-1].strip() if boxed else None
        # clean up final answer (remove \text{mph}, spacing, etc.)
        clean_answer = re.sub(r"\\text\{[^}]*\}", "", final_answer) if final_answer else None
        clean_answer = re.sub(r"[^\d]", "", clean_answer) if clean_answer else None
        
        label = (
            "STABLE_CORRECT" if clean_answer == GROUND_TRUTH_ANSWER
            else "PCC" if final_answer is not None
            else "NO_FINAL_ANSWER"
        )
        
        # Collect metrics
        def hidden_vec(step, layer):
            h = branch_out.hidden_states[step][layer][0]
            return h[-1]

        rows = []
        prev_hidden = {name: None for name in layer_idx}
        prev_logits = None

        for t in range(num_steps):
            token_id = branch_out.sequences[0][prefix_ids.shape[1] + t].item()
            token_text = tokenizer.decode([token_id])
            logits = branch_out.scores[t][0]

            row = {
                "step": t,
                "token": repr(token_text),
                "entropy": token_entropy(logits),
                "top2_margin": top2_margin(logits),
                "jsd_vs_prev": (
                    jensen_shannon_divergence(logits, prev_logits)
                    if prev_logits is not None else ""
                ),
            }

            for name, layer in layer_idx.items():
                h = hidden_vec(t, layer)
                prev_h = prev_hidden[name]
                row[f"l2_{name}"] = l2_transition(h, prev_h) if prev_h is not None else ""
                row[f"cos_{name}"] = cosine_progress(h, prev_h) if prev_h is not None else ""
                prev_hidden[name] = h

            prev_logits = logits
            rows.append(row)
            
        branches_data.append({
            "id": i,
            "label": label,
            "final_answer": final_answer,
            "continuation": continuation,
            "rows": rows
        })
        print(f"[Branch {i}] label={label}  final_answer={final_answer!r}  steps={num_steps}")

    n_stable = sum(1 for b in branches_data if b["label"] == "STABLE_CORRECT")
    n_pcc = sum(1 for b in branches_data if b["label"] == "PCC")
    
    print(f"\nSummary: {n_stable} Stable Correct, {n_pcc} PCC.")
    
    # Save a pair to CSV
    stable_branch = next((b for b in branches_data if b["label"] == "STABLE_CORRECT"), None)
    pcc_branch = next((b for b in branches_data if b["label"] == "PCC"), None)
    
    if stable_branch and pcc_branch:
        for b, name in [(stable_branch, "stable_correct"), (pcc_branch, "pcc")]:
            filename = f"branch_{b['id']}_{name}_metrics.csv"
            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(b["rows"][0].keys()))
                writer.writeheader()
                writer.writerows(b["rows"])
            print(f"Saved {name} metrics to {filename}")
    else:
        print("Could not find both a STABLE_CORRECT and a PCC branch in this run. Re-run to try again.")

if __name__ == "__main__":
    main()
