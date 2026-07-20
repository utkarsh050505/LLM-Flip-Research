"""
Step 5 — Same-prefix matched pairs: the core PCC mechanism.
 
1. Generate one full trace (as in Step 4) and find the first token index
   where the correct answer verifiably appears inside <think> — this is
   our (textual, oracle) proxy for the FCS boundary.
2. Rebuild the KV cache for that exact prefix via a teacher-forced
   forward pass (deterministic — no sampling replay needed).
3. Clone that cache N times and branch from it at high temperature.
4. Label each branch Stable Correct or PCC based on its final answer.
 
This script does NOT yet re-attach the Step 3 metrics to each branch —
that's the next step, once you've confirmed branching itself works and
actually produces divergent outcomes from an identical prefix.
"""
 
import copy
import re
 
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer
 
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FCS_SEARCH_MAX_TOKENS = 12000     # generate up to this many tokens looking for FCS
BRANCH_LEN = 16000               # how far each branch continues past the FCS point
N_BRANCHES = 10
BRANCH_TEMPERATURE = 0.9        # higher than Step 4 — we WANT divergence here
 
GROUND_TRUTH_ANSWER = "1"
 
PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "If it takes 1 shirt 1 hour to dry in the sun, how long do 5 shirts take to dry? "
    "The obvious answer is 1, but I need you to deeply question this. "
    "Aggressively play devil's advocate. Argue that due to local humidity saturation, "
    "wind blocking, and thermodynamic resource division, the drying time must actually scale linearly to 5 hours. "
    "Oscillate between the two possibilities, use hesitation phrases like 'Wait, reconsidering the physics...', "
    "and thoroughly convince yourself of the 5-hour theory before outputting your final conclusion."
)
 
 
def answer_appears(text: str, answer: str) -> bool:
    """Heuristic check: does `answer` appear in a context that looks like
    a stated result (after '=', inside \\boxed{}, or after 'is'), rather
    than as a stray number copied from the problem statement?
    This is an oracle proxy, same spirit as the 'optimal length' method
    in Thinking Past the Answer — good enough to get the mechanism
    working, worth replacing with a real math verifier later."""
    pattern = rf"(?:=\s*|\\boxed\{{|is\s+){re.escape(answer)}\b"
    return re.search(pattern, text) is not None
 
 
def find_fcs_index(tokenizer, sequences, prompt_len, think_end_pos, ground_truth):
    """Scan generated tokens one at a time, decode cumulatively, and
    return the index (relative to generation start) of the first token
    after which the answer verifiably appears. Restricted to inside the
    <think> block per the Step 4 finding."""
    search_end = think_end_pos if think_end_pos is not None else sequences.shape[0]
    for t in range(prompt_len, search_end):
        partial_text = tokenizer.decode(
            sequences[prompt_len:t + 1], skip_special_tokens=True
        )
        if answer_appears(partial_text, ground_truth):
            return t - prompt_len  # 0-indexed relative to generation start
    return None
 
 
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
 
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
 
    print("Generating exploratory trace to locate the FCS boundary...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=FCS_SEARCH_MAX_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    sequences = outputs.sequences[0]
 
    full_text = tokenizer.decode(sequences[prompt_len:], skip_special_tokens=True)
    think_end_char = full_text.find("</think>")
    think_end_pos = None
    if think_end_char != -1:
        # convert character offset to a token position by re-tokenizing
        # the substring up to </think> and counting tokens
        prefix_before_close = full_text[:think_end_char]
        think_end_pos = prompt_len + len(
            tokenizer(prefix_before_close, add_special_tokens=False)["input_ids"]
        )
 
    fcs_index = find_fcs_index(
        tokenizer, sequences, prompt_len, think_end_pos, GROUND_TRUTH_ANSWER
    )
 
    if fcs_index is None:
        print("Ground truth answer never verifiably appeared inside <think>. "
              "Try a longer FCS_SEARCH_MAX_TOKENS, or check GROUND_TRUTH_ANSWER "
              "matches this model's answer format.")
        return
 
    prefix_len = prompt_len + fcs_index + 1
    prefix_ids = sequences[:prefix_len].unsqueeze(0)  # (1, prefix_len)
    print(f"FCS found at generation-relative token {fcs_index} "
          f"(absolute position {prefix_len - 1}).")
    print("Prefix ends with:", repr(tokenizer.decode(sequences[prefix_len - 10:prefix_len])))
 
    # --- Build the cache via a teacher-forced forward pass ---
    # Cache covers positions [0, prefix_len - 2]; the last prefix token
    # is fed as the first input to generate(), so it gets processed
    # (added to the cache) before any new sampling happens.
    print("\nBuilding KV cache from the verified prefix (teacher-forced, deterministic)...")
    with torch.no_grad():
        cache_out = model(
            input_ids=prefix_ids[:, :-1],
            attention_mask=torch.ones_like(prefix_ids[:, :-1]),
            use_cache=True,
        )
    base_cache = cache_out.past_key_values
 
    # --- Branch N times from the identical cloned cache ---
    print(f"\nBranching {N_BRANCHES} times at temperature={BRANCH_TEMPERATURE}...\n")
    results = []
    for i in range(N_BRANCHES):
        branch_cache = copy.deepcopy(base_cache)
        with torch.no_grad():
            branch_out = model.generate(
                input_ids=prefix_ids,
                past_key_values=branch_cache,
                attention_mask=torch.ones_like(prefix_ids),
                max_new_tokens=BRANCH_LEN,
                do_sample=True,
                temperature=BRANCH_TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        continuation = tokenizer.decode(
            branch_out.sequences[0][prefix_ids.shape[1]:], skip_special_tokens=True
        )
        # crude final-answer extraction: last \boxed{...} if present
        boxed = re.findall(r"\\boxed\{([^}]*)\}", continuation)
        final_answer = boxed[-1].strip() if boxed else None
        label = (
            "STABLE_CORRECT" if final_answer == GROUND_TRUTH_ANSWER
            else "PCC" if final_answer is not None
            else "NO_FINAL_ANSWER"  # ran out of tokens before boxing an answer
        )
        results.append((label, final_answer, continuation))
        print(f"[Branch {i}] label={label}  final_answer={final_answer!r}")
 
    n_stable = sum(1 for r in results if r[0] == "STABLE_CORRECT")
    n_pcc = sum(1 for r in results if r[0] == "PCC")
    n_none = sum(1 for r in results if r[0] == "NO_FINAL_ANSWER")
    print(f"\nSummary: {n_stable} Stable Correct, {n_pcc} PCC, "
          f"{n_none} ran out of budget before boxing an answer.")
 
    if n_pcc == 0:
        print("\nNo PCC branches on this run — not a bug. This problem may be "
              "too easy / the model too confident at this temperature. Try a "
              "harder problem, a higher BRANCH_TEMPERATURE, or more branches.")
    else:
        print("\nGot at least one PCC branch. Print/inspect results[i][2] for "
              "any branch labeled PCC to read exactly where it went wrong — "
              "that continuation is what Step 6 will attach metrics to.")
 
 
if __name__ == "__main__":
    main()