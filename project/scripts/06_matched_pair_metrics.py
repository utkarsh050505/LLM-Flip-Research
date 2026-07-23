"""
Step 6 (v2) — Matched pair metrics, with real budget forcing.

Two fixes from the previous version:

1. REAL "Wait" injection, not just token-banning. Banning </think>/eos
   only stops the model from CONCLUDING — it doesn't push it toward the
   reconsideration behavior that actually produces PCC. The s1 paper's
   budget forcing literally inserts the text "Wait, ..." when the model
   tries to stop early. We now detect that moment (the model's raw,
   unmasked argmax is </think> or eos) and inject a real Wait phrase
   instead of just masking the logit and letting it pick something else.

2. Manual per-token loop instead of a single long generate() call with
   output_scores=True. At this vocab size (151,936), retaining scores
   for ~3000 tokens x 6 branches is 10+ GB of logits alone — that's
   almost certainly what was making this crawl. Now we compute entropy/
   margin/L2/cosine from each step's raw tensors immediately, keep only
   the resulting floats, and let the tensors get garbage collected
   before the next step.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import copy
import csv
import re

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

# pyrefly: ignore [missing-import]
from configs.metrics import (
    token_entropy,
    top2_margin,
    jensen_shannon_divergence,
    l2_transition,
    cosine_progress,
)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

FCS_SEARCH_CEILING = 8000     # safety net for the (cheap) FCS search phase
N_BRANCHES = 1
BRANCH_TEMPERATURE = 1.1

# Start lower than the paper's 2500 — this is a 1.5B model, not a 32B one,
# and every extra forced token costs real wall-clock time. Raise once
# you've confirmed the mechanism actually produces PCC at this budget.
TARGET_BUDGET = 8000
MAX_CEILING = TARGET_BUDGET + 600   # hard stop even if nothing concludes
CHECK_EVERY = 4
CONCLUDE_PROB_THRESHOLD = 0.05  # trigger Wait if combined P(</think>) + P(eos) exceeds this

WAIT_PHRASE = " Wait,"
GROUND_TRUTH_ANSWER = "55"

PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "Alice chooses a set A of positive integers. Then Bob lists all finite nonempty" 
    "sets B of positive integers with the property that the maximum element of B" 
    "belongs to A. Bob's list has 2024 sets. Find the sum of the elements of A."
)


def get_layer_indices(model):
    n = model.config.num_hidden_layers
    return {"early": 1, "mid": n // 2, "late": n}


def answer_appears(text: str, answer: str) -> bool:
    pattern = (
        rf"(?i)(?:final\s+answer\s+is|the\s+answer\s+is|=\s*|\\boxed\{{)"
        rf"\s*{re.escape(answer)}\b"
    )
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
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        if self.check_fn(text):
            self.stopped_at = input_ids.shape[1]
            return True
        return False


def manual_branch_generate(
    model, tokenizer, prefix_ids, base_cache, layer_idx,
    target_budget, max_ceiling, temperature,
    wait_ids, think_close_id, eos_id, check_every=4,
):
    """One branch, one token at a time. Returns (generated_ids, metric_rows)."""
    device = prefix_ids.device
    cache = copy.deepcopy(base_cache)
    cur_input = prefix_ids[:, -1:]
    wait_injections = 0

    generated_ids = []
    rows = []
    prev_hidden = {name: None for name in layer_idx}
    prev_logits = None
    wait_queue = []
    think_closed = False

    for step in range(max_ceiling):
        with torch.no_grad():
            out = model(
                input_ids=cur_input,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
        cache = out.past_key_values
        logits = out.logits[0, -1, :].float()  # just this step's (vocab,) vector

        row = {
            "step": step,
            "entropy": token_entropy(logits),
            "top2_margin": top2_margin(logits),
            "jsd_vs_prev": (
                jensen_shannon_divergence(logits, prev_logits)
                if prev_logits is not None else ""
            ),
        }
        for name, layer in layer_idx.items():
            h = out.hidden_states[layer][0, -1, :]
            prev_h = prev_hidden[name]
            row[f"l2_{name}"] = l2_transition(h, prev_h) if prev_h is not None else ""
            row[f"cos_{name}"] = cosine_progress(h, prev_h) if prev_h is not None else ""
            prev_hidden[name] = h
        prev_logits = logits
        rows.append(row)

        if wait_queue:
            next_id = wait_queue.pop(0)
        else:
            under_budget = step < target_budget
            if under_budget:
                probs_raw = torch.softmax(logits, dim=-1)
                conclude_prob = (probs_raw[think_close_id] + probs_raw[eos_id]).item()
                if conclude_prob > CONCLUDE_PROB_THRESHOLD:
                    # the model has real intent to wrap up, even if it's not
                    # literally rank-1 yet -- inject Wait now, don't wait for
                    # a condition that masking makes nearly impossible to reach
                    wait_queue = list(wait_ids)
                    wait_injections += 1
                    next_id = wait_queue.pop(0)
                else:
                    masked = logits.clone()
                    masked[think_close_id] = -float("inf")
                    masked[eos_id] = -float("inf")
                    probs = torch.softmax(masked / temperature, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

        generated_ids.append(next_id)
        if next_id == think_close_id:
            think_closed = True
        cur_input = torch.tensor([[next_id]], device=device)

        del out, logits  # explicit, matches the doc's "drop raw tensors" advice

        if step % check_every == 0 or next_id == eos_id:
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
            if think_closed and re.search(r"\\boxed\{[^}]*\}", decoded):
                break
        if next_id == eos_id:
            break

    return generated_ids, rows, wait_injections


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    layer_idx = get_layer_indices(model)
    eos_id = tokenizer.eos_token_id
    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    if think_close_id is None or think_close_id == tokenizer.unk_token_id:
        raise RuntimeError(
            "Couldn't resolve </think> to a token id for this tokenizer — "
            "check the special token name for this model."
        )
    wait_ids = tokenizer(WAIT_PHRASE, add_special_tokens=False)["input_ids"]

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    fcs_criteria = StopOnPattern(
        tokenizer, prompt_len,
        check_fn=lambda text: answer_appears(text, GROUND_TRUTH_ANSWER),
        check_every=CHECK_EVERY, min_tokens=15,
    )
    print("Searching for FCS boundary...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=FCS_SEARCH_CEILING,
            do_sample=True,
            temperature=0.7,
            pad_token_id=eos_id,
            return_dict_in_generate=True,
            stopping_criteria=StoppingCriteriaList([fcs_criteria]),
        )
    sequences = outputs.sequences[0]
    if fcs_criteria.stopped_at is None:
        print("Answer never appeared — raise FCS_SEARCH_CEILING or check the answer format.")
        return

    prefix_len = fcs_criteria.stopped_at
    prefix_ids = sequences[:prefix_len].unsqueeze(0)
    print(f"FCS found after {prefix_len - prompt_len} tokens.")

    print("Building KV cache from the verified prefix...")
    with torch.no_grad():
        cache_out = model(
            input_ids=prefix_ids[:, :-1],
            attention_mask=torch.ones_like(prefix_ids[:, :-1]),
            use_cache=True,
        )
    base_cache = cache_out.past_key_values

    print(f"\nBranching {N_BRANCHES} times (target budget={TARGET_BUDGET} forced tokens)...\n")
    branches_data = []
    for i in range(N_BRANCHES):
        gen_ids, rows, wait_injections = manual_branch_generate(
            model, tokenizer, prefix_ids, base_cache, layer_idx,
            TARGET_BUDGET, MAX_CEILING, BRANCH_TEMPERATURE,
            wait_ids, think_close_id, eos_id, CHECK_EVERY,
        )
        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        with open(f"branch_{i}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(continuation)
        boxed = re.findall(r"\\boxed\{([^}]*)\}", continuation)
        raw_answer = boxed[-1].strip() if boxed else None
        clean_answer = re.sub(r"[^\d]", "", raw_answer) if raw_answer else None
        label = (
            "STABLE_CORRECT" if clean_answer == GROUND_TRUTH_ANSWER
            else "PCC" if raw_answer is not None
            else "NO_FINAL_ANSWER"
        )
        branches_data.append({"id": i, "label": label, "final_answer": raw_answer,
                               "continuation": continuation, "rows": rows})
        print(f"[Branch {i}, wait_injections={wait_injections}] label={label}  final_answer={raw_answer!r}  "
              f"tokens={len(gen_ids)}")

    n_stable = sum(1 for b in branches_data if b["label"] == "STABLE_CORRECT")
    n_pcc = sum(1 for b in branches_data if b["label"] == "PCC")
    print(f"\nSummary: {n_stable} Stable Correct, {n_pcc} PCC.")

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
        print("Still no matched pair this run — see notes below before re-running.")


if __name__ == "__main__":
    main()