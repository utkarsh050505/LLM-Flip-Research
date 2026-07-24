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

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-llama-8B"

FCS_SEARCH_CEILING = 8000     # safety net for the (cheap) FCS search phase
FCS_MAX_ATTEMPTS = 6          # hard problems often don't land on the answer
                               # inside <think> on the first try -- retry
N_BRANCHES = 5
BRANCH_TEMPERATURE = 1.1

# Budget is now set ADAPTIVELY once we know how long this problem's
# natural derivation actually was (see main()) -- a fixed budget forces
# the same amount of reasoning onto every problem regardless of
# difficulty, which is exactly the "uniform compute allocation is
# suboptimal" mistake the paper argues against. An easy problem forced
# to 5000 tokens doesn't overthink, it degrades into garbage.
BUDGET_MULTIPLIER = 3     # force up to ~3x the natural derivation length
MIN_TARGET_BUDGET = 400   # floor, so trivially-fast problems still get SOME forcing
MAX_TARGET_BUDGET = 4000  # ceiling, so a slow FCS search doesn't run away
CHECK_EVERY = 4
CONCLUDE_PROB_THRESHOLD = 0.05  # trigger Wait if combined P(</think>) + P(eos) exceeds this
MAX_REPEATED_BOXED = 3   # stop a branch once it repeats the same boxed
                          # answer this many times -- that's degeneration
                          # under forcing, not reconsideration, and forcing
                          # further just produces more garbage, not signal

WAIT_PHRASE_VARIANTS = [
    " Wait,",
    " Wait, actually, let me try solving this a completely different way to check:",
    " Wait, hold on — let me reconsider whether I've even set up the problem correctly:",
    " Wait, let me question my core assumption here and see if another interpretation fits:",
]
GROUND_TRUTH_ANSWER = "204"

PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "Every morning Aya goes for a 9-kilometer-long walk and stops at a coffee shop " 
    "afterwards. When she walks at a constant speed of s kilometers per hour, the " 
    "walk takes her 4 hours, including t minutes spent in the coffee shop. When she "
    "walks s+2 kilometers per hour, the walk takes her 2 hours and 24 minutes, "
    "including t minutes spent in the coffee shop. Suppose Aya walks at s+1/2 "
    "kilometers per hour. Find the number of minutes the walk takes her, including "
    "the t minutes spent in the coffee shop."
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


class FCSStoppingCriteria(StoppingCriteria):
    """Stops as soon as the answer verifiably appears INSIDE <think> --
    this is the restriction that was in the Step 5 fixed version and got
    dropped in the Step 6 rewrite. Also fails fast (stops, unfound) the
    moment </think> closes without a match, instead of burning the full
    ceiling on a trace that's already left the reasoning phase -- forcing
    generation after that point just gets you reformatting, not real
    reasoning, which is exactly what produced the "two boxed answers"
    loop."""

    def __init__(self, tokenizer, prompt_len, answer, check_every=4, min_tokens=15):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.answer = answer
        self.check_every = check_every
        self.min_tokens = min_tokens
        self.stopped_at = None
        self.found = False

    def __call__(self, input_ids, scores, **kwargs):
        new_len = input_ids.shape[1] - self.prompt_len
        if new_len < self.min_tokens or new_len % self.check_every != 0:
            return False
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        think_pos = text.find("</think>")
        search_region = text if think_pos == -1 else text[:think_pos]
        if answer_appears(search_region, self.answer):
            self.stopped_at = input_ids.shape[1]
            self.found = True
            return True
        if think_pos != -1:
            self.stopped_at = None
            self.found = False
            return True
        return False


def manual_branch_generate(
    model, tokenizer, prefix_ids, base_cache, layer_idx,
    target_budget, max_ceiling, temperature,
    wait_id_variants, think_close_id, eos_id, check_every=4,
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
    degenerate = False

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
            # only force while still inside <think> -- once it's closed,
            # further forcing just produces reformatting loops, not real
            # reasoning (this is the fix for the "two boxed answers" bug)
            under_budget = step < target_budget and not think_closed
            if under_budget:
                probs_raw = torch.softmax(logits, dim=-1)
                conclude_prob = (probs_raw[think_close_id] + probs_raw[eos_id]).item()
                if conclude_prob > CONCLUDE_PROB_THRESHOLD:
                    # the model has real intent to wrap up, even if it's not
                    # literally rank-1 yet -- inject Wait now, don't wait for
                    # a condition that masking makes nearly impossible to reach.
                    # Escalate the phrase with repeated injections -- bare
                    # "Wait," tends to trigger re-verification of the SAME
                    # derivation, not exploration of a different one.
                    variant_idx = min(wait_injections, len(wait_id_variants) - 1)
                    wait_queue = list(wait_id_variants[variant_idx])
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
            # degeneration guard: forcing well past the natural derivation
            # doesn't always produce a garbled string of DIFFERENT wrong
            # answers -- it can also just make the model loop on writing
            # "Final Answer" blocks that repeat the SAME boxed value over
            # and over inside <think>, never actually closing it. That's
            # not reconsideration, it's an artifact of the forced budget
            # being too large for this problem -- stop instead of
            # continuing to burn tokens on garbage.
            boxed_so_far = re.findall(r"\\boxed\{([^}]*)\}", decoded)
            if len(boxed_so_far) >= MAX_REPEATED_BOXED:
                recent = boxed_so_far[-MAX_REPEATED_BOXED:]
                if len(set(recent)) == 1:
                    degenerate = True
                    break
        if next_id == eos_id:
            break

    return generated_ids, rows, wait_injections, degenerate


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"Loading {MODEL_NAME}...")
    print("NOTE: 8B models require ~16GB of free RAM/VRAM. If the script crashes silently here, your system ran out of memory.")
    try:
        # pyrefly: ignore [missing-import]
        from transformers import BitsAndBytesConfig
        # Try loading in 4-bit to save massive amounts of memory (requires bitsandbytes)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
        print("Successfully loaded in 4-bit mode.")
    except ImportError:
        print("bitsandbytes not found, falling back to standard bfloat16 loading...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
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
    wait_id_variants = [
        tokenizer(phrase, add_special_tokens=False)["input_ids"]
        for phrase in WAIT_PHRASE_VARIANTS
    ]

    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    print("Searching for FCS boundary (inside <think> only)...")
    prefix_ids = None
    for attempt in range(1, FCS_MAX_ATTEMPTS + 1):
        fcs_criteria = FCSStoppingCriteria(
            tokenizer, prompt_len, GROUND_TRUTH_ANSWER,
            check_every=CHECK_EVERY, min_tokens=15,
        )
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
        if fcs_criteria.found:
            sequences = outputs.sequences[0]
            prefix_len = fcs_criteria.stopped_at
            prefix_ids = sequences[:prefix_len].unsqueeze(0)
            print(f"FCS found on attempt {attempt} after {prefix_len - prompt_len} tokens.")
            break
        print(f"Attempt {attempt}/{FCS_MAX_ATTEMPTS}: model concluded without ever "
              f"deriving {GROUND_TRUTH_ANSWER!r} inside <think> -- retrying with a fresh sample.")

    if prefix_ids is None:
        print(f"\nNo usable FCS prefix found in {FCS_MAX_ATTEMPTS} attempts. This is itself "
              f"informative -- if this keeps happening, the model may rarely/never derive "
              f"the correct answer on this problem, which makes it a bad branching candidate "
              f"(check its pass rate with the Step 7 triage script).")
        return

    natural_tokens = prefix_len - prompt_len
    target_budget = int(min(max(natural_tokens * BUDGET_MULTIPLIER, MIN_TARGET_BUDGET),
                             MAX_TARGET_BUDGET))
    max_ceiling = target_budget + 1000
    print(f"Natural derivation took {natural_tokens} tokens -> "
          f"adaptive target_budget={target_budget} (was a fixed 5000 before)")

    print("Building KV cache from the verified prefix...")
    with torch.no_grad():
        cache_out = model(
            input_ids=prefix_ids[:, :-1],
            attention_mask=torch.ones_like(prefix_ids[:, :-1]),
            use_cache=True,
        )
    base_cache = cache_out.past_key_values

    print(f"\nBranching {N_BRANCHES} times (target budget={target_budget} forced tokens)...\n")
    branches_data = []
    for i in range(N_BRANCHES):
        gen_ids, rows, wait_injections, degenerate = manual_branch_generate(
            model, tokenizer, prefix_ids, base_cache, layer_idx,
            target_budget, max_ceiling, BRANCH_TEMPERATURE,
            wait_id_variants, think_close_id, eos_id, CHECK_EVERY,
        )
        full_ids = prefix_ids[0].tolist() + gen_ids
        transcript = tokenizer.decode(full_ids, skip_special_tokens=False)
        with open(f"branch_{i}_transcript.txt", "w", encoding="utf-8") as f:
            f.write(transcript)

        continuation = tokenizer.decode(gen_ids, skip_special_tokens=True)
        boxed = re.findall(r"\\boxed\{([^}]*)\}", continuation)
        raw_answer = boxed[-1].strip() if boxed else None
        clean_answer = re.sub(r"[^\d]", "", raw_answer) if raw_answer else None
        distinct_answers = sorted(set(re.sub(r"[^\d]", "", b) for b in boxed))
        label = (
            "DEGENERATE" if degenerate
            else "STABLE_CORRECT" if clean_answer == GROUND_TRUTH_ANSWER
            else "PCC" if raw_answer is not None
            else "NO_FINAL_ANSWER"
        )
        branches_data.append({"id": i, "label": label, "final_answer": raw_answer,
                               "continuation": continuation, "rows": rows})
        print(f"[Branch {i}, wait_injections={wait_injections}] label={label}  "
              f"final_answer={raw_answer!r}  distinct_boxed_answers={distinct_answers}  "
              f"tokens={len(gen_ids)}")

    n_stable = sum(1 for b in branches_data if b["label"] == "STABLE_CORRECT")
    n_pcc = sum(1 for b in branches_data if b["label"] == "PCC")
    n_degenerate = sum(1 for b in branches_data if b["label"] == "DEGENERATE")
    print(f"\nSummary: {n_stable} Stable Correct, {n_pcc} PCC, {n_degenerate} Degenerate.")

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