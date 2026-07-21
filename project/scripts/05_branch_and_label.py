"""
Step 5 (fixed) — Same-prefix matched pairs, with real early stopping.

Fixes from the previous run:
  1. FCS search now halts the MOMENT the answer verifiably appears,
     instead of generating a full 12k-token ceiling and scanning after
     the fact. Same idea for branches: stop once a \\boxed{} closes.
  2. input_ids reverted to prefix_ids[:, -1:] — feeding the full prefix
     back in redoes prefill work the cache already has.
  3. Prompt reverted to a neutral problem. The adversarial "argue
     yourself into the wrong answer" prompt is fine as a one-off
     plumbing stress test (and it proved branches CAN diverge), but it
     can never be real PCC data — the model was instructed to produce
     the failure you'd be claiming to discover.
  4. FCS matching tightened to require a concluding marker (boxed/=/
     "answer is"), and now REQUIRES a minimum number of reasoning
     tokens before it's eligible — cheap insurance against matching an
     early echo of anything answer-shaped in the prompt.
"""

import copy
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

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
FCS_SEARCH_CEILING = 12000     # safety net only — stopping criteria should trigger well before this
BRANCH_CEILING = 16000         # safety net only — stops as soon as \boxed{} closes
N_BRANCHES = 6
BRANCH_TEMPERATURE = 0.9
CHECK_EVERY = 4              # decode+check every N tokens, not every single one
MIN_TOKENS_BEFORE_FCS = 15   # ignore matches before this many tokens have been generated

GROUND_TRUTH_ANSWER = "45"

PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "A train travels 120 miles in the same time a car travels 90 miles. "
    "If the train's speed is 15 mph more than the car's speed, "
    "what is the car's speed in mph?"
)


def answer_appears(text: str, answer: str) -> bool:
    """Requires a concluding marker (boxed / '=' / 'answer is') right
    before the number — tighter than a bare 'is X' check, which is what
    got fooled by a prompt that stated the answer up front last time."""
    pattern = rf"(?:=\s*|\\boxed\{{|answer\s+is\s+){re.escape(answer)}\b"
    return re.search(pattern, text) is not None


class StopOnPattern(StoppingCriteria):
    """Stops generation as soon as `check_fn(decoded_new_text)` is True.
    Only decodes every `check_every` tokens to keep overhead down."""

    def __init__(self, tokenizer, prompt_len, check_fn, check_every=4, min_tokens=0):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.check_fn = check_fn
        self.check_every = check_every
        self.min_tokens = min_tokens
        self.stopped_at = None  # absolute sequence length when stopped

    def __call__(self, input_ids, scores, **kwargs):
        new_len = input_ids.shape[1] - self.prompt_len
        if new_len < self.min_tokens or new_len % self.check_every != 0:
            return False
        text = self.tokenizer.decode(
            input_ids[0, self.prompt_len:], skip_special_tokens=True
        )
        if self.check_fn(text):
            self.stopped_at = input_ids.shape[1]
            return True
        return False


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

    fcs_criteria = StopOnPattern(
        tokenizer, prompt_len,
        check_fn=lambda text: answer_appears(text, GROUND_TRUTH_ANSWER),
        check_every=CHECK_EVERY,
        min_tokens=MIN_TOKENS_BEFORE_FCS,
    )

    print("Searching for FCS boundary (stops the moment the answer appears)...")
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
        print(f"Answer never appeared within {FCS_SEARCH_CEILING} tokens — "
              "raise the ceiling or check GROUND_TRUTH_ANSWER's format.")
        return

    prefix_len = fcs_criteria.stopped_at
    prefix_ids = sequences[:prefix_len].unsqueeze(0)
    fcs_index = prefix_len - prompt_len - 1
    print(f"FCS found at generation-relative token {fcs_index} "
          f"(after {prefix_len - prompt_len} tokens generated).")
    print("Prefix ends with:", repr(tokenizer.decode(sequences[prefix_len - 10:prefix_len])))

    print("\nBuilding KV cache from the verified prefix (teacher-forced, deterministic)...")
    with torch.no_grad():
        cache_out = model(
            input_ids=prefix_ids[:, :-1],
            attention_mask=torch.ones_like(prefix_ids[:, :-1]),
            use_cache=True,
        )
    base_cache = cache_out.past_key_values

    print(f"\nBranching {N_BRANCHES} times at temperature={BRANCH_TEMPERATURE}...\n")
    results = []
    for i in range(N_BRANCHES):
        branch_cache = copy.deepcopy(base_cache)
        branch_criteria = StopOnPattern(
            tokenizer, prefix_len,
            check_fn=lambda text: re.search(r"\\boxed\{[^}]*\}", text) is not None,
            check_every=CHECK_EVERY,
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
                stopping_criteria=StoppingCriteriaList([branch_criteria]),
            )
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
        results.append((label, final_answer, continuation))
        print(f"[Branch {i}] label={label}  final_answer={final_answer!r}  "
              f"tokens_used={branch_out.sequences.shape[1] - prefix_ids.shape[1]}")

    n_stable = sum(1 for r in results if r[0] == "STABLE_CORRECT")
    n_pcc = sum(1 for r in results if r[0] == "PCC")
    n_none = sum(1 for r in results if r[0] == "NO_FINAL_ANSWER")
    print(f"\nSummary: {n_stable} Stable Correct, {n_pcc} PCC, {n_none} no final answer.")


if __name__ == "__main__":
    main()