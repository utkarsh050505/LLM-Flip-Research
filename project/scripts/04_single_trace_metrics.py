"""
Step 4 — Apply the Step 3 metrics to one real generation trace.

Still no branching, no FCS detection, no causal claims — just: generate
one trace, compute entropy/margin/JSD/L2/cosine at every step across
three layers, and write it to a CSV so you can actually look at the
numbers before building anything more complicated on top of them.

What to look for once you open the CSV:
  - Do entropy/margin spikes line up with hesitation words in the
    decoded text ("wait", "let me reconsider", "hmm")?
  - Are the early/mid/late layers noisy in the same way, or does one
    layer look much spikier than the others? (This matters for which
    layer(s) you pick to instrument densely in Step 5.)
  - What's a "normal" L2 transition magnitude on this model, so you
    have a baseline before you go looking for anomalies?
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import csv
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.metrics import (
    token_entropy,
    top2_margin,
    jensen_shannon_divergence,
    l2_transition,
    cosine_progress,
)

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MAX_NEW_TOKENS = 400
OUTPUT_CSV = "trace_metrics.csv"

# A problem with more room for multi-step reasoning than "17*24-15".
# Swap this out for a real AIME/MATH problem once you're past inspection.
PROMPT = (
    "Solve step by step and put your final answer in \\boxed{}: "
    "A train travels 120 miles in the same time a car travels 90 miles. "
    "If the train's speed is 15 mph more than the car's speed, "
    "what is the car's speed in mph?"
)


def get_layer_indices(model):
    """Early / mid / late layer indices, skipping the raw embedding
    layer (index 0) since it hasn't done any computation yet."""
    n = model.config.num_hidden_layers  # e.g. 28
    return {"early": 1, "mid": n // 2, "late": n}


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

    print("Generating (this is the slow part — hidden_states for every "
          "layer at every step is a lot of tensors)...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )

    num_steps = len(outputs.scores)
    prompt_len = inputs["input_ids"].shape[1]

    # helper: get this step's hidden vector for a given layer, handling
    # the step-0 (prefill, full prompt) vs step-1+ (single token) split
    # from Step 2.
    def hidden_vec(step, layer):
        h = outputs.hidden_states[step][layer][0]  # (seq_len, hidden_dim)
        return h[-1]  # last position = the token this step produced

    rows = []
    prev_hidden = {name: None for name in layer_idx}
    prev_logits = None

    for t in range(num_steps):
        token_id = outputs.sequences[0][prompt_len + t].item()
        token_text = tokenizer.decode([token_id])
        logits = outputs.scores[t][0]

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

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    full_text = tokenizer.decode(
        outputs.sequences[0][prompt_len:], skip_special_tokens=True
    )
    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print("\n----- FULL TRACE -----\n")
    print(full_text)


if __name__ == "__main__":
    main()