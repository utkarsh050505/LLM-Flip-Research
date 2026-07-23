"""
Step 7 — Problem triage: find problems this model is actually unsure about.

No forcing, no branching yet. Just: sample each candidate problem N times
at a normal temperature, see how often the model gets it right. Problems
it nails every time, or misses every time, are bad candidates for
branching — there's no genuine internal conflict to isolate. Problems it
gets right ~40-60% of the time are exactly where you'd expect natural
hesitation and (occasionally) real collapse to show up, no forcing
required.

Pulls from MATH-500, the same benchmark used in the paper you uploaded,
so ground truth answers are sourced, not hand-typed — don't trust
hand-picked problems' answers without checking them yourself, math
mistakes are easy to make and expensive to build a pipeline on top of.

If "HuggingFaceH4/MATH-500" doesn't resolve on your machine, search
huggingface.co/datasets for "MATH-500" and swap in the exact repo id —
naming can drift and I can't verify it from here.
"""

import random
import re

# pyrefly: ignore [missing-import]
import torch
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
N_SAMPLES_PER_PROBLEM = 8
N_CANDIDATE_PROBLEMS = 15   # how many problems to screen this run
MAX_NEW_TOKENS = 700        # natural solve length ceiling, no forcing
TEMPERATURE = 0.8
RANDOM_SEED = 0


def extract_boxed_answer(text: str):
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    return matches[-1].strip() if matches else None


def normalize_answer(ans):
    if ans is None:
        return None
    return re.sub(r"[^\dA-Za-z\.\-]", "", str(ans))


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print("Sample record from the dataset (check these field names match "
          "what the code below expects):")
    print(ds[0])
    print()

    random.seed(RANDOM_SEED)
    candidates = random.sample(list(ds), N_CANDIDATE_PROBLEMS)

    results = []
    for idx, item in enumerate(candidates):
        problem_text = item["problem"]
        gt_answer = normalize_answer(item["answer"])

        messages = [{"role": "user", "content":
                     f"Solve step by step and put your final answer in \\boxed{{}}: {problem_text}"}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        correct = 0
        for _ in range(N_SAMPLES_PER_PROBLEM):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            pred = normalize_answer(extract_boxed_answer(text))
            if pred == gt_answer:
                correct += 1

        acc = correct / N_SAMPLES_PER_PROBLEM
        results.append({
            "acc": acc, "problem": problem_text, "answer": gt_answer,
            "level": item.get("level"), "subject": item.get("subject"),
        })
        print(f"[{idx+1}/{N_CANDIDATE_PROBLEMS}] acc={acc:.2f}  "
              f"level={item.get('level')}  subject={item.get('subject')}  "
              f"{problem_text[:70]}...")

    # closest to 50% accuracy = maximum genuine uncertainty
    results.sort(key=lambda r: abs(r["acc"] - 0.5))

    print("\n" + "=" * 60)
    print("Best candidates for branching (ranked by uncertainty):")
    print("=" * 60)
    for r in results[:5]:
        print(f"\nacc={r['acc']:.2f}  level={r['level']}  subject={r['subject']}")
        print(f"answer={r['answer']!r}")
        print(r["problem"])


if __name__ == "__main__":
    main()