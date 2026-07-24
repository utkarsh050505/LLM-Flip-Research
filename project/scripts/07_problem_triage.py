import json
import random
import re
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
from datasets import load_dataset
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-llama-8B"
N_SAMPLES_PER_PROBLEM = 8
N_CANDIDATE_PROBLEMS = 10
MAX_NEW_TOKENS = 7000
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    
    try:
        # pyrefly: ignore [missing-import]
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            quantization_config=bnb_config, 
            device_map="auto",
            attn_implementation="sdpa"
        )
        print("Loaded model in 4-bit mode with PyTorch SDPA")
    except (ValueError, ImportError):
        print("Failed to load in 4-bit/SDPA, falling back to standard bfloat16 loading...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("Loaded model with standard attention (bfloat16)")
        
    model.eval()

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    problems_dir = Path(__file__).parent.parent / "problems"
    problems_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving surviving problems to {problems_dir}")

    random.seed(RANDOM_SEED)
    candidates = random.sample(list(ds), N_CANDIDATE_PROBLEMS)

    results = []
    for idx, item in enumerate(candidates):
        problem_text = item["problem"]
        gt_answer = normalize_answer(item["answer"])

        messages = [{"role": "user", "content": f"Solve step by step and put your final answer in \\boxed{{}}: {problem_text}"}]
        
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        # 1. BATCHING: Duplicate the input tensor N times to generate in parallel
        batched_input_ids = inputs["input_ids"].repeat(N_SAMPLES_PER_PROBLEM, 1)
        batched_attention_mask = torch.ones_like(batched_input_ids)
        
        correct = 0
        
        # 2. GENERATE ALL AT ONCE
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batched_input_ids,
                attention_mask=batched_attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
            )
            
        # 3. DECODE AND EVALUATE BATCH
        prompt_length = inputs["input_ids"].shape[1]
        for out in outputs:
            text = tokenizer.decode(out[prompt_length:], skip_special_tokens=True)
            pred = normalize_answer(extract_boxed_answer(text))
            if pred == gt_answer:
                correct += 1

        acc = correct / N_SAMPLES_PER_PROBLEM
        
        problem_record = {
            "problem": problem_text,
            "answer": gt_answer,
            "subject": item.get("subject"),
            "level": item.get("level"),
            "pass_rate": acc
        }
        
        results.append(problem_record)
        print(f"[{idx+1}/{N_CANDIDATE_PROBLEMS}] acc={acc:.2f}  "
              f"level={item.get('level')}  subject={item.get('subject')}  "
              f"{problem_text[:70]}...")

        if 0.3 <= acc <= 0.7:
            safe_subject = str(item.get("subject") or "unknown").replace(" ", "_").lower()
            file_name = f"problem_{idx:03d}_{safe_subject}.json"
            file_path = problems_dir / file_name
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(problem_record, f, indent=4)

    surviving_problems = [r for r in results if 0.3 <= r["pass_rate"] <= 0.7]
    surviving_problems.sort(key=lambda r: abs(r["pass_rate"] - 0.5))

    print("\n" + "=" * 60)
    print(f"Found {len(surviving_problems)} surviving problems (pass rate 0.3-0.7).")
    print("=" * 60)
    for r in surviving_problems[:5]:
        print(f"\npass_rate={r['pass_rate']:.2f}  level={r['level']}  subject={r['subject']}")
        print(f"answer={r['answer']!r}")
        print(r["problem"])


if __name__ == "__main__":
    main()