"""
Find exact token offset of the second-guessing phrase in Sample 8 and align event window to it.
"""
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)

with open("results/proper_run2/r1_distill_qwen1_5b/gsm8k/seed_42/budget_prompt_Therefore__the_final_answer_is/difficulty_difficulty_utterance_granularity_15.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        if item.get("idx") == 8 and item.get("difficulty_idx") == 0:
            prompt = item.get("source_actual_query", "")
            gen = item.get("source_model_output", "")
            
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            gen_tokens = tokenizer.encode(gen, add_special_tokens=False)
            
            print("Prompt token len:", len(prompt_tokens))
            print("Gen token len:", len(gen_tokens))
            
            # Find the character index of first doubt phrase
            doubt_phrases = [
                "Wait, that's conflicting",
                "Wait, hold on",
                "Hmm. So, now I'm confused",
                "Wait, let me think again",
                "let me check if I missed something"
            ]
            for phrase in doubt_phrases:
                char_idx = gen.find(phrase)
                if char_idx != -1:
                    # Token index corresponding to this char
                    sub_text = gen[:char_idx]
                    token_idx = len(tokenizer.encode(sub_text, add_special_tokens=False))
                    print(f"Phrase '{phrase}' found at char {char_idx} -> Token #{token_idx} (out of {len(gen_tokens)})")
            break
