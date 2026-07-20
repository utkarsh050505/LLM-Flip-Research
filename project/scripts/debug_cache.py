# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_cache_branching():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    
    prompt = "Hello, how are you doing today?"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("Original input length:", inputs.input_ids.shape[1])
    
    # 1. Prefill
    with torch.no_grad():
        out = model(
            input_ids=inputs.input_ids[:, :-1],
            attention_mask=inputs.attention_mask[:, :-1],
            use_cache=True
        )
    past = out.past_key_values
    
    # 2. Try to generate from cache
    try:
        # Method A: input_ids = last token, past_key_values = past, full attention_mask
        gen_out = model.generate(
            input_ids=inputs.input_ids[:, -1:],
            attention_mask=inputs.attention_mask,
            past_key_values=past,
            max_new_tokens=5,
            do_sample=False
        )
        print("Method A worked!")
    except Exception as e:
        print("Method A failed:", type(e).__name__, e)
        
    try:
        # Method B: input_ids = full, past_key_values = past, full attention_mask
        gen_out = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            past_key_values=past,
            max_new_tokens=5,
            do_sample=False
        )
        print("Method B worked!")
    except Exception as e:
        print("Method B failed:", type(e).__name__, e)

if __name__ == "__main__":
    test_cache_branching()
