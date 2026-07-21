# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
input_text = "The quick brown fox jumps over the lazy"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
prefix_ids = inputs.input_ids
print("prefix_ids shape:", prefix_ids.shape)
with torch.no_grad():
    cache_out = model(input_ids=prefix_ids[:, :-1], use_cache=True)
past_key_values = cache_out.past_key_values
print("cache length:", past_key_values[0][0].shape[2])

print("Trying generation with input_ids=prefix_ids[:, -1:]...")
try:
    out = model.generate(
        input_ids=prefix_ids[:, -1:],
        past_key_values=past_key_values,
        max_new_tokens=5
    )
    print("Success with -1:", out.shape)
except Exception as e:
    print("Error with -1:", e)

print("Trying generation with input_ids=prefix_ids...")
try:
    out = model.generate(
        input_ids=prefix_ids,
        past_key_values=past_key_values,
        max_new_tokens=5
    )
    print("Success with full prefix:", out.shape)
except Exception as e:
    print("Error with full prefix:", type(e).__name__, e)
