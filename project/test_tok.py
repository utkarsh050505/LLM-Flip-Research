# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print("</think> id:", tokenizer.encode("</think>", add_special_tokens=False))
print("boxed id:", tokenizer.encode("\\boxed", add_special_tokens=False))
print("Wait id:", tokenizer.encode("Wait", add_special_tokens=False))
