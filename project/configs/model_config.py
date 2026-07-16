# configs/model_config.py
import os

# Registry of supported models and their Hugging Face paths
MODEL_REGISTRY = {
    "qwen_1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen_3b": "Qwen/Qwen2.5-3B-Instruct",
    "deepseek_7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "llama_8b": "meta-llama/Llama-3.1-8B-Instruct"
}

# The active model key (change this to select a different model from the registry)
ACTIVE_MODEL_KEY = "qwen_1.5b"

# Active Model HF Path
MODEL_NAME = MODEL_REGISTRY[ACTIVE_MODEL_KEY]

# Hardware settings
DEVICE = "cuda"  # Use 'cuda' or 'cpu'
LOAD_IN_4BIT = True  # Enable bitsandbytes 4-bit loading (ignored on CPU)

# Hugging Face local cache directory (Redirected to A: drive to save C: space)
HF_CACHE_DIR = "A:\\LLMResearch\\hf_cache"

# Output directory for saving model outputs, logits, and hidden states
OUTPUT_DIR = "A:\\LLMResearch\\outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)