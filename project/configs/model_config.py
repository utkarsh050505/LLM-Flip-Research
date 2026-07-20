# configs/model_config.py
import os

# Registry of supported models and their Hugging Face paths, families, and quantization settings
MODEL_REGISTRY = {
    "deepseek_qwen_1.5b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "family": "qwen",
        "load_in_4bit": False
    },
    "qwen_math_1.5b": {
        "hf_path": "Qwen/Qwen2.5-Math-1.5B-Instruct",
        "family": "qwen",
        "load_in_4bit": False
    },
    "metastone_s1_1.5b": {
        "hf_path": "MetaStoneTec/MetaStone-S1-1.5B",
        "family": "qwen",
        "load_in_4bit": False
    },
    "t0_s1_1.5b": {
        "hf_path": "alan-turing-institute/t0-s1-1.5B",
        "family": "qwen",
        "load_in_4bit": False
    },
    "deepseek_qwen_7b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "family": "qwen",
        "load_in_4bit": True
    },
    "deepseek_llama_8b": {
        "hf_path": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "family": "llama",
        "load_in_4bit": True
    }
}

# The active model key (change this to select a different model from the registry)
ACTIVE_MODEL_KEY = "deepseek_qwen_1.5b"

# Active Model config
ACTIVE_MODEL_CONFIG = MODEL_REGISTRY[ACTIVE_MODEL_KEY]
MODEL_NAME = ACTIVE_MODEL_CONFIG["hf_path"]
LOAD_IN_4BIT = ACTIVE_MODEL_CONFIG["load_in_4bit"]

# Hardware settings
DEVICE = "cuda"  # Use 'cuda' or 'cpu'

# Hugging Face local cache directory (Redirected to A: drive to save C: space)
HF_CACHE_DIR = "A:\\LLMResearch\\hf_cache"

# Output directory for saving model outputs, logits, and hidden states
OUTPUT_DIR = "A:\\LLMResearch\\outputs"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
