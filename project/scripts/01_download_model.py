# scripts/01_download_model.py
import sys
import os

# Append the project root to sys.path to allow imports from src/ and configs/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import MODEL_NAME, HF_CACHE_DIR
from src.utils import setup_logging
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    logger = setup_logging()
    logger.info(f"Starting download of model: {MODEL_NAME}")
    logger.info(f"Target HF cache directory: {HF_CACHE_DIR}")

    os.makedirs(HF_CACHE_DIR, exist_ok=True)

    # 1. Download and cache tokenizer
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True
    )
    logger.info("Tokenizer downloaded and cached successfully.")

    # 2. Download and cache model
    logger.info("Downloading model checkpoint files (this may take a few minutes)...")
    # Load model weight parameters (without actually loading onto GPU, keeping on CPU or just download cache)
    # We do a fast load on CPU or just map metadata. Since target is cache prep, we can just download it.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=HF_CACHE_DIR,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map=None  # Load to CPU to keep it light
    )
    logger.info("Model downloaded and cached successfully.")
    logger.info("Download completed successfully!")

if __name__ == "__main__":
    main()
