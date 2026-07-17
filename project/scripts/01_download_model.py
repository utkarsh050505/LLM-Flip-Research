"""
Script 01 — Download Model

Downloads and caches the active model from Hugging Face.
No GPU required — runs on CPU for download only.

Usage:
    conda activate llmresearch
    python scripts/01_download_model.py
"""

import sys
import os

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from configs.model_config import MODEL_NAME, HF_CACHE_DIR, ACTIVE_MODEL_KEY
from src.utils.logger import setup_logging


def main() -> None:
    """Download and cache the active model and tokenizer."""
    logger = setup_logging(level=logging.INFO)

    logger.info("=" * 60)
    logger.info("Model Download Script")
    logger.info("=" * 60)
    logger.info("Active model key : %s", ACTIVE_MODEL_KEY)
    logger.info("HuggingFace ID   : %s", MODEL_NAME)
    logger.info("Cache directory  : %s", HF_CACHE_DIR)

    cache_path = Path(HF_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    # ---- Download tokenizer ----
    logger.info("Downloading tokenizer ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
        )
        logger.info(
            "Tokenizer ready. Vocab size: %d",
            tokenizer.vocab_size,
        )
    except Exception as e:
        logger.error("Failed to download tokenizer: %s", e)
        raise

    # ---- Download model weights ----
    logger.info("Downloading model weights (this may take several minutes) ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=None,  # CPU only — just downloading
        )
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model downloaded. Parameters: %.2fM",
            param_count / 1e6,
        )
    except Exception as e:
        logger.error("Failed to download model: %s", e)
        raise

    # ---- Verify cache ----
    cache_size_mb = sum(
        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
    ) / (1024 ** 2)
    logger.info("Total cache size: %.1f MB", cache_size_mb)

    logger.info("=" * 60)
    logger.info("Download complete!")
    logger.info("=" * 60)

    # Free memory
    del model
    del tokenizer


if __name__ == "__main__":
    main()
