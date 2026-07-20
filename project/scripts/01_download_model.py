"""
Script 01 — Download Model & GPU Smoke Test

Downloads the active model configured in configs/model_config.py,
loads it on the GPU (applying 4-bit quantization if specified),
and runs a prompt generation to verify correctness.

Usage:
    conda activate llmresearch
    python project/01_download_model.py
"""

import sys
import os

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from pathlib import Path
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from configs.model_config import (
    MODEL_REGISTRY,
    HF_CACHE_DIR,
    DEVICE,
)
from src.utils.logger import setup_logging
from src.utils.cli import prompt_model_and_precision


def main() -> None:
    """Download, load, and verify the active model."""
    logger = setup_logging(level=logging.INFO)

    # Prompt user for model and precision selection
    selected_key, selected_config, precision_mode = prompt_model_and_precision(MODEL_REGISTRY)
    model_name = selected_config["hf_path"]

    logger.info("=" * 60)
    logger.info("Model Unified Download & GPU Smoke Test")
    logger.info("=" * 60)
    logger.info("Selected model key : %s", selected_key)
    logger.info("HuggingFace ID     : %s", model_name)
    logger.info("Cache directory    : %s", HF_CACHE_DIR)
    logger.info("Target Device      : %s", DEVICE)
    logger.info("Precision/Quant    : %s", precision_mode)

    # ---- Validate GPU if using CUDA ----
    if DEVICE == "cuda":
        cuda_ok = torch.cuda.is_available()
        logger.info("CUDA available: %s", cuda_ok)
        if not cuda_ok:
            logger.error("No GPU visible to torch — fix this before continuing.")
            sys.exit(1)
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    cache_path = Path(HF_CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)

    # ---- Download and Load Tokenizer ----
    logger.info("Loading tokenizer ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=HF_CACHE_DIR,
            trust_remote_code=True,
        )
        logger.info(
            "Tokenizer ready. Vocab size: %d",
            tokenizer.vocab_size,
        )
    except Exception as e:
        logger.error("Failed to download/load tokenizer: %s", e)
        raise

    # ---- Download and Load Model Weights ----
    logger.info("Loading model weights (first run will download) ...")
    try:
        load_kwargs = {
            "cache_dir": HF_CACHE_DIR,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if DEVICE == "cuda":
            if precision_mode == "4bit":
                logger.info("Configuring 4-bit quantization (BitsAndBytes NF4) ...")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kwargs["device_map"] = "auto"
            elif precision_mode == "8bit":
                logger.info("Configuring 8-bit quantization (BitsAndBytes) ...")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                load_kwargs["device_map"] = "auto"
            else:
                logger.info("Loading model in native BF16 ...")
                load_kwargs["torch_dtype"] = torch.bfloat16
                load_kwargs["device_map"] = "cuda"
        else:
            logger.info("Loading model on CPU ...")
            load_kwargs["device_map"] = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        logger.info(
            "Model loaded successfully. Total parameters: %.2fM",
            param_count / 1e6,
        )
    except Exception as e:
        logger.error("Failed to load model weights: %s", e)
        raise

    # ---- Generate reasoning trace ----
    prompt = (
        "Solve step by step and put your final answer in \\boxed{}: "
        "What is 17 * 24 - 15?"
    )
    logger.info("Preparing prompt: %r", prompt)
    messages = [{"role": "user", "content": prompt}]
    
    try:
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except Exception as e:
        logger.warning("Could not apply chat template: %s. Using raw prompt instead.", e)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    logger.info(
        "Generating response (this can take 20-60s during first run kernel warm-up) ..."
    )
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
            )

        response = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )

        print("\n" + "=" * 25 + " MODEL OUTPUT " + "=" * 25)
        print(response)
        print("=" * 64 + "\n")

        if DEVICE == "cuda":
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
            logger.info("Peak VRAM used: %.2f GB", peak_vram)
    except Exception as e:
        logger.error("Error during generation: %s", e)
        raise

    # ---- Verify Cache ----
    cache_size_gb = sum(
        f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
    ) / (1024 ** 3)
    logger.info("Total cache size: %.2f GB", cache_size_gb)

    logger.info("=" * 60)
    logger.info("Unified download & smoke test complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()