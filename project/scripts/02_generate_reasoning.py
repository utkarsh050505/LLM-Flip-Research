# scripts/02_generate_reasoning.py
import sys
import os
# pyrefly: ignore [missing-import]
import torch

# Append the project root to sys.path to allow imports from src/ and configs/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.model_config import MODEL_NAME, DEVICE, LOAD_IN_4BIT, HF_CACHE_DIR, OUTPUT_DIR
from configs.experiment_config import SEED, GENERATION_PARAMS, DEFAULT_PROMPT
from src.utils import set_seed, setup_logging, get_vram_usage
from src.adapters.qwen import QwenAdapter
from src.generation import ReasoningTrace

def main():
    # 1. Setup logging
    log_file = os.path.join(OUTPUT_DIR, "generate_reasoning.log")
    logger = setup_logging(log_file)
    logger.info("=" * 60)
    logger.info("Starting reasoning trace generation & feature extraction...")
    logger.info(f"Model Name: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE} (CUDA available: {torch.cuda.is_available()})")
    logger.info(f"Load in 4-bit: {LOAD_IN_4BIT}")
    
    # 2. Set deterministic seed
    set_seed(SEED)
    logger.info(f"Deterministic seed set to {SEED}")
    
    # Log VRAM before loading
    logger.info(f"VRAM usage (before model load): {get_vram_usage()}")

    # 3. Instantiate QwenAdapter
    logger.info("Initializing QwenAdapter...")
    adapter = QwenAdapter(
        model_name=MODEL_NAME,
        device=DEVICE,
        load_in_4bit=LOAD_IN_4BIT,
        cache_dir=HF_CACHE_DIR
    )

    # 4. Load the model and tokenizer
    logger.info("Loading model and tokenizer...")
    adapter.load_model()
    logger.info("Model loaded successfully.")
    
    # Log VRAM after loading
    logger.info(f"VRAM usage (after model load): {get_vram_usage()}")

    # 5. Generate reasoning trace
    logger.info(f"Prompt: {DEFAULT_PROMPT}")
    logger.info(f"Generation parameters: {GENERATION_PARAMS}")
    
    # Generate and extract
    outputs = adapter.generate(
        prompt=DEFAULT_PROMPT,
        max_new_tokens=GENERATION_PARAMS["max_new_tokens"],
        temperature=GENERATION_PARAMS["temperature"],
        top_p=GENERATION_PARAMS["top_p"],
        do_sample=GENERATION_PARAMS["do_sample"],
        apply_template=True
    )
    
    logger.info("Generation and extraction completed successfully!")
    logger.info(f"Generated text snippet: {outputs['generated_text'][:150]}...")
    logger.info(f"Number of generated tokens: {len(outputs['tokens'])}")
    logger.info(f"VRAM usage (post generation): {get_vram_usage()}")

    # 6. Create ReasoningTrace container
    trace = ReasoningTrace(
        prompt=DEFAULT_PROMPT,
        generated_text=outputs["generated_text"],
        tokens=outputs["tokens"],
        token_ids=outputs["token_ids"],
        token_probs=outputs["token_probs"],
        logits=outputs["logits"],
        hidden_states=outputs["hidden_states"],
        prompt_length=outputs["prompt_length"]
    )

    # 7. Save reasoning trace and metadata
    filepath_base = os.path.join(OUTPUT_DIR, "trace_qwen_1.5b")
    logger.info(f"Saving trace to base path: {filepath_base} (.json and .pt)")
    trace.save(filepath_base)
    logger.info("Trace saved successfully.")

    # 8. Quick validation load check
    logger.info("Verifying saved files by reloading...")
    loaded_trace = ReasoningTrace.load(filepath_base)
    logger.info("Reload verification successful!")
    logger.info(f"Reloaded trace generated length: {len(loaded_trace.tokens)} tokens.")
    logger.info(f"Reloaded logits shape: {loaded_trace.logits[0].shape if loaded_trace.logits else 'None'}")
    logger.info(f"Reloaded hidden states shape: {loaded_trace.hidden_states[0].shape if loaded_trace.hidden_states else 'None'}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
