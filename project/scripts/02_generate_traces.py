"""
Script 02 — Generate Reasoning Traces

Loads the active model, generates a reasoning trace from a prompt,
and saves it as JSON to datasets/traces/.

Phase 2: Uses the custom autoregressive decoder with per-step
instrumentation (top-k logits, entropy, hidden states).

Usage:
    conda activate llmresearch
    python scripts/02_generate_traces.py

Output:
    datasets/traces/trace_<timestamp>_<uuid>.json
"""

import sys
import os

# Ensure project root is on sys.path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

from configs.model_config import MODEL_NAME, HF_CACHE_DIR, ACTIVE_MODEL_KEY
from configs.experiment_config import DEFAULT_PROMPT, SEED
from configs.generation_config import (
    TEMPERATURE,
    MAX_NEW_TOKENS,
    TOP_K_LOGITS,
    EXTRACT_HIDDEN_STATES,
    CHECKPOINT_WINDOW,
    TRACE_OUTPUT_DIR,
    DEFAULT_BENCHMARK,
    DEFAULT_PROBLEM_ID,
)
from src.utils.logger import setup_logging
from src.utils.seed import set_deterministic_seed
from src.adapters.qwen_adapter import QwenAdapter
from src.generation.generator import TraceGenerator
from src.checkpoints.checkpoint_builder import CheckpointBuilder


def main() -> None:
    """Generate a single reasoning trace and save it to disk."""
    logger = setup_logging(level=logging.INFO)

    logger.info("=" * 60)
    logger.info("Trace Generation Script — Phase 2 (Custom Decoder)")
    logger.info("=" * 60)
    logger.info("Model key      : %s", ACTIVE_MODEL_KEY)
    logger.info("Model ID       : %s", MODEL_NAME)
    logger.info("Temperature    : %.2f", TEMPERATURE)
    logger.info("Max tokens     : %d", MAX_NEW_TOKENS)
    logger.info("Top-k logits   : %d", TOP_K_LOGITS)
    logger.info("Hidden states  : %s", EXTRACT_HIDDEN_STATES)
    logger.info("Checkpt window : %d", CHECKPOINT_WINDOW)
    logger.info("Seed           : %d", SEED)
    logger.info("Output dir     : %s", TRACE_OUTPUT_DIR)
    logger.info("-" * 60)

    # Set seeds for reproducibility
    set_deterministic_seed(SEED)

    # Create adapter and generator
    adapter = QwenAdapter(
        model_name=MODEL_NAME,
        device="cuda",
        cache_dir=HF_CACHE_DIR,
    )

    checkpoint_builder = CheckpointBuilder(window_size=CHECKPOINT_WINDOW)

    generator = TraceGenerator(
        adapter=adapter,
        output_dir=TRACE_OUTPUT_DIR,
        checkpoint_builder=checkpoint_builder,
    )

    try:
        # Load model
        generator.setup()

        # Log prompt
        prompt_preview = DEFAULT_PROMPT[:100] + "..." if len(DEFAULT_PROMPT) > 100 else DEFAULT_PROMPT
        logger.info("Prompt: %s", prompt_preview)

        # Generate and save trace
        trace, output_path = generator.generate_and_save(
            prompt=DEFAULT_PROMPT,
            benchmark=DEFAULT_BENCHMARK,
            problem_id=DEFAULT_PROBLEM_ID,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            seed=SEED,
        )

        # Report results
        logger.info("=" * 60)
        logger.info("RESULTS")
        logger.info("=" * 60)
        logger.info("Output file     : %s", output_path)
        logger.info("Steps generated : %d", trace.num_steps)
        logger.info("Checkpoints     : %d", len(trace.checkpoints))
        logger.info("Time elapsed    : %.2f s", trace.generation.timing.total_seconds)
        logger.info("Prefill time    : %.3f s", trace.generation.timing.prefill_seconds)
        logger.info("Throughput      : %.1f tok/s", trace.generation.timing.tokens_per_second)
        logger.info("-" * 60)

        # Phase 3: Trajectory statistics
        if trace.trajectory:
            t = trace.trajectory
            logger.info("Trajectory ID   : %s", t.trajectory_id)
            logger.info("Trajectory len  : %d checkpoints", t.trajectory_length)
            logger.info("Duration        : %.2f s", t.trajectory_duration)
            logger.info("Avg entropy     : %.4f", t.avg_entropy)
            logger.info("Max entropy     : %.4f", t.max_entropy)
            logger.info("Entropy var     : %.6f", t.entropy_variance)
            logger.info("Avg confidence  : %.4f", t.avg_confidence)
            if t.velocity_profile:
                logger.info("Velocity range  : [%.4f, %.4f]",
                            min(t.velocity_profile), max(t.velocity_profile))
            logger.info("-" * 60)

        # Entropy statistics
        entropies = [e for e in trace.entropies if e is not None]
        if entropies:
            logger.info("Entropy — min: %.3f, max: %.3f, mean: %.3f",
                        min(entropies), max(entropies),
                        sum(entropies) / len(entropies))

        # Print a preview of the reasoning text
        preview_len = 500
        reasoning = trace.generation.reasoning_text
        if len(reasoning) > preview_len:
            logger.info("Reasoning preview:\n%s\n... [truncated]", reasoning[:preview_len])
        else:
            logger.info("Reasoning:\n%s", reasoning)

        # Show first few steps as a sample
        logger.info("-" * 60)
        logger.info("Sample steps (first 5):")
        for step in trace.generation.steps[:5]:
            top_token = step.top_k_logits[0].token if step.top_k_logits else "?"
            top_prob = step.top_k_logits[0].probability if step.top_k_logits else 0
            logger.info(
                "  [%3d] token=%-12r  entropy=%.3f  top1=%r (p=%.4f)",
                step.step_index,
                step.generated_token,
                step.entropy if step.entropy is not None else 0.0,
                top_token,
                top_prob,
            )

        logger.info("=" * 60)
        logger.info("Trace generation complete!")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Trace generation failed.")
        raise
    finally:
        # Always clean up GPU memory
        if adapter.is_loaded():
            generator.teardown()


if __name__ == "__main__":
    main()
