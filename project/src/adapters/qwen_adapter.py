"""
Qwen Adapter

Concrete implementation of the BaseAdapter interface
for Qwen-family reasoning models (Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-8B).

Phase 2: Uses the custom autoregressive Decoder instead of model.generate().
Every forward pass is individually instrumented.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src.adapters.base_adapter import BaseAdapter
from src.trace.reasoning_trace import (
    ReasoningTrace,
    TraceMetadata,
)
from src.generation.decoder import Decoder, DecoderConfig

logger = logging.getLogger("sbtf")


class QwenAdapter(BaseAdapter):
    """
    Adapter for Qwen-family instruction-tuned models.

    Supports fp16 loading with device_map='auto' for single-GPU setups.
    Uses the custom Decoder for autoregressive generation with
    per-step instrumentation.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        super().__init__(model_name, device, cache_dir)

    # --------------------------------------------------
    # Model Loading
    # --------------------------------------------------

    def load_model(self) -> None:
        """
        Load the Qwen tokenizer and model into memory.

        Uses fp16 precision with device_map='auto' to fit within
        available GPU VRAM (8 GB RTX 4060).
        """
        if self.is_loaded():
            logger.warning("Model already loaded, skipping reload.")
            return

        logger.info("Loading tokenizer for %s ...", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )

        logger.info("Loading model %s (fp16, device_map=auto) ...", self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model.eval()

        # Report VRAM usage
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            logger.info(
                "GPU memory — allocated: %.2f GB, reserved: %.2f GB",
                allocated_gb,
                reserved_gb,
            )

        logger.info("Model loaded successfully: %s", self.model_name)

    # --------------------------------------------------

    def unload_model(self) -> None:
        """Free GPU memory by deleting the model and tokenizer."""
        logger.info("Unloading model %s ...", self.model_name)
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded, GPU cache cleared.")

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_model_name(self) -> str:
        return self.model_name

    def count_tokens(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not loaded. Call load_model() first."
            )
        return len(
            self.tokenizer.encode(text, add_special_tokens=False)
        )

    # --------------------------------------------------
    # Generation (Phase 2 — Custom Decoder)
    # --------------------------------------------------

    def generate_trace(
        self,
        prompt: str,
        benchmark: str = "debug",
        problem_id: str = "0",
        temperature: float = 0.6,
        max_new_tokens: int = 512,
        seed: int = 42,
        top_k_logits: int = 5,
        extract_hidden_states: bool = True,
    ) -> ReasoningTrace:
        """
        Generate a reasoning trace using the custom autoregressive decoder.

        Every forward pass is individually instrumented, producing a
        GenerationStep with top-k logits, entropy, and optional hidden states.

        Args:
            prompt: The reasoning question to answer.
            benchmark: Benchmark identifier.
            problem_id: Problem identifier within the benchmark.
            temperature: Sampling temperature (0 = greedy decoding).
            max_new_tokens: Maximum number of new tokens.
            seed: Random seed.
            top_k_logits: Number of top logits to record per step.
            extract_hidden_states: Whether to extract hidden state layers.

        Returns:
            A fully instrumented ReasoningTrace.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Model not loaded. Call load_model() before generate_trace()."
            )

        logger.info("Generating trace for problem '%s' ...", problem_id)

        # Build metadata
        metadata = TraceMetadata(
            model_name=self.model_name,
            benchmark=benchmark,
            problem_id=problem_id,
            prompt_text=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        trace = ReasoningTrace(metadata)

        # Format prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        )
        prompt_ids = inputs.input_ids.to(self.model.device)

        logger.info("Prompt length: %d tokens", prompt_ids.shape[1])

        # Configure and run decoder
        decoder_config = DecoderConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k_logits=top_k_logits,
            extract_hidden_states=extract_hidden_states,
            hidden_state_layers=("first", "middle", "last"),
            seed=seed,
        )

        decoder = Decoder(
            model=self.model,
            tokenizer=self.tokenizer,
            config=decoder_config,
        )

        steps, timing = decoder.decode(prompt_ids)

        # Populate trace
        for step in steps:
            trace.add_step(step)

        trace.generation.timing = timing

        logger.info(
            "Generated %d tokens in %.2fs (%.1f tok/s)",
            timing.num_generated_tokens,
            timing.total_seconds,
            timing.tokens_per_second,
        )

        return trace