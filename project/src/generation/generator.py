"""
Trace Generator

Orchestrates the end-to-end flow:
    Prompt → Adapter → ReasoningTrace → Serializer → JSON

This is the main entry point for trace collection.
It coordinates between the adapter (model-specific) and
the serializer (format-specific), keeping both decoupled.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from src.adapters.base_adapter import BaseAdapter
from src.trace.reasoning_trace import ReasoningTrace
from src.trace.serializer import save_trace
from src.utils.io import ensure_directory
from src.checkpoints.checkpoint_builder import CheckpointBuilder

logger = logging.getLogger("sbtf")


class TraceGenerator:
    """
    High-level orchestrator for generating and saving reasoning traces.

    Wraps an adapter and provides a clean interface for scripts.

    Usage:
        adapter = QwenAdapter("Qwen/Qwen2.5-1.5B-Instruct")
        generator = TraceGenerator(adapter, output_dir="datasets/traces/")
        generator.setup()
        trace, path = generator.generate_and_save(prompt="What is 2+2?")
        generator.teardown()
    """

    def __init__(
        self,
        adapter: BaseAdapter,
        output_dir: Union[str, Path],
        checkpoint_builder: Optional[CheckpointBuilder] = None,
    ):
        """
        Args:
            adapter: A concrete BaseAdapter subclass (e.g., QwenAdapter).
            output_dir: Directory where trace JSON files will be saved.
            checkpoint_builder: Optional builder to extract reasoning checkpoints.
        """
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.checkpoint_builder = checkpoint_builder

    def setup(self) -> None:
        """
        Load the model and prepare output directory.

        Call this once before generating traces.
        """
        ensure_directory(self.output_dir)
        self.adapter.load_model()
        logger.info("TraceGenerator ready. Output dir: %s", self.output_dir)

    def teardown(self) -> None:
        """
        Unload the model and free GPU memory.

        Call this when done generating traces.
        """
        self.adapter.unload_model()
        logger.info("TraceGenerator torn down.")

    def generate_and_save(
        self,
        prompt: str,
        benchmark: str = "debug",
        problem_id: str = "0",
        temperature: float = 0.6,
        max_new_tokens: int = 512,
        seed: int = 42,
        filename: Optional[str] = None,
    ) -> tuple[ReasoningTrace, Path]:
        """
        Generate a reasoning trace and save it to disk.

        Args:
            prompt: The question / prompt text.
            benchmark: Benchmark name.
            problem_id: Problem identifier.
            temperature: Sampling temperature.
            max_new_tokens: Max tokens to generate.
            seed: Random seed.
            filename: Optional custom filename (without extension).
                      If None, a timestamp + UUID filename is generated.

        Returns:
            Tuple of (ReasoningTrace, Path to saved JSON file).

        Raises:
            RuntimeError: If setup() has not been called.
        """
        if not self.adapter.is_loaded():
            raise RuntimeError(
                "Adapter not loaded. Call generator.setup() first."
            )

        # Generate trace via the adapter
        trace = self.adapter.generate_trace(
            prompt=prompt,
            benchmark=benchmark,
            problem_id=problem_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

        # Build checkpoints if enabled
        if self.checkpoint_builder:
            self.checkpoint_builder.build_checkpoints(trace)
            logger.info("Extracted %d checkpoints", len(trace.checkpoints))

        # Build filename
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            short_uuid = uuid.uuid4().hex[:8]
            filename = f"trace_{timestamp}_{short_uuid}"

        output_path = self.output_dir / f"{filename}.json"

        # Save
        save_trace(trace, output_path)

        logger.info(
            "Trace generated and saved: %s (%d steps)",
            output_path.name,
            trace.num_steps,
        )

        return trace, output_path
