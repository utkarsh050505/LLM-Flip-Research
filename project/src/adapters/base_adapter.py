"""
Base Adapter Interface

Every reasoning model (Qwen, Llama, DeepSeek, etc.)
must implement this interface.

This keeps the rest of the codebase model-agnostic.
No code outside of adapters/ should ever call
AutoModelForCausalLM.from_pretrained() directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.trace.reasoning_trace import ReasoningTrace


class BaseAdapter(ABC):
    """
    Abstract interface for all reasoning model adapters.

    Subclasses must implement: load_model, unload_model,
    generate_trace, count_tokens, get_model_name.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            model_name: HuggingFace model identifier.
            device: Target device ('cuda' or 'cpu').
            cache_dir: Local HuggingFace cache directory.
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir

        self.model = None
        self.tokenizer = None

    # --------------------------------------------------
    # Model Lifecycle
    # --------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Load tokenizer and model into memory."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Free GPU/CPU memory."""
        ...

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    @abstractmethod
    def generate_trace(
        self,
        prompt: str,
        benchmark: str = "debug",
        problem_id: str = "0",
        temperature: float = 0.6,
        max_new_tokens: int = 512,
        seed: int = 42,
    ) -> ReasoningTrace:
        """
        Generate a complete ReasoningTrace from a prompt.

        This is the main entry point used by the generation pipeline.

        Args:
            prompt: The user prompt / question.
            benchmark: Name of the benchmark dataset.
            problem_id: Identifier within the benchmark.
            temperature: Sampling temperature (0 = greedy).
            max_new_tokens: Maximum tokens to generate.
            seed: Random seed for reproducibility.

        Returns:
            A fully populated ReasoningTrace.
        """
        ...

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """Return human-readable model name."""
        ...

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self.model is not None and self.tokenizer is not None