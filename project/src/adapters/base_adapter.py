"""
Base Adapter Interface

Every reasoning model (Qwen, Llama, DeepSeek, etc.)
must implement this interface.

This keeps the rest of the codebase model-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from src.trace.reasoning_trace import ReasoningTrace


class BaseAdapter(ABC):
    """
    Abstract interface for all reasoning models.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ):

        self.model_name = model_name
        self.device = device

        self.model = None
        self.tokenizer = None

    # --------------------------------------------------
    # Model Lifecycle
    # --------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        Load tokenizer and model into memory.
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """
        Free GPU memory.
        """
        pass

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    @abstractmethod
    def generate_trace(
        self,
        prompt: str,
        benchmark: str,
        problem_id: str,
        temperature: float,
        max_new_tokens: int,
        checkpoint_interval: int = 32,
    ) -> ReasoningTrace:
        """
        Generate a complete ReasoningTrace.

        This is the main entry point used by the project.
        """
        pass

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    @abstractmethod
    def count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count tokens using the model tokenizer.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Return human-readable model name.
        """
        pass