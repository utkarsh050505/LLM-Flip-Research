"""
BaseBenchmark — abstract interface every benchmark adapter must implement.

Standard sample schema:
    {
        "pid": str,          # unique problem ID
        "query": str,        # the question text
        "answer": str,       # ground truth answer
        "subject": str,      # optional subject/category
        "level": str,        # optional difficulty level
    }
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseBenchmark(ABC):
    def __init__(
        self,
        benchmark_name: str,
        additional_args: Optional[str] = None,
        debug_limit: Optional[int] = None,
    ):
        self.benchmark_name = benchmark_name
        self.debug_limit = debug_limit
        self.samples: list[dict[str, Any]] = []

        if additional_args:
            self.parse_additional_args(additional_args)
        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        """Load the dataset and populate self.samples with raw records."""
        ...

    @abstractmethod
    def preprocess_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert raw dataset records to the standard sample schema.

        Each returned dict must contain at minimum:
            - "pid": unique problem ID
            - "query": the question text
            - "answer": ground truth answer string
        """
        ...

    def parse_additional_args(self, additional_args: str) -> None:
        """Parse semicolon-separated key=value pairs into attributes."""
        for arg in additional_args.split(";"):
            if "=" in arg:
                key, value = arg.split("=", 1)
                setattr(self, key.strip(), value.strip())

    def get_samples(self) -> list[dict[str, Any]]:
        """Return preprocessed samples, optionally truncated for debug."""
        processed = self.preprocess_samples(self.samples)
        if self.debug_limit is not None and self.debug_limit > 0:
            processed = processed[: self.debug_limit]
        return processed

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.benchmark_name!r}, n={len(self.samples)})"
