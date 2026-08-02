"""
MATH-500 benchmark adapter.

Loads the HuggingFace MATH-500 dataset (curated 500-problem subset of
the MATH benchmark). Extracts answers from \\boxed{} in solution text.
"""
from __future__ import annotations

import re
from typing import Any

from benchmarking.base_benchmark import BaseBenchmark
from benchmarking import benchmark


@benchmark
class MATH500(BaseBenchmark):
    """MATH-500 — 500-problem subset of MATH benchmark."""

    hf_name = "HuggingFaceH4/MATH-500"
    split = "test"

    def load_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset(self.hf_name, split=self.split)
        self.samples = [dict(row) for row in ds]

    def preprocess_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = []
        for idx, sample in enumerate(samples):
            # Extract answer from \boxed{} in solution, or use 'answer' field
            answer = sample.get("answer", "")
            if not answer:
                solution = sample.get("solution", "")
                answer = self._extract_boxed(solution) or ""

            processed.append({
                "pid": f"math500_{idx}",
                "query": sample.get("problem", sample.get("question", "")),
                "answer": answer,
                "subject": sample.get("subject", sample.get("type", "math")),
                "level": sample.get("level", "unknown"),
            })
        return processed

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Extract the last \\boxed{...} content from text."""
        idx = text.rfind(r"\boxed{")
        if idx == -1:
            return None
        start = idx + len(r"\boxed{")
        depth = 1
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:pos].strip()
        return None
