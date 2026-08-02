"""
GSM8K benchmark adapter.

Loads the GSM8K dataset (grade-school math, ~8.5K test problems) from
HuggingFace. Extracts numeric answers from the '#### <answer>' format.
"""
from __future__ import annotations

import re
from typing import Any

from benchmarking.base_benchmark import BaseBenchmark
from benchmarking import benchmark


@benchmark
class GSM8K(BaseBenchmark):
    """GSM8K — Grade School Math 8K."""

    hf_name = "openai/gsm8k"
    split = "test"

    def load_data(self) -> None:
        from datasets import load_dataset

        ds = load_dataset(self.hf_name, "main", split=self.split)
        self.samples = [dict(row) for row in ds]

    def preprocess_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = []
        for idx, sample in enumerate(samples):
            answer_text = sample.get("answer", "")
            # GSM8K format: reasoning steps ending with "#### <number>"
            match = re.search(r"####\s*(.+)", answer_text)
            final_answer = match.group(1).strip() if match else answer_text.strip()

            processed.append({
                "pid": f"gsm8k_{idx}",
                "query": sample["question"],
                "answer": final_answer,
                "subject": "math",
                "level": "grade_school",
            })
        return processed
