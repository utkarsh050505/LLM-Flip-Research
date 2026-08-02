"""
Local problems benchmark adapter.

Loads problems from the project's existing problems/ directory
(JSON files with {problem, answer, subject, level, pass_rate}).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarking.base_benchmark import BaseBenchmark
from benchmarking import benchmark


@benchmark
class LocalProblems(BaseBenchmark):
    """Loads problems from the local problems/ directory."""

    def load_data(self) -> None:
        problems_dir = Path(__file__).parent.parent / "problems"
        if not problems_dir.exists():
            self.samples = []
            return

        self.samples = []
        for json_file in sorted(problems_dir.glob("*.json")):
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["_filename"] = json_file.stem
            self.samples.append(data)

    def preprocess_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        processed = []
        for idx, sample in enumerate(samples):
            processed.append({
                "pid": sample.get("_filename", f"local_{idx}"),
                "query": sample.get("problem", sample.get("question", "")),
                "answer": str(sample.get("answer", "")),
                "subject": sample.get("subject", "unknown"),
                "level": sample.get("level", "unknown"),
            })
        return processed
