"""
DifficultyExperiment — run prefix-continuation difficulty analysis.

For each sample's reasoning trace:
1. Split into progressively longer prefixes
2. For each prefix: build prompt + prefix + budget_forcing_prompt
3. Generate a continuation from the model
4. Evaluate the answer from each continuation
5. Produce accuracy-over-prefix-length data showing the flip point
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from difficulty.config import DifficultyConfig
from difficulty.splitting import split_with_granularity
from evaluation.parsing import ParsingHelper


@dataclass
class PrefixResult:
    """Result of evaluating one prefix continuation."""
    prefix_idx: int
    prefix_length_chars: int
    prefix_length_lines: int
    continuation: str
    extracted_answer: Optional[str]
    ground_truth: str
    correct: Optional[bool]
    raw_text: str


@dataclass
class DifficultyResult:
    """Full difficulty analysis result for one sample."""
    pid: str
    query: str
    ground_truth: str
    original_answer: Optional[str]
    original_correct: Optional[bool]
    num_prefixes: int
    prefix_results: list[PrefixResult]
    last_correct_prefix_idx: Optional[int]
    first_incorrect_prefix_idx: Optional[int]
    flipped: bool


class DifficultyExperiment:
    """
    Run the difficulty prefix-continuation experiment.

    Follows the paper's architecture: for each sample that had a full
    reasoning trace generated, replay the trace as progressively longer
    prefixes, force an early answer at each, and track accuracy.
    """

    def __init__(self, backend: Any, config: DifficultyConfig):
        self.backend = backend
        self.config = config

    def run_single(
        self,
        sample: dict[str, Any],
        reasoning_trace: str,
    ) -> DifficultyResult:
        """
        Run difficulty analysis on a single sample.

        Args:
            sample: Benchmark sample with 'pid', 'query', 'answer'.
            reasoning_trace: The model's full reasoning trace for this sample.

        Returns:
            DifficultyResult with per-prefix accuracy data.
        """
        query = sample["query"]
        ground_truth = sample["answer"]
        pid = sample["pid"]

        # Get the tokenizer for token-level splitting
        tokenizer = getattr(self.backend, "tokenizer", None)

        # Split reasoning trace into progressively longer prefixes
        prefixes = split_with_granularity(
            text=reasoning_trace,
            tokenizer=tokenizer,
            difficulty_level=self.config.difficulty_level,
            granularity=self.config.granularity,
        )

        if not prefixes:
            return DifficultyResult(
                pid=pid,
                query=query,
                ground_truth=ground_truth,
                original_answer=None,
                original_correct=None,
                num_prefixes=0,
                prefix_results=[],
                last_correct_prefix_idx=None,
                first_incorrect_prefix_idx=None,
                flipped=False,
            )

        # Subsample prefixes if there are too many (evenly spaced + always include last)
        if len(prefixes) > self.config.max_prefixes:
            total = len(prefixes)
            step = total / (self.config.max_prefixes - 1)
            indices = [int(i * step) for i in range(self.config.max_prefixes - 1)]
            if indices[-1] != total - 1:
                indices.append(total - 1)
            # Deduplicate while preserving order
            seen = set()
            unique_indices = []
            for i in indices:
                if i not in seen:
                    seen.add(i)
                    unique_indices.append(i)
            prefixes = [prefixes[i] for i in unique_indices]

        # Evaluate the original full trace answer
        original_answer = ParsingHelper.extract_answer(reasoning_trace)
        original_correct = (
            ParsingHelper.compare_answers(original_answer, ground_truth)
            if original_answer
            else None
        )

        # Run prefix continuations
        prefix_results = []
        last_correct_idx = None
        first_incorrect_idx = None

        for prefix_idx, prefix in enumerate(prefixes):
            # Skip prefixes that are too long (would cause OOM)
            if len(prefix) > self.config.max_prefix_chars:
                continue

            # Free GPU memory before each generation
            self._clear_cuda_cache()

            result = self._evaluate_prefix(
                query=query,
                prefix=prefix,
                prefix_idx=prefix_idx,
                ground_truth=ground_truth,
            )
            prefix_results.append(result)

            if result.correct is True:
                last_correct_idx = prefix_idx
            if result.correct is False and first_incorrect_idx is None:
                first_incorrect_idx = prefix_idx

        # Determine if there was a flip (correct → incorrect)
        flipped = last_correct_idx is not None and first_incorrect_idx is not None

        return DifficultyResult(
            pid=pid,
            query=query,
            ground_truth=ground_truth,
            original_answer=original_answer,
            original_correct=original_correct,
            num_prefixes=len(prefix_results),
            prefix_results=prefix_results,
            last_correct_prefix_idx=last_correct_idx,
            first_incorrect_prefix_idx=first_incorrect_idx,
            flipped=flipped,
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        """Free unused GPU memory between generations."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _evaluate_prefix(
        self,
        query: str,
        prefix: str,
        prefix_idx: int,
        ground_truth: str,
    ) -> PrefixResult:
        """Generate continuation from a prefix and evaluate the answer."""
        # Build the full prompt: original question + reasoning prefix + budget forcing
        # The model sees: <user>query</user><assistant><think>prefix</think>\n\boxed{
        assistant_content = f"<think>\n{prefix}\n{self.config.budget_forcing_prompt}"

        # Use the backend to generate continuation
        continuation = self.backend.generate_text(
            prompt=query,
            max_new_tokens=self.config.max_continuation_tokens,
            temperature=self.config.continuation_temperature,
            assistant_prefill=assistant_content,
        )

        # The full text includes the prefix + budget forcing + continuation
        full_text = assistant_content + continuation

        # Extract and evaluate the answer
        extracted = ParsingHelper.extract_answer(full_text)
        correct = (
            ParsingHelper.compare_answers(extracted, ground_truth)
            if extracted
            else None
        )

        return PrefixResult(
            prefix_idx=prefix_idx,
            prefix_length_chars=len(prefix),
            prefix_length_lines=prefix.count("\n") + 1,
            continuation=continuation,
            extracted_answer=extracted,
            ground_truth=ground_truth,
            correct=correct,
            raw_text=full_text,
        )

    def run_batch(
        self,
        samples_and_traces: list[tuple[dict[str, Any], str]],
        output_path: Optional[Path] = None,
        progress_callback: Any = None,
    ) -> list[DifficultyResult]:
        """
        Run difficulty analysis on a batch of samples.

        Args:
            samples_and_traces: List of (sample_dict, reasoning_trace) tuples.
            output_path: Optional path to write incremental JSONL output.
            progress_callback: Optional callable(idx, total) for progress updates.

        Returns:
            List of DifficultyResult.
        """
        results = []

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        for idx, (sample, trace) in enumerate(samples_and_traces):
            if progress_callback:
                progress_callback(idx, len(samples_and_traces))

            result = self.run_single(sample, trace)
            results.append(result)

            # Incremental write
            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    record = {
                        "pid": result.pid,
                        "query": result.query,
                        "ground_truth": result.ground_truth,
                        "original_answer": result.original_answer,
                        "original_correct": result.original_correct,
                        "num_prefixes": result.num_prefixes,
                        "last_correct_prefix_idx": result.last_correct_prefix_idx,
                        "first_incorrect_prefix_idx": result.first_incorrect_prefix_idx,
                        "flipped": result.flipped,
                        "prefix_results": [
                            {
                                "prefix_idx": pr.prefix_idx,
                                "prefix_length_chars": pr.prefix_length_chars,
                                "prefix_length_lines": pr.prefix_length_lines,
                                "extracted_answer": pr.extracted_answer,
                                "correct": pr.correct,
                            }
                            for pr in result.prefix_results
                        ],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return results
