"""
Parsing — robust answer extraction and normalization.

Mirrors the ParsingHelper from thinking-past-the-answer with
additional strategies, building on the existing pcc/answer.py logic.
"""
from __future__ import annotations

import re
from enum import Enum, auto
from typing import Optional


class Strategies(Enum):
    """Answer extraction strategies."""
    BOXED = auto()          # Extract from \\boxed{...}
    FINAL_ANSWER = auto()   # Extract from "FINAL ANSWER: ..."
    LAST_NUMBER = auto()    # Extract the last standalone number
    LLM_EXTRACT = auto()    # Use LLM-based extraction


class ParsingHelper:
    """Shared cleaning and extraction logic."""

    @staticmethod
    def extract_all_boxed(text: str) -> list[str]:
        """
        Extract all \\boxed{...} contents, handling nested braces.

        Matches the paper's implementation with full brace-depth tracking.
        """
        results = []
        start_marker = "boxed{"
        current_idx = 0

        while True:
            start_idx = text.find(start_marker, current_idx)
            if start_idx == -1:
                break

            scan_idx = start_idx + len(start_marker)
            brace_depth = 1
            content_start = scan_idx

            for i in range(scan_idx, len(text)):
                if text[i] == "{":
                    brace_depth += 1
                elif text[i] == "}":
                    brace_depth -= 1
                if brace_depth == 0:
                    results.append(text[content_start:i])
                    current_idx = i + 1
                    break
            else:
                break

        return results

    @staticmethod
    def extract_last_boxed(text: str) -> Optional[str]:
        """Extract the last \\boxed{...} content."""
        all_boxed = ParsingHelper.extract_all_boxed(text)
        return all_boxed[-1].strip() if all_boxed else None

    @staticmethod
    def clean(text: str) -> str:
        """
        Normalize an answer string for comparison.

        Steps:
        1. Strip LaTeX commands (\\text{}, \\frac{}{}, etc.)
        2. Lowercase
        3. Keep only: a-z, 0-9, dots, minus, slashes, spaces
        4. Normalize whitespace
        5. Strip leading/trailing whitespace
        """
        text = str(text)
        if not text:
            return ""

        # Strip LaTeX commands
        text = re.sub(r"\\[a-zA-Z]+", "", text)

        # Lowercase
        text = text.lower()

        # Keep only allowed characters
        text = re.sub(r"[^a-z0-9.\-/\s]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @staticmethod
    def extract_answer(
        text: str,
        strategies: list[Strategies] | None = None,
    ) -> Optional[str]:
        """
        Extract the final answer using a chain of strategies.

        Tries each strategy in order and returns the first successful extraction.
        """
        if strategies is None:
            strategies = [Strategies.BOXED, Strategies.FINAL_ANSWER, Strategies.LAST_NUMBER]

        for strategy in strategies:
            result = None

            if strategy == Strategies.BOXED:
                result = ParsingHelper.extract_last_boxed(text)

            elif strategy == Strategies.FINAL_ANSWER:
                matches = re.findall(
                    r"(?im)^\s*(?:FINAL\s+ANSWER|final\s+answer|The\s+answer\s+is)\s*:?\s*(.+?)\s*$",
                    text,
                )
                if matches:
                    result = matches[-1].strip()

            elif strategy == Strategies.LAST_NUMBER:
                # Find the last standalone number (including decimals, fractions)
                numbers = re.findall(
                    r"(?<![a-zA-Z])(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$",
                    text.strip(),
                )
                if numbers:
                    result = numbers[-1]

            if result is not None:
                return result

        return None

    @staticmethod
    def compare_answers(predicted: str, ground_truth: str) -> bool:
        """Compare two answers after normalization."""
        pred_clean = ParsingHelper.clean(predicted)
        gt_clean = ParsingHelper.clean(ground_truth)

        if not pred_clean or not gt_clean:
            return False

        # Direct match
        if pred_clean == gt_clean:
            return True

        # Try numeric comparison (handles "3.0" == "3", etc.)
        try:
            pred_num = float(pred_clean.replace(" ", ""))
            gt_num = float(gt_clean.replace(" ", ""))
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, ZeroDivisionError):
            pass

        return False
