"""
TaxonomyClassifier — classify overthinking failure modes.

Compares the last-correct prefix trace with the final trace to identify
what went wrong when the model flipped from correct to incorrect.

Adapted from thinking-past-the-answer's categorize_overthinking_from_last_true.py.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Optional


TAXONOMY_LABELS = [
    "calculation_error",
    "logical_error",
    "overthinking_flip",
    "degenerate_repetition",
    "format_error",
]

TAXONOMY_GUIDE = """Category guide:
- calculation_error: the continued trace makes an arithmetic, algebraic, counting, formula, unit-conversion, or numerical-computation mistake after the last correct prefix.
- logical_error: the continued trace uses flawed non-numerical reasoning, draws an unsupported conclusion, maps the right reasoning to the wrong answer, follows an irrelevant detour, repeats itself into a contradiction, or changes only the final answer without a calculation mistake.
- overthinking_flip: the model had the correct answer but continued reasoning, second-guessed itself, and changed to a wrong answer. The original reasoning was sound but additional deliberation introduced doubt.
- degenerate_repetition: the model enters a loop, repeating the same reasoning or answer pattern multiple times, eventually producing a wrong or garbled final answer.
- format_error: the model's final answer is correct in substance but incorrectly formatted, or the extraction failed to capture the intended answer.
"""

JUDGE_PROMPT = """Return exactly one compact JSON object and nothing else.
Do not think step by step in the visible response. Do not output analysis, markdown, or prose.
The first character must be "{{" and the last character must be "}}".

You are analyzing overthinking in a reasoning trace.

The first trace is the LAST prefix where the model's answer was still correct.
The second trace is the FINAL full trace where the model's answer became incorrect.

Task:
1. Compare only what changed after the last-correct prefix.
2. Identify the main failure mode introduced by the continued reasoning.
3. Choose the best available category even when the drift is small or ambiguous.

Allowed categories:
{categories}

{category_guide}

Severity:
- 0 = no real failure or final answer still correct
- 25 = minor drift but failure source is weak/ambiguous
- 50 = clear failure mode
- 75 = strong failure mode that dominates the final trace
- 100 = extreme failure mode

Return only valid JSON with this exact schema:
{{
  "category": "one_allowed_category",
  "severity": 0_to_100_integer,
  "went_wrong": "short explanation",
  "evidence": "short quote or paraphrase from the added trace"
}}

Metadata:
- pid: {pid}
- last_correct_prefix_idx: {last_correct_idx}
- ground_truth: {ground_truth}
- last_correct_answer: {last_correct_answer}
- final_answer: {final_answer}

Original question:
```text
{query}
```

Last-correct trace:
```text
{last_correct_trace}
```

Final/full trace:
```text
{final_trace}
```

New content after last-correct prefix:
```text
{added_suffix}
```"""


@dataclass
class TaxonomyResult:
    """Result of taxonomy classification for one sample."""
    pid: str
    category: str
    severity: int
    went_wrong: str
    evidence: str
    last_correct_prefix_idx: Optional[int]
    ground_truth: str
    last_correct_answer: Optional[str]
    final_answer: Optional[str]
    raw_judge_output: str


class TaxonomyClassifier:
    """
    Classify overthinking failure modes using the model as a judge.

    For each sample where a flip was detected (correct → incorrect),
    compares the last-correct trace with the final trace and classifies
    the failure mode.
    """

    def __init__(self, backend: Any = None):
        """
        Args:
            backend: A loaded TransformersBackend instance. If None,
                     uses heuristic-only classification.
        """
        self.backend = backend

    def classify(
        self,
        pid: str,
        query: str,
        ground_truth: str,
        last_correct_trace: str,
        last_correct_answer: Optional[str],
        final_trace: str,
        final_answer: Optional[str],
        last_correct_idx: Optional[int] = None,
    ) -> TaxonomyResult:
        """
        Classify the failure mode for a single flipped sample.
        """
        # Compute the added suffix (delta between traces)
        added_suffix = final_trace[len(last_correct_trace):].strip() if len(final_trace) > len(last_correct_trace) else final_trace

        # Try LLM-based classification first
        if self.backend is not None:
            result = self._llm_classify(
                pid=pid,
                query=query,
                ground_truth=ground_truth,
                last_correct_trace=last_correct_trace,
                last_correct_answer=last_correct_answer,
                final_trace=final_trace,
                final_answer=final_answer,
                added_suffix=added_suffix,
                last_correct_idx=last_correct_idx,
            )
            if result is not None:
                return result

        # Fall back to heuristic classification
        return self._heuristic_classify(
            pid=pid,
            query=query,
            ground_truth=ground_truth,
            last_correct_trace=last_correct_trace,
            last_correct_answer=last_correct_answer,
            final_trace=final_trace,
            final_answer=final_answer,
            added_suffix=added_suffix,
            last_correct_idx=last_correct_idx,
        )

    def _llm_classify(
        self, *, pid, query, ground_truth, last_correct_trace,
        last_correct_answer, final_trace, final_answer, added_suffix,
        last_correct_idx,
    ) -> Optional[TaxonomyResult]:
        """Use the model as a judge to classify the failure."""
        prompt = JUDGE_PROMPT.format(
            categories=", ".join(TAXONOMY_LABELS),
            category_guide=TAXONOMY_GUIDE,
            pid=pid,
            last_correct_idx=last_correct_idx,
            ground_truth=ground_truth,
            last_correct_answer=last_correct_answer or "N/A",
            final_answer=final_answer or "N/A",
            query=query,
            last_correct_trace=last_correct_trace,
            final_trace=final_trace,
            added_suffix=added_suffix,
        )

        try:
            raw_output = self.backend.generate_text(
                prompt=prompt,
                max_new_tokens=2048,
                temperature=0.1,
            )
            raw_output = raw_output.strip()

            # Remove any <think>...</think> blocks
            clean_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

            # Try to parse JSON
            json_match = re.search(r"\{.*\}", clean_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                category = parsed.get("category", "logical_error")
                if category not in TAXONOMY_LABELS:
                    category = "logical_error"

                return TaxonomyResult(
                    pid=pid,
                    category=category,
                    severity=int(parsed.get("severity", 50)),
                    went_wrong=parsed.get("went_wrong", ""),
                    evidence=parsed.get("evidence", ""),
                    last_correct_prefix_idx=last_correct_idx,
                    ground_truth=ground_truth,
                    last_correct_answer=last_correct_answer,
                    final_answer=final_answer,
                    raw_judge_output=raw_output,
                )
            else:
                print(f"\n[DEBUG] LLM Judge output did not contain JSON:\n{clean_output}")
        except Exception as e:
            print(f"\n[DEBUG] LLM Judge exception: {type(e).__name__}: {e}")
            if 'raw_output' in locals():
                print(f"[DEBUG] Raw output was:\n{raw_output}")

        return None

    def _heuristic_classify(
        self, *, pid, query, ground_truth, last_correct_trace,
        last_correct_answer, final_trace, final_answer, added_suffix,
        last_correct_idx,
    ) -> TaxonomyResult:
        """Heuristic-based classification when LLM judge is unavailable."""
        category = "logical_error"
        evidence = ""
        went_wrong = "Could not determine specific failure mode"

        # Check for degenerate repetition
        if added_suffix:
            lines = added_suffix.strip().split("\n")
            if len(lines) >= 3:
                unique_lines = set(line.strip() for line in lines if line.strip())
                if len(unique_lines) <= len(lines) // 3:
                    category = "degenerate_repetition"
                    went_wrong = "Model entered a repetitive loop"
                    evidence = lines[0][:100] if lines else ""

        # Check for calculation patterns in the suffix
        if category == "logical_error" and added_suffix:
            calc_patterns = re.findall(
                r"\d+\s*[+\-*/]\s*\d+\s*=\s*\d+", added_suffix
            )
            if calc_patterns:
                category = "calculation_error"
                went_wrong = "Arithmetic error in continued reasoning"
                evidence = calc_patterns[0]

        # Check for overthinking flip (correct answer was present but changed)
        if last_correct_answer and final_answer and last_correct_answer != final_answer:
            if "wait" in added_suffix.lower() or "actually" in added_suffix.lower() or "reconsider" in added_suffix.lower():
                category = "overthinking_flip"
                went_wrong = "Model second-guessed correct answer"
                evidence = added_suffix[:100]

        return TaxonomyResult(
            pid=pid,
            category=category,
            severity=50,
            went_wrong=went_wrong,
            evidence=evidence,
            last_correct_prefix_idx=last_correct_idx,
            ground_truth=ground_truth,
            last_correct_answer=last_correct_answer,
            final_answer=final_answer,
            raw_judge_output="heuristic",
        )

    def classify_batch(
        self,
        difficulty_results: list[Any],
    ) -> list[TaxonomyResult]:
        """Classify all flipped samples from difficulty results."""
        results = []
        for dr in difficulty_results:
            if not dr.flipped or not dr.prefix_results:
                continue

            last_correct_idx = dr.last_correct_prefix_idx
            if last_correct_idx is None:
                continue

            last_correct_pr = dr.prefix_results[last_correct_idx]
            final_pr = dr.prefix_results[-1]

            result = self.classify(
                pid=dr.pid,
                query=dr.query,
                ground_truth=dr.ground_truth,
                last_correct_trace=last_correct_pr.raw_text,
                last_correct_answer=last_correct_pr.extracted_answer,
                final_trace=final_pr.raw_text,
                final_answer=final_pr.extracted_answer,
                last_correct_idx=last_correct_idx,
            )
            results.append(result)

        return results
