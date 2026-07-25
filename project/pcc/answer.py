from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class AnswerEvaluation:
    raw_answer: str | None
    normalized_answer: str | None
    ground_truth: str
    correct: bool | None


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = str(answer).strip()
    answer = re.sub(r"\\text\{[^}]*\}", "", answer)
    answer = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", answer)
    answer = re.sub(r"\s+", "", answer)
    return re.sub(r"[^\dA-Za-z\.\-/]", "", answer)


def extract_boxed_answer(text: str) -> str | None:
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    for pos in range(start, len(text)):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos].strip()
    return None


def extract_final_answer(text: str) -> str | None:
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed

    final_patterns = [
        r"(?im)^\s*FINAL\s+ANSWER\s*:\s*(.+?)\s*$",
        r"(?i)(?:final\s+answer\s+is|the\s+answer\s+is)\s+([^\n\.]+)",
    ]
    for pattern in final_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    return None


def answer_appears(text: str, answer: str) -> bool:
    normalized = normalize_answer(answer)
    if normalized is None:
        return False
    escaped = re.escape(str(answer))
    marked_pattern = (
        rf"(?i)(?:final\s+answer\s+is|the\s+answer\s+is|answer\s*:\s*|=\s*|"
        rf"\\boxed\{{)\s*{escaped}\b"
    )
    if re.search(marked_pattern, text):
        return True
    extracted = extract_final_answer(text)
    return normalize_answer(extracted) == normalized


def evaluate_answer(text: str, ground_truth: str) -> AnswerEvaluation:
    raw = extract_final_answer(text)
    normalized = normalize_answer(raw)
    gt_normalized = normalize_answer(ground_truth) or ground_truth
    correct = None if normalized is None else normalized == gt_normalized
    return AnswerEvaluation(
        raw_answer=raw,
        normalized_answer=normalized,
        ground_truth=gt_normalized,
        correct=correct,
    )
