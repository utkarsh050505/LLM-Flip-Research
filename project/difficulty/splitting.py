"""
Splitting — split reasoning traces into progressively larger prefixes.

Direct implementation of split_with_granularity from thinking-past-the-answer.
"""
from __future__ import annotations

from typing import Any, Optional


def split_with_granularity(
    text: str,
    tokenizer: Any = None,
    difficulty_level: str = "utterance",
    granularity: int = 1,
) -> list[str]:
    """
    Split text into progressively larger cumulative prefixes.

    Args:
        text: The reasoning trace text to split.
        tokenizer: Tokenizer instance (required for 'token' level).
        difficulty_level: "utterance" (split by newlines) or "token" (split by tokens).
        granularity: Number of utterances or tokens per segment.

    Returns:
        List of progressively larger text prefixes. Each prefix contains
        all preceding segments plus the current one.

    Example with utterance level, granularity=1:
        Input: "Line 1\\nLine 2\\nLine 3"
        Output: ["Line 1", "Line 1\\nLine 2", "Line 1\\nLine 2\\nLine 3"]
    """
    if difficulty_level == "utterance":
        return _split_utterance(text, granularity)
    elif difficulty_level == "token":
        if tokenizer is None:
            raise ValueError("tokenizer is required for token-level splitting")
        return _split_token(text, tokenizer, granularity)
    else:
        raise ValueError(
            f"Invalid difficulty level: {difficulty_level!r}. "
            "Must be 'utterance' or 'token'."
        )


def _split_utterance(text: str, granularity: int) -> list[str]:
    """Split by newlines, grouping by granularity."""
    lines = text.split("\n")
    # Group lines into segments of size `granularity`
    segments = [
        "\n".join(lines[i : i + granularity])
        for i in range(0, len(lines), granularity)
    ]

    # Build cumulative prefixes
    result = []
    for i in range(1, len(segments) + 1):
        result.append("\n".join(segments[:i]))

    return result


def _split_token(text: str, tokenizer: Any, granularity: int) -> list[str]:
    """Split by tokens, grouping by granularity."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Build cumulative prefixes
    result = []
    for i in range(granularity, len(token_ids) + 1, granularity):
        prefix_ids = token_ids[:i]
        result.append(tokenizer.decode(prefix_ids))

    # Add the full text as the last prefix if not already included
    if result and len(token_ids) % granularity != 0:
        result.append(tokenizer.decode(token_ids))

    return result


def count_segments(
    text: str,
    difficulty_level: str = "utterance",
    granularity: int = 1,
    tokenizer: Any = None,
) -> int:
    """Return the number of prefix segments that would be generated."""
    return len(split_with_granularity(text, tokenizer, difficulty_level, granularity))
