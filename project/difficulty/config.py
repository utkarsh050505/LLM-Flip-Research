"""
DifficultyConfig — configuration for difficulty prefix-continuation experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DifficultyConfig:
    """Configuration for the difficulty prefix-continuation experiment."""

    # Splitting strategy: "utterance" (split by newlines) or "token" (split by tokens)
    difficulty_level: str = "utterance"

    # Number of utterances or tokens per segment
    granularity: int = 1

    # Budget-forcing prompt appended after each prefix to force early answer
    budget_forcing_prompt: str = (
        "</think>\n\n**Final Answer:**\n\\boxed{"
    )

    # The prompt string used in the output path construction
    path_budget_forcing_prompt: str = "Final_Answer"

    # Maximum tokens to generate for each continuation
    max_continuation_tokens: int = 512

    # Temperature for continuation generation
    continuation_temperature: float = 0.1

    # Maximum number of prefixes to evaluate per sample (prevents OOM on long traces)
    max_prefixes: int = 15

    # Maximum prefix length in characters (longer prefixes are skipped to prevent OOM)
    max_prefix_chars: int = 3000

    # Seed for reproducibility
    seed: int = 42

    # Experiment name for output organization
    experiment_name: str = "main"
