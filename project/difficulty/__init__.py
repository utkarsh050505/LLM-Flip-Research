"""
Difficulty — prefix-continuation analysis for detecting overthinking.

Core experiment from 'Thinking Past the Answer': split reasoning traces
into progressively longer prefixes, force an early answer at each prefix,
and measure accuracy over prefix length.
"""
from __future__ import annotations

from difficulty.config import DifficultyConfig
from difficulty.splitting import split_with_granularity
from difficulty.experiment import DifficultyExperiment, DifficultyResult, PrefixResult

__all__ = [
    "DifficultyConfig",
    "DifficultyExperiment",
    "DifficultyResult",
    "PrefixResult",
    "split_with_granularity",
]
