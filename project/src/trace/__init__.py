"""
Trace package.

Core data structures for reasoning trace collection and serialization.
"""

from src.trace.reasoning_trace import (
    ReasoningTrace,
    TraceMetadata,
    GenerationData,
    GenerationStep,
    GenerationTiming,
    TopKEntry,
    ReasoningCheckpoint,
    FeaturePlaceholders,
    LatentData,
    TraceEvents,
    TraceOutcome,
)
from src.trace.serializer import save_trace, load_trace

__all__ = [
    "ReasoningTrace",
    "TraceMetadata",
    "GenerationData",
    "GenerationStep",
    "GenerationTiming",
    "TopKEntry",
    "ReasoningCheckpoint",
    "FeaturePlaceholders",
    "LatentData",
    "TraceEvents",
    "TraceOutcome",
    "save_trace",
    "load_trace",
]
