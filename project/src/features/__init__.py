"""
Features package.

Phase 3: Latent state representations for mechanistic interpretability.

Provides:
    CheckpointStateVector  — per-checkpoint decoder dynamics
    ReasoningTrajectory    — ordered trajectory of state vectors
    ReasoningPhase         — lifecycle stage labels (UNKNOWN by default)
    ReasoningEvent[Type]   — discrete trajectory events
"""

from src.features.state_vector import CheckpointStateVector
from src.features.trajectory import ReasoningTrajectory
from src.features.reasoning_phase import ReasoningPhase
from src.features.reasoning_event import ReasoningEvent, ReasoningEventType

__all__ = [
    "CheckpointStateVector",
    "ReasoningTrajectory",
    "ReasoningPhase",
    "ReasoningEvent",
    "ReasoningEventType",
]
