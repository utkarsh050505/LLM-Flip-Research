"""
Reasoning Event

Phase 3: Discrete events detected along a reasoning trajectory.

Events represent significant state transitions identified from state vector
dynamics. They are NOT populated automatically by default — future detector
modules will emit them. This module defines only the data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ReasoningEventType(str, Enum):
    """Catalogue of detectable trajectory events."""
    STATE_SHIFT = "STATE_SHIFT"
    ANSWER_CHANGE = "ANSWER_CHANGE"
    ENTROPY_SPIKE = "ENTROPY_SPIKE"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"
    CONFIDENCE_DROP = "CONFIDENCE_DROP"
    LATENT_SHIFT = "LATENT_SHIFT"


@dataclass
class ReasoningEvent:
    """
    A single discrete event along the reasoning trajectory.

    event_type:       which kind of event was detected
    checkpoint_index: the checkpoint at which the event occurred
    timestamp:        wall-clock time (seconds since generation start)
    metadata:         arbitrary key-value payload (detector-specific)
    """
    event_type: str
    checkpoint_index: int
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
