"""
Reasoning Trajectory

Phase 3: The primary scientific object for mechanistic interpretability.

A ReasoningTrajectory is a sequence of CheckpointStateVectors enriched with
trajectory-level statistics (mean entropy, velocity profile, acceleration
profile, etc.). It is the central data structure for downstream analysis of
Post-Correctness Collapse, belief drift, and latent phase transitions.

Text is NOT required to construct a trajectory — it operates entirely on
decoder dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.features.state_vector import CheckpointStateVector


@dataclass
class ReasoningTrajectory:
    """
    Ordered sequence of state vectors with trajectory-level statistics.

    Built from a trace's checkpoints after state vectors have been computed.
    Future analyses (PCC detection, hazard prediction) operate on this object
    rather than on raw tokens or text.
    """

    trajectory_id: str = ""

    # ---- ordered state vectors ----
    checkpoint_vectors: List[CheckpointStateVector] = field(default_factory=list)

    # ---- trajectory-level aggregate statistics ----
    trajectory_length: int = 0               # number of checkpoints
    trajectory_duration: float = 0.0         # wall-clock seconds

    avg_entropy: float = 0.0
    max_entropy: float = 0.0
    entropy_variance: float = 0.0

    avg_confidence: float = 0.0

    # ---- dynamics profiles (one value per inter-checkpoint gap) ----
    velocity_profile: List[float] = field(default_factory=list)
    acceleration_profile: List[float] = field(default_factory=list)

    # ---- extensibility ----
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Construction helpers
    # ----------------------------------------------------------------

    def compute_statistics(self) -> None:
        """
        Populate aggregate statistics from the checkpoint_vectors list.
        Call this after all vectors have been appended.
        """
        n = len(self.checkpoint_vectors)
        self.trajectory_length = n
        if n == 0:
            return

        entropies = [v.entropy_mean for v in self.checkpoint_vectors]
        confidences = [v.confidence_mean for v in self.checkpoint_vectors]

        self.avg_entropy = sum(entropies) / n
        self.max_entropy = max(entropies)
        mean_e = self.avg_entropy
        self.entropy_variance = sum((e - mean_e) ** 2 for e in entropies) / n

        self.avg_confidence = sum(confidences) / n

        # Velocity / acceleration profiles are already stored per-vector;
        # collect them into the trajectory-level profiles.
        self.velocity_profile = [v.state_velocity for v in self.checkpoint_vectors]
        self.acceleration_profile = [v.state_acceleration for v in self.checkpoint_vectors]

        # Duration is the last checkpoint's token_rate-based estimate or
        # supplied externally — we leave it to the builder to set.
