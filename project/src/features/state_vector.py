"""
Checkpoint State Vector

Phase 3: Latent representation for each reasoning checkpoint.

A CheckpointStateVector captures the decoder dynamics (entropy, confidence,
logit margins, velocity, acceleration) at a checkpoint boundary. These
scalar features are text-independent and will later be augmented or replaced
with dense hidden-state embeddings.

The central insight: PCC is a latent state phenomenon, not a text phenomenon.
State vectors allow trajectory analysis without parsing generated text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CheckpointStateVector:
    """
    Latent feature vector summarising decoder dynamics for one checkpoint window.

    Scalar features (Phase 3):
        entropy_mean / entropy_std / entropy_trend
        confidence_mean / confidence_trend
        logit_margin_mean / logit_margin_std
        topk_concentration
        token_rate

    Dynamics (computed across consecutive checkpoints):
        state_velocity   — first-order change from the previous checkpoint
        state_acceleration — second-order change (velocity of velocity)

    Tensor references (populated in future phases):
        hidden_state_ref  — path / key to a stored hidden-state tensor
        attention_ref     — path / key to stored attention patterns
        activation_ref    — path / key to stored activation vectors

    future_feature_placeholders — dict for arbitrary downstream features
    """

    # ---- window-level scalar features ----
    entropy_mean: float = 0.0
    entropy_std: float = 0.0
    entropy_trend: float = 0.0          # slope across the window

    confidence_mean: float = 0.0
    confidence_trend: float = 0.0       # slope across the window

    logit_margin_mean: float = 0.0      # mean (top1 - top2) logit gap
    logit_margin_std: float = 0.0

    topk_concentration: float = 0.0     # sum of top-k probabilities
    token_rate: float = 0.0             # tokens per second in this window

    # ---- dynamics (inter-checkpoint) ----
    state_velocity: float = 0.0         # L2 of feature deltas from prev checkpoint
    state_acceleration: float = 0.0     # delta of velocity from prev checkpoint

    # ---- tensor references (future phases) ----
    hidden_state_ref: Optional[str] = None
    attention_ref: Optional[str] = None
    activation_ref: Optional[str] = None

    # ---- extensibility ----
    future_feature_placeholders: Dict[str, Any] = field(default_factory=dict)

    # ---- convenience ----

    def to_feature_list(self) -> List[float]:
        """Return a flat list of all scalar features (for distance computation)."""
        return [
            self.entropy_mean,
            self.entropy_std,
            self.entropy_trend,
            self.confidence_mean,
            self.confidence_trend,
            self.logit_margin_mean,
            self.logit_margin_std,
            self.topk_concentration,
            self.token_rate,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        """Labels matching to_feature_list() ordering."""
        return [
            "entropy_mean",
            "entropy_std",
            "entropy_trend",
            "confidence_mean",
            "confidence_trend",
            "logit_margin_mean",
            "logit_margin_std",
            "topk_concentration",
            "token_rate",
        ]
