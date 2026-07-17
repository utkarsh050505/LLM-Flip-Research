"""
Reasoning Phase

Phase 3: Lifecycle labels for the stages of a model's reasoning trajectory.

These are NOT automatically classified — they default to UNKNOWN.
Future phases will implement phase-transition detection algorithms that
populate these labels from state vector dynamics.
"""

from enum import Enum


class ReasoningPhase(str, Enum):
    """
    Semantic labels for the stage of a reasoning trajectory.

    UNKNOWN     — not yet classified (default)
    EXPLORATION — model is exploring the problem space
    NARROWING   — model is converging toward a specific approach
    COMMITMENT  — model has committed to a solution path
    STABLE      — model is confidently elaborating a committed answer
    REVISION    — model is revising its previous approach
    COLLAPSE    — model has abandoned a previously correct path
    """
    UNKNOWN = "UNKNOWN"
    EXPLORATION = "EXPLORATION"
    NARROWING = "NARROWING"
    COMMITMENT = "COMMITMENT"
    STABLE = "STABLE"
    REVISION = "REVISION"
    COLLAPSE = "COLLAPSE"
