"""
ReasoningTrace

Core data structure for storing a complete reasoning trajectory.

Everything in the project revolves around this object.

Phase 2: Replaced flat token_ids/tokens lists with a structured
List[GenerationStep] that captures per-token instrumentation data
(logits, entropy, hidden states, timing) at every decoding step.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.reasoning.reasoning_state import ReasoningState
from src.features.state_vector import CheckpointStateVector
from src.features.reasoning_phase import ReasoningPhase
from src.features.reasoning_event import ReasoningEvent
from src.features.trajectory import ReasoningTrajectory


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

@dataclass
class TraceMetadata:
    """Provenance metadata for a single reasoning trace."""

    model_name: str
    benchmark: str
    problem_id: str
    prompt_text: str
    temperature: float
    max_new_tokens: int
    seed: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------
# Generation Step (Phase 2)
# ---------------------------------------------------------

@dataclass
class TopKEntry:
    """A single entry in the top-k logits for a decoding step."""

    token: str
    token_id: int
    logit: float
    probability: float


@dataclass
class GenerationStep:
    """
    Per-token instrumentation record from one forward pass.

    This is the core data unit for mechanistic interpretability.
    Every decoding step produces exactly one GenerationStep.

    Fields:
        step_index: Zero-based position in the generated sequence.
        generated_token: Decoded string for this token.
        generated_token_id: Vocabulary index of the selected token.
        timestamp: Wall-clock time in seconds since generation start.
        top_k_logits: Top-k tokens with their logits and probabilities.
        entropy: Shannon entropy of the full probability distribution (nats).
        selected_hidden_states: Dict mapping layer name to hidden state vector.
                                Empty until explicitly populated.
        finish_reason: Why generation stopped at this step (None if not final).
    """

    step_index: int
    generated_token: str
    generated_token_id: int
    timestamp: float

    top_k_logits: List[TopKEntry] = field(default_factory=list)
    entropy: Optional[float] = None

    selected_hidden_states: Dict[str, Any] = field(default_factory=dict)

    finish_reason: Optional[str] = None


# ---------------------------------------------------------
# Generation Timing
# ---------------------------------------------------------

@dataclass
class GenerationTiming:
    """Wall-clock timing metrics for a single generation run."""

    total_seconds: float = 0.0
    tokens_per_second: float = 0.0
    num_generated_tokens: int = 0
    prefill_seconds: float = 0.0


# ---------------------------------------------------------
# Generation Data
# ---------------------------------------------------------

@dataclass
class GenerationData:
    """
    Complete generation output.

    Phase 2: The primary data is now in `steps`. The `reasoning_text`
    field is reconstructed from steps for convenience. The old
    `token_ids` and `tokens` flat lists have been replaced by
    structured GenerationStep objects.
    """

    reasoning_text: str = ""

    steps: List[GenerationStep] = field(default_factory=list)

    timing: GenerationTiming = field(default_factory=GenerationTiming)


# ---------------------------------------------------------
# Checkpoints (Phase 2.5 / 3)
# ---------------------------------------------------------

@dataclass
class FeaturePlaceholders:
    hidden_state_ref: Optional[str] = None
    latent_ref: Optional[str] = None
    mechanistic_features: Optional[Dict[str, Any]] = None

@dataclass
class ReasoningCheckpoint:
    checkpoint_index: int
    start_step: int
    end_step: int
    window_text: str

    reasoning_state: ReasoningState

    entropy_mean: float
    entropy_max: float
    entropy_std: float
    confidence_mean: float
    token_count: int
    timestamp: float

    # Phase 3: latent state representation
    state_vector: Optional[CheckpointStateVector] = None
    reasoning_phase: str = ReasoningPhase.UNKNOWN
    events: List[ReasoningEvent] = field(default_factory=list)

    event_flags: Dict[str, bool] = field(default_factory=dict)
    feature_placeholders: FeaturePlaceholders = field(default_factory=FeaturePlaceholders)


# ---------------------------------------------------------
# Latent Features
# ---------------------------------------------------------

@dataclass
class LatentData:

    hidden_states: List[Any] = field(default_factory=list)

    l2_transition: List[float] = field(default_factory=list)

    cosine_transition: List[float] = field(default_factory=list)

    curvature: List[float] = field(default_factory=list)


# ---------------------------------------------------------
# Events
# ---------------------------------------------------------

@dataclass
class TraceEvents:

    first_correct_solution: Optional[int] = None

    post_correctness_collapse: Optional[int] = None

    recovery_points: List[int] = field(default_factory=list)

    answer_change_points: List[int] = field(default_factory=list)

    hesitation_points: List[int] = field(default_factory=list)


# ---------------------------------------------------------
# Outcome
# ---------------------------------------------------------

@dataclass
class TraceOutcome:

    final_answer: Optional[str] = None

    final_correct: Optional[bool] = None

    trajectory_type: Optional[str] = None


# ---------------------------------------------------------
# Main Trace
# ---------------------------------------------------------

@dataclass
class ReasoningTrace:

    metadata: TraceMetadata

    generation: GenerationData = field(default_factory=GenerationData)

    checkpoints: List[ReasoningCheckpoint] = field(default_factory=list)

    latent: LatentData = field(default_factory=LatentData)

    events: TraceEvents = field(default_factory=TraceEvents)

    outcome: TraceOutcome = field(default_factory=TraceOutcome)

    # Phase 3: trajectory-level latent representation
    trajectory: Optional[ReasoningTrajectory] = None

    # -----------------------------------------------------
    # Step-Level Access (Phase 2)
    # -----------------------------------------------------

    def add_step(self, step: GenerationStep) -> None:
        """
        Append a GenerationStep and update reasoning_text.

        This is the primary way the decoder populates a trace.
        """
        self.generation.steps.append(step)
        self.generation.reasoning_text += step.generated_token

    @property
    def token_ids(self) -> List[int]:
        """Convenience: extract flat list of token IDs from steps."""
        return [s.generated_token_id for s in self.generation.steps]

    @property
    def tokens(self) -> List[str]:
        """Convenience: extract flat list of decoded tokens from steps."""
        return [s.generated_token for s in self.generation.steps]

    @property
    def entropies(self) -> List[Optional[float]]:
        """Convenience: extract entropy values from steps."""
        return [s.entropy for s in self.generation.steps]

    @property
    def num_steps(self) -> int:
        """Number of generated steps."""
        return len(self.generation.steps)

    # -----------------------------------------------------
    # Latent
    # -----------------------------------------------------

    def add_hidden_state(
        self,
        hidden_state,
        l2=None,
        cosine=None,
        curvature=None,
    ):

        self.latent.hidden_states.append(hidden_state)

        if l2 is not None:
            self.latent.l2_transition.append(l2)

        if cosine is not None:
            self.latent.cosine_transition.append(cosine)

        if curvature is not None:
            self.latent.curvature.append(curvature)

    # -----------------------------------------------------
    # Checkpoints
    # -----------------------------------------------------

    def add_checkpoint(self, checkpoint: ReasoningCheckpoint):

        self.checkpoints.append(checkpoint)

    # -----------------------------------------------------
    # Outcome
    # -----------------------------------------------------

    def finalize(
        self,
        final_answer: str,
        final_correct: bool,
        trajectory_type: str,
    ):

        self.outcome.final_answer = final_answer
        self.outcome.final_correct = final_correct
        self.outcome.trajectory_type = trajectory_type

    # -----------------------------------------------------
    # Serialization
    # -----------------------------------------------------

    def to_dict(self):

        return asdict(self)