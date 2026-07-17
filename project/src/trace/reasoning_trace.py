"""
ReasoningTrace

Core data structure for storing a complete reasoning trajectory.

Everything in the project revolves around this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

@dataclass
class TraceMetadata:
    model_name: str
    benchmark: str
    problem_id: str
    temperature: float
    max_new_tokens: int
    seed: int


# ---------------------------------------------------------
# Generation
# ---------------------------------------------------------

@dataclass
class GenerationData:

    reasoning_text: str = ""

    token_ids: List[int] = field(default_factory=list)

    tokens: List[str] = field(default_factory=list)

    top_k_logits: List[Dict[str, float]] = field(default_factory=list)

    entropy: List[float] = field(default_factory=list)

    token_times: List[float] = field(default_factory=list)


# ---------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------

@dataclass
class Checkpoint:

    token_position: int

    extracted_answer: Optional[str]

    is_correct: Optional[bool]

    confidence: Optional[float]

    answer_margin: Optional[float]


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

    checkpoints: List[Checkpoint] = field(default_factory=list)

    latent: LatentData = field(default_factory=LatentData)

    events: TraceEvents = field(default_factory=TraceEvents)

    outcome: TraceOutcome = field(default_factory=TraceOutcome)

    # -----------------------------------------------------
    # Generation
    # -----------------------------------------------------

    def add_token(
        self,
        token_id: int,
        token: str,
        entropy: Optional[float] = None,
        top_logits: Optional[Dict[str, float]] = None,
        token_time: Optional[float] = None,
    ):

        self.generation.token_ids.append(token_id)
        self.generation.tokens.append(token)

        self.generation.reasoning_text += token

        if entropy is not None:
            self.generation.entropy.append(entropy)

        if top_logits is not None:
            self.generation.top_k_logits.append(top_logits)

        if token_time is not None:
            self.generation.token_times.append(token_time)

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

    def add_checkpoint(self, checkpoint: Checkpoint):

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