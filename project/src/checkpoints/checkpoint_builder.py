"""
Checkpoint Builder

Phase 3: Creates semantic reasoning checkpoints with latent state vectors.

The builder now produces both the semantic layer (ReasoningState, window text,
event flags) AND the latent layer (CheckpointStateVector, velocity,
acceleration). After all checkpoints are built, it constructs a
ReasoningTrajectory that aggregates trajectory-level statistics.
"""
import math
import re
import statistics
import uuid
from typing import List, Optional, Dict

from src.trace.reasoning_trace import (
    ReasoningTrace,
    ReasoningCheckpoint,
    FeaturePlaceholders,
    GenerationStep,
)
from src.reasoning.reasoning_state import ReasoningStatus, ReasoningState
from src.features.state_vector import CheckpointStateVector
from src.features.reasoning_phase import ReasoningPhase
from src.features.reasoning_event import ReasoningEvent, ReasoningEventType
from src.features.trajectory import ReasoningTrajectory


# ================================================================
# Semantic Extractor (Phase 2.6 — preserved)
# ================================================================

class SemanticExtractor:
    """
    Multi-stage heuristic extractor to identify the semantic belief state.
    This does NOT use an LLM. It operates strictly offline using explicit rules.
    """

    @staticmethod
    def extract(window_text: str, full_text_so_far: str) -> ReasoningState:
        """
        Extract the reasoning state by examining both the current window and full context.

        Stage 1: Explicit markers (e.g., Final answer)
        Stage 2: Mathematical expressions/equalities
        Stage 3: Intermediate calculations -> REASONING
        Stage 4: Multiple candidates -> strongest one
        """
        explicit_patterns = [
            (r'\\boxed{([^}]+)}', 0.99, "explicit_boxed"),
            (r'(?i)(?:final\s+)?answer\s*[:\=]\s*(.+?)(?:\n|$)', 0.95, "explicit_answer_marker"),
            (r'(?i)(?:therefore|hence|thus|result)\s*[:,\s]\s*(.+?)(?:\n|$)', 0.90, "explicit_conclusion"),
        ]

        best_candidate = None
        best_confidence = 0.0
        best_evidence = ""

        for pattern, conf, evidence in explicit_patterns:
            matches = re.finditer(pattern, full_text_so_far)
            last_match = None
            for m in matches:
                last_match = m

            if last_match:
                if conf > best_confidence:
                    best_candidate = last_match.group(1).strip()
                    best_confidence = conf
                    best_evidence = evidence

        if best_candidate:
            status = ReasoningStatus.FINAL if best_confidence >= 0.95 else ReasoningStatus.HYPOTHESIS
            return ReasoningState(
                status=status,
                candidate_answer=best_candidate,
                confidence=best_confidence,
                evidence=best_evidence,
                notes=None,
            )

        math_patterns = [
            (r'([A-Za-z0-9_]+)\s*=\s*(-?\d+\.?\d*(?:/\d+)?)', 0.80, "equality_expression"),
            (r'(?:is|equals)\s+(-?\d+\.?\d*(?:/\d+)?)', 0.70, "semantic_equality"),
        ]

        for pattern, conf, evidence in math_patterns:
            matches = list(re.finditer(pattern, window_text))
            if matches:
                last_match = matches[-1]
                candidate = (
                    last_match.group(2)
                    if last_match.lastindex and last_match.lastindex >= 2
                    else last_match.group(1)
                )
                return ReasoningState(
                    status=ReasoningStatus.HYPOTHESIS,
                    candidate_answer=candidate.strip(),
                    confidence=conf,
                    evidence=evidence,
                    notes=None,
                )

        return ReasoningState(
            status=ReasoningStatus.REASONING,
            candidate_answer=None,
            confidence=0.0,
            evidence="intermediate_reasoning",
            notes=None,
        )


# ================================================================
# State Vector Computation Helpers
# ================================================================

def _linear_trend(values: List[float]) -> float:
    """
    Compute the slope of a least-squares linear fit over a sequence.
    Returns 0.0 if there are fewer than 2 data points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _compute_logit_margins(steps: List[GenerationStep]) -> List[float]:
    """
    Compute the logit margin (top-1 logit - top-2 logit) for each step.
    """
    margins = []
    for s in steps:
        if s.top_k_logits and len(s.top_k_logits) >= 2:
            margins.append(s.top_k_logits[0].logit - s.top_k_logits[1].logit)
    return margins


def _compute_topk_concentration(steps: List[GenerationStep], k: int = 5) -> float:
    """
    Average fraction of probability mass captured by the top-k tokens.
    """
    concentrations = []
    for s in steps:
        if s.top_k_logits:
            total = sum(entry.probability for entry in s.top_k_logits[:k])
            concentrations.append(total)
    return sum(concentrations) / len(concentrations) if concentrations else 0.0


def _compute_state_velocity(
    current: CheckpointStateVector,
    previous: CheckpointStateVector,
) -> float:
    """
    L2 distance between the scalar feature vectors of two consecutive checkpoints.
    This is a proxy for state velocity until hidden-state embeddings are available.
    """
    cur = current.to_feature_list()
    prev = previous.to_feature_list()
    return math.sqrt(sum((c - p) ** 2 for c, p in zip(cur, prev)))


# ================================================================
# CheckpointBuilder
# ================================================================

class CheckpointBuilder:
    """
    Constructs ReasoningCheckpoints with both semantic and latent layers,
    then assembles a ReasoningTrajectory.
    """

    def __init__(self, window_size: int = 16):
        self.window_size = window_size

    def build_checkpoints(self, trace: ReasoningTrace) -> None:
        """
        Process the trace steps, build checkpoints with state vectors,
        compute dynamics, and attach a ReasoningTrajectory to the trace.
        """
        steps = trace.generation.steps
        if not steps:
            return

        trace.checkpoints = []
        num_steps = len(steps)

        previous_candidate = None
        answer_has_emerged = False
        prev_vector: Optional[CheckpointStateVector] = None
        prev_velocity: float = 0.0

        for idx, start_idx in enumerate(range(0, num_steps, self.window_size)):
            end_idx = min(start_idx + self.window_size, num_steps)
            window_steps = steps[start_idx:end_idx]

            # ---- text layer (Phase 2.6) ----
            window_text = "".join(s.generated_token for s in window_steps)
            full_text_so_far = "".join(s.generated_token for s in steps[:end_idx])
            reasoning_state = SemanticExtractor.extract(window_text, full_text_so_far)
            current_candidate = reasoning_state.candidate_answer

            # ---- event flags ----
            event_flags: Dict[str, bool] = {
                "answer_emerged": False,
                "answer_changed": False,
                "possible_final_answer": reasoning_state.status == ReasoningStatus.FINAL,
                "high_entropy": False,
                "low_confidence": False,
            }

            if current_candidate is not None:
                if not answer_has_emerged:
                    answer_has_emerged = True
                    event_flags["answer_emerged"] = True
                if previous_candidate is not None and current_candidate != previous_candidate:
                    event_flags["answer_changed"] = True
                previous_candidate = current_candidate

            # ---- scalar features ----
            entropies = [s.entropy for s in window_steps if s.entropy is not None]
            if entropies:
                entropy_mean = sum(entropies) / len(entropies)
                entropy_max = max(entropies)
                entropy_std = statistics.stdev(entropies) if len(entropies) > 1 else 0.0
                entropy_trend = _linear_trend(entropies)
            else:
                entropy_mean = 0.0
                entropy_max = 0.0
                entropy_std = 0.0
                entropy_trend = 0.0

            confidences = []
            for s in window_steps:
                if s.top_k_logits and len(s.top_k_logits) > 0:
                    confidences.append(s.top_k_logits[0].probability)
            confidence_mean = sum(confidences) / len(confidences) if confidences else 0.0
            confidence_trend = _linear_trend(confidences) if confidences else 0.0

            logit_margins = _compute_logit_margins(window_steps)
            logit_margin_mean = sum(logit_margins) / len(logit_margins) if logit_margins else 0.0
            logit_margin_std = statistics.stdev(logit_margins) if len(logit_margins) > 1 else 0.0

            topk_concentration = _compute_topk_concentration(window_steps)

            # Token rate for this window
            if len(window_steps) >= 2:
                dt = window_steps[-1].timestamp - window_steps[0].timestamp
                token_rate = len(window_steps) / dt if dt > 0 else 0.0
            else:
                token_rate = 0.0

            # ---- state vector ----
            state_vector = CheckpointStateVector(
                entropy_mean=round(entropy_mean, 6),
                entropy_std=round(entropy_std, 6),
                entropy_trend=round(entropy_trend, 6),
                confidence_mean=round(confidence_mean, 6),
                confidence_trend=round(confidence_trend, 6),
                logit_margin_mean=round(logit_margin_mean, 6),
                logit_margin_std=round(logit_margin_std, 6),
                topk_concentration=round(topk_concentration, 6),
                token_rate=round(token_rate, 4),
            )

            # ---- dynamics ----
            if prev_vector is not None:
                velocity = _compute_state_velocity(state_vector, prev_vector)
                state_vector.state_velocity = round(velocity, 6)
                acceleration = velocity - prev_velocity
                state_vector.state_acceleration = round(acceleration, 6)
                prev_velocity = velocity
            else:
                state_vector.state_velocity = 0.0
                state_vector.state_acceleration = 0.0
                prev_velocity = 0.0

            prev_vector = state_vector

            # ---- threshold-based event flags ----
            if entropy_mean > 2.0 or entropy_max > 3.0:
                event_flags["high_entropy"] = True
            if confidence_mean < 0.5:
                event_flags["low_confidence"] = True

            timestamp = window_steps[-1].timestamp if window_steps else 0.0

            # ---- assemble checkpoint ----
            checkpoint = ReasoningCheckpoint(
                checkpoint_index=idx,
                start_step=start_idx,
                end_step=end_idx - 1,
                window_text=window_text,
                reasoning_state=reasoning_state,
                entropy_mean=round(entropy_mean, 6),
                entropy_max=round(entropy_max, 6),
                entropy_std=round(entropy_std, 6),
                confidence_mean=round(confidence_mean, 6),
                token_count=end_idx - start_idx,
                timestamp=round(timestamp, 6),
                state_vector=state_vector,
                reasoning_phase=ReasoningPhase.UNKNOWN,
                events=[],
                event_flags=event_flags,
                feature_placeholders=FeaturePlaceholders(),
            )
            trace.add_checkpoint(checkpoint)

        # ---- build trajectory ----
        trajectory = ReasoningTrajectory(
            trajectory_id=uuid.uuid4().hex[:12],
            checkpoint_vectors=[cp.state_vector for cp in trace.checkpoints],
        )
        if trace.checkpoints:
            trajectory.trajectory_duration = trace.checkpoints[-1].timestamp
        trajectory.compute_statistics()
        trace.trajectory = trajectory
