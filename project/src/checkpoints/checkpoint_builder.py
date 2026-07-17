"""
Checkpoint Builder

Phase 2.6: Creates semantic reasoning checkpoints over token-level GenerationSteps.
The checkpoint sequences now represent the model's actual reasoning trajectory.
"""
import re
import math
from typing import List, Optional, Tuple, Dict
import statistics

from src.trace.reasoning_trace import ReasoningTrace, ReasoningCheckpoint, FeaturePlaceholders, GenerationStep
from src.reasoning.reasoning_state import ReasoningStatus, ReasoningState


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
        # Look for explicit high-confidence markers first
        explicit_patterns = [
            (r'\\boxed{([^}]+)}', 0.99, "explicit_boxed"),
            (r'(?i)(?:final\s+)?answer\s*[:\=]\s*(.+?)(?:\n|$)', 0.95, "explicit_answer_marker"),
            (r'(?i)(?:therefore|hence|thus|result)\s*[:,\s]\s*(.+?)(?:\n|$)', 0.90, "explicit_conclusion"),
        ]

        # Stage 1: Explicit markers in the full text so far
        # (We check the full text because an answer might have been finalized in a previous window)
        best_candidate = None
        best_confidence = 0.0
        best_evidence = ""

        for pattern, conf, evidence in explicit_patterns:
            matches = re.finditer(pattern, full_text_so_far)
            # Take the last match as the most recent belief
            last_match = None
            for m in matches:
                last_match = m
            
            if last_match:
                # If we found a stronger confidence match, update
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
                notes=None
            )

        # Stage 2: Look for completed mathematical expressions in the *current window*
        # We don't want to extract random old numbers if the model is still reasoning
        math_patterns = [
            (r'([A-Za-z0-9_]+)\s*=\s*(-?\d+\.?\d*(?:/\d+)?)', 0.80, "equality_expression"),
            (r'(?:is|equals)\s+(-?\d+\.?\d*(?:/\d+)?)', 0.70, "semantic_equality"),
        ]
        
        for pattern, conf, evidence in math_patterns:
            matches = list(re.finditer(pattern, window_text))
            if matches:
                # Take the last match in the window
                last_match = matches[-1]
                candidate = last_match.group(2) if last_match.lastindex and last_match.lastindex >= 2 else last_match.group(1)
                return ReasoningState(
                    status=ReasoningStatus.HYPOTHESIS,
                    candidate_answer=candidate.strip(),
                    confidence=conf,
                    evidence=evidence,
                    notes=None
                )

        # Stage 3: If no explicit hypothesis or answer, we are just reasoning.
        # Do not extract plain numbers as answers.
        return ReasoningState(
            status=ReasoningStatus.REASONING,
            candidate_answer=None,
            confidence=0.0,
            evidence="intermediate_reasoning",
            notes=None
        )


class CheckpointBuilder:
    """
    Constructs ReasoningCheckpoints from raw GenerationSteps.
    """

    def __init__(self, window_size: int = 16):
        self.window_size = window_size

    def build_checkpoints(self, trace: ReasoningTrace) -> None:
        """
        Process the trace steps and append semantic checkpoints to the trace.
        """
        steps = trace.generation.steps
        if not steps:
            return

        trace.checkpoints = []
        num_steps = len(steps)
        
        previous_candidate = None
        answer_has_emerged = False

        for idx, start_idx in enumerate(range(0, num_steps, self.window_size)):
            end_idx = min(start_idx + self.window_size, num_steps)
            window_steps = steps[start_idx:end_idx]
            
            # Reconstruct text
            window_text = "".join(s.generated_token for s in window_steps)
            full_text_so_far = "".join(s.generated_token for s in steps[:end_idx])
            
            # Extract semantic state
            reasoning_state = SemanticExtractor.extract(window_text, full_text_so_far)
            
            current_candidate = reasoning_state.candidate_answer
            
            # Event Flags
            event_flags = {
                "answer_emerged": False,
                "answer_changed": False,
                "possible_final_answer": reasoning_state.status == ReasoningStatus.FINAL,
                "high_entropy": False,
                "low_confidence": False,
            }
            
            # Answer evolution logic (only based on candidate_answer changes)
            if current_candidate is not None:
                if not answer_has_emerged:
                    answer_has_emerged = True
                    event_flags["answer_emerged"] = True
                
                if previous_candidate is not None and current_candidate != previous_candidate:
                    event_flags["answer_changed"] = True
                    
                previous_candidate = current_candidate
            
            # Compute features for this window
            entropies = [s.entropy for s in window_steps if s.entropy is not None]
            if entropies:
                entropy_mean = sum(entropies) / len(entropies)
                entropy_max = max(entropies)
                entropy_std = statistics.stdev(entropies) if len(entropies) > 1 else 0.0
            else:
                entropy_mean = 0.0
                entropy_max = 0.0
                entropy_std = 0.0
                
            # Confidence is derived from top-1 probability
            confidences = []
            for s in window_steps:
                if s.top_k_logits and len(s.top_k_logits) > 0:
                    confidences.append(s.top_k_logits[0].probability)
            
            confidence_mean = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Set threshold flags
            if entropy_mean > 2.0 or entropy_max > 3.0:
                event_flags["high_entropy"] = True
            if confidence_mean < 0.5:
                event_flags["low_confidence"] = True
            
            # Calculate timestamp (time elapsed since start of trace)
            timestamp = window_steps[-1].timestamp if window_steps else 0.0

            checkpoint = ReasoningCheckpoint(
                checkpoint_index=idx,
                start_step=start_idx,
                end_step=end_idx - 1, # Inclusive
                window_text=window_text,
                reasoning_state=reasoning_state,
                event_flags=event_flags,
                entropy_mean=round(entropy_mean, 6),
                entropy_max=round(entropy_max, 6),
                entropy_std=round(entropy_std, 6),
                confidence_mean=round(confidence_mean, 6),
                token_count=end_idx - start_idx,
                timestamp=round(timestamp, 6),
                feature_placeholders=FeaturePlaceholders()
            )
            trace.add_checkpoint(checkpoint)
