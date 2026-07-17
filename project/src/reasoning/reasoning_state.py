"""
Reasoning State

Phase 2.6: Provides semantic state representation for reasoning checkpoints.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ReasoningStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    REASONING = "REASONING"
    HYPOTHESIS = "HYPOTHESIS"
    FINAL = "FINAL"
    ERROR = "ERROR"


@dataclass
class ReasoningState:
    """
    Semantic state of reasoning at a given checkpoint.
    
    status: The current stage of reasoning (UNKNOWN, REASONING, HYPOTHESIS, FINAL, ERROR).
    candidate_answer: The extracted belief or answer (if any).
    confidence: Extractor heuristic confidence (0.0 to 1.0).
    evidence: Description of why this answer/status was chosen.
    notes: Any additional notes.
    """
    status: str
    candidate_answer: Optional[str]
    confidence: float
    evidence: str
    notes: Optional[str] = None
