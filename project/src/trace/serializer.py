"""
Trace Serializer

JSON-based save/load for ReasoningTrace objects.

Design decisions:
    - JSON over pickle/torch for human readability and cross-platform portability.
    - Hidden states within GenerationStep are excluded from JSON by default
      (they are large float vectors — will use .pt format for bulk analysis).
    - The serializer filters out empty placeholder lists to keep files clean.
    - TopKEntry objects are serialized as compact dicts.

Phase 2: Updated for step-based trace structure (List[GenerationStep]).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Union

from src.reasoning.reasoning_state import ReasoningState
from src.trace.reasoning_trace import (
    ReasoningTrace,
    TraceMetadata,
    GenerationData,
    GenerationStep,
    GenerationTiming,
    TopKEntry,
    ReasoningCheckpoint,
    FeaturePlaceholders,
    LatentData,
    TraceEvents,
    TraceOutcome,
)

logger = logging.getLogger("sbtf")


def _clean_step_for_json(step_dict: dict) -> dict:
    """
    Clean a single GenerationStep dict for JSON output.

    - Removes selected_hidden_states (large vectors, saved as .pt).
    - Removes None finish_reason for non-terminal steps.
    - Keeps all other fields.
    """
    cleaned = {}
    for key, value in step_dict.items():
        # Skip hidden states in JSON — they are saved separately
        if key == "selected_hidden_states":
            continue
        # Skip None finish_reason for non-terminal steps
        if key == "finish_reason" and value is None:
            continue
        cleaned[key] = value
    return cleaned


def _clean_for_json(data: dict) -> dict:
    """
    Recursively remove None values and empty lists from the dict
    to produce compact, readable JSON output.

    Preserves empty strings and zero values (they are meaningful).

    Args:
        data: Dictionary from dataclasses.asdict().

    Returns:
        Cleaned dictionary.
    """
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, dict):
            nested = _clean_for_json(value)
            if nested:
                cleaned[key] = nested
        elif isinstance(value, list):
            if len(value) > 0:
                # If it's a list of step dicts, clean each step
                if key == "steps" and value and isinstance(value[0], dict):
                    cleaned[key] = [
                        _clean_step_for_json(s) for s in value
                    ]
                else:
                    cleaned[key] = value
        elif value is not None:
            cleaned[key] = value
    return cleaned


def save_trace(
    trace: ReasoningTrace,
    path: Union[str, Path],
    compact: bool = False,
) -> Path:
    """
    Serialize a ReasoningTrace to a JSON file.

    Hidden state vectors within GenerationStep are excluded from JSON.
    They should be saved separately via save_hidden_states() for bulk analysis.

    Args:
        trace: The ReasoningTrace to save.
        path: File path (will be created, parents included).
        compact: If True, write minified JSON. Default is indented for readability.

    Returns:
        The Path where the trace was saved.

    Raises:
        TypeError: If trace is not a ReasoningTrace.
        OSError: If the file cannot be written.
    """
    if not isinstance(trace, ReasoningTrace):
        raise TypeError(
            f"Expected ReasoningTrace, got {type(trace).__name__}"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(trace)

    # Remove hidden_states from the latent section (tensor data)
    if "latent" in data and "hidden_states" in data["latent"]:
        data["latent"]["hidden_states"] = []

    cleaned = _clean_for_json(data)

    indent = None if compact else 2
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=indent, ensure_ascii=False)

    logger.info("Trace saved to %s (%.1f KB)", path, path.stat().st_size / 1024)
    return path


def load_trace(path: Union[str, Path]) -> ReasoningTrace:
    """
    Deserialize a ReasoningTrace from a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Reconstructed ReasoningTrace instance.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Reconstruct nested dataclasses from the JSON dict.

    metadata = TraceMetadata(**data.get("metadata", {}))

    # Reconstruct GenerationTiming
    timing_data = data.get("generation", {}).get("timing", {})
    timing = GenerationTiming(**timing_data)

    # Reconstruct GenerationSteps
    steps_raw = data.get("generation", {}).get("steps", [])
    steps = []
    for s in steps_raw:
        # Reconstruct TopKEntry objects
        top_k_raw = s.get("top_k_logits", [])
        top_k = [TopKEntry(**entry) for entry in top_k_raw]

        steps.append(GenerationStep(
            step_index=s.get("step_index", 0),
            generated_token=s.get("generated_token", ""),
            generated_token_id=s.get("generated_token_id", 0),
            timestamp=s.get("timestamp", 0.0),
            top_k_logits=top_k,
            entropy=s.get("entropy"),
            selected_hidden_states=s.get("selected_hidden_states", {}),
            finish_reason=s.get("finish_reason"),
        ))

    generation = GenerationData(
        reasoning_text=data.get("generation", {}).get("reasoning_text", ""),
        steps=steps,
        timing=timing,
    )

    checkpoints = []
    for cp in data.get("checkpoints", []):
        fp_data = cp.pop("feature_placeholders", {})
        fp = FeaturePlaceholders(**fp_data)
        
        rs_data = cp.pop("reasoning_state", {})
        rs = ReasoningState(**rs_data) if rs_data else None
        
        checkpoints.append(ReasoningCheckpoint(feature_placeholders=fp, reasoning_state=rs, **cp))

    latent = LatentData(**{
        k: v for k, v in data.get("latent", {}).items()
        if k in LatentData.__dataclass_fields__
    })

    events = TraceEvents(**{
        k: v for k, v in data.get("events", {}).items()
        if k in TraceEvents.__dataclass_fields__
    })

    outcome = TraceOutcome(**{
        k: v for k, v in data.get("outcome", {}).items()
        if k in TraceOutcome.__dataclass_fields__
    })

    trace = ReasoningTrace(
        metadata=metadata,
        generation=generation,
        checkpoints=checkpoints,
        latent=latent,
        events=events,
        outcome=outcome,
    )

    logger.info("Trace loaded from %s (%d steps)", path, len(steps))
    return trace
