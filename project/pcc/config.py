from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReasoningFormat:
    name: str = "deepseek_r1"
    think_close: str | None = "</think>"
    final_answer_prompt: str = "</think>\n\n**Final Answer:**\n\\boxed{"
    requires_visible_reasoning: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    backend: str = "transformers"
    model_id: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device_map: str = "auto"
    dtype: str = "bfloat16"
    quantization: str | None = None
    attn_implementation: str | None = "sdpa"
    max_fcs_tokens: int = 8000
    fcs_attempts: int = 6
    min_tokens_before_fcs: int = 15
    num_branches: int = 6
    branch_temperature: float = 1.0
    fcs_temperature: float = 0.7
    budget_multiplier: int = 3
    min_target_budget: int = 400
    max_target_budget: int = 4000
    extra_branch_tokens: int = 1000
    check_every: int = 4
    conclude_probability_threshold: float = 0.05
    max_repeated_boxed: int = 3
    output_jsonl: str = "project/results/pcc_branch_runs.jsonl"
    reasoning_format: ReasoningFormat = field(default_factory=ReasoningFormat)
    wait_phrases: tuple[str, ...] = (
        " Wait,",
        " Wait, actually, let me try solving this a completely different way to check:",
        " Wait, let me question my core assumption here and see if another interpretation fits:",
    )


def load_experiment_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return _config_from_mapping(data)


def _config_from_mapping(data: dict[str, Any]) -> ExperimentConfig:
    reasoning_data = data.pop("reasoning_format", None)
    if reasoning_data is not None:
        if not isinstance(reasoning_data, dict):
            raise ValueError("reasoning_format must be an object")
        data["reasoning_format"] = ReasoningFormat(**reasoning_data)

    valid_names = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        raise ValueError(f"Unknown experiment config keys: {unknown}")

    if "wait_phrases" in data:
        data["wait_phrases"] = tuple(data["wait_phrases"])
    return ExperimentConfig(**data)
