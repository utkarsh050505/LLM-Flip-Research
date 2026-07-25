from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BackendCapabilities:
    supports_logits: bool
    supports_hidden_states: bool
    supports_kv_cache: bool
    supports_manual_step: bool
    supports_chat_template: bool

    @property
    def supports_full_pcc_branching(self) -> bool:
        return (
            self.supports_logits
            and self.supports_hidden_states
            and self.supports_kv_cache
            and self.supports_manual_step
        )


class ModelBackend(Protocol):
    capabilities: BackendCapabilities

    @property
    def model_id(self) -> str:
        ...

    @property
    def device(self) -> Any:
        ...

    @property
    def eos_token_id(self) -> int | None:
        ...

    def load(self) -> None:
        ...

    def encode_chat(self, prompt: str) -> dict[str, Any]:
        ...

    def tokenize_text(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
        ...

    def token_id_for_literal(self, text: str) -> int | None:
        ...

    def prefill(self, input_ids: Any, *, output_hidden_states: bool = False) -> Any:
        ...

    def step(
        self,
        input_ids: Any,
        past_key_values: Any,
        *,
        output_hidden_states: bool = False,
    ) -> Any:
        ...

    def clone_cache(self, past_key_values: Any) -> Any:
        ...
