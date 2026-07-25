from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .base import BackendCapabilities, BackendError


@dataclass(frozen=True)
class TransformersBackendConfig:
    model_id: str
    device_map: str = "auto"
    dtype: str = "bfloat16"
    quantization: str | None = None
    attn_implementation: str | None = "sdpa"
    padding_side: str = "left"


class TransformersBackend:
    capabilities = BackendCapabilities(
        supports_logits=True,
        supports_hidden_states=True,
        supports_kv_cache=True,
        supports_manual_step=True,
        supports_chat_template=True,
    )

    def __init__(self, config: TransformersBackendConfig):
        self.config = config
        self.tokenizer = None
        self.model = None

    @property
    def model_id(self) -> str:
        return self.config.model_id

    @property
    def device(self) -> Any:
        self._require_loaded()
        return self.model.device

    @property
    def eos_token_id(self) -> int | None:
        self._require_loaded()
        return self.tokenizer.eos_token_id

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "TransformersBackend requires torch and transformers to be installed."
            ) from exc

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }.get(self.config.dtype)
        if dtype is None:
            raise BackendError(f"Unsupported dtype: {self.config.dtype!r}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            padding_side=self.config.padding_side,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": self.config.device_map,
        }
        if self.config.attn_implementation:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        if self.config.quantization == "4bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise BackendError(
                    "4-bit loading requires bitsandbytes support from transformers."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        elif self.config.quantization == "8bit":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:
                raise BackendError(
                    "8-bit loading requires bitsandbytes support from transformers."
                ) from exc
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config.quantization not in (None, "none"):
            raise BackendError(f"Unsupported quantization: {self.config.quantization!r}")
        else:
            model_kwargs["torch_dtype"] = dtype

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )
        except (ValueError, ImportError):
            if "attn_implementation" not in model_kwargs:
                raise
            model_kwargs.pop("attn_implementation")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                **model_kwargs,
            )
        self.model.eval()

    def encode_chat(self, prompt: str) -> dict[str, Any]:
        self._require_loaded()
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        return {key: value.to(self.device) for key, value in encoded.items()}

    def tokenize_text(self, text: str) -> list[int]:
        self._require_loaded()
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, token_ids: Any, *, skip_special_tokens: bool = True) -> str:
        self._require_loaded()
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_id_for_literal(self, text: str) -> int | None:
        self._require_loaded()
        converted = self.tokenizer.convert_tokens_to_ids(text)
        if converted is not None and converted != self.tokenizer.unk_token_id:
            return int(converted)
        ids = self.tokenize_text(text)
        if len(ids) == 1:
            return int(ids[0])
        return None

    def prefill(self, input_ids: Any, *, output_hidden_states: bool = False) -> Any:
        self._require_loaded()
        import torch

        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )

    def step(
        self,
        input_ids: Any,
        past_key_values: Any,
        *,
        output_hidden_states: bool = False,
    ) -> Any:
        self._require_loaded()
        import torch

        with torch.no_grad():
            return self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=output_hidden_states,
            )

    def clone_cache(self, past_key_values: Any) -> Any:
        return copy.deepcopy(past_key_values)

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise BackendError("Backend is not loaded. Call load() first.")
