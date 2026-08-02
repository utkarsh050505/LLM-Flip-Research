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

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        assistant_prefill: str | None = None,
    ) -> str:
        """
        Generate text from a prompt using the chat template.

        Args:
            prompt: The user message content.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            assistant_prefill: Optional text to prepend as the assistant's
                response start (for prefix-continuation experiments).

        Returns:
            Generated text string (only the new tokens).
        """
        self._require_loaded()
        import torch

        messages = [{"role": "user", "content": prompt}]

        if hasattr(self.tokenizer, "apply_chat_template"):
            # Always apply the template with add_generation_prompt=True
            # This produces: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.device)

            # If we have an assistant prefill, manually append its tokens
            # This avoids continue_final_message which some templates don't support
            if assistant_prefill:
                prefill_ids = self.tokenizer.encode(
                    assistant_prefill, add_special_tokens=False, return_tensors="pt"
                ).to(self.device)
                inputs["input_ids"] = torch.cat(
                    [inputs["input_ids"], prefill_ids], dim=1
                )
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        else:
            text = prompt
            if assistant_prefill:
                text = f"{prompt}\n{assistant_prefill}"
            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def generate_continuation(
        self,
        prompt: str,
        prefix_text: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.1,
        budget_forcing_suffix: str = "",
    ) -> str:
        """
        Generate a continuation from a prompt + reasoning prefix.

        This is the core method for difficulty prefix-continuation experiments.
        The model sees the original question and a partial reasoning trace,
        and must continue from that point.

        Args:
            prompt: The original question/user message.
            prefix_text: The reasoning trace prefix to continue from.
            max_new_tokens: Maximum tokens for the continuation.
            temperature: Sampling temperature.
            budget_forcing_suffix: Budget-forcing prompt to append after the prefix.

        Returns:
            Generated continuation text.
        """
        assistant_content = prefix_text + budget_forcing_suffix
        return self.generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            assistant_prefill=assistant_content,
        )

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise BackendError("Backend is not loaded. Call load() first.")
