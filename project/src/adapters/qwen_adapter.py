"""
Qwen Adapter

Concrete implementation of the BaseAdapter interface
for Qwen reasoning models.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from src.adapters.base_adapter import BaseAdapter
from src.trace.reasoning_trace import (
    ReasoningTrace,
    TraceMetadata,
)


class QwenAdapter(BaseAdapter):

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
    ):

        super().__init__(model_name, device)

    # --------------------------------------------------
    # Model Loading
    # --------------------------------------------------

    def load_model(self):

        print(f"Loading {self.model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

        print("Model loaded.")

    # --------------------------------------------------

    def unload_model(self):

        del self.model
        del self.tokenizer

        torch.cuda.empty_cache()

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_model_name(self):

        return self.model_name

    # --------------------------------------------------

    def count_tokens(
        self,
        text: str,
    ):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    # --------------------------------------------------
    # Generation
    # --------------------------------------------------

    def generate_trace(
        self,
        prompt: str,
        benchmark: str,
        problem_id: str,
        temperature: float,
        max_new_tokens: int,
        checkpoint_interval: int = 32,
    ) -> ReasoningTrace:

        metadata = TraceMetadata(
            model_name=self.model_name,
            benchmark=benchmark,
            problem_id=problem_id,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=42,
        )

        trace = ReasoningTrace(metadata)

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]

        decoded = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        trace.generation.reasoning_text = decoded

        trace.outcome.final_answer = decoded

        return trace