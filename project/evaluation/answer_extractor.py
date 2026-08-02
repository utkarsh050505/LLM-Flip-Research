"""
Answer Extractor — uses the model itself to extract concise answers
from verbose reasoning traces.

Self-hosted alternative to the paper's OpenAI-compatible vLLM server.
Uses the same TransformersBackend for extraction.
"""
from __future__ import annotations

import re
from typing import Optional, Any


class AnswerExtractionPipeline:
    """
    Extract concise final answers from model reasoning output.

    Uses the loaded model backend to run a secondary extraction prompt.
    Falls back to regex-based extraction if the model is unavailable.
    """

    EXTRACTION_PROMPT = (
        "You are a helpful assistant that extracts concise answers from text. "
        "Extract only the direct answer, removing explanations.\n\n"
        "Given the following answer, extract ONLY the final answer in a concise format.\n"
        "Do not reason. Do not explain. Do not repeat the question. "
        "Do not include words like \"answer\".\n"
        "Return only the extracted answer, for example: 2 or A or \\frac{{1}}{{3}}.\n\n"
        "Model Answer: {model_answer}\n\n"
        "Extract the answer (just the answer itself, no explanations):"
    )

    def __init__(self, backend: Any = None):
        """
        Args:
            backend: A loaded TransformersBackend instance. If None,
                     falls back to regex-only extraction.
        """
        self.backend = backend

    def extract(self, model_answer: str) -> Optional[str]:
        """
        Extract the final answer from a model's reasoning output.

        First tries regex-based extraction, then falls back to LLM
        extraction if regex doesn't find anything.
        """
        # Try regex-based extraction first (fast path)
        from evaluation.parsing import ParsingHelper, Strategies

        regex_answer = ParsingHelper.extract_answer(
            model_answer,
            strategies=[Strategies.BOXED, Strategies.FINAL_ANSWER],
        )
        if regex_answer is not None:
            return regex_answer

        # Fall back to LLM-based extraction if backend available
        if self.backend is not None:
            return self._llm_extract(model_answer)

        # Last resort: try extracting last number
        return ParsingHelper.extract_answer(
            model_answer,
            strategies=[Strategies.LAST_NUMBER],
        )

    def _llm_extract(self, model_answer: str) -> Optional[str]:
        """Use the model itself to extract a concise answer."""
        # Truncate very long answers to avoid OOM
        truncated = model_answer[:2000] if len(model_answer) > 2000 else model_answer
        prompt = self.EXTRACTION_PROMPT.format(model_answer=truncated)

        try:
            output = self.backend.generate_text(
                prompt=prompt,
                max_new_tokens=50,
                temperature=0.1,
            )
            # Clean the extraction output
            answer = output.strip()
            # Remove any reasoning artifacts
            answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
            return answer if answer else None
        except Exception:
            return None

    def batch_extract(self, model_answers: list[str]) -> list[Optional[str]]:
        """Extract answers from a batch of model outputs."""
        return [self.extract(answer) for answer in model_answers]
