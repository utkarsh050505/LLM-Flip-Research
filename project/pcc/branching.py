from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backends.base import BackendError, ModelBackend
from pcc.answer import answer_appears, evaluate_answer
from pcc.config import ExperimentConfig

@dataclass(frozen=True)
class FCSResult:
    found: bool
    attempt: int
    prefix_ids: Any | None
    prompt_len: int
    generated_tokens: int
    reason: str
    prefix_tail: str = ""


@dataclass(frozen=True)
class BranchResult:
    branch_id: int
    label: str
    raw_answer: str | None
    normalized_answer: str | None
    correct: bool | None
    tokens_generated: int
    wait_injections: int
    degenerate: bool
    distinct_boxed_answers: list[str]
    transcript: str
    metrics: list[dict[str, Any]]


@dataclass(frozen=True)
class PCCBranchRunResult:
    model_id: str
    backend: str
    problem: str
    ground_truth: str
    fcs: FCSResult
    target_budget: int | None
    branches: list[BranchResult]
    config: dict[str, Any]


class PCCBranchExperiment:
    def __init__(self, backend: ModelBackend, config: ExperimentConfig):
        self.backend = backend
        self.config = config

    def run(self, prompt: str, ground_truth: str, *, seed: int | None = None) -> PCCBranchRunResult:
        self._require_supported_backend()
        if seed is not None:
            # pyrefly: ignore [missing-import]
            import torch

            torch.manual_seed(seed)

        fcs = self.find_fcs_prefix(prompt, ground_truth)
        if not fcs.found:
            return PCCBranchRunResult(
                model_id=self.backend.model_id,
                backend=self.config.backend,
                problem=prompt,
                ground_truth=ground_truth,
                fcs=fcs,
                target_budget=None,
                branches=[],
                config=asdict(self.config),
            )

        target_budget = self._target_budget(fcs.generated_tokens)
        branches = [
            self.generate_branch(
                branch_id=branch_id,
                prefix_ids=fcs.prefix_ids,
                prompt_len=fcs.prompt_len,
                ground_truth=ground_truth,
                target_budget=target_budget,
            )
            for branch_id in range(self.config.num_branches)
        ]
        return PCCBranchRunResult(
            model_id=self.backend.model_id,
            backend=self.config.backend,
            problem=prompt,
            ground_truth=ground_truth,
            fcs=fcs,
            target_budget=target_budget,
            branches=branches,
            config=asdict(self.config),
        )

    def find_fcs_prefix(self, prompt: str, ground_truth: str) -> FCSResult:
        # pyrefly: ignore [missing-import]
        import torch

        for attempt in range(1, self.config.fcs_attempts + 1):
            inputs = self.backend.encode_chat(prompt)
            prompt_ids = inputs["input_ids"]
            prompt_len = prompt_ids.shape[1]
            out = self.backend.prefill(prompt_ids, output_hidden_states=False)
            cache = out.past_key_values
            logits = out.logits[0, -1, :].float().detach().clone()
            generated: list[int] = []

            for step in range(self.config.max_fcs_tokens):
                next_id = _sample_from_logits(logits, temperature=self.config.fcs_temperature)
                generated.append(next_id)

                if self._should_check_fcs(step, next_id):
                    decoded = self.backend.decode(generated, skip_special_tokens=False)
                    reasoning_region = self._reasoning_region(decoded)
                    if answer_appears(reasoning_region, ground_truth):
                        generated_tensor = torch.tensor([generated], device=prompt_ids.device)
                        prefix_ids = torch.cat([prompt_ids, generated_tensor], dim=1)
                        tail = self.backend.decode(
                            prefix_ids[0, -min(prefix_ids.shape[1], 24):],
                            skip_special_tokens=False,
                        )
                        return FCSResult(
                            found=True,
                            attempt=attempt,
                            prefix_ids=prefix_ids,
                            prompt_len=prompt_len,
                            generated_tokens=len(generated),
                            reason="answer appeared before reasoning close",
                            prefix_tail=tail,
                        )
                    if self._reasoning_closed(decoded):
                        break

                if next_id == self.backend.eos_token_id:
                    break

                cur_input = torch.tensor([[next_id]], device=prompt_ids.device)
                out = self.backend.step(cur_input, cache, output_hidden_states=False)
                cache = out.past_key_values
                logits = out.logits[0, -1, :].float().detach().clone()

        return FCSResult(
            found=False,
            attempt=self.config.fcs_attempts,
            prefix_ids=None,
            prompt_len=0,
            generated_tokens=0,
            reason="no correct answer found before reasoning close or token ceiling",
        )

    def generate_branch(
        self,
        *,
        branch_id: int,
        prefix_ids: Any,
        prompt_len: int,
        ground_truth: str,
        target_budget: int,
    ) -> BranchResult:
        import re
        # pyrefly: ignore [missing-import]
        import torch

        out = self.backend.prefill(prefix_ids, output_hidden_states=True)
        cache = self.backend.clone_cache(out.past_key_values)
        logits = out.logits[0, -1, :].float().detach().clone()
        current_hidden_states = out.hidden_states

        layer_idx = _layer_indices(self.backend)
        think_close = self.config.reasoning_format.think_close
        think_close_id = (
            self.backend.token_id_for_literal(think_close) if think_close else None
        )
        eos_id = self.backend.eos_token_id
        wait_variants = [
            self.backend.tokenize_text(phrase)
            for phrase in self.config.wait_phrases
        ]

        generated: list[int] = []
        metrics: list[dict[str, Any]] = []
        prev_logits = None
        prev_hidden = {name: None for name in layer_idx}
        wait_queue: list[int] = []
        wait_injections = 0
        think_closed = False
        degenerate = False
        max_tokens = target_budget + self.config.conclusion_buffer_tokens

        for step in range(max_tokens):
            row = _metric_row(
                step=step,
                logits=logits,
                hidden_states=current_hidden_states,
                layer_idx=layer_idx,
                prev_logits=prev_logits,
                prev_hidden=prev_hidden,
            )
            metrics.append(row)
            prev_logits = logits.detach().clone()

            if wait_queue:
                next_id = wait_queue.pop(0)
            else:
                next_id = self._choose_branch_token(
                    logits=logits,
                    step=step,
                    target_budget=target_budget,
                    think_closed=think_closed,
                    think_close_id=think_close_id,
                    eos_id=eos_id,
                    wait_variants=wait_variants,
                    wait_injections=wait_injections,
                )
                if isinstance(next_id, list):
                    wait_queue = next_id
                    wait_injections += 1
                    next_id = wait_queue.pop(0)

            generated.append(next_id)
            decoded = self.backend.decode(generated, skip_special_tokens=False)
            if think_close and think_close in decoded:
                think_closed = True
            if think_close_id is not None and next_id == think_close_id:
                think_closed = True

            if self._should_check_branch(step, next_id):
                if think_closed and re.search(r"\\boxed\{[^}]*\}", decoded):
                    break
                boxed = re.findall(r"\\boxed\{([^}]*)\}", decoded)
                if len(boxed) >= self.config.max_repeated_boxed:
                    recent = boxed[-self.config.max_repeated_boxed:]
                    if len(set(recent)) == 1:
                        degenerate = True
                        break

            if next_id == eos_id:
                break

            cur_input = torch.tensor([[next_id]], device=prefix_ids.device)
            out = self.backend.step(cur_input, cache, output_hidden_states=True)
            cache = out.past_key_values
            logits = out.logits[0, -1, :].float().detach().clone()
            current_hidden_states = out.hidden_states

        transcript_ids = prefix_ids[0].tolist() + generated
        transcript = self.backend.decode(transcript_ids, skip_special_tokens=False)
        continuation = self.backend.decode(generated, skip_special_tokens=True)
        evaluation = evaluate_answer(continuation, ground_truth)
        boxed_values = re.findall(r"\\boxed\{([^}]*)\}", continuation)
        distinct_boxed = sorted(set(boxed_values))
        label = _label(evaluation.correct, evaluation.raw_answer, degenerate)

        return BranchResult(
            branch_id=branch_id,
            label=label,
            raw_answer=evaluation.raw_answer,
            normalized_answer=evaluation.normalized_answer,
            correct=evaluation.correct,
            tokens_generated=len(generated),
            wait_injections=wait_injections,
            degenerate=degenerate,
            distinct_boxed_answers=distinct_boxed,
            transcript=transcript,
            metrics=metrics,
        )

    def _choose_branch_token(
        self,
        *,
        logits: Any,
        step: int,
        target_budget: int,
        think_closed: bool,
        think_close_id: int | None,
        eos_id: int | None,
        wait_variants: list[list[int]],
        wait_injections: int,
    ) -> int | list[int]:
        # pyrefly: ignore [missing-import]
        import torch

        can_force = think_close_id is not None and eos_id is not None
        under_budget = step < target_budget and not think_closed and can_force
        if under_budget:
            probs_raw = torch.softmax(logits, dim=-1)
            conclude_prob = probs_raw[think_close_id].item() + probs_raw[eos_id].item()
            if conclude_prob > self.config.conclude_probability_threshold:
                variant_idx = min(wait_injections, len(wait_variants) - 1)
                return list(wait_variants[variant_idx])
            return _sample_from_logits(
                logits,
                temperature=self.config.branch_temperature,
                banned_token_ids=[think_close_id, eos_id],
            )
        return _sample_from_logits(logits, temperature=self.config.branch_temperature)

    def _target_budget(self, natural_tokens: int) -> int:
        return int(
            min(
                max(
                    natural_tokens * self.config.budget_multiplier,
                    self.config.min_target_budget,
                ),
                self.config.max_target_budget,
            )
        )

    def _reasoning_region(self, text: str) -> str:
        think_close = self.config.reasoning_format.think_close
        if not think_close:
            return text
        pos = text.find(think_close)
        return text if pos == -1 else text[:pos]

    def _reasoning_closed(self, text: str) -> bool:
        think_close = self.config.reasoning_format.think_close
        return bool(think_close and think_close in text)

    def _should_check_fcs(self, step: int, next_id: int) -> bool:
        return (
            step >= self.config.min_tokens_before_fcs
            and step % self.config.check_every == 0
        ) or next_id == self.backend.eos_token_id

    def _should_check_branch(self, step: int, next_id: int) -> bool:
        return step % self.config.check_every == 0 or next_id == self.backend.eos_token_id

    def _require_supported_backend(self) -> None:
        if not self.backend.capabilities.supports_full_pcc_branching:
            raise BackendError(
                "PCC branch experiments require logits, hidden states, KV cache access, "
                "and manual token stepping. Use a Transformers-style local backend."
            )


def _sample_from_logits(logits: Any, *, temperature: float, banned_token_ids: list[int] | None = None) -> int:
    # pyrefly: ignore [missing-import]
    import torch

    working = logits.clone()
    for token_id in banned_token_ids or []:
        if token_id is not None:
            working[token_id] = -float("inf")
    probs = torch.softmax(working / temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _layer_indices(backend: ModelBackend) -> dict[str, int]:
    model = getattr(backend, "model", None)
    if model is None:
        return {}
    n_layers = model.config.num_hidden_layers
    return {"early": 1, "mid": n_layers // 2, "late": n_layers}


def _metric_row(
    *,
    step: int,
    logits: Any,
    hidden_states: Any,
    layer_idx: dict[str, int],
    prev_logits: Any,
    prev_hidden: dict[str, Any],
) -> dict[str, Any]:
    from configs.metrics import (
        cosine_progress,
        jensen_shannon_divergence,
        l2_transition,
        token_entropy,
        top2_margin,
    )

    row = {
        "step": step,
        "entropy": token_entropy(logits),
        "top2_margin": top2_margin(logits),
        "jsd_vs_prev": (
            jensen_shannon_divergence(logits, prev_logits)
            if prev_logits is not None
            else None
        ),
    }
    for name, layer in layer_idx.items():
        hidden = hidden_states[layer][0, -1, :]
        old_hidden = prev_hidden[name]
        row[f"l2_{name}"] = (
            l2_transition(hidden, old_hidden) if old_hidden is not None else None
        )
        row[f"cos_{name}"] = (
            cosine_progress(hidden, old_hidden) if old_hidden is not None else None
        )
        prev_hidden[name] = hidden.detach().clone()
    return row


def _label(correct: bool | None, raw_answer: str | None, degenerate: bool) -> str:
    if degenerate:
        return "DEGENERATE"
    if correct is True:
        return "STABLE_CORRECT"
    if raw_answer is not None:
        return "PCC"
    return "NO_FINAL_ANSWER"
