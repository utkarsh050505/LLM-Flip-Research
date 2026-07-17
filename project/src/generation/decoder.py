"""
Custom Autoregressive Decoder

Scientific instrumentation layer for mechanistic interpretability research.

Replaces HuggingFace model.generate() with a manual decoding loop that
exposes every forward pass, enabling per-token capture of:
    - Top-k logits and probabilities
    - Shannon entropy of the full distribution
    - Selected hidden state layers (first / middle / last)
    - Per-step wall-clock timing

Algorithm
---------
1. Tokenize prompt and run a single prefill forward pass.
   This populates the KV cache and produces logits for the first generated token.

2. For each subsequent token:
   a. Feed ONLY the last generated token into the model (1 token input).
   b. Pass the existing past_key_values to avoid recomputing the prefix.
   c. The model returns logits for the next token and updated past_key_values.
   d. Apply temperature scaling to logits.
   e. Sample or argmax the next token from the distribution.
   f. Compute Shannon entropy from the full probability distribution.
   g. Extract top-k logits and their probabilities.
   h. Optionally extract selected hidden state layers.
   i. Record everything in a GenerationStep.

3. Stop when EOS token is generated or max_new_tokens is reached.

Complexity
----------
- Time:  O(T) forward passes, where T = number of generated tokens.
         Each forward pass is O(1) in input length due to KV cache reuse.
         Total compute matches model.generate() — no redundant work.

- Space: O(T * V) for logits at each step (V = vocab size), but only
         top-k are retained in the trace. KV cache grows as O(T * L * D)
         where L = num_layers, D = hidden_dim. This is identical to
         model.generate().

Memory Management
-----------------
- past_key_values is maintained across steps (no recomputation).
- Hidden states are immediately moved to CPU and detached from the graph.
- Only top-k logits are stored per step (not the full vocab distribution).
- Entropy is computed on GPU before discarding the logit tensor.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.trace.reasoning_trace import (
    GenerationStep,
    GenerationTiming,
    TopKEntry,
)

logger = logging.getLogger("sbtf")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

@dataclass
class DecoderConfig:
    """
    Configuration for the autoregressive decoder.

    Args:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. 0 = greedy (argmax).
        top_k_logits: Number of top logits to record per step.
        extract_hidden_states: Whether to extract hidden states.
        hidden_state_layers: Which layers to extract ('first', 'middle', 'last').
        seed: Random seed for reproducibility.
    """
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_k_logits: int = 5
    extract_hidden_states: bool = True
    hidden_state_layers: Tuple[str, ...] = ("first", "middle", "last")
    seed: int = 42


# ---------------------------------------------------------
# Decoder
# ---------------------------------------------------------

class Decoder:
    """
    Custom autoregressive decoder for scientific trace collection.

    This class replaces model.generate() with a manual decoding loop
    that instruments every forward pass. It is designed for mechanistic
    interpretability research, not production inference.

    Usage:
        decoder = Decoder(model, tokenizer, config)
        steps, timing = decoder.decode(prompt_ids)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        config: DecoderConfig,
    ):
        """
        Args:
            model: A loaded HuggingFace causal LM (in eval mode).
            tokenizer: The corresponding tokenizer.
            config: Decoder configuration.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

        # Resolve EOS token ID(s)
        self.eos_token_ids = self._resolve_eos_ids()

        # Compute number of layers for hidden state extraction
        self._num_model_layers: Optional[int] = None
        if config.extract_hidden_states:
            self._num_model_layers = self._count_layers()

        logger.info(
            "Decoder initialized — max_tokens=%d, temperature=%.2f, "
            "top_k=%d, hidden_states=%s",
            config.max_new_tokens,
            config.temperature,
            config.top_k_logits,
            config.extract_hidden_states,
        )

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def decode(
        self,
        prompt_ids: torch.Tensor,
    ) -> Tuple[List[GenerationStep], GenerationTiming]:
        """
        Run autoregressive decoding on a tokenized prompt.

        Args:
            prompt_ids: Tokenized prompt as a tensor of shape (1, seq_len).
                        Must already be on the correct device.

        Returns:
            Tuple of:
                - List[GenerationStep]: One step per generated token.
                - GenerationTiming: Aggregate timing metrics.

        Raises:
            RuntimeError: If prompt_ids has wrong shape.
        """
        if prompt_ids.dim() != 2 or prompt_ids.shape[0] != 1:
            raise RuntimeError(
                f"prompt_ids must have shape (1, seq_len), got {prompt_ids.shape}"
            )

        # Set seed
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        steps: List[GenerationStep] = []
        t_generation_start = time.perf_counter()

        # ----- Phase 1: Prefill -----
        # Run the full prompt through the model in one pass.
        # This populates the KV cache and produces logits for the first token.

        logger.info(
            "Prefill: %d prompt tokens ...", prompt_ids.shape[1]
        )
        t_prefill_start = time.perf_counter()

        with torch.no_grad():
            prefill_output = self.model(
                input_ids=prompt_ids,
                use_cache=True,
                output_hidden_states=self.config.extract_hidden_states,
            )

        past_key_values = prefill_output.past_key_values
        logits = prefill_output.logits  # (1, seq_len, vocab_size)

        t_prefill_end = time.perf_counter()
        prefill_seconds = t_prefill_end - t_prefill_start
        logger.info("Prefill complete in %.3fs", prefill_seconds)

        # The logits at position [-1] predict the first generated token
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # ----- Phase 2: Autoregressive decoding -----

        for step_idx in range(self.config.max_new_tokens):

            t_step_start = time.perf_counter()

            # -- Sample or greedy select --
            next_token_id, step_logits_raw = self._select_token(
                next_token_logits
            )

            # -- Compute entropy from the full distribution --
            entropy = self._compute_entropy(next_token_logits)

            # -- Extract top-k logits --
            top_k_entries = self._extract_top_k(
                next_token_logits, step_logits_raw
            )

            # -- Extract hidden states (from prefill or current step) --
            hidden_dict = {}
            if self.config.extract_hidden_states:
                hidden_states_tuple = (
                    prefill_output.hidden_states
                    if step_idx == 0
                    else current_output.hidden_states  # noqa: F821
                )
                hidden_dict = self._extract_hidden_states(hidden_states_tuple)

            # -- Decode token string --
            token_str = self.tokenizer.decode(
                [next_token_id.item()],
                skip_special_tokens=False,
            )

            # -- Timing --
            t_step_end = time.perf_counter()
            step_timestamp = t_step_end - t_generation_start

            # -- Check stopping condition --
            is_eos = next_token_id.item() in self.eos_token_ids
            is_last = step_idx == self.config.max_new_tokens - 1

            finish_reason = None
            if is_eos:
                finish_reason = "eos"
            elif is_last:
                finish_reason = "max_tokens"

            # -- Build step --
            step = GenerationStep(
                step_index=step_idx,
                generated_token=token_str,
                generated_token_id=next_token_id.item(),
                timestamp=round(step_timestamp, 6),
                top_k_logits=top_k_entries,
                entropy=round(entropy, 6),
                selected_hidden_states=hidden_dict,
                finish_reason=finish_reason,
            )
            steps.append(step)

            # Log progress periodically
            if (step_idx + 1) % 50 == 0:
                elapsed = time.perf_counter() - t_generation_start
                tps = (step_idx + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "Step %d — %.1f tok/s — entropy=%.3f",
                    step_idx + 1, tps, entropy,
                )

            # -- Stop if EOS --
            if is_eos:
                logger.info(
                    "EOS token generated at step %d", step_idx
                )
                break

            # -- Next forward pass (single token) --
            next_input = next_token_id.unsqueeze(0).unsqueeze(0)  # (1, 1)

            with torch.no_grad():
                current_output = self.model(
                    input_ids=next_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_hidden_states=self.config.extract_hidden_states,
                )

            past_key_values = current_output.past_key_values
            next_token_logits = current_output.logits[:, -1, :]

        # ----- Aggregate timing -----

        t_generation_end = time.perf_counter()
        total_seconds = t_generation_end - t_generation_start
        num_tokens = len(steps)
        tokens_per_second = num_tokens / total_seconds if total_seconds > 0 else 0.0

        timing = GenerationTiming(
            total_seconds=round(total_seconds, 3),
            tokens_per_second=round(tokens_per_second, 2),
            num_generated_tokens=num_tokens,
            prefill_seconds=round(prefill_seconds, 3),
        )

        logger.info(
            "Decoding complete: %d tokens in %.2fs (%.1f tok/s), "
            "prefill=%.3fs",
            num_tokens, total_seconds, tokens_per_second, prefill_seconds,
        )

        # Free KV cache
        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return steps, timing

    # --------------------------------------------------
    # Private: Token Selection
    # --------------------------------------------------

    def _select_token(
        self,
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the next token from logits.

        Args:
            logits: Raw logits of shape (1, vocab_size).

        Returns:
            Tuple of (selected_token_id, temperature-scaled_logits).
            selected_token_id has shape ().
            scaled_logits has shape (1, vocab_size).
        """
        if self.config.temperature <= 0:
            # Greedy decoding — deterministic
            token_id = logits.argmax(dim=-1).squeeze()
            return token_id, logits
        else:
            # Temperature-scaled sampling
            scaled = logits / self.config.temperature
            probs = F.softmax(scaled, dim=-1)
            token_id = torch.multinomial(probs, num_samples=1).squeeze()
            return token_id, scaled

    # --------------------------------------------------
    # Private: Entropy
    # --------------------------------------------------

    def _compute_entropy(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        Compute Shannon entropy of the token probability distribution.

        Uses log-softmax for numerical stability.

        H(p) = -Σ p(x) * log(p(x))

        Returns entropy in nats (natural log).

        Args:
            logits: Raw logits of shape (1, vocab_size).

        Returns:
            Scalar entropy value.
        """
        log_probs = F.log_softmax(logits, dim=-1)  # (1, V)
        probs = log_probs.exp()

        # Mask zeros to avoid NaN in log
        entropy = -(probs * log_probs).sum(dim=-1)

        return entropy.item()

    # --------------------------------------------------
    # Private: Top-K Extraction
    # --------------------------------------------------

    def _extract_top_k(
        self,
        raw_logits: torch.Tensor,
        scaled_logits: torch.Tensor,
    ) -> List[TopKEntry]:
        """
        Extract the top-k tokens by logit value.

        Stores both the raw logit and the probability (after softmax
        of the scaled logits) for each of the top-k tokens.

        Args:
            raw_logits: Pre-temperature logits (1, V).
            scaled_logits: Post-temperature logits (1, V).

        Returns:
            List of TopKEntry, sorted by descending logit.
        """
        k = min(self.config.top_k_logits, raw_logits.shape[-1])

        # Top-k by raw logit value
        top_values, top_indices = torch.topk(raw_logits[0], k)

        # Probabilities from scaled logits
        probs = F.softmax(scaled_logits, dim=-1)[0]

        entries = []
        for i in range(k):
            tid = top_indices[i].item()
            token_str = self.tokenizer.decode([tid], skip_special_tokens=False)
            entries.append(TopKEntry(
                token=token_str,
                token_id=tid,
                logit=round(top_values[i].item(), 4),
                probability=round(probs[tid].item(), 6),
            ))

        return entries

    # --------------------------------------------------
    # Private: Hidden State Extraction
    # --------------------------------------------------

    def _extract_hidden_states(
        self,
        hidden_states: Tuple[torch.Tensor, ...],
    ) -> dict:
        """
        Extract selected hidden state layers for the LAST token position.

        Only keeps the first, middle, and last layers as specified in config.
        Immediately moves tensors to CPU and detaches from the computation graph.

        The hidden_states tuple has (num_layers + 1) entries:
            index 0 = embedding layer output
            index 1..N = transformer layer 1..N outputs

        We select:
            'first'  → layer 1 (first transformer layer, index 1)
            'middle' → layer N//2 (middle transformer layer)
            'last'   → layer N   (final transformer layer, index -1)

        Each extracted state is a 1D tensor of shape (hidden_dim,).

        Args:
            hidden_states: Tuple of tensors from model output.

        Returns:
            Dict mapping layer name to list of floats.
        """
        if not hidden_states:
            return {}

        num_layers = len(hidden_states) - 1  # Exclude embedding layer

        layer_map = {
            "first": 1,
            "middle": num_layers // 2,
            "last": num_layers,  # == len(hidden_states) - 1
        }

        result = {}
        for layer_name in self.config.hidden_state_layers:
            if layer_name not in layer_map:
                logger.warning("Unknown hidden state layer: %s", layer_name)
                continue

            idx = layer_map[layer_name]
            if idx >= len(hidden_states):
                continue

            # hidden_states[idx] shape: (batch=1, seq_len, hidden_dim)
            # Take the last token position
            state = hidden_states[idx][0, -1, :].detach().cpu()

            # Store as list of floats for JSON serialization
            result[layer_name] = state.tolist()

        return result

    # --------------------------------------------------
    # Private: Utilities
    # --------------------------------------------------

    def _resolve_eos_ids(self) -> set:
        """
        Resolve EOS token ID(s) from the tokenizer.

        Some models have multiple EOS tokens (e.g., Qwen uses
        both eos_token_id and additional_special_tokens).

        Returns:
            Set of integer token IDs that signal end-of-sequence.
        """
        eos_ids = set()

        if self.tokenizer.eos_token_id is not None:
            eos_ids.add(self.tokenizer.eos_token_id)

        # Qwen models may use additional stop tokens
        if hasattr(self.model, "generation_config"):
            gc = self.model.generation_config
            if hasattr(gc, "eos_token_id") and gc.eos_token_id is not None:
                if isinstance(gc.eos_token_id, list):
                    eos_ids.update(gc.eos_token_id)
                else:
                    eos_ids.add(gc.eos_token_id)

        if not eos_ids:
            logger.warning(
                "No EOS token found. Decoding will only stop at max_tokens."
            )

        logger.debug("EOS token IDs: %s", eos_ids)
        return eos_ids

    def _count_layers(self) -> int:
        """
        Count the number of transformer layers in the model.

        Inspects common HuggingFace model attribute names.

        Returns:
            Number of transformer layers.
        """
        config = self.model.config

        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            if hasattr(config, attr):
                n = getattr(config, attr)
                logger.info("Model has %d transformer layers", n)
                return n

        logger.warning(
            "Could not determine number of layers from model config. "
            "Hidden state extraction may not work correctly."
        )
        return 0
