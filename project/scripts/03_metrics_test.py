"""
Step 3 tests — sanity-check each metric against known answers.

Run: python test_metrics.py
If every line prints PASS, the functions are safe to point at real
generation output in Step 4. If anything prints FAIL, fix metrics.py
before going further — a silently-wrong metric will waste days of
compute before you notice.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import math
# pyrefly: ignore [missing-import]
import torch

from configs.metrics import (
    token_entropy,
    top2_margin,
    jensen_shannon_divergence,
    l2_transition,
    cosine_progress,
)

VOCAB = 100
HIDDEN = 16


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL':5s} {name}")


def main():
    torch.manual_seed(0)

    # --- entropy ---
    uniform_logits = torch.zeros(VOCAB)  # softmax(zeros) = uniform
    peaked_logits = torch.full((VOCAB,), -10.0)
    peaked_logits[0] = 10.0  # one token dominates

    h_uniform = token_entropy(uniform_logits)
    h_peaked = token_entropy(peaked_logits)
    check(
        "uniform distribution has near-max entropy "
        f"(got {h_uniform:.3f}, max possible = {math.log(VOCAB):.3f})",
        abs(h_uniform - math.log(VOCAB)) < 1e-3,
    )
    check(
        f"peaked distribution has near-zero entropy (got {h_peaked:.5f})",
        h_peaked < 0.01,
    )
    check("peaked entropy < uniform entropy", h_peaked < h_uniform)

    # --- top-2 margin ---
    margin_peaked = top2_margin(peaked_logits)
    tied_logits = torch.zeros(VOCAB)
    tied_logits[0] = 5.0
    tied_logits[1] = 5.0  # exact tie for first place
    margin_tied = top2_margin(tied_logits)
    check(
        f"peaked distribution has large top-2 margin (got {margin_peaked:.3f})",
        margin_peaked > 0.9,
    )
    check(
        f"exact tie has ~zero top-2 margin (got {margin_tied:.5f})",
        margin_tied < 1e-4,
    )

    # --- JSD ---
    jsd_same = jensen_shannon_divergence(uniform_logits, uniform_logits)
    jsd_diff = jensen_shannon_divergence(uniform_logits, peaked_logits)
    check(f"JSD of identical distributions is ~0 (got {jsd_same:.5f})", jsd_same < 1e-6)
    check(
        f"JSD of different distributions is > 0 and <= ln(2) "
        f"(got {jsd_diff:.3f}, ln(2)={math.log(2):.3f})",
        0 < jsd_diff <= math.log(2) + 1e-6,
    )

    # --- L2 transition ---
    h1 = torch.randn(HIDDEN)
    l2_same = l2_transition(h1, h1)
    l2_diff = l2_transition(h1, h1 + 5.0)
    check(f"L2 distance to itself is 0 (got {l2_same:.6f})", l2_same < 1e-5)
    check(
        f"L2 distance grows with an added offset (got {l2_diff:.3f})",
        l2_diff > l2_same,
    )

    # --- cosine progress ---
    cos_same = cosine_progress(h1, h1)
    cos_opposite = cosine_progress(h1, -h1)
    check(f"cosine similarity to itself is ~1 (got {cos_same:.4f})", cos_same > 0.999)
    check(
        f"cosine similarity to its own negation is ~-1 (got {cos_opposite:.4f})",
        cos_opposite < -0.999,
    )

    print("\nIf everything above says PASS, metrics.py is safe to use on "
          "real generation output in Step 4.")


if __name__ == "__main__":
    main()