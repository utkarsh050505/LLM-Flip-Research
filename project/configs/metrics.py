"""
Step 3 — Core metric functions.

Each function takes raw tensors straight out of generate()'s outputs
(one step's worth) and returns a plain Python float. Nothing here
touches the model or does any generation — that's on purpose. Keep
these pure so test_metrics.py can check them against known answers
before you ever point them at real output.

Shapes you'll be feeding these, per the Step 2 findings:
  logits            : (vocab_size,)      — outputs.scores[t][0]
  logits_prev       : (vocab_size,)      — outputs.scores[t-1][0]
  hidden            : (hidden_dim,)      — outputs.hidden_states[t][layer][0, -1, :]
  hidden_prev       : (hidden_dim,)      — outputs.hidden_states[t-1][layer][0, -1, :]
"""

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

def token_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy (nats) of the token distribution at one step.
    High = model is spread across many plausible next tokens.
    Low  = model is confident in one token."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy = -(probs * log_probs).sum()
    return entropy.item()


def top2_margin(logits: torch.Tensor) -> float:
    """Gap in probability between the top and second-place token.
    Large margin = model strongly prefers one token.
    Near-zero margin = model is torn between (at least) two options —
    exactly the kind of internal conflict we're hunting for."""
    probs = F.softmax(logits.float(), dim=-1)
    top2 = torch.topk(probs, k=2).values
    return (top2[0] - top2[1]).item()


def jensen_shannon_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    """JSD between two token distributions (e.g. consecutive steps).
    Symmetric and bounded (max ln(2) in nats), unlike KL divergence,
    which makes it easier to compare across positions."""
    p = F.softmax(logits_a.float(), dim=-1)
    q = F.softmax(logits_b.float(), dim=-1)
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * torch.log((a.clamp_min(1e-12)) / (b.clamp_min(1e-12)))).sum()

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).item()


def l2_transition(hidden: torch.Tensor, hidden_prev: torch.Tensor) -> float:
    """L2 distance between the same layer's hidden state at consecutive
    steps. Spikes here indicate the model's internal representation is
    undergoing a large, sudden reconfiguration."""
    return torch.norm(hidden.float() - hidden_prev.float(), p=2).item()


def cosine_progress(hidden: torch.Tensor, hidden_prev: torch.Tensor) -> float:
    """Cosine similarity between consecutive hidden states. Close to 1 =
    moving steadily in the same direction (TRACED's 'Progress'). Near 0
    or negative = direction is unstable / doubling back."""
    return F.cosine_similarity(
        hidden.float().unsqueeze(0), hidden_prev.float().unsqueeze(0)
    ).item()