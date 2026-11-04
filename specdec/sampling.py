"""The math of speculative sampling. No models and no caches live here.

Notation follows Leviathan et al. (2023): p is the target distribution and q is
the draft distribution. Chen et al. (2023) use the opposite letters.

All probabilities are float64 on the CPU. A model may run in float32 or float16
on the GPU, but acceptance ratios and residuals are computed here, because a
ratio of two small float32 numbers loses the precision the sampler depends on.
"""

from dataclasses import dataclass

import torch

from specdec.randomness import Randomness

# Below this residual mass, a rejection can only come from rounding, because the
# exact rejection probability is at most this mass. Falling back to p then moves
# the output law by at most this amount in total variation.
RESIDUAL_MASS_FLOOR = 1e-12


@dataclass(frozen=True)
class SamplingConfig:
    """The transforms both models agree on before any probability is compared.

    Greedy decoding is a separate mode (see `verify_greedy`), so temperature
    must be positive here.
    """

    temperature: float = 1.0
    top_k: int = 0  # 0 keeps every token.
    top_p: float = 1.0  # 1.0 keeps every token.

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature must be positive; use greedy mode for argmax")
        if self.top_k < 0:
            raise ValueError("top_k must be 0 or positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


def to_probabilities(logits: torch.Tensor, config: SamplingConfig) -> torch.Tensor:
    """Turn raw logits of shape (..., vocab) into the distribution actually sampled.

    Filtered tokens get probability exactly 0, so the sampler sees the same
    support the user asked for. Filter ties break by lowest token id, because
    the sort is stable.
    """
    logits = logits.to(torch.float64) / config.temperature
    if config.top_k > 0 and config.top_k < logits.shape[-1]:
        kth_largest = logits.topk(config.top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_largest, -torch.inf)
    probabilities = logits.softmax(dim=-1)
    if config.top_p < 1.0:
        probabilities = _keep_top_p(probabilities, config.top_p)
    return probabilities


def _keep_top_p(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest high-probability set whose mass reaches top_p, then renormalize."""
    order = probabilities.argsort(dim=-1, descending=True, stable=True)
    sorted_probabilities = probabilities.gather(-1, order)
    mass_before = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
    keep_sorted = mass_before < top_p  # The token that crosses top_p stays.
    keep = torch.zeros_like(keep_sorted).scatter(-1, order, keep_sorted)
    kept = probabilities.masked_fill(~keep, 0.0)
    return kept / kept.sum(dim=-1, keepdim=True)


def acceptance_probability(p_x: float, q_x: float) -> float:
    """Return min(1, p(x) / q(x)) for a token x that was drawn from q."""
    if q_x <= 0:
        raise ValueError("a drafted token must have positive draft probability")
    return min(1.0, p_x / q_x)


def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Return the distribution to sample from after a rejection: norm(max(0, p - q)).

    Why this is right: acceptance gives token x probability min(p(x), q(x)).
    The remaining p(x) - min(p(x), q(x)) = max(0, p(x) - q(x)) must come from
    the rejection branch, whose total probability is the sum of that same
    quantity, which is the total variation distance between p and q.

    Identical distributions never reach this function. A rejection needs
    p(x) < q(x) for some x, and then the positive part has mass at least
    q(x) - p(x) > 0. Rounding can still make it vanish, hence the floor.
    """
    excess = (p - q).clamp(min=0.0)
    mass = float(excess.sum())
    if mass <= RESIDUAL_MASS_FLOOR:
        return p
    return excess / mass


@dataclass(frozen=True)
class BlockOutcome:
    """What one draft-and-verify block decided.

    The delivered tokens are `draft_tokens[:accepted]` followed by `next_token`
    if there is one. Rejected draft tokens are never delivered.
    """

    accepted: int
    next_token: int | None  # From the residual or the bonus draw. None when EOS ended the block.
    rejected_at: int | None  # Index into the draft block, or None if nothing was rejected.


def verify_block(
    p_rows: torch.Tensor,
    q_rows: torch.Tensor,
    draft_tokens: list[int],
    rng: Randomness,
    eos_token_id: int | None = None,
) -> BlockOutcome:
    """Run the accept and reject chain over one draft block.

    p_rows has shape (k + 1, vocab). Row i is the target distribution given the
    committed prefix plus draft_tokens[:i], so the last row serves the bonus.
    q_rows has shape (k, vocab), with row i computed from that same prefix.

    The caller limits k so the bonus token always fits the generation budget.
    Every draw uses fresh randomness, which is what makes the block length free
    to depend on anything in the past.
    """
    block_length = len(draft_tokens)
    if p_rows.shape[0] != block_length + 1 or q_rows.shape[0] != block_length:
        raise ValueError("expected k + 1 target rows and k draft rows")

    for i, token in enumerate(draft_tokens):
        p_x, q_x = float(p_rows[i, token]), float(q_rows[i, token])
        if rng.coin(acceptance_probability(p_x, q_x)):
            if token == eos_token_id:
                return BlockOutcome(accepted=i + 1, next_token=None, rejected_at=None)
            continue
        residual = residual_distribution(p_rows[i], q_rows[i])
        return BlockOutcome(accepted=i, next_token=rng.categorical(residual), rejected_at=i)

    # Every draft token was accepted, so the target row after them is a free
    # sample that costs no extra target pass.
    return BlockOutcome(
        accepted=block_length,
        next_token=rng.categorical(p_rows[block_length]),
        rejected_at=None,
    )


def greedy_token(logits: torch.Tensor) -> int:
    """Return the argmax of one row. Ties go to the lowest token id, as torch documents."""
    return int(logits.argmax(dim=-1))


def verify_greedy(target_logits: torch.Tensor, draft_tokens: list[int]) -> BlockOutcome:
    """Greedy verification: keep draft tokens while they equal the target's argmax.

    This is a separate mode, not sampling with a small temperature. The result
    equals target-only greedy decoding token for token, as long as cached and
    uncached logits agree to the tie-breaking precision.
    """
    for i, token in enumerate(draft_tokens):
        best = greedy_token(target_logits[i])
        if best != token:
            return BlockOutcome(accepted=i, next_token=best, rejected_at=i)
    return BlockOutcome(
        accepted=len(draft_tokens),
        next_token=greedy_token(target_logits[len(draft_tokens)]),
        rejected_at=None,
    )
