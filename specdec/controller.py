"""A cost-based rule for the length of the next draft block.

Each block delivers E(k) tokens on average and costs C(k) seconds. The controller
picks the k that maximizes E(k) / C(k), which is the long-run token rate (the
renewal reward theorem) if acceptance and costs stay near their current estimates.

With acceptance alpha, a block of k draft tokens delivers the capped geometric
E(k) = 1 + alpha + ... + alpha^k (Leviathan et al., Equation 1). The cost is
C(k) = k * draft_pass + verify(k + 1), where verify is fit to measured passes.

Why this cannot change the output law: the choice is made before any token of the
block is drawn, and every draw in the block uses fresh randomness. Each delivered
token has the target's conditional law for every possible k, so the law of the
whole sequence does not depend on which k was chosen. Wall-clock timing can
therefore feed the rule freely. `tests/test_controller.py` checks this exactly.
"""

from collections import deque
from dataclasses import dataclass

import torch

from specdec.generate import BlockRecord, GenerationState

RECENT_PASSES = 64
RIDGE_STRENGTH = 4.0  # Equivalent to this many observations at the prior.
FEED_SCALE = 10.0  # Scales features to similar sizes so the ridge treats them evenly.
CONTEXT_SCALE = 500.0
DRAFT_EWMA_WEIGHT = 0.1


@dataclass(frozen=True)
class CostPrior:
    """Starting costs in seconds, from a calibration run on this machine and device."""

    draft_pass_s: float
    verify_base_s: float
    verify_per_token_s: float
    verify_per_context_token_s: float


@dataclass(frozen=True)
class ControllerConfig:
    max_block_length: int = 16
    prior_acceptance: float = 0.7
    prior_weight: float = 2.0  # Prior strength, in verified positions.
    half_life_positions: float = 8.0


def expected_tokens(acceptance: float, block_length: int) -> float:
    """Return the expected tokens from one block: 1 + alpha + ... + alpha^k."""
    return sum(acceptance**i for i in range(block_length + 1))


def best_block_length(
    acceptance: float,
    draft_pass_s: float,
    verify_s,
    max_block_length: int,
) -> int:
    """Return the k with the best expected token rate. Ties go to the shorter block."""
    best_length, best_rate = 0, 1.0 / verify_s(1)
    for block_length in range(1, max_block_length + 1):
        cost = block_length * draft_pass_s + verify_s(block_length + 1)
        rate = expected_tokens(acceptance, block_length) / cost
        if rate > best_rate:
            best_length, best_rate = block_length, rate
    return best_length


class AcceptanceEstimate:
    """A decayed average of per-position acceptance probabilities, pulled toward a prior.

    The evidence decays with every delivered token, including tokens from blocks
    with no draft. So a controller that fell back to k = 0 drifts back to the prior
    and tries drafting again, without a separate exploration rule.
    """

    def __init__(self, config: ControllerConfig):
        self.prior = config.prior_acceptance
        self.prior_weight = config.prior_weight
        self.decay = 0.5 ** (1.0 / config.half_life_positions)
        self.total = 0.0
        self.weight = 0.0

    def observe(self, block: BlockRecord) -> None:
        for overlap in block.overlaps:
            self.total = self.total * self.decay + overlap
            self.weight = self.weight * self.decay + 1.0
        for _ in range(block.delivered - len(block.overlaps)):
            self.total *= self.decay
            self.weight *= self.decay

    @property
    def value(self) -> float:
        evidence = self.total + self.prior * self.prior_weight
        return evidence / (self.weight + self.prior_weight)


class LatencyEstimate:
    """Online estimates of draft and verification cost, pulled toward a calibrated prior.

    Verification is fit by ridge regression toward the prior over recent passes.
    With no data the fit equals the prior, so there is no special start-up case.
    """

    def __init__(self, prior: CostPrior):
        self.draft_pass_s = prior.draft_pass_s
        self.prior_theta = torch.tensor(
            [
                prior.verify_base_s,
                prior.verify_per_token_s * FEED_SCALE,
                prior.verify_per_context_token_s * CONTEXT_SCALE,
            ],
            dtype=torch.float64,
        )
        self.theta = self.prior_theta.clone()
        self.recent: deque[tuple[float, float, float]] = deque(maxlen=RECENT_PASSES)

    def observe(self, block: BlockRecord) -> None:
        if block.draft_passes:
            sample = block.proposal_s / block.draft_passes
            self.draft_pass_s += DRAFT_EWMA_WEIGHT * (sample - self.draft_pass_s)
        seconds = block.verification_s + block.sampling_s
        self.recent.append((block.target_feed, block.context_length, seconds))
        self._refit()

    def _refit(self) -> None:
        rows = torch.tensor(
            [[1.0, feed / FEED_SCALE, context / CONTEXT_SCALE] for feed, context, _ in self.recent],
            dtype=torch.float64,
        )
        seconds = torch.tensor([sample[2] for sample in self.recent], dtype=torch.float64)
        residual = seconds - rows @ self.prior_theta
        system = rows.T @ rows + RIDGE_STRENGTH * torch.eye(3, dtype=torch.float64)
        self.theta = self.prior_theta + torch.linalg.solve(system, rows.T @ residual)

    def verify_s(self, feed: int, context: int) -> float:
        base, per_feed, per_context = self.theta.tolist()
        predicted = base + per_feed * feed / FEED_SCALE + per_context * context / CONTEXT_SCALE
        # A wild refit must not make a pass look free or negative.
        return max(predicted, 1e-5)


@dataclass(frozen=True)
class Decision:
    """One choice, kept so a recorded run can show why the block length changed."""

    block_index: int
    context_length: int
    acceptance: float
    draft_pass_s: float
    verify_one_s: float
    chosen: int


class AdaptiveController:
    """A block policy. Call it with the generation state to get the next block length."""

    def __init__(self, prior: CostPrior, config: ControllerConfig | None = None):
        self.config = config or ControllerConfig()
        self.acceptance = AcceptanceEstimate(self.config)
        self.latency = LatencyEstimate(prior)
        self.decisions: list[Decision] = []
        self.seen_blocks = 0

    def __call__(self, state: GenerationState) -> int:
        for block in state.blocks[self.seen_blocks :]:
            self.acceptance.observe(block)
            self.latency.observe(block)
        self.seen_blocks = len(state.blocks)

        context = len(state.tokens)
        acceptance = self.acceptance.value

        def verify_s(feed: int) -> float:
            return self.latency.verify_s(feed, context)

        chosen = best_block_length(
            acceptance, self.latency.draft_pass_s, verify_s, self.config.max_block_length
        )
        self.decisions.append(
            Decision(
                block_index=self.seen_blocks,
                context_length=context,
                acceptance=acceptance,
                draft_pass_s=self.latency.draft_pass_s,
                verify_one_s=verify_s(1),
                chosen=chosen,
            )
        )
        return chosen
