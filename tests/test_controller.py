"""The controller must choose sensibly and must not be able to change the output law."""

import pytest

from specdec.controller import (
    AcceptanceEstimate,
    AdaptiveController,
    ControllerConfig,
    CostPrior,
    LatencyEstimate,
    best_block_length,
    expected_tokens,
)
from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import BlockRecord, GenerationState
from specdec.sampling import SamplingConfig, to_probabilities
from tests.toy import run_engine, table_model

PRIOR = CostPrior(
    draft_pass_s=0.002,
    verify_base_s=0.02,
    verify_per_token_s=0.0002,
    verify_per_context_token_s=0.00001,
)


def block(overlaps=(), delivered=1, verification_s=0.02, proposal_s=0.0, passes=0, feed=1, ctx=10):
    return BlockRecord(
        block_length=passes,
        drafted=passes,
        accepted=len(overlaps),
        rejected_at=None,
        delivered=delivered,
        draft_tokens=(),
        context_length=ctx,
        target_feed=feed,
        draft_feed=passes,
        draft_passes=passes,
        overlaps=tuple(overlaps),
        proposal_s=proposal_s,
        verification_s=verification_s,
        sampling_s=0.0,
        rollback_s=0.0,
        finished_at_s=0.0,
    )


def flat_verify(seconds):
    return lambda feed: seconds


def test_expected_tokens_matches_the_closed_form():
    alpha, k = 0.6, 5
    assert expected_tokens(alpha, k) == pytest.approx((1 - alpha ** (k + 1)) / (1 - alpha))


def test_a_free_draft_with_perfect_acceptance_uses_the_longest_block():
    assert best_block_length(1.0, 0.0, flat_verify(0.02), 16) == 16


def test_a_draft_that_always_fails_is_not_used():
    assert best_block_length(0.0, 0.001, flat_verify(0.02), 16) == 0


def test_theorem_3_9_a_draft_costlier_than_its_acceptance_is_not_used():
    """Leviathan et al.: speculation needs alpha > c. Here alpha = 0.3 and c = 0.5."""
    assert best_block_length(0.3, 0.01, flat_verify(0.02), 16) == 0


def test_cheap_drafts_and_high_acceptance_lengthen_the_block():
    lengths = [best_block_length(alpha, 0.002, flat_verify(0.02), 32) for alpha in (0.5, 0.8, 0.95)]
    assert lengths == sorted(lengths)
    assert lengths[0] < lengths[-1]


def test_acceptance_estimate_starts_at_the_prior_and_follows_evidence():
    estimate = AcceptanceEstimate(ControllerConfig(prior_acceptance=0.7))
    assert estimate.value == pytest.approx(0.7)
    for _ in range(20):
        estimate.observe(block(overlaps=(0.1, 0.1), delivered=2))
    assert estimate.value < 0.25


def test_acceptance_estimate_returns_to_the_prior_without_drafting():
    """This is how a controller at k = 0 recovers: no exploration rule is needed."""
    estimate = AcceptanceEstimate(ControllerConfig(prior_acceptance=0.7, half_life_positions=4))
    for _ in range(20):
        estimate.observe(block(overlaps=(0.0,), delivered=1))
    low = estimate.value
    for _ in range(60):
        estimate.observe(block(delivered=1))
    assert low < 0.2
    assert estimate.value > 0.69


def test_latency_estimate_equals_the_prior_with_no_data():
    estimate = LatencyEstimate(PRIOR)
    expected = 0.02 + 5 * 0.0002 + 100 * 0.00001
    assert estimate.verify_s(5, 100) == pytest.approx(expected)


def test_latency_estimate_tracks_a_slower_machine():
    estimate = LatencyEstimate(PRIOR)
    for feed in (1, 3, 5, 9) * 16:
        estimate.observe(block(verification_s=0.06 + 0.001 * feed, feed=feed, ctx=100))
    assert estimate.verify_s(5, 100) == pytest.approx(0.065, rel=0.1)


def test_controller_shortens_after_rejections():
    controller = AdaptiveController(PRIOR)
    state = GenerationState(tokens=[1] * 50, prompt_length=10, remaining=100)
    opening = controller(state)
    state.blocks.extend(block(overlaps=(0.05,), passes=1, proposal_s=0.002) for _ in range(12))
    assert controller(state) < opening


def test_the_real_controller_cannot_change_the_output_law():
    target_fn, draft_fn = table_model(3, "target"), table_model(3, "draft")
    config = ControllerConfig(max_block_length=3, half_life_positions=1.0, prior_weight=0.5)

    def run(rng):
        controller = AdaptiveController(PRIOR, config)
        return tuple(run_engine(target_fn, draft_fn, rng, (0,), 3, 2, controller).new_tokens)

    def target_probabilities(prefix):
        return to_probabilities(target_fn(prefix), SamplingConfig())

    law = exact_law(run)
    assert total_variation(law, target_law(target_probabilities, (0,), 3, 2)) < 1e-11
