"""Edge cases of the block verification math."""

import pytest
import torch

from specdec.exhaustive import exact_law
from specdec.sampling import (
    RESIDUAL_MASS_FLOOR,
    SamplingConfig,
    acceptance_probability,
    greedy_token,
    residual_distribution,
    to_probabilities,
    verify_block,
    verify_greedy,
)


def distribution(*values):
    return torch.tensor(values, dtype=torch.float64)


class Scripted:
    def __init__(self, coins):
        self.coins = list(coins)

    def coin(self, probability):
        return self.coins.pop(0)

    def categorical(self, probabilities):
        return int(probabilities.argmax())


def test_acceptance_is_capped_at_one():
    assert acceptance_probability(0.8, 0.2) == 1.0
    assert acceptance_probability(0.2, 0.8) == pytest.approx(0.25)
    assert acceptance_probability(0.0, 0.8) == 0.0


def test_residual_is_the_normalized_positive_part():
    residual = residual_distribution(distribution(0.5, 0.3, 0.2), distribution(0.2, 0.6, 0.2))
    assert residual.tolist() == pytest.approx([1.0, 0.0, 0.0])


def test_residual_equals_target_on_disjoint_supports():
    p, q = distribution(0.5, 0.5, 0.0), distribution(0.0, 0.0, 1.0)
    assert residual_distribution(p, q).tolist() == p.tolist()


def test_residual_falls_back_to_target_when_mass_is_only_rounding():
    p = distribution(0.5, 0.5)
    almost_p = p + distribution(RESIDUAL_MASS_FLOOR / 4, -RESIDUAL_MASS_FLOOR / 4)
    assert residual_distribution(p, almost_p).tolist() == p.tolist()


def test_identical_distributions_never_reject():
    p = distribution(0.25, 0.75)
    outcomes = exact_law(
        lambda rng: (verify_block(p[None].repeat(3, 1), p[None].repeat(2, 1), [0, 1], rng).accepted,)
    )
    assert outcomes == {(2,): pytest.approx(1.0)}


def test_rejection_reports_its_position_and_keeps_earlier_tokens():
    p_rows = distribution(0.5, 0.5)[None].repeat(3, 1)
    q_rows = distribution(0.9, 0.1)[None].repeat(2, 1)
    outcome = verify_block(p_rows, q_rows, [0, 0], Scripted([True, False]))
    assert (outcome.accepted, outcome.rejected_at, outcome.next_token) == (1, 1, 1)


def test_accepted_eos_ends_the_block_without_a_bonus():
    p_rows = distribution(0.5, 0.5)[None].repeat(2, 1)
    q_rows = distribution(0.5, 0.5)[None]
    outcome = verify_block(p_rows, q_rows, [1], Scripted([True]), eos_token_id=1)
    assert (outcome.accepted, outcome.next_token) == (1, None)


def test_all_accepted_block_draws_the_bonus_from_the_last_row():
    p_rows = torch.stack([distribution(0.5, 0.5), distribution(0.1, 0.9)])
    q_rows = distribution(0.5, 0.5)[None]
    outcome = verify_block(p_rows, q_rows, [0], Scripted([True]))
    assert (outcome.accepted, outcome.next_token, outcome.rejected_at) == (1, 1, None)


def test_greedy_ties_go_to_the_lowest_token_id():
    assert greedy_token(distribution(1.0, 3.0, 3.0, 2.0)) == 1


def test_greedy_keeps_matches_and_corrects_the_first_mismatch():
    logits = torch.stack(
        [distribution(0.0, 9.0), distribution(9.0, 0.0), distribution(0.0, 9.0)]
    )
    outcome = verify_greedy(logits, [1, 1])
    assert (outcome.accepted, outcome.next_token, outcome.rejected_at) == (1, 0, 1)


def test_greedy_stops_at_an_accepted_eos():
    logits = torch.stack([distribution(0.0, 9.0), distribution(9.0, 0.0)])
    outcome = verify_greedy(logits, [1], eos_token_id=1)
    assert (outcome.accepted, outcome.next_token) == (1, None)


def test_sampling_config_refuses_values_that_mean_something_else():
    with pytest.raises(ValueError, match="greedy"):
        SamplingConfig(temperature=0.0)
    with pytest.raises(ValueError):
        SamplingConfig(top_p=0.0)


def test_filtered_tokens_get_exactly_zero():
    probabilities = to_probabilities(distribution(3.0, 2.0, 1.0), SamplingConfig(top_k=2))
    assert probabilities[2] == 0.0
    assert probabilities.sum() == pytest.approx(1.0)
