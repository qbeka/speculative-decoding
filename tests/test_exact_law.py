"""The output law of the engine must equal the target law exactly.

Each test walks every random path of the real engine, so a pass is a calculation
and not a statistical claim. The only tolerance is float64 rounding.
"""

import pytest
import torch

from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import fixed_block_length
from specdec.sampling import SamplingConfig, to_probabilities
from tests.toy import fixed_distribution_model, run_engine, table_model

ROUNDING = 1e-11


def assert_matches_target(
    target_fn, draft_fn, *, max_new_tokens=3, eos=None, policy=None, sampling=None
):
    prompt = (0,)
    sampling = sampling or SamplingConfig()

    def run(rng):
        generation = run_engine(
            target_fn, draft_fn, rng, prompt, max_new_tokens, eos, policy, sampling=sampling
        )
        return tuple(generation.new_tokens)

    def target_probabilities(prefix):
        return to_probabilities(target_fn(prefix), sampling)

    produced = exact_law(run)
    expected = target_law(target_probabilities, prompt, max_new_tokens, eos)
    assert sum(produced.values()) == pytest.approx(1.0, abs=ROUNDING)
    assert total_variation(produced, expected) < ROUNDING


@pytest.mark.parametrize("block_length", [0, 1, 2, 3, 4, 9])
def test_law_for_each_fixed_block_length(block_length):
    assert_matches_target(
        table_model(3, "target"),
        table_model(3, "draft"),
        policy=fixed_block_length(block_length),
    )


@pytest.mark.parametrize("eos", [0, 1, 2])
@pytest.mark.parametrize("block_length", [1, 2, 3])
def test_law_when_eos_can_land_anywhere_in_a_block(eos, block_length):
    assert_matches_target(
        table_model(3, "target"),
        table_model(3, "draft"),
        max_new_tokens=4,
        eos=eos,
        policy=fixed_block_length(block_length),
    )


@pytest.mark.parametrize(
    "sampling",
    [
        SamplingConfig(temperature=0.5),
        SamplingConfig(top_k=2),
        SamplingConfig(top_p=0.7),
        SamplingConfig(temperature=1.7, top_k=3, top_p=0.9),
    ],
)
def test_law_under_sampling_filters(sampling):
    assert_matches_target(table_model(4, "target"), table_model(4, "draft"), sampling=sampling)


def test_law_with_identical_distributions():
    model = table_model(3, "same")
    assert_matches_target(model, model)


def test_law_with_disjoint_supports():
    assert_matches_target(
        fixed_distribution_model([0.5, 0.5, 0.0, 0.0]),
        fixed_distribution_model([0.0, 0.0, 0.5, 0.5]),
    )


def test_law_when_draft_misses_part_of_target_support():
    assert_matches_target(
        fixed_distribution_model([0.2, 0.3, 0.5]),
        fixed_distribution_model([1.0, 0.0, 0.0]),
    )


def test_law_when_target_rules_out_a_draft_token():
    assert_matches_target(
        fixed_distribution_model([1.0, 0.0, 0.0]),
        fixed_distribution_model([0.2, 0.3, 0.5]),
    )


def test_law_when_a_policy_reacts_to_past_rejections():
    """Block length may depend on anything already decided, including rejections."""

    def reactive(state):
        if not state.blocks:
            return 3
        return 0 if state.blocks[-1].rejected_at is not None else 2

    assert_matches_target(
        table_model(3, "target"), table_model(3, "draft"), max_new_tokens=4, policy=reactive
    )


def test_law_when_a_policy_reacts_to_the_delivered_tokens():
    assert_matches_target(
        table_model(3, "target"),
        table_model(3, "draft"),
        max_new_tokens=4,
        policy=lambda state: state.tokens[-1] + 1,
    )


def test_every_path_stays_within_budget():
    target_fn, draft_fn = table_model(2, "target"), table_model(2, "draft")
    law = exact_law(
        lambda rng: tuple(
            run_engine(
                target_fn, draft_fn, rng, max_new_tokens=4, policy=fixed_block_length(9)
            ).new_tokens
        )
    )
    assert {len(sequence) for sequence in law} == {4}


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    logits = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64).log()
    filtered = to_probabilities(logits, SamplingConfig(top_p=0.6))
    assert filtered.tolist() == pytest.approx([0.625, 0.375, 0.0])
