"""Negative controls: broken samplers must be caught by the same checks.

If these tests stopped failing the broken code, the exact-law suite would be
proving nothing.
"""

import pytest
import torch

import specdec.generate
import specdec.sampling
from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import fixed_block_length
from specdec.sampling import SamplingConfig, to_probabilities, verify_block
from tests.toy import fixed_distribution_model, run_engine, table_model


def law_gap(target_fn, draft_fn, max_new_tokens=1, policy=None):
    """Return the total variation distance between engine law and target law."""

    def run(rng):
        return tuple(
            run_engine(target_fn, draft_fn, rng, (0,), max_new_tokens, None, policy).new_tokens
        )

    def target_probabilities(prefix):
        return to_probabilities(target_fn(prefix), SamplingConfig())

    expected = target_law(target_probabilities, (0,), max_new_tokens, None)
    return total_variation(exact_law(run), expected)


def test_resampling_from_the_target_after_rejection_is_wrong(monkeypatch):
    """The rejection counterexample, small enough to check by hand.

    Let p = (0.5, 0.5) and q = (1, 0). The draft always proposes token 0, and it
    is accepted half the time. Resampling from p then gives token 0 another
    0.5 * 0.5, for a total of 0.75 instead of 0.5. The residual gives token 1 all
    of the rejection branch, for exactly 0.5. The gap is 0.25.
    """
    target_fn = fixed_distribution_model([0.5, 0.5])
    draft_fn = fixed_distribution_model([1.0, 0.0])
    policy = fixed_block_length(1)

    assert law_gap(target_fn, draft_fn, 2, policy) < 1e-12

    monkeypatch.setattr(specdec.sampling, "residual_distribution", lambda p, q: p)
    # One block delivers one or two tokens, and the counterexample is about the first.
    assert law_gap(target_fn, draft_fn, 2, policy) == pytest.approx(0.25)


def test_a_residual_taken_from_the_wrong_row_is_caught(monkeypatch):
    original = specdec.sampling.residual_distribution
    monkeypatch.setattr(specdec.sampling, "residual_distribution", lambda p, q: original(q, p))
    assert law_gap(table_model(3, "target"), table_model(3, "draft"), 3) > 0.01


def test_an_off_by_one_target_row_is_caught(monkeypatch):
    original = specdec.generate.verify_block

    def shifted(p_rows, q_rows, draft_tokens, rng, eos_token_id=None):
        return original(p_rows.roll(1, dims=0), q_rows, draft_tokens, rng, eos_token_id)

    monkeypatch.setattr(specdec.generate, "verify_block", shifted)
    assert law_gap(table_model(3, "target"), table_model(3, "draft"), 3) > 0.01


def test_a_session_that_skips_rollback_is_caught(monkeypatch):
    monkeypatch.setattr("specdec.sessions.ToySession.rollback", lambda self, length: None)
    with pytest.raises((AssertionError, ValueError, IndexError)):
        assert law_gap(table_model(3, "target"), table_model(3, "draft"), 3) < 1e-9


def test_verify_block_rejects_a_token_the_draft_could_not_have_drawn():
    p_rows = torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    q_rows = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    class AlwaysAccept:
        def coin(self, probability):
            return True

        def categorical(self, probabilities):
            return 0

    with pytest.raises(ValueError, match="positive draft probability"):
        verify_block(p_rows, q_rows, [1], AlwaysAccept())
