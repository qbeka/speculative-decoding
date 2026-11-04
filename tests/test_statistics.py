"""Stochastic draws must match the exact law, with the real random source.

Seeds are fixed so the test is repeatable. Failure means the counts sit in the far
tail of the exact law, which a correct sampler does about one time in 1000.
"""

import math
from collections import Counter

import torch

from specdec.exhaustive import target_law
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig, to_probabilities
from tests.toy import run_engine, table_model

DRAWS = 20_000
ALPHA = 1e-3


def chi_square_p_value(observed: Counter, expected: dict, draws: int) -> float:
    """Pearson test, with the survival function from the regularized gamma function."""
    statistic = sum(
        (observed.get(outcome, 0) - draws * probability) ** 2 / (draws * probability)
        for outcome, probability in expected.items()
    )
    degrees = len(expected) - 1
    return float(torch.special.gammaincc(torch.tensor(degrees / 2.0), torch.tensor(statistic / 2.0)))


def sample_counts(target_fn, draft_fn, draws, seed, block_length=2):
    from specdec.generate import fixed_block_length

    rng = TorchRandom(seed)
    return Counter(
        tuple(
            run_engine(
                target_fn, draft_fn, rng, (0,), 2, None, fixed_block_length(block_length)
            ).new_tokens
        )
        for _ in range(draws)
    )


def expected_law(target_fn):
    return target_law(lambda prefix: to_probabilities(target_fn(prefix), SamplingConfig()), (0,), 2, None)


def test_draws_match_the_exact_target_law():
    target_fn, draft_fn = table_model(3, "target", 1.0), table_model(3, "draft", 1.0)
    expected = expected_law(target_fn)
    assert min(expected.values()) * DRAWS > 5  # The chi-square approximation needs this.
    observed = sample_counts(target_fn, draft_fn, DRAWS, seed=1234)
    assert chi_square_p_value(observed, expected, DRAWS) > ALPHA


def test_the_statistical_check_detects_a_wrong_law():
    """Negative control: the draft's own law must fail against the target's."""
    target_fn, draft_fn = table_model(3, "target", 1.0), table_model(3, "draft", 1.0)
    observed = sample_counts(draft_fn, draft_fn, DRAWS, seed=1234)
    assert chi_square_p_value(observed, expected_law(target_fn), DRAWS) < ALPHA


def test_chi_square_helper_on_a_known_value():
    # 549 of 1000 fair flips gives a statistic of 9.604, which is 3.1 standard
    # deviations and a two-sided p value of about 0.0019.
    p_value = chi_square_p_value(Counter(a=549, b=451), {"a": 0.5, "b": 0.5}, 1000)
    assert math.isclose(p_value, 0.0019, abs_tol=0.0001)
