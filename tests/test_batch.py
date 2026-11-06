"""The batch engine must give each request the same law and logits as a lone run."""

import pytest
import torch

from specdec.batch import BatchSession, ToyBatch, generate_batch
from specdec.controller import AdaptiveController, ControllerConfig, CostPrior
from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import GenerationConfig, fixed_block_length, generate
from specdec.models import ModelSession
from specdec.sampling import SamplingConfig, to_probabilities
from tests.test_model_session import tiny_llama
from tests.toy import table_model

TOLERANCE = 1e-9


class Shared:
    """One branching source used by every request, so paths cover the joint outcome."""

    def __init__(self, rng):
        self.rng = rng

    def coin(self, probability):
        return self.rng.coin(probability)

    def categorical(self, probabilities):
        return self.rng.categorical(probabilities)


@pytest.fixture(scope="module")
def target_model():
    return tiny_llama(0)


@pytest.fixture(scope="module")
def draft_model():
    return tiny_llama(1, layers=1, hidden=8)


def test_padded_batch_logits_match_lone_sessions(target_model):
    prompts = [[1, 2, 3, 4], [2], [3, 1]]
    batch = BatchSession(target_model, 3)
    rows = batch.extend(prompts, [len(prompt) for prompt in prompts])
    for prompt, row in zip(prompts, rows, strict=True):
        lone = ModelSession(target_model).extend(prompt, rows=len(prompt))
        assert torch.allclose(row, lone, atol=TOLERANCE)


def test_logits_stay_right_across_holes_and_compaction(target_model):
    """Reject tokens on one sequence, continue, compact, and compare with a lone session."""
    batch = BatchSession(target_model, 2)
    batch.extend([[1, 2, 3], [4, 3]], [0, 0])
    batch.extend([[4, 0, 1], [2, 2, 2]], [0, 0])
    batch.rollback(0, 4)  # Sequence 0 keeps 1 2 3 4 and leaves two holes.
    batch.rollback(1, 3)  # Sequence 1 keeps 4 3 2.

    rows = batch.extend([[2], [1, 0]], [1, 1])
    assert torch.allclose(
        rows[0], ModelSession(target_model).extend([1, 2, 3, 4, 2]), atol=TOLERANCE
    )
    assert torch.allclose(
        rows[1], ModelSession(target_model).extend([4, 3, 2, 1, 0]), atol=TOLERANCE
    )

    batch.compact()
    assert batch.valid.shape[1] == max(batch.lengths)
    rows = batch.extend([[3], [3]], [1, 1])
    assert torch.allclose(
        rows[0], ModelSession(target_model).extend([1, 2, 3, 4, 2, 3]), atol=TOLERANCE
    )
    assert torch.allclose(
        rows[1], ModelSession(target_model).extend([4, 3, 2, 1, 0, 3]), atol=TOLERANCE
    )


def test_keep_drops_sequences_and_preserves_the_rest(target_model):
    batch = BatchSession(target_model, 3)
    batch.extend([[1, 2], [3, 4], [2, 1]], [0, 0, 0])
    batch.keep([2, 0])
    rows = batch.extend([[3], [3]], [1, 1])
    assert torch.allclose(rows[0], ModelSession(target_model).extend([2, 1, 3]), atol=TOLERANCE)
    assert torch.allclose(rows[1], ModelSession(target_model).extend([1, 2, 3]), atol=TOLERANCE)


def toy_batch_gap(make_policy, max_new_tokens):
    """Return the distance between the batch's joint law and the product of target laws."""
    target_fn, draft_fn = table_model(3, "target"), table_model(3, "draft")
    prompts = [[0, 1], [1, 2]]
    config = GenerationConfig(max_new_tokens=max_new_tokens, eos_token_id=2)

    def run(rng):
        shared = Shared(rng)
        generations = generate_batch(
            ToyBatch(target_fn, 2),
            ToyBatch(draft_fn, 2),
            prompts,
            config,
            make_policy(),
            [shared, shared],
        )
        return tuple(tuple(generation.new_tokens) for generation in generations)

    sampling = SamplingConfig()
    first, second = (
        target_law(
            lambda prefix: to_probabilities(target_fn(prefix), sampling),
            tuple(p),
            max_new_tokens,
            2,
        )
        for p in prompts
    )
    expected = {(a, b): pa * pb for a, pa in first.items() for b, pb in second.items()}
    return total_variation(exact_law(run), expected)


@pytest.mark.parametrize("block_length", [0, 1, 2])
def test_batched_toy_law_is_the_product_of_target_laws(block_length):
    # Three new tokens leave room for a block of two, so the length is not clamped.
    assert toy_batch_gap(lambda: fixed_block_length(block_length), 3) < 1e-11


def test_the_real_controller_keeps_the_joint_law_of_a_batch():
    prior = CostPrior(0.003, 0.03, 0.001, 0.0)
    config = ControllerConfig(max_block_length=2, half_life_positions=1.0, prior_weight=0.5)
    assert toy_batch_gap(lambda: AdaptiveController(prior, config), 3) < 1e-11


def test_a_batch_of_one_keeps_the_law_at_length_two_on_a_real_model(target_model, draft_model):
    """A joint law over two requests at this length is too large to enumerate here."""
    prompt = [1, 2, 3]
    config = GenerationConfig(max_new_tokens=3, eos_token_id=3)

    def run(rng):
        generation = generate_batch(
            BatchSession(target_model, 1),
            BatchSession(draft_model, 1),
            [prompt],
            config,
            fixed_block_length(2),
            [rng],
        )[0]
        return tuple(generation.new_tokens)

    sampling = SamplingConfig()

    def probabilities(prefix):
        return to_probabilities(ModelSession(target_model).extend(list(prefix))[-1], sampling)

    expected = target_law(probabilities, tuple(prompt), 3, 3)
    assert total_variation(exact_law(run), expected) < 1e-8


@pytest.mark.parametrize("block_length", [0, 1])
def test_batched_law_on_a_real_cached_model(target_model, draft_model, block_length):
    """Exact law through padding, holes, masks and early finishers on a real model."""
    prompts = [[1, 2, 3], [4, 3]]
    config = GenerationConfig(max_new_tokens=2, eos_token_id=3)

    def run(rng):
        shared = Shared(rng)
        generations = generate_batch(
            BatchSession(target_model, 2),
            BatchSession(draft_model, 2),
            prompts,
            config,
            fixed_block_length(block_length),
            [shared, shared],
        )
        return tuple(tuple(generation.new_tokens) for generation in generations)

    sampling = SamplingConfig()

    def probabilities(prefix):
        return to_probabilities(ModelSession(target_model).extend(list(prefix))[-1], sampling)

    first, second = (target_law(probabilities, tuple(p), 2, 3) for p in prompts)
    expected = {(a, b): pa * pb for a, pa in first.items() for b, pb in second.items()}
    assert total_variation(exact_law(run), expected) < 1e-8


@pytest.mark.parametrize("block_length", [0, 2, 4])
def test_batched_greedy_equals_lone_greedy_with_different_finish_times(
    target_model, draft_model, block_length
):
    prompts = [[1, 2], [3, 4, 2, 1, 0], [2, 2, 2]]
    config = GenerationConfig(max_new_tokens=14, eos_token_id=3, greedy=True)
    lone = [
        generate(ModelSession(target_model), None, prompt, config, fixed_block_length(0), None)
        for prompt in prompts
    ]
    batched = generate_batch(
        BatchSession(target_model, 3),
        BatchSession(draft_model, 3),
        prompts,
        config,
        fixed_block_length(block_length),
        [None, None, None],
    )
    assert [g.tokens for g in batched] == [g.tokens for g in lone]


def test_prompts_of_one_token_are_refused_in_a_batch(target_model):
    with pytest.raises(ValueError, match="two tokens"):
        generate_batch(
            BatchSession(target_model, 1),
            None,
            [[1]],
            GenerationConfig(max_new_tokens=1),
            fixed_block_length(0),
            [None],
        )
