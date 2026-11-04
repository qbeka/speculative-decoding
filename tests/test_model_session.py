"""Cached logits must equal full-prefix recomputation, including after any rollback.

These tests use a tiny randomly initialized Llama in float64 on the CPU, so they
need no download and rounding cannot hide a real difference.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import GenerationConfig, fixed_block_length, generate
from specdec.models import ModelSession
from specdec.sampling import SamplingConfig, to_probabilities

VOCAB = 5
TOLERANCE = 1e-10


def tiny_llama(seed: int, layers: int = 2, hidden: int = 16) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=hidden,
        intermediate_size=2 * hidden,
        num_hidden_layers=layers,
        num_attention_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config).to(torch.float64).eval()


@pytest.fixture(scope="module")
def target_model():
    return tiny_llama(0)


@pytest.fixture(scope="module")
def draft_model():
    return tiny_llama(1, layers=1, hidden=8)


def full_prefix_logits(model, tokens):
    """Logits with no cache at all, the reference for every cached result."""
    return ModelSession(model).extend(tokens)[-1]


def test_cached_logits_match_full_prefix_recomputation(target_model):
    tokens = [1, 2, 3, 4, 0, 1, 2]
    session = ModelSession(target_model)
    session.extend(tokens[:2])
    for i in range(2, len(tokens)):
        logits = session.extend([tokens[i]])[-1]
        assert torch.allclose(
            logits, full_prefix_logits(target_model, tokens[: i + 1]), atol=TOLERANCE
        )


def test_a_batched_block_matches_one_token_at_a_time(target_model):
    prefix, block = [1, 2, 3], [4, 0, 1, 2]
    batched = ModelSession(target_model)
    batched.extend(prefix[:-1])
    rows = batched.extend([prefix[-1], *block], rows=len(block) + 1)
    for i in range(len(block) + 1):
        expected = full_prefix_logits(target_model, prefix + block[:i])
        assert torch.allclose(rows[i], expected, atol=TOLERANCE)


@pytest.mark.parametrize("keep", range(1, 7))
def test_rollback_to_every_position_restores_the_exact_state(target_model, keep):
    """Force a rejection at every position, then continue with a different token."""
    drafted = [1, 2, 3, 4, 0, 1]
    corrected = (drafted[keep] + 1) % VOCAB if keep < len(drafted) else 2
    session = ModelSession(target_model)
    session.extend(drafted)
    session.rollback(keep)
    assert session.length == keep
    assert session.cache.get_seq_length() == keep

    logits = session.extend([corrected])[-1]
    expected = full_prefix_logits(target_model, [*drafted[:keep], corrected])
    assert torch.allclose(logits, expected, atol=TOLERANCE)


def test_repeated_rollback_and_extend_do_not_drift(target_model):
    session = ModelSession(target_model)
    session.extend([1, 2, 3])
    for token in [4, 0, 1, 2, 3, 4]:
        session.extend([token, token, token])
        session.rollback(3)
    logits = session.extend([2])[-1]
    assert torch.allclose(logits, full_prefix_logits(target_model, [1, 2, 3, 2]), atol=TOLERANCE)


def test_rollback_refuses_to_move_forward(target_model):
    session = ModelSession(target_model)
    session.extend([1, 2])
    with pytest.raises(ValueError):
        session.rollback(3)


def session_probabilities(model, sampling):
    def probabilities(prefix):
        return to_probabilities(ModelSession(model).extend(list(prefix))[-1], sampling)

    return probabilities


@pytest.mark.parametrize("block_length", [1, 2, 3])
def test_engine_law_on_a_real_cached_model_matches_the_target(
    target_model, draft_model, block_length
):
    """The exact-law check, now through real caches, rollback and masks."""
    sampling = SamplingConfig(temperature=1.0)
    prompt = (1, 2)
    config = GenerationConfig(max_new_tokens=3, eos_token_id=3, sampling=sampling)

    def run(rng):
        generation = generate(
            ModelSession(target_model),
            ModelSession(draft_model),
            list(prompt),
            config,
            fixed_block_length(block_length),
            rng,
        )
        return tuple(generation.new_tokens)

    expected = target_law(session_probabilities(target_model, sampling), prompt, 3, 3)
    assert total_variation(exact_law(run), expected) < 1e-9


@pytest.mark.parametrize("block_length", [0, 1, 2, 5])
def test_greedy_equals_target_only_on_a_real_cached_model(target_model, draft_model, block_length):
    config = GenerationConfig(max_new_tokens=12, greedy=True)
    reference = generate(
        ModelSession(target_model), None, [1, 2], config, fixed_block_length(0), rng=None
    )
    speculative = generate(
        ModelSession(target_model),
        ModelSession(draft_model),
        [1, 2],
        config,
        fixed_block_length(block_length),
        rng=None,
    )
    assert speculative.tokens == reference.tokens


def test_draft_that_lagged_behind_catches_up(target_model, draft_model):
    """A policy may skip the draft for a while, so its cache must refill correctly."""
    schedule = iter([2, 0, 0, 3, 0, 2, 2, 2, 2, 2, 2, 2])
    config = GenerationConfig(max_new_tokens=10, greedy=True)
    draft = ModelSession(draft_model)
    generation = generate(
        ModelSession(target_model), draft, [1, 2], config, lambda state: next(schedule), None
    )
    reference = generate(
        ModelSession(target_model), None, [1, 2], config, fixed_block_length(0), None
    )
    assert generation.tokens == reference.tokens
    assert draft.length <= len(generation.tokens) - 1
