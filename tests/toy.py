"""Tiny categorical models whose every sequence probability can be enumerated."""

import hashlib

import torch

from specdec.generate import (
    BlockPolicy,
    GenerationConfig,
    fixed_block_length,
    generate,
)
from specdec.randomness import Randomness
from specdec.sessions import ToySession


def table_model(vocab: int, name: str, sharpness: float = 2.0):
    """Return a deterministic prefix-to-logits function.

    The logits are a hash of the prefix, so each context has its own distribution
    and any cache that answers for the wrong prefix gives a different number.
    """

    def logits_for_prefix(prefix: tuple[int, ...]) -> torch.Tensor:
        digest = hashlib.sha256(f"{name}:{prefix}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63)
        generator = torch.Generator().manual_seed(seed)
        return sharpness * torch.randn(vocab, generator=generator, dtype=torch.float64)

    return logits_for_prefix


def fixed_distribution_model(probabilities: list[float]):
    """Return a model that ignores context, with exact zeros where requested."""
    table = torch.tensor(probabilities, dtype=torch.float64)
    logits = table.log()
    return lambda prefix: logits.clone()


def run_engine(
    target_fn,
    draft_fn,
    rng: Randomness,
    prompt: tuple[int, ...] = (0,),
    max_new_tokens: int = 3,
    eos_token_id: int | None = None,
    policy: BlockPolicy | None = None,
    **config_kwargs,
):
    """Run the real engine on toy sessions and return its Generation."""
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, **config_kwargs
    )
    return generate(
        ToySession(target_fn),
        ToySession(draft_fn),
        list(prompt),
        config,
        policy if policy is not None else fixed_block_length(2),
        rng,
    )
