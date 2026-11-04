"""The two kinds of random draw the sampler makes.

The engine never calls a generator directly. It asks for a coin flip or a
categorical draw through this small interface, so a test can replace it with
`specdec.exhaustive.BranchingRandom` and walk every possible outcome.
"""

from typing import Protocol

import torch


class Randomness(Protocol):
    def coin(self, probability: float) -> bool:
        """Return True with the given probability."""
        ...

    def categorical(self, probabilities: torch.Tensor) -> int:
        """Return an index drawn from a normalized 1-D distribution."""
        ...


class TorchRandom:
    """Real randomness from a seeded CPU generator."""

    def __init__(self, seed: int):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)

    def coin(self, probability: float) -> bool:
        # uniform() is in [0, 1), so probability 1 always passes and 0 never does.
        return float(torch.rand(1, generator=self.generator, dtype=torch.float64)) < probability

    def categorical(self, probabilities: torch.Tensor) -> int:
        return int(torch.multinomial(probabilities, 1, generator=self.generator))
