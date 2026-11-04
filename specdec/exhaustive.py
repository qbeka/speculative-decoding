"""Compute the exact output law of the real engine by running every random path.

`BranchingRandom` replaces the random source. Each coin and each categorical
draw becomes a branch point, and the engine is rerun once per path with a
scripted list of choices. The probability of a path is the product of its branch
probabilities. Summing paths that deliver the same sequence gives the exact law
of the code under test, with no sampling error and no separate reimplementation
of the algorithm.

This is only practical for tiny models, which is all it is for.
"""

from collections import defaultdict
from collections.abc import Callable

import torch


class BranchingRandom:
    """Follows a scripted list of branches and records what it saw."""

    def __init__(self, script: list[int]):
        self.script = script
        self.position = 0
        self.weight = 1.0
        self.branch_counts: list[int] = []

    def _choose(self, weights: list[float]) -> int:
        """Pick a branch among those with positive weight and multiply its weight in."""
        options = [i for i, weight in enumerate(weights) if weight > 0]
        if not options:
            raise ValueError("a draw had no outcome with positive probability")
        scripted = self.script[self.position] if self.position < len(self.script) else 0
        self.branch_counts.append(len(options))
        self.position += 1
        choice = options[scripted]
        self.weight *= weights[choice]
        return choice

    def coin(self, probability: float) -> bool:
        return self._choose([probability, 1.0 - probability]) == 0

    def categorical(self, probabilities: torch.Tensor) -> int:
        return self._choose(probabilities.tolist())


def exact_law(run: Callable[[BranchingRandom], tuple[int, ...]]) -> dict[tuple[int, ...], float]:
    """Return {output: probability} for a run function, by depth-first path enumeration."""
    law: dict[tuple[int, ...], float] = defaultdict(float)
    script: list[int] = []
    while True:
        rng = BranchingRandom(script)
        output = run(rng)
        law[output] += rng.weight

        # Advance the path like an odometer: bump the deepest branch that has an
        # unused option, and drop everything after it.
        script = _pad(rng.script, len(rng.branch_counts))
        depth = len(script) - 1
        while depth >= 0 and script[depth] + 1 >= rng.branch_counts[depth]:
            depth -= 1
        if depth < 0:
            return dict(law)
        script = [*script[:depth], script[depth] + 1]


def _pad(script: list[int], length: int) -> list[int]:
    return script[:length] + [0] * (length - len(script))


def target_law(
    target_probabilities: Callable[[tuple[int, ...]], torch.Tensor],
    prompt: tuple[int, ...],
    max_new_tokens: int,
    eos_token_id: int | None,
) -> dict[tuple[int, ...], float]:
    """Return the law of plain autoregressive sampling by the chain rule.

    This is the reference. It shares no code with the engine.
    """
    law: dict[tuple[int, ...], float] = {}

    def walk(sequence: tuple[int, ...], probability: float, remaining: int) -> None:
        if remaining == 0 or (sequence and sequence[-1] == eos_token_id):
            law[sequence] = law.get(sequence, 0.0) + probability
            return
        distribution = target_probabilities((*prompt, *sequence))
        for token, token_probability in enumerate(distribution.tolist()):
            if token_probability > 0:
                walk((*sequence, token), probability * token_probability, remaining - 1)

    walk((), 1.0, max_new_tokens)
    return law


def total_variation(
    first: dict[tuple[int, ...], float], second: dict[tuple[int, ...], float]
) -> float:
    """Half the L1 distance between two laws over token sequences."""
    keys = first.keys() | second.keys()
    return 0.5 * sum(abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in keys)
