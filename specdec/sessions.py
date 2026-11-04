"""A session is one model plus the cached state of one token sequence.

The engine only needs two operations, so the toy models used in tests and real
Hugging Face models are interchangeable:

- `extend` feeds new tokens and returns logits for the next token after each.
- `rollback` forgets every token past a given length.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import torch


class Session(Protocol):
    length: int  # Number of tokens the session has seen.

    def extend(self, tokens: Sequence[int], rows: int = 1) -> torch.Tensor:
        """Feed tokens and return the last `rows` rows of logits, shape (rows, vocab).

        Row -1 is the distribution after the final token.
        """
        ...

    def rollback(self, length: int) -> None:
        """Forget every token after the first `length` tokens."""
        ...


class ToySession:
    """A session over a pure function from prefix to logits.

    It holds its tokens explicitly, so a wrong rollback gives wrong logits
    instead of going unnoticed. It also counts the passes the engine makes.
    """

    def __init__(self, logits_for_prefix: Callable[[tuple[int, ...]], torch.Tensor]):
        self.logits_for_prefix = logits_for_prefix
        self.tokens: list[int] = []
        self.passes = 0

    @property
    def length(self) -> int:
        return len(self.tokens)

    def extend(self, tokens: Sequence[int], rows: int = 1) -> torch.Tensor:
        if not tokens:
            raise ValueError("extend needs at least one token")
        if not 1 <= rows <= len(tokens):
            raise ValueError("rows must be between 1 and the number of new tokens")
        self.passes += 1
        self.tokens.extend(tokens)
        ends = range(len(self.tokens) - rows + 1, len(self.tokens) + 1)
        return torch.stack([self.logits_for_prefix(tuple(self.tokens[:end])) for end in ends])

    def rollback(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot roll back to {length} from {self.length}")
        del self.tokens[length:]
