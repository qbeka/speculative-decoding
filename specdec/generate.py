"""One speculative generation loop for one sequence.

Invariant at the top of every block: `tokens` is the committed sequence, and
each session holds a prefix of `tokens` that does not include the last token.
The target therefore always has at least one token left to feed, and the row it
returns for that token is the distribution of the next position.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import torch

from specdec.randomness import Randomness
from specdec.sampling import (
    BlockOutcome,
    SamplingConfig,
    greedy_token,
    to_probabilities,
    verify_block,
    verify_greedy,
)
from specdec.sessions import Session

# Decides the next draft length from history only. It runs before any draft
# token of the block exists, so it cannot see the randomness of that block.
BlockPolicy = Callable[["GenerationState"], int]


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    eos_token_id: int | None = None
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    greedy: bool = False


@dataclass(frozen=True)
class BlockRecord:
    """Everything measured about one block. Times are seconds."""

    block_length: int  # Draft tokens requested after clamping to the budget.
    drafted: int  # Draft tokens produced, which is fewer if the draft hit EOS.
    accepted: int
    rejected_at: int | None
    delivered: int  # Tokens committed by this block.
    context_length: int  # Committed tokens when the block started.
    target_feed: int  # Tokens fed to the target in its one pass.
    draft_feed: int  # Tokens fed to the draft across its passes.
    proposal_s: float
    verification_s: float
    sampling_s: float
    rollback_s: float
    finished_at_s: float  # Since the start of generation.


@dataclass
class GenerationState:
    """What a block policy may read: the past and nothing else."""

    tokens: list[int]
    prompt_length: int
    remaining: int
    blocks: list[BlockRecord] = field(default_factory=list)


@dataclass
class Generation:
    tokens: list[int]  # Prompt plus delivered tokens.
    prompt_length: int
    blocks: list[BlockRecord]
    prefill_s: float
    wall_s: float
    finish_reason: str  # "eos" or "length".

    @property
    def new_tokens(self) -> list[int]:
        return self.tokens[self.prompt_length :]


def fixed_block_length(block_length: int) -> BlockPolicy:
    """The baseline policy: the same draft length for every block."""
    return lambda state: block_length


def propose(
    draft: Session,
    tokens: list[int],
    block_length: int,
    config: GenerationConfig,
    rng: Randomness,
) -> tuple[list[int], torch.Tensor | None, int]:
    """Draw up to block_length tokens from the draft, one pass per token after the first.

    Returns the draft tokens, their q rows and the number of tokens fed. In greedy
    mode the rows are raw logits and are not used by verification.
    """
    feed = tokens[draft.length :]
    logits = draft.extend(feed)
    fed = len(feed)
    drafted: list[int] = []
    rows: list[torch.Tensor] = []
    while True:
        if config.greedy:
            rows.append(logits[-1])
            token = greedy_token(logits[-1])
        else:
            row = to_probabilities(logits[-1], config.sampling)
            rows.append(row)
            token = rng.categorical(row)
        drafted.append(token)
        # A drafted EOS ends the block. Tokens after EOS would mean nothing.
        if len(drafted) == block_length or token == config.eos_token_id:
            return drafted, torch.stack(rows), fed
        logits = draft.extend([token])
        fed += 1


def generate(
    target: Session,
    draft: Session | None,
    prompt: Sequence[int],
    config: GenerationConfig,
    policy: BlockPolicy,
    rng: Randomness,
) -> Generation:
    """Generate up to config.max_new_tokens tokens. A block length of 0 is plain decoding."""
    if not prompt:
        raise ValueError("the prompt needs at least one token")
    if config.max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    started = perf_counter()
    tokens = list(prompt)
    target.extend(tokens[:-1], rows=1) if len(tokens) > 1 else None
    if draft is not None and len(tokens) > 1:
        draft.extend(tokens[:-1], rows=1)
    prefill_s = perf_counter() - started

    state = GenerationState(
        tokens=tokens, prompt_length=len(prompt), remaining=config.max_new_tokens
    )
    finish_reason = "length"
    while state.remaining > 0:
        # Clamping to remaining - 1 means the bonus token always fits the budget.
        requested = policy(state) if draft is not None else 0
        block_length = min(requested, state.remaining - 1)
        context_length = len(tokens)

        tick = perf_counter()
        draft_tokens: list[int] = []
        q_rows = None
        draft_feed = 0
        if block_length > 0:
            draft_tokens, q_rows, draft_feed = propose(draft, tokens, block_length, config, rng)
        proposal_s = perf_counter() - tick

        tick = perf_counter()
        feed = tokens[target.length :] + draft_tokens
        target_logits = target.extend(feed, rows=len(draft_tokens) + 1)
        verification_s = perf_counter() - tick

        tick = perf_counter()
        outcome = _verify(target_logits, q_rows, draft_tokens, config, rng)
        sampling_s = perf_counter() - tick

        delivered = draft_tokens[: outcome.accepted]
        if outcome.next_token is not None:
            delivered = [*delivered, outcome.next_token]
        tokens.extend(delivered)
        state.remaining -= len(delivered)

        # Rejected tokens leave both caches here. The last committed token stays
        # out of the caches so the next block has a token to feed.
        tick = perf_counter()
        target.rollback(len(tokens) - 1)
        if draft is not None and draft.length > len(tokens) - 1:
            draft.rollback(len(tokens) - 1)
        rollback_s = perf_counter() - tick

        state.blocks.append(
            BlockRecord(
                block_length=block_length,
                drafted=len(draft_tokens),
                accepted=outcome.accepted,
                rejected_at=outcome.rejected_at,
                delivered=len(delivered),
                context_length=context_length,
                target_feed=len(feed),
                draft_feed=draft_feed,
                proposal_s=proposal_s,
                verification_s=verification_s,
                sampling_s=sampling_s,
                rollback_s=rollback_s,
                finished_at_s=perf_counter() - started,
            )
        )
        if delivered and delivered[-1] == config.eos_token_id:
            finish_reason = "eos"
            break

    return Generation(
        tokens=tokens,
        prompt_length=len(prompt),
        blocks=state.blocks,
        prefill_s=prefill_s,
        wall_s=perf_counter() - started,
        finish_reason=finish_reason,
    )


def _verify(
    target_logits: torch.Tensor,
    q_rows: torch.Tensor | None,
    draft_tokens: list[int],
    config: GenerationConfig,
    rng: Randomness,
) -> BlockOutcome:
    if config.greedy:
        return verify_greedy(target_logits, draft_tokens)
    p_rows = to_probabilities(target_logits, config.sampling)
    if q_rows is None:
        q_rows = p_rows[:0]
    return verify_block(p_rows, q_rows, draft_tokens, rng, config.eos_token_id)
