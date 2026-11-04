"""Speculative decoding for a static batch of requests.

All requests in a batch start together and share target passes. Requests are not
added while a batch runs, so this is static batching, not continuous batching.

`BatchSession` keeps one cache for the whole batch plus a mask of which cache
columns belong to which sequence. Rolling back a sequence clears mask bits and
moves no tensors. The rejected columns stay as holes that attention ignores, until
`compact` removes them. Each sequence gets its own random source, so a request's
output law does not depend on the other requests in its batch.
"""

from collections import Counter
from dataclasses import replace
from time import perf_counter
from typing import Protocol

import torch
from transformers import DynamicCache, PreTrainedModel

from specdec.generate import (
    BlockPolicy,
    BlockRecord,
    Generation,
    GenerationConfig,
    GenerationState,
    _verify,
)
from specdec.models import synchronize
from specdec.randomness import Randomness
from specdec.sampling import greedy_token, to_probabilities
from specdec.sessions import ToySession

PAD_TOKEN = 0  # Padding positions are masked out, so the token id has no effect.
MAX_HOLES = 32  # Compact once the cache is this much wider than the longest sequence.


class BatchedSessions(Protocol):
    lengths: list[int]

    def extend(self, feeds: list[list[int]], rows: list[int]) -> list[torch.Tensor | None]:
        """Feed each sequence its tokens and return the last rows[i] rows for sequence i."""
        ...

    def rollback(self, index: int, length: int) -> None: ...

    def keep(self, indices: list[int]) -> None:
        """Keep only these sequences, in this order."""
        ...


class ToyBatch:
    """A batch of toy sessions, so the exact-law tests can run the batch loop."""

    def __init__(self, logits_for_prefix, batch_size: int):
        self.sessions = [ToySession(logits_for_prefix) for _ in range(batch_size)]

    @property
    def lengths(self) -> list[int]:
        return [session.length for session in self.sessions]

    def extend(self, feeds, rows):
        return [
            session.extend(feed, count) if count else _extend_only(session, feed)
            for session, feed, count in zip(self.sessions, feeds, rows, strict=True)
        ]

    def rollback(self, index, length):
        self.sessions[index].rollback(length)

    def keep(self, indices):
        self.sessions = [self.sessions[i] for i in indices]


def _extend_only(session: ToySession, feed: list[int]) -> None:
    if feed:
        session.extend(feed)
    return None


class BatchSession:
    """One model, one batched cache and a validity mask with shape (batch, width)."""

    def __init__(self, model: PreTrainedModel, batch_size: int):
        self.model = model
        self.device = next(model.parameters()).device
        self.cache = DynamicCache(config=model.config)
        self.valid = torch.zeros(batch_size, 0, dtype=torch.bool)
        self.lengths = [0] * batch_size
        self.seconds: Counter[str] = Counter()

    def extend(self, feeds: list[list[int]], rows: list[int]) -> list[torch.Tensor | None]:
        width = max(len(feed) for feed in feeds)
        input_ids = torch.full((len(feeds), width), PAD_TOKEN, dtype=torch.long)
        position_ids = torch.zeros((len(feeds), width), dtype=torch.long)
        feed_valid = torch.zeros((len(feeds), width), dtype=torch.bool)
        for i, feed in enumerate(feeds):
            input_ids[i, : len(feed)] = torch.tensor(feed, dtype=torch.long)
            feed_valid[i, : len(feed)] = True
            # Positions count real tokens only, so holes never shift a sequence.
            position_ids[i] = torch.arange(self.lengths[i], self.lengths[i] + width)
        self.valid = torch.cat([self.valid, feed_valid], dim=1)

        started = perf_counter()
        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids.to(self.device),
                position_ids=position_ids.to(self.device),
                attention_mask=self.valid.long().to(self.device),
                past_key_values=self.cache,
                use_cache=True,
                logits_to_keep=width,
            )
        forward_done = perf_counter()
        synchronize(self.device)
        sync_done = perf_counter()
        logits = output.logits.cpu().to(torch.float64)
        self.seconds["forward"] += forward_done - started
        self.seconds["sync"] += sync_done - forward_done
        self.seconds["transfer"] += perf_counter() - sync_done

        results: list[torch.Tensor | None] = []
        for i, (feed, count) in enumerate(zip(feeds, rows, strict=True)):
            self.lengths[i] += len(feed)
            results.append(logits[i, len(feed) - count : len(feed)] if count else None)
        return results

    def rollback(self, index: int, length: int) -> None:
        if not 0 <= length <= self.lengths[index]:
            raise ValueError(f"cannot roll back to {length} from {self.lengths[index]}")
        columns = self.valid[index].nonzero().squeeze(1)
        self.valid[index, columns[length:]] = False
        self.lengths[index] = length

    @torch.inference_mode()
    def compact(self) -> None:
        """Move each sequence's valid columns to the left and cut the cache width."""
        width = max(self.lengths)
        index = torch.zeros(len(self.lengths), width, dtype=torch.long)
        valid = torch.zeros(len(self.lengths), width, dtype=torch.bool)
        for i, length in enumerate(self.lengths):
            index[i, :length] = self.valid[i].nonzero().squeeze(1)
            valid[i, :length] = True
        index = index.to(self.device)
        for layer in self.cache.layers:
            gather = index[:, None, :, None].expand(
                -1, layer.keys.shape[1], -1, layer.keys.shape[3]
            )
            layer.keys = layer.keys.gather(2, gather)
            layer.values = layer.values.gather(2, gather)
        self.valid = valid

    def compact_if_needed(self) -> None:
        if self.valid.shape[1] > max(self.lengths) + MAX_HOLES:
            self.compact()

    def keep(self, indices: list[int]) -> None:
        self.cache.batch_select_indices(torch.tensor(indices, device=self.device))
        self.valid = self.valid[indices]
        self.lengths = [self.lengths[i] for i in indices]
        self.compact_if_needed()


def propose_batch(
    draft: BatchedSessions,
    sequences: list[list[int]],
    block_lengths: list[int],
    config: GenerationConfig,
    rngs: list[Randomness],
) -> tuple[list[list[int]], list[torch.Tensor | None], int]:
    """Draft every sequence at once, one shared pass per draft position."""
    count = len(sequences)
    drafted: list[list[int]] = [[] for _ in range(count)]
    rows: list[list[torch.Tensor]] = [[] for _ in range(count)]
    feeds = [
        sequence[draft.lengths[i] :] if block_lengths[i] > 0 else []
        for i, sequence in enumerate(sequences)
    ]
    passes = 0
    while any(feeds):
        wanted = [1 if feed else 0 for feed in feeds]
        logits = draft.extend(feeds, wanted)
        passes += 1
        feeds = [[] for _ in range(count)]
        for i, row in enumerate(logits):
            if row is None:
                continue
            if config.greedy:
                rows[i].append(row[-1])
                token = greedy_token(row[-1])
            else:
                probabilities = to_probabilities(row[-1], config.sampling)
                rows[i].append(probabilities)
                token = rngs[i].categorical(probabilities)
            drafted[i].append(token)
            if len(drafted[i]) < block_lengths[i] and token != config.eos_token_id:
                feeds[i] = [token]
    stacked = [torch.stack(row_list) if row_list else None for row_list in rows]
    return drafted, stacked, passes


def generate_batch(
    target: BatchedSessions,
    draft: BatchedSessions | None,
    prompts: list[list[int]],
    config: GenerationConfig,
    policy: BlockPolicy,
    rngs: list[Randomness],
) -> list[Generation]:
    """Generate for all prompts. One policy chooses one block length per round.

    The block length is shared because a pass costs the whole batch the same, and
    the longest block sets that cost. A single policy object sees every request's
    blocks, so its acceptance estimate pools the batch.
    """
    if any(len(prompt) < 2 for prompt in prompts):
        raise ValueError("batched prompts need at least two tokens")

    started = perf_counter()
    target.extend([prompt[:-1] for prompt in prompts], [0] * len(prompts))
    if draft is not None:
        draft.extend([prompt[:-1] for prompt in prompts], [0] * len(prompts))
    prefill_s = perf_counter() - started

    tokens = [list(prompt) for prompt in prompts]
    remaining = [config.max_new_tokens] * len(prompts)
    blocks: list[list[BlockRecord]] = [[] for _ in prompts]
    finish = ["length"] * len(prompts)
    wall = [0.0] * len(prompts)
    shared_state = GenerationState(tokens=tokens[0], prompt_length=0, remaining=0)

    active = list(range(len(prompts)))  # active[slot] is the request in that slot.
    while active:
        shared_state.tokens = max((tokens[r] for r in active), key=len)
        requested = policy(shared_state) if draft is not None else 0
        block_lengths = [min(requested, remaining[r] - 1) for r in active]
        sequences = [tokens[r] for r in active]
        active_rngs = [rngs[r] for r in active]

        tick = perf_counter()
        drafted: list[list[int]] = [[] for _ in active]
        q_rows: list[torch.Tensor | None] = [None] * len(active)
        draft_passes = 0
        if any(block_lengths):
            drafted, q_rows, draft_passes = propose_batch(
                draft, sequences, block_lengths, config, active_rngs
            )
        proposal_s = perf_counter() - tick

        tick = perf_counter()
        feeds = [
            sequence[target.lengths[slot] :] + drafted[slot]
            for slot, sequence in enumerate(sequences)
        ]
        logits = target.extend(feeds, [len(d) + 1 for d in drafted])
        verification_s = perf_counter() - tick

        finished_now = perf_counter() - started
        still_active: list[int] = []
        tick = perf_counter()
        records: list[tuple[int, BlockRecord]] = []
        for slot, request in enumerate(active):
            context_length = len(tokens[request])
            outcome, overlaps = _verify(
                logits[slot], q_rows[slot], drafted[slot], config, active_rngs[slot]
            )
            delivered = drafted[slot][: outcome.accepted]
            if outcome.next_token is not None:
                delivered = [*delivered, outcome.next_token]
            tokens[request].extend(delivered)
            remaining[request] -= len(delivered)

            target.rollback(slot, len(tokens[request]) - 1)
            if draft is not None and draft.lengths[slot] > len(tokens[request]) - 1:
                draft.rollback(slot, len(tokens[request]) - 1)

            records.append(
                (
                    request,
                    BlockRecord(
                        block_length=block_lengths[slot],
                        drafted=len(drafted[slot]),
                        accepted=outcome.accepted,
                        rejected_at=outcome.rejected_at,
                        delivered=len(delivered),
                        draft_tokens=tuple(drafted[slot]),
                        context_length=context_length,
                        target_feed=max(len(feed) for feed in feeds),
                        draft_feed=len(drafted[slot]),
                        draft_passes=draft_passes,
                        overlaps=overlaps,
                        proposal_s=proposal_s,
                        verification_s=verification_s,
                        sampling_s=0.0,
                        rollback_s=0.0,
                        finished_at_s=finished_now,
                    ),
                )
            )
            if delivered and delivered[-1] == config.eos_token_id:
                finish[request] = "eos"
            elif remaining[request] > 0:
                still_active.append(slot)
        sampling_s = perf_counter() - tick

        for request, record in records:
            record = replace(record, sampling_s=sampling_s / len(records))
            blocks[request].append(record)
            shared_state.blocks.append(record)

        finished_at = perf_counter() - started
        for slot, request in enumerate(active):
            if slot not in still_active:
                wall[request] = finished_at
        if len(still_active) < len(active):
            if still_active:
                target.keep(still_active)
                if draft is not None:
                    draft.keep(still_active)
        active = [active[slot] for slot in still_active]
        if active:
            for sessions in (target, draft):
                if isinstance(sessions, BatchSession):
                    sessions.compact_if_needed()

    return [
        Generation(
            tokens=tokens[request],
            prompt_length=len(prompts[request]),
            blocks=blocks[request],
            prefill_s=prefill_s,
            wall_s=wall[request],
            finish_reason=finish[request],
        )
        for request in range(len(prompts))
    ]
