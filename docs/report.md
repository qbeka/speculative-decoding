# Benchmark report

Timing numbers come from `runs/final/v2`, unless a section names another file. Commit `0070a9a`
produced the run, and the manifest records its hashes, hardware and command.
`runs/final/v2/report/tables.md` has every table. Rebuild it with `make site`.

## Setup

- **Target:** SmolLM2 1.7B, float32 on Metal, temperature 0.7, 128 token limit.
- **Drafts:** stock SmolLM2 135M, and two distilled drafts of it, 8 and 30 layers.
- **Workloads:** 24 receipts from three unseen template families, and 30 general prompts.
- **Replication:** 3 repeats with independent seeds. One unit is one prompt at batch size 1
  and one static batch at batch size 4.
- **Intervals:** the plan named a two-level bootstrap, and the primary decision uses it. That
  interval conditions on these three passes, because one repeat of one arm is one process, and a
  slow pass hits every prompt in it. So this report quotes a crossed bootstrap everywhere else,
  which resamples prompts and repeats separately. Three repeats estimate that variation only
  roughly, and batch size 4 has just 6 or 8 units. Per-repeat ratios are in `tables.md`.

## Primary decision

| Arm | Baseline | Ratio | Interval | Rule | Result |
|---|---|---|---|---|---|
| `adaptive/distilled-8` | `fixed/distilled-8/k12` | 0.915 | 0.866 to 0.968 | Lower bound above 1.00 | Not supported |

The crossed interval is 0.858 to 0.972, and the three repeats gave 0.90, 0.90 and 0.95. In this
run the fixed length was faster. The discarded first run gave 1.04, interval 0.97 to 1.12, and it
is not valid evidence, but it is a reason not to lean on the size of this gap. The claim is
unsupported either way.

## Speed against the large model alone

Batch size 1, ratio of verified tokens per second.

| Arm | Receipts | General text |
|---|---|---|
| `fixed/distilled-30/k12` | 2.27 (2.05 to 2.55) | 0.48 (0.43 to 0.52) |
| `fixed/distilled-8/k12` | 2.02 (1.82 to 2.27) | 0.37 (0.32 to 0.41) |
| `adaptive/distilled-30` | 1.93 (1.86 to 2.02) | 1.07 (0.94 to 1.18) |
| `adaptive/distilled-8` | 1.85 (1.71 to 2.01) | 0.87 (0.83 to 0.92) |
| `fixed/135m/k3` | 1.73 (1.58 to 1.98) | 1.23 (1.16 to 1.31) |
| `adaptive/135m` | 1.68 (1.54 to 1.92) | 1.22 (1.16 to 1.29) |
| `fixed/135m/k12` | 1.51 (1.36 to 1.70) | 0.74 (0.69 to 0.81) |

The large model alone ran 16 percent slower on all 24 receipts in its first repeat, which lifts
that repeat's ratios and widens these intervals.

![Speed against the large model alone](../runs/final/v2/report/speedup.png)

## What the evidence shows

**A tuned fixed length beats the controller on its own workload.** Acceptance on receipts is not
steady: at length 12, 45 percent of blocks rejected at the first position and 19 percent accepted
all 12. A long block still pays, because a target pass costs far more than a draft pass. The
controller's loss is too many verification passes. It averaged a length of 4.8 and used 46 percent
more target passes (1,812 against 1,240). That added 3.6 ms per token in verification and saved
2.4 ms in proposals. Its estimate also ran low, at 0.73 against a true overlap of 0.84, so it
stayed cautious after rejections. It restarts from its prior on every request. Its own computation
is about 0.14 ms per block, so that is not the cost.

**The controller limits the damage from wrong tuning.** With the 8-layer draft on general
text, a fixed length of 12 ran at 0.37x. The controller shortened its drafts, to length 1 in 77
percent of blocks, and lost less at 0.87x, a gain of 2.37x (2.07 to 2.64). That is still slower than
no speculation. The 30-layer draft shows the same effect: 2.23x (1.89 to 2.66), ending at 1.07x of
the large model alone (0.94 to 1.18).

**It does not replace a sensible short length.** With the stock draft, `adaptive/135m` was close
to `fixed/135m/k3` on both workloads: 0.97 on receipts (0.94 to 1.01) and 0.99 on general text
(0.96 to 1.02). `fixed/135m/k3` needed no controller and matched or beat every adaptive arm on
general text. The controller only helps when the chosen length is far from right.

**Distillation is the larger gain.** At the same size and draft cost, the 30-layer distilled
draft beat the stock draft at the better of its two tested lengths by 1.31x (1.23 to 1.39). This
was computed after the run and was not in the frozen plan. The cost is that the same draft lost to
the stock draft on general text.

**A shallow draft overfits templates.** Mean overlap with the target on training, then
validation, then test receipts was 0.993, 0.907 and 0.842 for the 8-layer draft, against 0.992,
0.974 and 0.956 for the 30-layer draft. The 8-layer draft costs less per pass, but it generalized
worse, and the 30-layer draft won on the test split.

**The engine matches a mature implementation.** At the same draft and length, it ran at 1.03x
(0.97 to 1.08) of assisted generation in Transformers on receipts, and 0.99x (0.96 to 1.02) on
general text. The planned comparison with the library's heuristic schedule was not delivered: that
arm ran the library defaults, a constant length of 20 with a confidence threshold of 0.4. See
decision 7.

## Where speculation gets slower

| Condition | Ratio to the large model alone |
|---|---|
| Distilled draft at fixed length, general text, batch size 1 | 0.37 to 0.48 |
| Stock draft at length 12, general text, batch size 1 | 0.74 |
| Adaptive with the 8-layer draft, general text, batch size 1 | 0.87 (0.83 to 0.92) |
| Distilled draft at fixed length, general text, batch size 4 | 0.37 to 0.40 |
| Adaptive with the 8-layer draft, general text, batch size 4 | 0.84 |

The stock draft stayed above 1.00 at batch size 1 at length 3. At batch size 4 on general text,
`fixed/135m/k2` reached 1.11 (1.05 to 1.17). `adaptive/135m` reached 1.04 (0.97 to 1.10), which
is not clearly faster than no speculation.

## Concurrency

Static batching shares one draft length across the batch, so one hard sequence shortens or
wastes work for all four. The best batch size 4 ratio on receipts was 1.49 (1.28 to 1.65),
against 2.27 at batch size 1. On receipts, the large model alone gained the most from batching,
2.93x on the same prompts against at most 2.18x for the other arms, which narrows what speculation
can add. On general text, the gains were closer.

## Latency and memory

| Receipts, batch size 1 | Median TTFT | Median gap | p95 gap | Tokens per pass |
|---|---|---|---|---|
| Large model alone | 0.168 s | 34.4 ms | 43.0 ms | 1.00 |
| `fixed/distilled-30/k12` | 0.277 s | 0.0 ms | 149.0 ms | 9.43 |
| `adaptive/distilled-30` | 0.207 s | 0.0 ms | 115.6 ms | 5.63 |

A median gap of zero means most tokens arrive in a group with the token before them. Requests
finish sooner, but the stream is burstier and the first token comes later.

GPU memory was 7.6 GB for the large model alone and at most 7.9 GB with a draft at batch size
1, and up to 9.3 GB at batch size 4. Process peak RSS read 10.7 GB in every arm, because the
peak happens while loading weights, so it does not separate the arms.

## Correctness

- **Exact law on tiny models.** `specdec/exhaustive.py` walks every random path through the
  real engine and compares the result with the target's probabilities.
- **Exact law on real weights.** `runs/law_audit/mps.json`: 1,500 draws per arm over 3 new tokens,
  135M draft, batch size 1, drafts of up to 2 tokens. p of 0.150 and 0.084 at a fixed length, 0.608
  and 0.594 adaptive. A draft-only control failed on both prompts, with p too small for float32 to
  hold, and a repeat of the target scored 0.109 and 0.414. Length 12, distilled drafts and
  batches were not audited on real weights.
- **Batches on tiny models.** Exact joint-law tests cover a fixed length of 2 and the real
  controller, and a real tiny model in a batch of one at length 2.
- **Batch equals single.** At equal seeds, batch size 1 and batch size 4 gave identical tokens on
  324 of 324 requests compared: the target and `fixed/distilled-30/k12` on both workloads.
- **Precision.** float16 changes the target's distribution with the block size, so the benchmark
  uses float32. float32 still varies by up to 4e-5 total variation per position, so exactness holds
  up to that rounding. See decision 2.

## Task accuracy

Receipt field accuracy was 0.637 for the large model alone, and 0.611 to 0.661 across arms at
batch size 1. All arms share one output law, so these differences are sampling noise. This score
measures the model, not the sampler.

## Failures that changed the work

See `DECISIONS.md`. The main ones are a discarded final run caused by a seed bug, float16 failing
the law check, a training run that exhausted memory and a mislabeled baseline. `reviews.md` lists
what the review found.
