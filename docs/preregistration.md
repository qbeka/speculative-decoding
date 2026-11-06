# Preregistration

Committed before any test prompt ran. `config/frozen.json` holds the exact arms, and
`benchmarks/freeze_plan.py` regenerates it from the validation sweep in `runs/validation/v1`.

## Question

Can a controller that accounts for verification cost and recent acceptance beat the best
tuned fixed draft length, on the same hardware and workload, while preserving the target's
sampling law?

## Primary decision

| Item | Frozen value |
|---|---|
| Arm | `adaptive/distilled-8`, half-life 16 positions, prior acceptance 0.8 |
| Baseline | `fixed/distilled-8/k12`, the best fixed arm on validation receipts |
| Workload | `receipts_test`, 24 prompts from families never used in training or tuning |
| Batch size | 1 |
| Metric | Verified tokens per second, ratio of paired geometric means |
| Interval | 95 percent two-level bootstrap over prompts and repeats, 10,000 samples, seed 20260314 |
| Rule | Supported only if the interval's lower bound exceeds 1.00 |

Validation already points against this claim: on validation receipts, the adaptive arm ran
at 70.5 tokens/s and the fixed arm at 72.6. The claim stays primary because it is the
question this project set out to answer, and a negative result is a valid result.

## Secondary comparisons

These are reported with intervals and do not change the primary decision.

- Each adaptive arm against the fixed arm with the same draft, on both workloads and at
  batch sizes 1 and 4.
- One setting on both workloads: `adaptive/135m` against `fixed/135m/k3`, the best on
  general text, and against `fixed/135m/k12`, the best on receipts.
- Every arm against target-only, to show where speculation gets slower.
- This project's fixed arms against assisted generation in Transformers at the same draft
  and block length, and the library's own heuristic schedule. Assisted generation supports
  batch size 1 only.

## Data

| Split | Receipt families | Use |
|---|---|---|
| Train | grocery, restaurant, pharmacy, electronics, coffee, online order | Teacher sequences for distillation |
| Validation | fuel, hardware, ride | Choosing arms and settings |
| Test | hotel, bookstore, parking | This run only |

General text has 30 validation and 30 test prompts, and it measures what specialization costs.

## Measurement

Same target, tokenizer, float32 precision, temperature 0.7, 128 token limit, prompts and
machine for every arm. Each arm runs in a fresh process, in a shuffled order on every repeat.
Three repeats, plus one cold pass at batch size 1. Batch size 4 uses static batching with one
shared block length per round, which is a real limit of this engine. The 1 minute load average is
recorded with every request.

Recorded: request wall time, time to first token, median and p95 inter-token gap, verified
tokens per second, peak memory, accepted tokens per target pass and task accuracy. Rejected
tokens are never counted. The unit of replication is one prompt at batch size 1 and one
static batch above that.

## Not tested

EAGLE is left out, because it needs a trained draft head that does not exist for this target.
Latency under open-loop arrival is left out: batching here is static.

## Amendment after the first final run

The plan above did not change. `runs/final/v1` was discarded for a seed bug, and
`runs/final/v2` repeats the same plan with independent seeds. See decision 6 in
`DECISIONS.md`. The primary claim was not supported in v1, and that result was known
before v2 ran. Nothing was tuned between the two runs.

