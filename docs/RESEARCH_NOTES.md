# Research notes

What each primary source says, what this project took from it, and where this
project differs. All sources were read on 2026-11-04.

Notation warning: Leviathan et al. call the target p and the draft q. Chen et al.
use the opposite letters. This project uses Leviathan's.

## Leviathan, Kalman and Matias, ICML 2023

Version read: PMLR 202:19274-19286, the arXiv 2211.17192 v2 text.

Adopted:

- Acceptance min(1, p(x)/q(x)) and the residual norm(max(0, p - q)), in
  `specdec/sampling.py`.
- The capped geometric token count (1 - a^(k+1)) / (1 - a), Equation 1, in
  `expected_tokens`.
- Corollary 3.6, acceptance equals the expected sum of min(p, q). The controller
  uses this per position in place of noisy accept and reject outcomes.
- Theorem 3.8, the improvement factor (1 - a^(k+1)) / ((1 - a)(kc + 1)), and
  Corollary 3.9, speculation helps only if acceptance exceeds c.

Assumptions in the paper that do not hold here:

- The paper treats c as negligible (below 0.05). Here c is about 0.3 to 0.5 for the
  off-the-shelf draft. See decision 1 in `DECISIONS.md`.
- The paper assumes the target's pass cost does not grow with block size. Here it
  grows a little: a 17-token pass costs about 1.3 times a 1-token pass.
- The paper's model has acceptance independent across positions. The controller
  makes the same simplification, but only within one block, and re-estimates
  acceptance between blocks.

## Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling

Version read: arXiv 2302.01318.

Adopted:

- The role of parallel scoring. Chen explains why scoring k tokens costs about the
  same as one: the work is bound by memory and the cache size does not change. The
  profile in `runs/profile/` reproduces this on Metal.
- The indexing of Algorithm 2. Each loop returns at most K + 1 tokens, because the
  last target row gives a free sample if every draft token is accepted. At least
  one token always comes out.
- Sampling transforms are applied before the rejection test.

Differences:

- Chen's result is exact "within hardware numerics" and leaves that unquantified.
  This project measures it. At float16 on Metal the numerics are not negligible,
  which is decision 2 in `DECISIONS.md`.
- Chen trains an 8-layer draft for latency: 1.8 ms against 14.1 ms per token, so c
  is about 0.13. That matches the lesson of decision 1: depth drives cost.
- Chen serves on 16 TPU v4s. This project is a single Apple M5 Pro.

## Li et al., EAGLE-2

Version read: arXiv 2406.16858 v2.

EAGLE-2 chooses a draft tree per context. It uses the draft's confidence score as a
cheap estimate of acceptance and ranks tree nodes by the product of confidences
along the path. It reports 3.05x to 4.26x speedups across three model series.

This project has a narrower scope on purpose:

| | EAGLE-2 | This project |
|---|---|---|
| Shape | Tree of candidates | Single chain |
| Decides | Which nodes to expand, during drafting | One block length, before drafting |
| Signal | Draft confidence for the current context | Recent exact acceptance, cost and context length |
| Draft | Trained jointly with target features | An ordinary causal model |
| Cost model | Fixed node budget | Measured seconds on this machine |

The single chain keeps the exactness argument short: the block length is chosen
before any draw of the block. A context-aware rule like EAGLE-2's would have to
reason about stopping on a drafted token.

Gap: EAGLE needs a draft trained for each target, and none exists for SmolLM2. The
[SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) license is not a standard
SPDX license, so its code is not bundled. No direct EAGLE comparison was run.

## Hugging Face assisted generation

Version read: `transformers` 5.22.1, `generation/utils.py`.

This is the mature implementation used as a baseline. With sampling it calls
`_speculative_sampling`, which is the same rule in float32 on the device. Its
"heuristic" schedule adds 2 to the draft length after a fully accepted block and
subtracts 1 otherwise. It raises an error above batch size 1, so it can only be
compared at batch size 1.
