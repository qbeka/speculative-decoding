# Speculative decoding

A local inference engine that makes a large language model faster without changing what
it says. A small model drafts tokens, the large model checks them in one pass, and a
controller decides how far to draft.

The project asks one question: can a controller that tracks verification cost and recent
acceptance beat the best tuned fixed draft length? On the frozen test, **the answer is no.**
A well tuned fixed length won. The controller's value showed up somewhere else: when the
text stopped looking like what the system was tuned for, it lost far less than the fixed
length did. A short fixed length on the stock draft avoided that loss too, so the controller
is protection against a badly chosen length, not a speedup.

[Open the replay](site/index.html) to watch three decoders work the same recorded prompt.

| | |
|---|---|
| **Fastest on unseen receipts** | 2.27x the large model alone, 95% interval 2.05 to 2.55. Median 62.7 tokens/s against 28.1, with a distilled draft at a fixed length of 12 |
| **Main claim** | Not supported. Adaptive ran at 0.92x of the best fixed length, interval 0.87 to 0.97, and the rule needed the whole interval above 1.00 |
| **When the text changes** | The same fixed length fell to 0.37x of the large model alone. Adaptive lost less at 0.87x, a gain of 2.37x, interval 2.07 to 2.64 |
| **Distillation** | 1.31x faster than the stock draft at the better of its two tested lengths, at the same size and draft cost, interval 1.23 to 1.39. This comparison was not in the frozen plan |
| **Accepted work** | 9.4 tokens delivered per large model pass on unseen receipts |
| **Exact sampling** | Every path enumerated on small cases. On real weights, 1,500 draws per arm at drafts of up to 2 tokens gave p from 0.08 to 0.61, and a control that used the draft alone failed |
| **Against Transformers** | Same speed as assisted generation at the same draft and length: 1.03x on receipts, interval 0.97 to 1.08, and 0.99x on general text, interval 0.96 to 1.02 |
| **Concurrent requests** | 1.49x at batch size 4, interval 1.28 to 1.65, with static batching |
| **Tests** | 123 tests, including exact-law checks that walk every path through the real engine |
| **Scale** | 2,754 timed requests: 17 arms, 54 prompts, 3 repeats |

Measured on one Apple M5 Pro with 48 GB of memory, using float32 on Metal and SmolLM2 models.
The test receipts are synthetic. [What the numbers leave out](#what-the-numbers-leave-out)
has the details. Intervals here resample prompts and repeats separately, except the main claim,
which uses the interval the plan named.

## Run it

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the locked
dependencies.

```bash
uv sync --all-extras
```

Run the smoke test. It checks the exact law on a tiny model, then downloads the models, about
4 GB, and runs real inference.

```bash
uv run specdec smoke
```

Generate text with a draft of length 4, then compare it with the large model alone.

```bash
uv run specdec generate "Write one sentence about the sea." --block-length 4
```

```bash
uv run specdec generate "Write one sentence about the sea." --block-length 0
```

## Check it

Run the correctness suite. It needs no downloaded weights and takes about 30 seconds.

```bash
uv run pytest -q
```

Rebuild every table, plot and the replay from the raw records. No model runs, but the
tokenizer downloads on first use.

```bash
make site
```

| Command | What it does | Time |
|---|---|---:|
| `uv run pytest -q` | Runs the correctness suite | 30 s |
| `uv run python benchmarks/law_audit.py --out runs/law_audit/new.json` | Tests the exact law on real weights | Hours |
| `uv run specdec teacher` | Records teacher sequences for distillation | 20 min |
| `uv run specdec distill --layers 8 --out artifacts/drafts/distilled-8` | Trains a draft. Repeat with `--layers 30 --steps 750` | 1 to 2 hours each |
| `uv run python benchmarks/freeze_plan.py --validation runs/validation/v1` | Writes `config/frozen.json` from a validation sweep | Seconds |
| `uv run specdec final --out runs/final/new` | Runs the frozen plan. Close other work first | 2.8 hours |

## How it works

1. The controller picks a draft length from past acceptance and measured latency, before any
   token of the block is drawn. That order is what keeps the output exact.
2. The small model samples that many tokens.
3. The large model scores them all in one pass.
4. Each draft token is accepted with probability min(1, p/q). The first rejection is
   resampled from the residual distribution. If all are accepted, a bonus token is drawn.
5. Both models roll their caches back to the last accepted token.

The controller estimates acceptance as a moving average of the overlap between the two
models, and fits verification cost as a line in block size and context length. It then
picks the length with the highest expected tokens per second.

| Path | What it holds |
|---|---|
| `specdec/sampling.py` | Accept, residual and bonus math. No models or caches |
| `specdec/generate.py`, `specdec/batch.py` | The decoding loops for one sequence and for a batch |
| `specdec/controller.py` | The block length rule |
| `specdec/exhaustive.py` | Walks every random path through the real engine |
| `specdec/distill.py` | Trains a draft from the target on training receipts |
| `tests/test_exact_law.py`, `tests/test_counterexamples.py` | The law, and broken samplers that must fail |
| `config/frozen.json` | The arms frozen before the test split ran |
| `runs/final/v2/` | Raw records, manifest, tables and plots of the final run |
| [`docs/report.md`](docs/report.md) | Every result and what it means |
| [`docs/preregistration.md`](docs/preregistration.md) | The plan, frozen first |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | Decisions and failed experiments |
| [`docs/RESEARCH_NOTES.md`](docs/RESEARCH_NOTES.md) | What the papers say and where this differs |
| [`docs/artifacts.md`](docs/artifacts.md) | Which command produced which file |
| [`docs/reviews.md`](docs/reviews.md) | Review findings and fixes |

## What the numbers leave out

- **One machine, three passes.** Every number comes from one laptop. Other work ran on it, with
  a median 1 minute load average of 3.8 and a peak of 16.8, sampled once per pass. Arm order was
  shuffled, but a loaded pass runs slower, and the large model alone was 16 percent slower on
  every receipt in its first pass. Three repeats give only a rough view of that variation.
- **The first final run was thrown out.** A bug gave nearly every request the same seed. The
  rerun used the same frozen plan and the same arm order, and only it is quoted here. The
  discarded run had the main ratio at 1.04x, interval 0.97 to 1.12, so the size of adaptive's
  loss is less certain than one run suggests. See decision 6 in `docs/DECISIONS.md`.
- **Validation was thin.** It ran the first 12 prompts of each file, so the `ride` family never
  ran, on one repeat with the same seed bug. It only chose settings and was not rerun, so those
  settings may be weaker than they look. It picked the 8-layer draft for the main test, and the
  30-layer draft then beat it on the test split.
- **One baseline ran the wrong settings.** The arm meant to test the library's heuristic schedule
  ran the library defaults instead, so that comparison was not delivered. See decision 7.
- **Speculation is burstier.** Tokens arrive in groups. On receipts, the fastest arm's median
  time to first token was 0.28 s against 0.17 s for the large model alone, and its p95 gap
  between tokens was 149 ms against 43 ms.
- **Distilled drafts pay for their focus.** On general text, the fastest receipt setup ran at
  0.48x of the large model alone. At batch size 4 on general text, every distilled setup tested
  was slower than no speculation at all, and only the stock draft at a fixed length beat it.
- **Batching is static.** One draft length is shared per round, and new requests cannot join
  a running batch. This is not an open-loop serving test.
- **Exact up to rounding.** float32 still varies a little with block size, up to 4e-5 total
  variation per position. The law audit on real weights covered short drafts at batch size 1.
  Longer drafts and batches are checked by exact tests on tiny models.
- **Small models.** The target is SmolLM2 1.7B. Larger targets change the cost ratio that all
  of this depends on. Field accuracy on receipts was about 64 percent for every arm, and the
  arms differ only by sampling noise because they share one output law.
- **EAGLE was not tested.** It needs a trained draft head that does not exist for this target.

## Credits

Speculative sampling is from Leviathan, Kalman and Matias (ICML 2023) and Chen et al.
(arXiv 2302.01318). The models are
[SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) from Hugging Face under
Apache 2.0. The replay uses Geist Serif under the SIL Open Font License. The receipts are
synthetic and contain no customer data.

MIT License.
