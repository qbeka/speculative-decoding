# Contributing

## Setup

```bash
uv sync --extra bench
uv run pytest -q
```

## Where things go

- Sampler math: `specdec/sampling.py`. No models or caches.
- The loop for one sequence: `specdec/generate.py`.
- The loop for a batch: `specdec/batch.py`.
- Block length rule: `specdec/controller.py`.
- Benchmark settings: `config/frozen.json`. Change it only before a final run.

## Rules

- A sampler change needs an exact-law test and a negative control that fails.
- A timing claim needs a raw record under `runs/`. Rerun the report from raw data.
- A final run never overwrites raw records. It gets a new folder.
- Weights and credentials stay out of git. Inputs are synthetic or written by hand.
- Prose has no em dashes. Commits are Conventional Commits, in lower case, with no
  period: `type(scope): imperative summary`. Scopes: `sampler`, `engine`, `batch`,
  `controller`, `models`, `data`, `distill`, `bench`, `demo`, `cli`, `docs`, `tests`,
  `infra`. No AI attribution trailers.

## How we write

Write for a reader who has never seen the project. Start with what they need. Use
"you", the active voice and short sentences. Give real numbers and say when they
were measured.

## Before you commit

```bash
uv run ruff format --check . && uv run ruff check . && uv run pytest -q
```
