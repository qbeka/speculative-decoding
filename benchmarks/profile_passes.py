"""Measure the cost of one model pass by feed size and context length.

Run from the repository root:
    uv run python benchmarks/profile_passes.py --out runs/profile/mps_float16.json
"""

import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from specdec.models import DRAFT_135M, DRAFT_360M, TARGET, ModelSession, best_device, load_model
from specdec.sampling import SamplingConfig, to_probabilities

MODELS = {"target": TARGET, "draft_135m": DRAFT_135M, "draft_360m": DRAFT_360M}
DTYPES = {"float16": torch.float16, "float32": torch.float32}
CONTEXTS = [32, 128, 512]
FEEDS = [1, 2, 3, 5, 9, 17]
REPEATS = 20


def measure_pass(session: ModelSession, context: int, feed: int) -> dict:
    """Time extend then rollback at a fixed context. Each repeat restores the state."""
    tokens = list(range(1000, 1000 + feed))
    samples = []
    for repeat in range(REPEATS + 3):
        session.seconds.clear()
        started = perf_counter()
        session.extend(tokens, rows=feed)
        session.rollback(context)
        total = perf_counter() - started
        if repeat >= 3:  # The first passes compile Metal shaders for this shape.
            samples.append({"total_s": total, **session.seconds})
    return {
        "context": context,
        "feed": feed,
        "samples": len(samples),
        "median_total_s": median(sample["total_s"] for sample in samples),
        "median_seconds": {
            part: median(sample[part] for sample in samples)
            for part in ("forward", "sync", "transfer", "rollback")
        },
    }


def measure_sampling_cost(vocab: int) -> dict:
    logits = torch.randn(vocab, dtype=torch.float64)
    results = {}
    configs = {"temperature": SamplingConfig(0.7), "top_p": SamplingConfig(0.7, top_p=0.95)}
    for name, config in configs.items():
        started = perf_counter()
        for _ in range(200):
            probabilities = to_probabilities(logits, config)
            torch.multinomial(probabilities, 1)
        results[name] = (perf_counter() - started) / 200
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for name, spec in MODELS.items():
        model = load_model(spec, args.device, DTYPES[args.dtype])
        for context in CONTEXTS:
            session = ModelSession(model)
            session.extend(list(range(2000, 2000 + context)))
            for feed in FEEDS:
                row = measure_pass(session, context, feed)
                rows.append({"model": name, **row})
                print(name, context, feed, f"{row['median_total_s'] * 1000:.2f} ms", flush=True)
        del model

    result = {
        "device": args.device,
        "dtype": args.dtype,
        "repeats": REPEATS,
        "unit": "seconds per pass",
        "rows": rows,
        "sampling_seconds_per_draw": measure_sampling_cost(49152),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
