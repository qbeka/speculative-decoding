"""Measure how far a real model's block logits sit from one-token-at-a-time logits.

The sampler is exact for the distributions it is given. On real hardware the
target's distribution at a position depends slightly on how many tokens shared
the pass, because rounding differs. This script reports the total variation
distance between the two, per dtype.

Run from the repository root:
    uv run python benchmarks/numeric_gap.py --out runs/numeric_gap/mps.json
"""

import argparse
import json
from pathlib import Path

import torch

from specdec.models import (
    TARGET,
    ModelSession,
    best_device,
    load_model,
    load_tokenizer,
    render_prompt,
)
from specdec.sampling import SamplingConfig, to_probabilities

DTYPES = {"float16": torch.float16, "float32": torch.float32}
PROMPT = "Explain why the sky is blue, then give the answer as a JSON object."
BLOCK = 8
CONTINUATIONS = 40


def row_gaps(model, prompt_tokens: list[int], continuation: list[int]) -> list[float]:
    """Return the TV distance at each position of a block against single-token passes."""
    sampling = SamplingConfig(temperature=0.7)
    gaps = []
    batched = ModelSession(model)
    single = ModelSession(model)
    batched.extend(prompt_tokens[:-1])
    single.extend(prompt_tokens[:-1])
    feed = [prompt_tokens[-1], *continuation]
    block_rows = batched.extend(feed, rows=len(feed))
    single_rows = torch.cat([single.extend([token]) for token in feed])
    for block_row, single_row in zip(block_rows, single_rows, strict=True):
        p = to_probabilities(block_row, sampling)
        q = to_probabilities(single_row, sampling)
        gaps.append(0.5 * float((p - q).abs().sum()))
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(TARGET)
    prompt_tokens = render_prompt(tokenizer, PROMPT)
    rows = []
    for name, dtype in DTYPES.items():
        model = load_model(TARGET, args.device, dtype)
        generator = torch.Generator().manual_seed(0)
        gaps = []
        for _ in range(CONTINUATIONS):
            continuation = torch.randint(1000, 40000, (BLOCK,), generator=generator).tolist()
            gaps.extend(row_gaps(model, prompt_tokens, continuation))
        tensor = torch.tensor(gaps)
        rows.append(
            {
                "dtype": name,
                "positions": len(gaps),
                "median_tv": float(tensor.median()),
                "max_tv": float(tensor.max()),
            }
        )
        print(rows[-1])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"device": args.device, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
