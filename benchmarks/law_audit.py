"""Audit the output law on real weights with a two-sample test.

The exact-law tests cover tiny models. This asks the same question at full size:
do the first tokens from speculative decoding look like draws from target-only
decoding? A pure draft run is the negative control and must fail the test.

Run from the repository root:
    uv run python benchmarks/law_audit.py --out runs/law_audit/mps.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from specdec.bench import draft_spec
from specdec.controller import AdaptiveController, CostPrior
from specdec.generate import GenerationConfig, fixed_block_length, generate
from specdec.models import (
    TARGET,
    ModelSession,
    best_device,
    load_model,
    load_tokenizer,
    render_prompt,
)
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig

PROMPTS = {
    "receipt": "Reply with one JSON object for this receipt. Inkwell Books 2026-01-02 Total 18.40",
    "general": "Write a four line poem about a rainy morning in the city.",
}
PRIOR = CostPrior(0.004, 0.03, 0.0003, 0.00001)


def draw(target, draft, prompt, config, policy_factory, draws, seed):
    counts = Counter()
    for index in range(draws):
        generation = generate(
            ModelSession(target),
            ModelSession(draft) if draft is not None else None,
            prompt,
            config,
            policy_factory(),
            TorchRandom(seed + index),
        )
        counts[tuple(generation.new_tokens)] += 1
    return counts


def two_sample_p_value(first: Counter, second: Counter, minimum: int = 10) -> tuple[float, int]:
    """Chi-square test of equal laws. Rare sequences fold into one bin."""
    pooled = first + second
    common = [key for key, count in pooled.items() if count >= minimum]

    def binned(counts):
        kept = [counts[key] for key in common]
        return [*kept, sum(counts.values()) - sum(kept)]

    a, b = binned(first), binned(second)
    total_a, total_b = sum(a), sum(b)
    statistic = 0.0
    bins = 0
    for x, y in zip(a, b, strict=True):
        if x + y == 0:
            continue
        bins += 1
        expected_a = (x + y) * total_a / (total_a + total_b)
        expected_b = (x + y) * total_b / (total_a + total_b)
        statistic += (x - expected_a) ** 2 / expected_a + (y - expected_b) ** 2 / expected_b
    degrees = bins - 1
    p_value = torch.special.gammaincc(torch.tensor(degrees / 2.0), torch.tensor(statistic / 2.0))
    return float(p_value), bins


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", default="135m")
    parser.add_argument("--draws", type=int, default=1500)
    parser.add_argument("--new-tokens", type=int, default=3)
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = load_tokenizer(TARGET)
    target = load_model(TARGET, args.device, torch.float32)
    draft = load_model(draft_spec(args.draft), args.device, torch.float32)
    config = GenerationConfig(
        max_new_tokens=args.new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        sampling=SamplingConfig(temperature=0.7),
    )
    rows = []
    started = perf_counter()
    for name, text in PROMPTS.items():
        prompt = render_prompt(tokenizer, text)
        base = lambda: fixed_block_length(0)  # noqa: E731
        reference = draw(target, None, prompt, config, base, args.draws, 1_000_000)
        checks = {
            "fixed_k2": draw(
                target, draft, prompt, config, lambda: fixed_block_length(2), args.draws, 2_000_000
            ),
            "adaptive": draw(
                target,
                draft,
                prompt,
                config,
                lambda: AdaptiveController(PRIOR),
                args.draws,
                3_000_000,
            ),
            "negative_control_draft_only": draw(
                draft, None, prompt, config, base, args.draws, 4_000_000
            ),
            "repeat_of_target": draw(target, None, prompt, config, base, args.draws, 5_000_000),
        }
        for check, counts in checks.items():
            p_value, bins = two_sample_p_value(reference, counts)
            rows.append({"prompt": name, "check": check, "p_value": p_value, "bins": bins})
            print(rows[-1], flush=True)

    result = {
        "draws_per_arm": args.draws,
        "new_tokens": args.new_tokens,
        "draft": args.draft,
        "elapsed_s": perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
