"""Turn raw benchmark records into paired comparisons. Nothing here touches a model.

The unit of replication is one chunk of requests served together: a single prompt
at batch size 1, or one static batch above that. Requests in a batch share their
timing, so counting them separately would overstate the evidence.

The verified token rate of a unit is its delivered tokens divided by its wall
time, which includes prefill. Comparisons use the log of the ratio between two
arms on the same unit and repeat, so a slow prompt does not dominate the mean.
"""

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import torch

BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260314


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between closest ranks."""
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def unit_rates(records: list[dict]) -> dict[tuple[int, int], float]:
    """Return {(chunk, repeat): verified tokens per second} for one arm and workload."""
    tokens: dict[tuple[int, int], int] = defaultdict(int)
    wall: dict[tuple[int, int], float] = defaultdict(float)
    for record in records:
        unit = (record["chunk"], record["repeat"])
        tokens[unit] += record["new_tokens"]
        wall[unit] = max(wall[unit], record["wall_s"])  # A batch ends with its last request.
    return {unit: tokens[unit] / wall[unit] for unit in tokens}


def bootstrap_ratio(log_ratios: dict[tuple[int, int], float]) -> tuple[float, float, float]:
    """Return the geometric mean ratio and a 95 percent interval.

    Two levels: resample units, then resample repeats inside each chosen unit. This
    carries the variation between prompts and between runs into the interval.
    """
    by_unit: dict[int, list[float]] = defaultdict(list)
    for (unit, _repeat), value in log_ratios.items():
        by_unit[unit].append(value)
    units = sorted(by_unit)
    point = sum(log_ratios.values()) / len(log_ratios)

    rng = random.Random(BOOTSTRAP_SEED)
    means = []
    for _ in range(BOOTSTRAP_SAMPLES):
        total, count = 0.0, 0
        for unit in rng.choices(units, k=len(units)):
            values = by_unit[unit]
            for value in rng.choices(values, k=len(values)):
                total += value
                count += 1
        means.append(total / count)
    return math.exp(point), math.exp(percentile(means, 0.025)), math.exp(percentile(means, 0.975))


@dataclass(frozen=True)
class Comparison:
    arm: str
    baseline: str
    workload: str
    batch_size: int
    ratio: float
    low: float
    high: float
    units: int
    repeats: int


def compare(records: list[dict], arm: str, baseline: str, workload: str, batch_size: int):
    """Compare one arm with a baseline on matching units and repeats."""
    selected = [
        record
        for record in records
        if record["workload"] == workload and record["batch_size"] == batch_size
    ]
    first = unit_rates([record for record in selected if record["arm"] == arm])
    second = unit_rates([record for record in selected if record["arm"] == baseline])
    shared = sorted(first.keys() & second.keys())
    if not shared:
        raise ValueError(f"no shared units for {arm} and {baseline} on {workload}")
    log_ratios = {unit: math.log(first[unit] / second[unit]) for unit in shared}
    ratio, low, high = bootstrap_ratio(log_ratios)
    return Comparison(
        arm,
        baseline,
        workload,
        batch_size,
        ratio,
        low,
        high,
        units=len({unit for unit, _ in shared}),
        repeats=len({repeat for _, repeat in shared}),
    )


def arm_summary(records: list[dict]) -> dict:
    """Summarize one arm on one workload and batch size. Gaps pool all requests."""
    rates = list(unit_rates(records).values())
    gaps: list[float] = []
    passes = accepted = drafted = 0
    for record in records:
        times = record["token_times_s"]
        gaps.extend(later - earlier for earlier, later in zip(times, times[1:], strict=False))
        passes += len(record["blocks"])
        accepted += sum(block["accepted"] for block in record["blocks"])
        drafted += sum(block["drafted"] for block in record["blocks"])
    first_tokens = [record["ttft_s"] for record in records if record["ttft_s"] is not None]
    delivered = sum(record["new_tokens"] for record in records)
    return {
        "requests": len(records),
        "tokens": delivered,
        "median_tokens_per_s": median(rates),
        "median_wall_s": median(record["wall_s"] for record in records),
        "median_ttft_s": median(first_tokens),
        "median_gap_s": median(gaps) if gaps else None,
        "p95_gap_s": percentile(gaps, 0.95) if gaps else None,
        "accepted_per_pass": accepted / passes if passes else None,
        "tokens_per_pass": delivered / passes if passes else None,
        "acceptance": accepted / drafted if drafted else None,
        "peak_rss_gb": max(record["peak_rss_bytes"] for record in records) / 1e9,
    }


def fit_cost_prior(records: list[dict]) -> dict:
    """Fit the controller's starting costs from one arm's measured blocks.

    Draft cost is the median per pass. Verification is a least squares line in feed
    size and context length, using the normal equations for three unknowns.
    """
    draft_samples: list[float] = []
    rows: list[tuple[float, float, float]] = []
    for record in records:
        for block in record["blocks"]:
            if block["draft_passes"]:
                draft_samples.append(block["proposal_s"] / block["draft_passes"])
            seconds = block["verification_s"] + block["sampling_s"]
            rows.append((block["target_feed"], block["context_length"], seconds))
    base, per_token, per_context = _least_squares(rows)
    return {
        "draft_pass_s": median(draft_samples) if draft_samples else 0.0,
        "verify_base_s": base,
        "verify_per_token_s": per_token,
        "verify_per_context_token_s": per_context,
    }


def _least_squares(rows: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    data = torch.tensor(rows, dtype=torch.float64)
    design = torch.stack([torch.ones(len(rows), dtype=torch.float64), data[:, 0], data[:, 1]], 1)
    solution = torch.linalg.lstsq(design, data[:, 2:], driver="gelsd").solution.squeeze(1)
    base, per_token, per_context = solution.tolist()
    return base, per_token, per_context
