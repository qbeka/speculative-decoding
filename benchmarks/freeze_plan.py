"""Write config/frozen.json from the validation sweep. It reads no test prompt.

Run from the repository root:
    uv run python benchmarks/freeze_plan.py --validation runs/validation/v1
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from specdec.analysis import fit_cost_prior, load_records
from specdec.bench import Arm
from specdec.experiments import best_arm

TEMPERATURE = 0.7
RECEIPTS = "receipts_validation"
GENERAL = "general_validation"
FINAL_WORKLOADS = ["receipts_test", "general_test"]
REPEATS = 3
ORDER_SEED = 20260314
MINIMUM_RATIO = 1.0


def best_setting(records: list[dict], draft: str, workload: str, batch_size: int) -> dict:
    """Pick the controller setting with the best validation rate for this draft."""
    names = sorted(
        {
            record["arm"]
            for record in records
            if record["arm"].startswith(f"adaptive/{draft}/") and record["batch_size"] == batch_size
        }
    )
    name = best_arm(records, names, workload, batch_size)
    return next(record["controller"] for record in records if record["arm"] == name)


def adaptive_arm(records: list[dict], draft: str, workload: str, batch_size: int) -> Arm:
    family = [
        record
        for record in records
        if record["arm"].startswith(f"fixed/{draft}/") and record["batch_size"] == batch_size
    ]
    return Arm(
        "adaptive",
        draft,
        0,
        TEMPERATURE,
        batch_size,
        controller=best_setting(records, draft, workload, batch_size),
        cost_prior=fit_cost_prior(family),
    )


def fixed_arm(draft: str, block_length: int, batch_size: int) -> Arm:
    return Arm("fixed", draft, block_length, TEMPERATURE, batch_size)


def build_plan(records: list[dict]) -> dict:
    target_one = Arm("target", temperature=TEMPERATURE)
    target_four = replace(target_one, batch_size=4)
    # Each draft's controller is tuned on the workload that draft serves best.
    adaptive_small = adaptive_arm(records, "distilled-8", RECEIPTS, 1)
    adaptive_full = adaptive_arm(records, "distilled-30", RECEIPTS, 1)
    adaptive_stock = adaptive_arm(records, "135m", GENERAL, 1)
    adaptive_small_four = adaptive_arm(records, "distilled-8", RECEIPTS, 4)
    adaptive_stock_four = adaptive_arm(records, "135m", GENERAL, 4)

    best_small = fixed_arm("distilled-8", 12, 1)
    best_full = fixed_arm("distilled-30", 12, 1)
    best_stock_general = fixed_arm("135m", 3, 1)
    best_stock_receipts = fixed_arm("135m", 12, 1)
    best_receipts_four = fixed_arm("distilled-30", 12, 4)
    small_four = fixed_arm("distilled-8", 8, 4)
    best_general_four = fixed_arm("135m", 2, 4)

    assisted = [
        Arm("assisted", "distilled-8", 12, TEMPERATURE, 1),
        Arm("assisted", "135m", 3, TEMPERATURE, 1),
        Arm("assisted", "135m", 3, TEMPERATURE, 1, assisted_schedule="heuristic"),
    ]
    arms = [
        target_one,
        best_small,
        best_full,
        best_stock_general,
        best_stock_receipts,
        adaptive_small,
        adaptive_full,
        adaptive_stock,
        *assisted,
        target_four,
        best_receipts_four,
        small_four,
        best_general_four,
        adaptive_small_four,
        adaptive_stock_four,
    ]

    def pair(arm: Arm, baseline: Arm, workload: str) -> dict:
        return {
            "arm": arm.name,
            "baseline": baseline.name,
            "workload": workload,
            "batch_size": arm.batch_size,
        }

    comparisons = []
    for workload in FINAL_WORKLOADS:
        comparisons += [pair(arm, target_one, workload) for arm in arms[1:11]]
        comparisons += [pair(arm, target_four, workload) for arm in arms[12:]]
        comparisons += [
            pair(adaptive_small, best_small, workload),
            pair(adaptive_full, best_full, workload),
            pair(adaptive_stock, best_stock_general, workload),
            pair(adaptive_stock, best_stock_receipts, workload),
            pair(adaptive_small_four, small_four, workload),
            pair(adaptive_small_four, best_receipts_four, workload),
            pair(adaptive_stock_four, best_general_four, workload),
            pair(best_stock_general, assisted[1], workload),
            pair(best_small, assisted[0], workload),
        ]
    return {
        "temperature": TEMPERATURE,
        "workloads": FINAL_WORKLOADS,
        "repeats": REPEATS,
        "order_seed": ORDER_SEED,
        "test_families": "hotel, bookstore and parking",
        "arms": [asdict(arm) for arm in arms],
        "comparisons": comparisons,
        "decision": {
            "adaptive": adaptive_small.name,
            "best_fixed": best_small.name,
            "workload": "receipts_test",
            "batch_size": 1,
            "minimum_ratio": MINIMUM_RATIO,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("config/frozen.json"))
    args = parser.parse_args()
    records = load_records(args.validation / "raw.jsonl")
    plan = build_plan(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2) + "\n")
    for arm in plan["arms"]:
        print(arm["kind"], arm["draft"], arm["block_length"], arm["batch_size"], arm["controller"])


if __name__ == "__main__":
    main()
