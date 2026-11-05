"""Orchestrate many arms. Each arm runs in a fresh process through `specdec arm`.

`run_validation` sweeps on validation prompts and selects settings. `run_final`
runs the frozen plan on test prompts with the arm order shuffled in each repeat, so
slow drift on the machine does not always fall on the same arm.
"""

import gzip
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from specdec.analysis import fit_cost_prior, load_records, unit_rates
from specdec.bench import Arm
from specdec.manifest import build_manifest, write_manifest

BLOCK_LENGTHS = [1, 2, 3, 4, 6, 8, 12, 16]
VALIDATION_WORKLOADS = ["receipts_validation", "general_validation"]
VALIDATION_LIMIT = 12
VALIDATION_REPEATS = 1  # Selection only. The frozen test run carries the evidence.
CONTROLLER_GRID = [
    {"half_life_positions": half_life, "prior_acceptance": prior}
    for half_life in (4.0, 8.0, 16.0)
    for prior in (0.6, 0.8)
]


def spawn_arm(
    arm: Arm,
    workload: str,
    repeat: int,
    out: Path,
    limit: int | None = None,
    cold: bool = False,
) -> None:
    command = [
        sys.executable,
        "-m",
        "specdec.cli",
        "arm",
        "--arm",
        json.dumps(asdict(arm)),
        "--workload",
        workload,
        "--repeat",
        str(repeat),
        "--out",
        str(out),
    ]
    if limit is not None:
        command += ["--limit", str(limit)]
    if cold:
        command.append("--cold")
    subprocess.run(command, check=True)


def mean_log_rate(records: list[dict]) -> float:
    rates = unit_rates(records)
    return sum(math.log(rate) for rate in rates.values()) / len(rates)


def records_for(records: list[dict], arm_name: str, workload: str, batch_size: int) -> list[dict]:
    return [
        record
        for record in records
        if record["arm"] == arm_name
        and record["workload"] == workload
        and record["batch_size"] == batch_size
    ]


def best_arm(records: list[dict], names: list[str], workload: str, batch_size: int) -> str:
    return max(
        names, key=lambda name: mean_log_rate(records_for(records, name, workload, batch_size))
    )


def sweep(arms: list[Arm], raw: Path, repeats: int, done: set) -> None:
    """Run arms not already recorded, so an interrupted sweep can resume."""
    for repeat in range(repeats):
        for arm in arms:
            for workload in VALIDATION_WORKLOADS:
                key = (arm.name, arm.batch_size, workload, repeat)
                if key not in done:
                    spawn_arm(arm, workload, repeat, raw, VALIDATION_LIMIT)


def recorded_keys(raw: Path) -> set:
    if not raw.exists():
        return set()
    return {
        (record["arm"], record["batch_size"], record["workload"], record["repeat"])
        for record in load_records(raw)
    }


def run_validation(out_dir: Path, drafts: list[dict], temperature: float, batch_sizes: list[int]):
    """Sweep fixed lengths, then the controller, and write the selection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"
    selection: dict = {"temperature": temperature, "choices": {}}

    for batch_size in batch_sizes:
        target_arm = Arm("target", temperature=temperature, batch_size=batch_size)
        fixed = [
            Arm(
                "fixed",
                draft["draft"],
                block_length,
                temperature,
                batch_size,
                draft.get("draft_device"),
            )
            for draft in drafts
            for block_length in BLOCK_LENGTHS
        ]
        sweep([target_arm, *fixed], raw, VALIDATION_REPEATS, recorded_keys(raw))
        records = load_records(raw)

        tuned = []
        for draft in drafts:
            family = [arm for arm in fixed if arm.draft == draft["draft"]]
            family = [arm for arm in family if arm.draft_device == draft.get("draft_device")]
            # Cost prior from the arm that verified the widest range of feed sizes.
            priors = fit_cost_prior(
                [
                    record
                    for arm in family
                    for workload in VALIDATION_WORKLOADS
                    for record in records_for(records, arm.name, workload, batch_size)
                ]
            )
            for controller in CONTROLLER_GRID:
                tuned.append(
                    replace(
                        family[0],
                        kind="adaptive",
                        block_length=0,
                        controller=controller,
                        cost_prior=priors,
                    )
                )
        for arm in tuned:
            tag = (
                f"h{arm.controller['half_life_positions']:g}a{arm.controller['prior_acceptance']:g}"
            )
            run_tagged(arm, tag, raw)
        records = load_records(raw)

        for workload in VALIDATION_WORKLOADS:
            names = [target_arm.name, *[arm.name for arm in fixed]]
            selection["choices"][f"{workload}/b{batch_size}"] = {
                "best_fixed": best_arm(records, names, workload, batch_size),
                "rates": {
                    name: mean_log_rate(records_for(records, name, workload, batch_size))
                    for name in names
                },
            }
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return selection


def run_tagged(arm: Arm, tag: str, raw: Path) -> None:
    """Run a controller setting and label its records. A recorded setting is skipped."""
    tagged = f"{arm.name}/{tag}"
    if any(key[:2] == (tagged, arm.batch_size) for key in recorded_keys(raw)):
        return
    scratch = raw.with_suffix(".tmp")
    scratch.unlink(missing_ok=True)  # Left by an interrupted setting.
    for repeat in range(VALIDATION_REPEATS):
        for workload in VALIDATION_WORKLOADS:
            spawn_arm(arm, workload, repeat, scratch, VALIDATION_LIMIT)
    with raw.open("a") as file:
        for record in load_records(scratch):
            record["arm"] = tagged
            record["controller"] = arm.controller
            file.write(json.dumps(record) + "\n")
    scratch.unlink()


def compress_raw(raw: Path) -> None:
    """Write the copy Git keeps. mtime=0 makes the bytes depend only on the records."""
    compressed = gzip.compress(raw.read_bytes(), mtime=0)
    raw.with_name(raw.name + ".gz").write_bytes(compressed)


def run_final(frozen_path: Path, out_dir: Path) -> None:
    """Run the frozen plan once and write raw records plus a manifest."""
    frozen = json.loads(frozen_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "raw.jsonl"
    if raw.exists():
        raise FileExistsError(f"{raw} exists; a final run never overwrites raw records")

    arms = [Arm(**arm) for arm in frozen["arms"]]
    order = random.Random(frozen["order_seed"])
    started = perf_counter()
    for arm in arms:  # Cold runs: fresh process, first request, no warm-up.
        for workload in frozen["workloads"]:
            if arm.batch_size == 1:
                spawn_arm(arm, workload, 0, out_dir / "cold.jsonl", cold=True)
    for repeat in range(frozen["repeats"]):
        for workload in frozen["workloads"]:
            shuffled = order.sample(arms, k=len(arms))
            for arm in shuffled:
                spawn_arm(arm, workload, repeat, raw)
    manifest = build_manifest(frozen, sys.argv, perf_counter() - started, raw)
    write_manifest(out_dir / "manifest.json", manifest)
    compress_raw(raw)
    compress_raw(out_dir / "cold.jsonl")


def fixed_rate_table(records: list[dict]) -> dict:
    """Group mean log rates by workload and arm, for reading a sweep."""
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for key in {(r["workload"], r["arm"], r["batch_size"]) for r in records}:
        workload, arm, batch_size = key
        grouped[f"{workload}/b{batch_size}"][arm] = mean_log_rate(
            records_for(records, arm, workload, batch_size)
        )
    return grouped
