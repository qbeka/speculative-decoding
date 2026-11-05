"""Regenerate every table and plot of a run from its raw records.

Nothing here reads a model. Deleting the report folder and running this again
gives the same files, because the bootstrap seed is fixed.
"""

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from specdec.accuracy import FIELDS, score_output  # noqa: E402
from specdec.analysis import arm_summary, compare, load_records  # noqa: E402
from specdec.data import read_jsonl, workload_path  # noqa: E402
from specdec.models import TARGET, load_tokenizer  # noqa: E402

# The page's tokens. The four hues pass the colour-blind checks on this paper.
SURFACE = "#ffffff"
INK = "#111511"
MUTED = "#5a5e5a"
GRID = "#e9e6e1"
KIND_COLORS = {
    "target": "#8d918e",
    "fixed": "#2f6fc4",
    "adaptive": "#157a5a",
    "assisted": "#e0702f",
}
PART_COLORS = {
    "draft": "#157a5a",
    "verify": "#e0702f",
    "sampling": "#2f6fc4",
    "prefill": "#a98200",
}
matplotlib.rcParams["font.family"] = ["Helvetica Neue", "Arial", "sans-serif"]


def style_axis(axis) -> None:
    axis.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=MUTED, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def group_key(record: dict) -> tuple[str, int, str]:
    return record["workload"], record["batch_size"], record["arm"]


def summaries(records: list[dict]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)
    return {key: arm_summary(group) for key, group in grouped.items()}


def task_accuracy(records: list[dict]) -> dict:
    """Return field accuracy per arm on receipt workloads."""
    tokenizer = load_tokenizer(TARGET)
    truths: dict[str, dict] = {}
    for workload in {record["workload"] for record in records}:
        if workload.startswith("receipts"):
            truths |= {row["id"]: row["truth"] for row in read_jsonl(workload_path(workload))}
    right: dict[tuple, int] = defaultdict(int)
    total: dict[tuple, int] = defaultdict(int)
    parsed: dict[tuple, int] = defaultdict(int)
    requests: dict[tuple, int] = defaultdict(int)
    for record in records:
        if record["prompt_id"] not in truths:
            continue
        text = tokenizer.decode(record["tokens"], skip_special_tokens=True)
        score = score_output(text, truths[record["prompt_id"]])
        key = (record["workload"], record["batch_size"], record["arm"])
        right[key] += sum(score["fields"].values())
        total[key] += len(FIELDS)
        parsed[key] += score["parsed"]
        requests[key] += 1
    return {
        key: {
            "field_accuracy": right[key] / total[key],
            "fields_scored": total[key],
            "json_parse_rate": parsed[key] / requests[key],
            "requests": requests[key],
        }
        for key in total
    }


def machine_load(records: list[dict]) -> str:
    """Describe the one minute load average seen at the end of each pass."""
    loads = sorted(record["load_1m"] for record in records if "load_1m" in record)
    if not loads:
        return "Load was not recorded."
    middle = loads[len(loads) // 2]
    return f"One minute load average: median {middle:.1f}, range {loads[0]:.1f} to {loads[-1]:.1f}."


def time_shares(records: list[dict]) -> dict:
    """Return mean seconds per delivered token by part, for arms with block records."""
    parts: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    tokens: dict[tuple, int] = defaultdict(int)
    for record in records:
        if not record["blocks"] or record["batch_size"] != 1:
            continue
        key = group_key(record)
        tokens[key] += record["new_tokens"]
        parts[key]["prefill"] += record["prefill_s"]
        for block in record["blocks"]:
            parts[key]["draft"] += block["proposal_s"]
            parts[key]["verify"] += block["verification_s"]
            parts[key]["sampling"] += block["sampling_s"] + block["rollback_s"]
    return {
        key: {part: seconds / tokens[key] for part, seconds in values.items()}
        for key, values in parts.items()
    }


def plot_comparisons(comparisons: list, out_path: Path, kinds: dict[str, str]) -> None:
    """Ratio to target-only, one panel per workload and batch size."""
    panels: dict[tuple, list] = defaultdict(list)
    for item in comparisons:
        panels[(item.workload, item.batch_size)].append(item)
    keys = sorted(panels)
    tallest = max(len(items) for items in panels.values())
    figure, axes = plt.subplots(
        len(keys), 1, figsize=(8.4, 0.9 + (0.34 * tallest + 0.6) * len(keys)), squeeze=False
    )
    figure.patch.set_facecolor(SURFACE)
    for axis, key in zip(axes[:, 0], keys, strict=True):
        style_axis(axis)
        items = sorted(panels[key], key=lambda item: item.ratio)
        for row, item in enumerate(items):
            color = KIND_COLORS[kinds[item.arm]]
            axis.plot([item.low, item.high], [row, row], color=color, linewidth=2, zorder=2)
            axis.plot(item.ratio, row, "o", color=color, markersize=8, zorder=3)
            axis.annotate(
                f"{item.ratio:.2f}",
                (item.high, row),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                color=MUTED,
                fontsize=8.5,
            )
        axis.axvline(1.0, color=INK, linewidth=1)
        axis.set_yticks(range(len(items)), [item.arm for item in items], fontsize=8.5, color=INK)
        axis.set_ylim(-0.7, tallest - 0.3)
        axis.set_title(
            f"{key[0]}, batch size {key[1]}", loc="left", fontsize=10.5, color=INK, weight="bold"
        )
    handles = [
        plt.Line2D([], [], marker="o", color=color, label=kind, linewidth=2)
        for kind, color in KIND_COLORS.items()
        if kind in kinds.values()
    ]
    axes[-1, 0].set_xlabel(
        "Verified tokens per second, ratio to target-only, with 95 percent interval", color=MUTED
    )
    figure.legend(handles=handles, loc="upper right", frameon=False, ncols=len(handles), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def plot_time_shares(shares: dict, out_path: Path) -> None:
    keys = sorted(shares, key=lambda key: (key[0], sum(shares[key].values())))
    figure, axis = plt.subplots(figsize=(8.4, 0.9 + 0.3 * len(keys)))
    figure.patch.set_facecolor(SURFACE)
    style_axis(axis)
    for row, key in enumerate(keys):
        left = 0.0
        for part, color in PART_COLORS.items():
            width = shares[key].get(part, 0.0) * 1000
            axis.barh(
                row, width, left=left, color=color, height=0.62, edgecolor=SURFACE, linewidth=2
            )
            left += width
    axis.set_yticks(
        range(len(keys)), [f"{key[0]}: {key[2]}" for key in keys], fontsize=8, color=INK
    )
    axis.set_xlabel("Milliseconds per delivered token, batch size 1", color=MUTED)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=p) for p, color in PART_COLORS.items()
    ]
    axis.legend(
        handles=handles,
        frameon=False,
        ncols=4,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    figure.tight_layout()
    figure.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def number(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(run_dir: Path) -> None:
    records = load_records(run_dir / "raw.jsonl")
    frozen = json.loads((run_dir / "manifest.json").read_text())["config"]
    out_dir = run_dir / "report"
    out_dir.mkdir(exist_ok=True)

    kinds = {record["arm"]: record["kind"] for record in records}
    summary = summaries(records)
    comparisons = [
        compare(records, c["arm"], c["baseline"], c["workload"], c["batch_size"])
        for c in frozen["comparisons"]
    ]
    accuracy = task_accuracy(records)

    primary = frozen["decision"]
    decision_item = compare(
        records,
        primary["adaptive"],
        primary["best_fixed"],
        primary["workload"],
        primary["batch_size"],
    )
    supported = decision_item.low > primary["minimum_ratio"]

    rows = [
        [
            workload,
            batch,
            arm,
            s["requests"],
            s["tokens"],
            number(s["median_tokens_per_s"], 1),
            number(s["median_wall_s"]),
            number(s["median_ttft_s"], 3),
            number(s["median_gap_s"] and s["median_gap_s"] * 1000, 1),
            number(s["p95_gap_s"] and s["p95_gap_s"] * 1000, 1),
            number(s["acceptance"]),
            number(s["tokens_per_pass"]),
            number(s["accepted_per_pass"]),
            number(s["peak_rss_gb"], 1),
        ]
        for (workload, batch, arm), s in sorted(summary.items())
    ]
    headers = [
        "Workload",
        "Batch",
        "Arm",
        "Requests",
        "Tokens",
        "Median tokens/s",
        "Median wall s",
        "Median TTFT s",
        "Median gap ms",
        "p95 gap ms",
        "Acceptance",
        "Tokens per pass",
        "Accepted per pass",
        "Peak RSS GB",
    ]
    comparison_rows = [
        [
            item.workload,
            item.batch_size,
            item.arm,
            item.baseline,
            f"{item.ratio:.3f}",
            f"{item.low:.3f} to {item.high:.3f}",
            item.units,
            item.repeats,
        ]
        for item in comparisons
    ]
    comparison_headers = [
        "Workload",
        "Batch",
        "Arm",
        "Baseline",
        "Ratio",
        "95 percent interval",
        "Units",
        "Repeats",
    ]
    accuracy_rows = [
        [
            workload,
            batch,
            arm,
            f"{value['field_accuracy']:.3f}",
            value["fields_scored"],
            f"{value['json_parse_rate']:.3f}",
        ]
        for (workload, batch, arm), value in sorted(accuracy.items())
    ]
    text = "\n\n".join(
        [
            "# Generated tables",
            "Generated by `specdec report` from `raw.jsonl`. Do not edit by hand.",
            "## Primary decision",
            f"{primary['adaptive']} against {primary['best_fixed']} on {primary['workload']}, "
            f"batch size {primary['batch_size']}: ratio {decision_item.ratio:.3f}, 95 percent "
            f"interval {decision_item.low:.3f} to {decision_item.high:.3f}. Rule: supported if "
            f"the lower bound exceeds {primary['minimum_ratio']}. "
            f"Result: {'supported' if supported else 'not supported'}.",
            "## Machine load",
            f"{machine_load(records)} Arms ran in a shuffled order on every repeat, so any "
            "other work on the machine falls on all of them, but it adds noise.",
            "## Paired comparisons",
            "One unit is one prompt at batch size 1, or one static batch above that.",
            markdown_table(comparison_headers, comparison_rows),
            "## Arm summaries",
            markdown_table(headers, rows),
            "## Task accuracy",
            "Receipts only. This measures text quality, not sampler correctness.",
            markdown_table(
                ["Workload", "Batch", "Arm", "Field accuracy", "Fields scored", "JSON parse rate"],
                accuracy_rows,
            ),
        ]
    )
    (out_dir / "tables.md").write_text(text + "\n")
    (out_dir / "decision.json").write_text(
        json.dumps(
            {"rule": primary, "result": asdict(decision_item), "supported": supported}, indent=2
        )
        + "\n"
    )
    plottable = [item for item in comparisons if item.baseline.startswith("target")]
    plot_comparisons(plottable, out_dir / "speedup.png", kinds)
    plot_time_shares(time_shares(records), out_dir / "time_per_token.png")
