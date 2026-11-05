"""Build a single offline HTML replay from a recorded run.

The page has no network requests and no live inference. Every number comes from
raw records, and it says so at the top.
"""

import base64
import json
from importlib import resources
from pathlib import Path

from specdec.analysis import load_records
from specdec.controller import expected_tokens
from specdec.data import read_jsonl, workload_path
from specdec.models import TARGET, load_tokenizer

MAX_TRACES_PER_WORKLOAD = 4


def decode_tokens(tokenizer, record: dict) -> list[dict]:
    """Return each delivered token with its text, time and source."""
    tokens = []
    position = 0
    if not record["blocks"]:
        for token, time in zip(record["tokens"], record["token_times_s"], strict=False):
            tokens.append({"text": tokenizer.decode([token]), "t": time, "source": "target"})
        return tokens
    for block in record["blocks"]:
        for index in range(block["delivered"]):
            token = record["tokens"][position]
            if index < block["accepted"]:
                source = "draft"
            elif block["rejected_at"] is not None:
                source = "corrected"
            else:
                source = "target"
            text = tokenizer.decode([token], skip_special_tokens=True)
            tokens.append({"text": text, "t": block["finished_at_s"], "source": source})
            position += 1
    return tokens


def block_view(tokenizer, record: dict) -> list[dict]:
    views = []
    start = record["prefill_s"]
    for block in record["blocks"]:
        rejected_at = block["rejected_at"]
        rejected_text = None
        if rejected_at is not None:
            rejected_text = tokenizer.decode([block["draft_tokens"][rejected_at]])
        views.append(
            {
                "start_s": start,
                "end_s": block["finished_at_s"],
                "drafted": block["drafted"],
                "accepted": block["accepted"],
                "delivered": block["delivered"],
                "rejected_at": rejected_at,
                "rejected_text": rejected_text,
            }
        )
        start = block["finished_at_s"]
    return views


def check_decisions(record: dict, max_block_length: int) -> None:
    """Refuse to publish an explanation that disagrees with what the controller chose."""
    for decision in record.get("decisions", []):
        best_length, best_rate = 0, 1.0 / decision["verify_one_s"]
        for k in range(1, max_block_length + 1):
            seconds = (
                k * decision["draft_pass_s"]
                + decision["verify_one_s"]
                + k * decision["verify_per_token_s"]
            )
            rate = expected_tokens(decision["acceptance"], k) / seconds
            if rate > best_rate:
                best_length, best_rate = k, rate
        if best_length != decision["chosen"]:
            raise ValueError(f"replay disagrees with decision {decision}")


def arm_view(tokenizer, record: dict, label: str, max_block_length: int) -> dict:
    check_decisions(record, max_block_length)
    return {
        "label": label,
        "wall_s": record["wall_s"],
        "tokens": decode_tokens(tokenizer, record),
        "blocks": block_view(tokenizer, record),
        "decisions": record.get("decisions"),
    }


def block_variety(record: dict) -> int:
    """Count changes of draft length. A trace with changes is the one worth watching."""
    lengths = [block["block_length"] for block in record["blocks"]]
    return sum(first != second for first, second in zip(lengths, lengths[1:], strict=False))


def trace_label(example: dict, workload: str) -> str:
    """Name a recording the way a viewer would: by what kind of document it is."""
    if workload.startswith("general"):
        return f"General text: {example['prompt'][:52]}"
    return f"{example['family'].replace('_', ' ').capitalize()} receipt, {example['id'][-4:]}"


def page_text(run_dir: Path) -> dict:
    """Read the recorded numbers shown on the page. A run without a report shows none."""
    path = run_dir / "report" / "page.json"
    text = json.loads(path.read_text()) if path.exists() else {}
    fallback = "Recorded performance from a saved run. Nothing runs in your browser."
    return {
        "provenance": text.get("provenance", fallback),
        "headline": text.get("headline", []),
        "claims": text.get("claims", []),
        "footer": text.get("footer", []),
    }


def build_demo(
    run_dir: Path, arms: dict[str, str], out_path: Path, max_block_length: int = 16
) -> None:
    """Write the replay. `arms` maps recorded arm names to the labels shown to viewers."""
    records = [r for r in load_records(run_dir / "raw.jsonl") if r["batch_size"] == 1]
    tokenizer = load_tokenizer(TARGET)
    adaptive_name = next(name for name in arms if name.startswith("adaptive"))
    traces = []
    for workload in sorted({record["workload"] for record in records}):
        examples = {row["id"]: row for row in read_jsonl(workload_path(workload))}
        chosen = [r for r in records if r["workload"] == workload and r["arm"] == adaptive_name]
        chosen.sort(key=block_variety, reverse=True)
        for anchor in chosen[:MAX_TRACES_PER_WORKLOAD]:
            matching = {
                r["arm"]: r
                for r in records
                if r["prompt_id"] == anchor["prompt_id"]
                and r["repeat"] == anchor["repeat"]
                and r["workload"] == workload
            }
            if not all(name in matching for name in arms):
                continue
            example = examples[anchor["prompt_id"]]
            traces.append(
                {
                    "label": trace_label(example, workload),
                    "prompt": example["prompt"],
                    "arms": [
                        arm_view(tokenizer, matching[name], label, max_block_length)
                        for name, label in arms.items()
                    ],
                }
            )
    if not traces:
        raise ValueError(f"no prompt was recorded by every arm in {list(arms)}")
    data = {**page_text(run_dir), "max_block_length": max_block_length, "traces": traces}
    package = resources.files("specdec")
    template = package.joinpath("demo_template.html").read_text()
    # The one typeface is embedded, so the page makes no network requests.
    font = package.joinpath("fonts/GeistSerif-Regular.woff2").read_bytes()
    template = template.replace("/*FONT*/", base64.b64encode(font).decode())
    # Escaping < keeps a token containing </script> from ending the data block.
    payload = json.dumps(data).replace("<", "\\u003c")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template.replace("__DATA__", payload))
