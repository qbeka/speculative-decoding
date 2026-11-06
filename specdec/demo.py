"""Build a single offline HTML replay from a recorded run.

The page has no network requests and no live inference. The stage shows raw records.
Results and limits are written in the template, in Details.
"""

import base64
import json
from importlib import resources
from pathlib import Path

from specdec.analysis import load_records
from specdec.controller import expected_tokens
from specdec.data import read_jsonl, workload_path
from specdec.models import TARGET, load_tokenizer

MAX_PROMPTS_PER_CASE = 4
# The cases a viewer can pick, in the order the page offers them.
CASES = {"receipts_test": "Receipt", "general_test": "General text"}
FONTS = ("GeistSerif-Regular", "Geist-Regular", "Geist-Medium")


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
    """Return each draft as it was proposed, with the token the large model added after it."""
    views = []
    position = 0
    for block in record["blocks"]:
        position += block["delivered"]
        rejected_at = block["rejected_at"]
        views.append(
            {
                "end_s": block["finished_at_s"],
                "draft_ms": 1000 * block["proposal_s"],
                "check_ms": 1000 * block["verification_s"],
                "draft": [tokenizer.decode([token]) for token in block["draft_tokens"]],
                "accepted": block["accepted"],
                "rejected_at": rejected_at,
                "written": tokenizer.decode([record["tokens"][position - 1]]),
            }
        )
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
    decisions = record.get("decisions")
    return {
        "label": label,
        "wall_s": record["wall_s"],
        "tokens": decode_tokens(tokenizer, record),
        "blocks": block_view(tokenizer, record),
        "acceptance": [d["acceptance"] for d in decisions] if decisions else None,
    }


def block_variety(record: dict) -> int:
    """Count changes of draft length. A trace with changes is the one worth watching."""
    lengths = [block["block_length"] for block in record["blocks"]]
    return sum(first != second for first, second in zip(lengths, lengths[1:], strict=False))


def rejects_after_accepting(record: dict) -> bool:
    """A first draft that shows an accepted token and a rejection teaches the idea at once."""
    first = record["blocks"][0]
    return first["accepted"] > 0 and first["rejected_at"] is not None


def trace_label(example: dict, workload: str) -> str:
    """Name a recording the way a viewer would: by what the text is."""
    if workload.startswith("general"):
        return example["prompt"]
    family = example["family"].replace("_", " ").capitalize()
    return f"{family} receipt, {example['truth']['merchant']}"


def pick_recordings(records: list[dict], label_of) -> list[dict]:
    """Keep the recording with the most length changes per label, then the top labels."""
    best = {}
    for record in sorted(records, key=block_variety, reverse=True):
        best.setdefault(label_of(record), record)
    picked = list(best.values())[:MAX_PROMPTS_PER_CASE]
    return sorted(picked, key=rejects_after_accepting, reverse=True)


def split_prompt(prompt: str) -> tuple[str, str]:
    """Separate a receipt prompt's first instruction from the receipt it was given."""
    task, separator, receipt = prompt.partition("\n\nReceipt:\n")
    if not separator:
        return "", prompt
    return task.split(". ")[0] + ".", receipt


def case_view(tokenizer, records, workload, arms, adaptive_name, max_block_length) -> list[dict]:
    examples = {row["id"]: row for row in read_jsonl(workload_path(workload))}
    here = [r for r in records if r["workload"] == workload]
    label_of = lambda record: trace_label(examples[record["prompt_id"]], workload)  # noqa: E731
    traces = []
    for anchor in pick_recordings([r for r in here if r["arm"] == adaptive_name], label_of):
        matching = {
            r["arm"]: r
            for r in here
            if r["prompt_id"] == anchor["prompt_id"] and r["repeat"] == anchor["repeat"]
        }
        if not all(name in matching for name in arms):
            continue
        example = examples[anchor["prompt_id"]]
        task, source = split_prompt(example["prompt"])
        traces.append(
            {
                "label": label_of(anchor),
                "task": task,
                "source": source,
                "arms": [
                    arm_view(tokenizer, matching[name], label, max_block_length)
                    for name, label in arms.items()
                ],
            }
        )
    return traces


def embedded_styles(package) -> str:
    """Inline the shared stylesheet and its fonts, so the page is one file."""
    css = package.joinpath("demo_foundation.css").read_text()
    for name in FONTS:
        font = package.joinpath(f"fonts/{name}.woff2").read_bytes()
        css = css.replace(f"/*{name}*/", base64.b64encode(font).decode())
    return css


def build_demo(
    run_dir: Path, arms: dict[str, str], out_path: Path, max_block_length: int = 16
) -> None:
    """Write the replay. `arms` maps recorded arm names to the labels shown to viewers."""
    records = [r for r in load_records(run_dir / "raw.jsonl") if r["batch_size"] == 1]
    tokenizer = load_tokenizer(TARGET)
    adaptive_name = next(name for name in arms if name.startswith("adaptive"))
    cases = []
    for workload, name in CASES.items():
        traces = case_view(tokenizer, records, workload, arms, adaptive_name, max_block_length)
        if traces:
            cases.append({"name": name, "traces": traces})
    if not cases:
        raise ValueError(f"no prompt was recorded by every arm in {list(arms)}")
    package = resources.files("specdec")
    template = package.joinpath("demo_template.html").read_text()
    template = template.replace("/*FOUNDATION*/", embedded_styles(package))
    # Escaping < keeps a token containing </script> from ending the data block.
    payload = json.dumps({"cases": cases}).replace("<", "\\u003c")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(template.replace("__DATA__", payload))
