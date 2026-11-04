"""Saved inputs. The benchmark reads these files, so a run never depends on a generator."""

import hashlib
import json
from pathlib import Path

from specdec.receipts import build_examples

DATA_DIR = Path("data")
GENERAL_PROMPTS = DATA_DIR / "general_prompts.txt"
RECEIPT_SEED = 20260314
TRAIN_PER_FAMILY = 200
EVALUATION_PER_FAMILY = 8


def workload_path(name: str) -> Path:
    return DATA_DIR / f"{name}.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def general_examples() -> list[dict]:
    """Split the general prompts by position: even lines validate, odd lines test."""
    prompts = GENERAL_PROMPTS.read_text().splitlines()
    return [
        {
            "id": f"general-{index:03d}",
            "split": "validation" if index % 2 == 0 else "test",
            "family": "general",
            "prompt": prompt,
        }
        for index, prompt in enumerate(prompts)
    ]


def prepare_data() -> list[Path]:
    """Write every workload file. The same code always writes the same bytes."""
    written = []
    sizes = {"train": TRAIN_PER_FAMILY, "validation": EVALUATION_PER_FAMILY}
    sizes["test"] = EVALUATION_PER_FAMILY
    for split, per_family in sizes.items():
        path = workload_path(f"receipts_{split}")
        write_jsonl(path, build_examples(split, per_family, RECEIPT_SEED))
        written.append(path)
    general = general_examples()
    for split in ("validation", "test"):
        path = workload_path(f"general_{split}")
        write_jsonl(path, [row for row in general if row["split"] == split])
        written.append(path)
    return written
