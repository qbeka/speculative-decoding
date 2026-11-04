"""A machine-readable record of exactly what a run used."""

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
import transformers

from specdec.data import file_sha256
from specdec.models import DRAFT_135M, DRAFT_360M, TARGET


def run_git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], capture_output=True, text=True, check=True
    ).stdout.strip()


def sysctl(name: str) -> str:
    result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True)
    return result.stdout.strip()


def hardware() -> dict:
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "logical_cpus": sysctl("hw.ncpu"),
        "memory_bytes": sysctl("hw.memsize"),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def draft_hashes() -> dict:
    return {
        str(path): file_sha256(path)
        for path in sorted(Path("artifacts/drafts").glob("*/model.safetensors"))
    }


def build_manifest(config: dict, command: list[str], elapsed_s: float, raw_path: Path) -> dict:
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "commit": run_git("rev-parse", "HEAD"),
        "uncommitted_changes": bool(run_git("status", "--porcelain", "--", "specdec", "config")),
        "command": command,
        "elapsed_s": elapsed_s,
        "config": config,
        "hardware": hardware(),
        "models": {spec.repository: spec.revision for spec in (TARGET, DRAFT_135M, DRAFT_360M)},
        "draft_weights_sha256": draft_hashes(),
        "inputs_sha256": {str(path): file_sha256(path) for path in sorted(Path("data").glob("*"))},
        "raw_records": {"path": raw_path.name, "sha256": file_sha256(raw_path)},
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")
