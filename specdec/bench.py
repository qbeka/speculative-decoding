"""Run one benchmark arm in this process and append raw per-request records.

One process runs one arm. That keeps peak memory clean and makes every arm start
from the same state. Orchestration across arms is in `specdec/experiments.py`.

Units: every `_s` field is seconds. Rejected draft tokens are never counted as
output: `new_tokens` comes from the committed sequence only.
"""

import json
import os
import resource
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter

import torch
from transformers.generation.streamers import BaseStreamer

from specdec.batch import BatchSession, generate_batch
from specdec.controller import AdaptiveController, ControllerConfig, CostPrior
from specdec.data import read_jsonl, workload_path
from specdec.generate import (
    Generation,
    GenerationConfig,
    fixed_block_length,
    generate,
)
from specdec.models import (
    DRAFT_135M,
    DRAFT_360M,
    TARGET,
    ModelSession,
    ModelSpec,
    best_device,
    check_compatible,
    load_model,
    load_tokenizer,
    local_model,
    render_prompt,
)
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig

MAX_NEW_TOKENS = 128
WARMUP_REQUESTS = 2  # Discarded. They compile Metal shaders for common shapes.
WARMUP_PROMPT = "Write one sentence about the sea."


def draft_spec(name: str) -> ModelSpec:
    named = {"135m": DRAFT_135M, "360m": DRAFT_360M}
    return named[name] if name in named else local_model(f"artifacts/drafts/{name}")


@dataclass(frozen=True)
class Arm:
    """One configuration under test. `kind` is target, fixed, adaptive or assisted."""

    kind: str
    draft: str | None = None
    block_length: int = 0
    temperature: float = 0.7
    batch_size: int = 1
    draft_device: str | None = None
    controller: dict = field(default_factory=dict)
    cost_prior: dict = field(default_factory=dict)
    assisted_schedule: str = "constant"

    @property
    def name(self) -> str:
        parts = [self.kind]
        if self.draft:
            parts.append(self.draft)
        if self.kind in ("fixed", "assisted"):
            parts.append(f"k{self.block_length}")
        if self.kind == "assisted":
            parts.append(self.assisted_schedule)
        if self.draft_device:
            parts.append(self.draft_device)
        return "/".join(parts)


class TimingStreamer(BaseStreamer):
    """Records when each token leaves `transformers.generate`, for the assisted arm."""

    def __init__(self, started: float):
        self.started = started
        self.times: list[float] = []
        self.first_call = True

    def put(self, value: torch.Tensor) -> None:
        if self.first_call:  # The first call carries the prompt.
            self.first_call = False
            return
        self.times.extend([perf_counter() - self.started] * value.numel())

    def end(self) -> None:
        pass


def request_seed(prompt_id: str, repeat: int) -> int:
    """The same prompt and repeat get the same seed in every arm."""
    return int.from_bytes(f"{prompt_id}:{repeat}".encode()[-7:], "little") % (2**31)


def generation_record(generation: Generation, controller=None) -> dict:
    token_times: list[float] = []
    for block in generation.blocks:
        token_times.extend([block.finished_at_s] * block.delivered)
    record = {
        "tokens": generation.new_tokens,
        "new_tokens": len(generation.new_tokens),
        "prompt_tokens": generation.prompt_length,
        "wall_s": generation.wall_s,
        "prefill_s": generation.prefill_s,
        "ttft_s": token_times[0] if token_times else None,
        "token_times_s": token_times,
        "finish_reason": generation.finish_reason,
        "blocks": [asdict(block) for block in generation.blocks],
    }
    if controller is not None:
        record["decisions"] = [asdict(decision) for decision in controller.decisions]
    return record


class ArmRunner:
    """Holds loaded models and runs requests for one arm."""

    def __init__(self, arm: Arm, device: str):
        self.arm = arm
        self.device = device
        started = perf_counter()
        self.tokenizer = load_tokenizer(TARGET)
        self.target = load_model(TARGET, device, torch.float32)
        self.draft = None
        if arm.draft:
            check_compatible(self.tokenizer, load_tokenizer(draft_spec(arm.draft)))
            draft_device = arm.draft_device or device
            self.draft = load_model(draft_spec(arm.draft), draft_device, torch.float32)
        self.load_s = perf_counter() - started
        self.config = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.tokenizer.eos_token_id,
            sampling=SamplingConfig(temperature=arm.temperature),
        )

    def policy(self):
        if self.arm.kind == "adaptive":
            config = ControllerConfig(**self.arm.controller)
            return AdaptiveController(CostPrior(**self.arm.cost_prior), config)
        return fixed_block_length(self.arm.block_length)

    def run_lone(self, prompt: list[int], seed: int) -> dict:
        if self.arm.kind == "assisted":
            return self.run_assisted(prompt, seed)
        policy = self.policy()
        generation = generate(
            ModelSession(self.target),
            ModelSession(self.draft) if self.draft is not None else None,
            prompt,
            self.config,
            policy,
            TorchRandom(seed),
        )
        return generation_record(generation, policy if self.arm.kind == "adaptive" else None)

    def run_batch(self, prompts: list[list[int]], seeds: list[int]) -> list[dict]:
        policy = self.policy()
        generations = generate_batch(
            BatchSession(self.target, len(prompts)),
            BatchSession(self.draft, len(prompts)) if self.draft is not None else None,
            prompts,
            self.config,
            policy,
            [TorchRandom(seed) for seed in seeds],
        )
        shared = policy if self.arm.kind == "adaptive" else None
        return [generation_record(generation, shared) for generation in generations]

    def run_assisted(self, prompt: list[int], seed: int) -> dict:
        """Run `transformers` assisted generation. Its loop and sampler are upstream code."""
        torch.manual_seed(seed)
        draft_config = self.draft.generation_config
        if self.arm.assisted_schedule == "constant":
            draft_config.num_assistant_tokens = self.arm.block_length
            draft_config.num_assistant_tokens_schedule = "constant"
            draft_config.assistant_confidence_threshold = 0.0
        input_ids = torch.tensor([prompt], device=self.device)
        started = perf_counter()
        streamer = TimingStreamer(started)
        output = self.target.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            assistant_model=self.draft,
            do_sample=True,
            temperature=self.arm.temperature,
            top_k=None,
            top_p=None,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=streamer,
        )
        wall_s = perf_counter() - started
        tokens = output[0, len(prompt) :].tolist()
        return {
            "tokens": tokens,
            "new_tokens": len(tokens),
            "prompt_tokens": len(prompt),
            "wall_s": wall_s,
            "prefill_s": None,
            "ttft_s": streamer.times[0] if streamer.times else None,
            "token_times_s": streamer.times,
            "finish_reason": "eos" if tokens[-1] == self.tokenizer.eos_token_id else "length",
            "blocks": [],
        }

    def warm_up(self) -> None:
        prompt = render_prompt(self.tokenizer, WARMUP_PROMPT)
        for index in range(WARMUP_REQUESTS):
            if self.arm.batch_size == 1:
                self.run_lone(prompt, index)
            else:
                size = self.arm.batch_size
                self.run_batch([prompt] * size, [index + i for i in range(size)])


def memory_snapshot(device: str) -> dict:
    """Return peak process memory and machine load. macOS reports ru_maxrss in bytes.

    The one minute load average is kept with every record, so a report can show how
    much other work shared the machine during a pass.
    """
    snapshot = {
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "load_1m": os.getloadavg()[0],
    }
    if device == "mps":
        snapshot["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
    return snapshot


def run_arm(
    arm: Arm,
    workload: str,
    repeat: int,
    out_path: Path,
    device: str | None = None,
    limit: int | None = None,
    cold: bool = False,
) -> None:
    """Run every prompt of a workload once and append records to out_path.

    With cold set, it runs one request right after loading, with no warm-up.
    """
    device = device or best_device()
    runner = ArmRunner(arm, device)
    examples = read_jsonl(workload_path(workload))
    if limit is not None:
        examples = examples[:limit]
    if cold:
        examples = examples[:1]
    else:
        runner.warm_up()
    prompts = [render_prompt(runner.tokenizer, example["prompt"]) for example in examples]
    seeds = [request_seed(example["id"], repeat) for example in examples]

    records = []
    for start in range(0, len(examples), arm.batch_size):
        chunk = examples[start : start + arm.batch_size]
        chunk_prompts = prompts[start : start + arm.batch_size]
        chunk_seeds = seeds[start : start + arm.batch_size]
        if arm.batch_size == 1:
            results = [runner.run_lone(chunk_prompts[0], chunk_seeds[0])]
        else:
            results = runner.run_batch(chunk_prompts, chunk_seeds)
        for example, seed, result in zip(chunk, chunk_seeds, results, strict=True):
            records.append(
                {
                    "arm": arm.name,
                    "kind": arm.kind,
                    "draft": arm.draft,
                    "block_length": arm.block_length,
                    "temperature": arm.temperature,
                    "batch_size": arm.batch_size,
                    "workload": workload,
                    "prompt_id": example["id"],
                    "family": example["family"],
                    "repeat": repeat,
                    "seed": seed,
                    "cold": cold,
                    "chunk": start // arm.batch_size,
                    **result,
                }
            )

    memory = memory_snapshot(device)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as file:
        for record in records:
            file.write(json.dumps({**record, **memory, "load_s": runner.load_s}) + "\n")

    delivered = sum(record["new_tokens"] for record in records)
    seconds = sum(record["wall_s"] for record in records) / arm.batch_size
    print(
        f"{arm.name} {workload} repeat {repeat}: {delivered} tokens, {seconds:.1f} s, "
        f"{delivered / seconds:.1f} tokens/s",
        file=sys.stderr,
        flush=True,
    )
