"""Train a domain draft from the target, using only training template families.

Step 1 records teacher sequences from the target with the same sampling settings
used at serving time. Step 2 trains the draft on those sequences.

The loss is KL(p || q) on the distributions after the serving temperature. The
acceptance rate is 1 - TV(p, q), and Pinsker's inequality bounds TV by the square
root of KL / 2, so lowering this loss raises the bound on acceptance. The target
runs alongside the draft during training, because storing a full distribution for
every position would take far too much disk.
"""

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.nn.functional import kl_div, log_softmax, softmax

from specdec.batch import BatchSession, generate_batch
from specdec.data import read_jsonl, workload_path, write_jsonl
from specdec.generate import GenerationConfig, fixed_block_length
from specdec.models import DRAFT_135M, TARGET, load_model, load_tokenizer, render_prompt
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig


def record_teacher_sequences(
    out_path: Path,
    device: str,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
) -> dict:
    """Generate teacher sequences for the training split and return their cost."""
    tokenizer = load_tokenizer(TARGET)
    model = load_model(TARGET, device, torch.float32)
    examples = read_jsonl(workload_path("receipts_train"))
    config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        sampling=SamplingConfig(temperature=temperature),
    )
    rows = []
    started = perf_counter()
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [render_prompt(tokenizer, example["prompt"]) for example in chunk]
        rngs = [TorchRandom(seed + start + i) for i in range(len(chunk))]
        generations = generate_batch(
            BatchSession(model, len(chunk)), None, prompts, config, fixed_block_length(0), rngs
        )
        for example, generation in zip(chunk, generations, strict=True):
            rows.append(
                {
                    "id": example["id"],
                    "family": example["family"],
                    "tokens": generation.tokens,
                    "prompt_length": generation.prompt_length,
                }
            )
        print(f"teacher {len(rows)}/{len(examples)} {perf_counter() - started:.0f} s", flush=True)
    write_jsonl(out_path, rows)
    return {
        "sequences": len(rows),
        "generated_tokens": sum(len(row["tokens"]) - row["prompt_length"] for row in rows),
        "wall_s": perf_counter() - started,
        "device": device,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
        "seed": seed,
    }


def truncated_draft(layers: int, device: str):
    """Load the 135M draft and keep its first layers. The embeddings stay pretrained."""
    model = load_model(DRAFT_135M, device, torch.float32)
    model.model.layers = model.model.layers[:layers]
    model.config.num_hidden_layers = layers
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = model.config.layer_types[:layers]
    return model


@dataclass(frozen=True)
class TrainConfig:
    layers: int
    steps: int = 1500
    batch_size: int = 12
    learning_rate: float = 3e-4
    temperature: float = 0.7
    seed: int = 0


WIDTH_BUCKET = 64


def make_batch(rows: list[dict], pad_token: int, device: str):
    """Right-pad sequences and mark which positions predict a generated token.

    Widths round up to WIDTH_BUCKET because MPS keeps a compiled graph for every
    tensor shape it sees.
    """
    longest = max(len(row["tokens"]) for row in rows)
    width = math.ceil(longest / WIDTH_BUCKET) * WIDTH_BUCKET
    input_ids = torch.full((len(rows), width), pad_token, dtype=torch.long)
    attention = torch.zeros((len(rows), width), dtype=torch.long)
    trained = torch.zeros((len(rows), width), dtype=torch.bool)
    for i, row in enumerate(rows):
        length = len(row["tokens"])
        input_ids[i, :length] = torch.tensor(row["tokens"], dtype=torch.long)
        attention[i, :length] = 1
        # Logit i predicts token i + 1, so generated tokens map to these rows.
        trained[i, row["prompt_length"] - 1 : length - 1] = True
    return input_ids.to(device), attention.to(device), trained.to(device)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Average the values at the True positions of a boolean mask."""
    return (values * mask).sum() / mask.sum()


def train_draft(teacher_path: Path, out_dir: Path, config: TrainConfig, device: str) -> dict:
    """Train a truncated draft and save it with a training log."""
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    rows = read_jsonl(teacher_path)
    tokenizer = load_tokenizer(TARGET)
    teacher = load_model(TARGET, device, torch.float32)
    student = truncated_draft(config.layers, device).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1 + math.cos(math.pi * step / config.steps))
    )

    log = []
    started = perf_counter()
    for step in range(config.steps):
        batch = rng.sample(rows, config.batch_size)
        input_ids, attention, trained = make_batch(batch, tokenizer.pad_token_id, device)
        with torch.no_grad():
            p_logits = teacher(input_ids=input_ids, attention_mask=attention).logits
            p = softmax(p_logits / config.temperature, dim=-1)
        q_logits = student(input_ids=input_ids, attention_mask=attention).logits
        log_q = log_softmax(q_logits / config.temperature, dim=-1)
        # A mask, not logits[trained], keeps every tensor shape fixed within a bucket.
        loss = masked_mean(kl_div(log_q, p, reduction="none").sum(dim=-1), trained)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        if device == "mps":
            # Without this the CPU queues steps faster than the GPU finishes them, and
            # the buffers of every queued step stay alive. The 30-layer draft hit 51 GB.
            torch.mps.synchronize()
            torch.mps.empty_cache()

        if step % 25 == 0 or step == config.steps - 1:
            overlap = masked_mean(torch.minimum(p, log_q.exp()).sum(dim=-1), trained)
            log.append({"step": step, "kl": loss.item(), "acceptance": overlap.item()})
            print(
                f"step {step} kl {loss.item():.4f} acceptance {overlap.item():.3f} "
                f"{perf_counter() - started:.0f} s",
                flush=True,
            )

    student.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)  # The arm runner checks it against the target's.
    summary = {
        "config": asdict(config),
        "train_s": perf_counter() - started,
        "parameters": sum(p.numel() for p in student.parameters()),
        "device": device,
        "log": log,
    }
    (out_dir / "training.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
