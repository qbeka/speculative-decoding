"""Pinned model revisions and the session that runs them.

Weights load through safetensors only, and remote code is never trusted.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, PreTrainedModel


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    revision: str


# Same tokenizer and vocabulary, all Apache-2.0. `check_compatible` enforces this
# at load time rather than trusting this comment.
TARGET = ModelSpec(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", "31b70e2e869a7173562077fd711b654946d38674"
)
DRAFT_135M = ModelSpec(
    "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"
)
DRAFT_360M = ModelSpec(
    "HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951"
)


def best_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_tokenizer(spec: ModelSpec):
    return AutoTokenizer.from_pretrained(spec.repository, revision=spec.revision)


def load_model(spec: ModelSpec, device: str, dtype: torch.dtype) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(
        spec.repository,
        revision=spec.revision,
        dtype=dtype,
        use_safetensors=True,
        trust_remote_code=False,
    )
    return model.to(device).eval()


def check_compatible(target_tokenizer, draft_tokenizer) -> None:
    """Refuse a pair whose token ids do not mean the same thing.

    Speculative sampling compares p and q token by token, so a difference here
    silently breaks the sampler instead of raising an error.
    """
    if target_tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("target and draft tokenizers have different vocabularies")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        if getattr(target_tokenizer, name) != getattr(draft_tokenizer, name):
            raise ValueError(f"target and draft disagree on {name}")
    if target_tokenizer.chat_template != draft_tokenizer.chat_template:
        raise ValueError("target and draft disagree on the chat template")


def synchronize(device: torch.device) -> None:
    """Wait until the device has finished queued work, so a timer means what it says."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


class ModelSession:
    """A Hugging Face causal model plus a key-value cache for one sequence.

    Positions and the attention mask are always passed explicitly, so a rollback
    cannot leave them out of step with the cache.

    `seconds` accumulates time for the profile:
    forward: encoding the model call. sync: waiting for the device to finish.
    transfer: moving logits to the CPU as float64. rollback: cropping the cache.
    """

    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.device = next(model.parameters()).device
        self.cache = DynamicCache(config=model.config)
        self.length = 0
        self.seconds: Counter[str] = Counter()

    @torch.inference_mode()
    def extend(self, tokens: Sequence[int], rows: int = 1) -> torch.Tensor:
        if not tokens:
            raise ValueError("extend needs at least one token")
        if not 1 <= rows <= len(tokens):
            raise ValueError("rows must be between 1 and the number of new tokens")
        new_length = self.length + len(tokens)

        started = perf_counter()
        output = self.model(
            input_ids=torch.tensor([list(tokens)], device=self.device),
            position_ids=torch.arange(self.length, new_length, device=self.device)[None],
            attention_mask=torch.ones(1, new_length, dtype=torch.long, device=self.device),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=rows,  # Skips the large output head for all other positions.
        )
        forward_done = perf_counter()
        synchronize(self.device)
        sync_done = perf_counter()
        # MPS has no float64, so the move to the CPU comes first.
        logits = output.logits[0].cpu().to(torch.float64)
        transfer_done = perf_counter()

        self.length = new_length
        self.seconds["forward"] += forward_done - started
        self.seconds["sync"] += sync_done - forward_done
        self.seconds["transfer"] += transfer_done - sync_done
        return logits

    def rollback(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot roll back to {length} from {self.length}")
        started = perf_counter()
        # In transformers 5 crop takes a negative count of tokens to remove.
        self.cache.crop(length - self.length)
        self.length = length
        self.seconds["rollback"] += perf_counter() - started
