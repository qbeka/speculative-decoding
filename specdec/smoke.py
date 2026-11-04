"""A fast end-to-end check of every layer. It needs no download for its first part."""

import torch
from huggingface_hub import try_to_load_from_cache
from transformers import LlamaConfig, LlamaForCausalLM

from specdec.batch import BatchSession, generate_batch
from specdec.controller import AdaptiveController, CostPrior
from specdec.exhaustive import exact_law, target_law, total_variation
from specdec.generate import GenerationConfig, fixed_block_length, generate
from specdec.models import (
    DRAFT_135M,
    TARGET,
    ModelSession,
    best_device,
    load_model,
    load_tokenizer,
    render_prompt,
)
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig, to_probabilities
from specdec.sessions import ToySession

PRIOR = CostPrior(0.001, 0.002, 0.0001, 0.000001)


def tiny_model(seed: int, layers: int) -> LlamaForCausalLM:
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=2,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(config).eval()


def check_exact_law() -> None:
    def target(prefix):
        return torch.tensor([0.1, 0.2, 0.7][:: 1 if len(prefix) % 2 else -1]).double().log()

    def draft(prefix):
        return torch.tensor([0.5, 0.3, 0.2]).double().log()

    config = GenerationConfig(max_new_tokens=3)

    def run(rng):
        generation = generate(
            ToySession(target), ToySession(draft), [0], config, fixed_block_length(2), rng
        )
        return tuple(generation.new_tokens)

    expected = target_law(lambda p: to_probabilities(target(p), SamplingConfig()), (0,), 3, None)
    gap = total_variation(exact_law(run), expected)
    if gap > 1e-11:
        raise AssertionError(f"exact law check failed with gap {gap}")
    print(f"ok   exact law on a tiny model, total variation {gap:.1e}")


def check_tiny_engine() -> None:
    target, draft = tiny_model(0, 2), tiny_model(1, 1)
    config = GenerationConfig(max_new_tokens=24, greedy=True)
    prompt = [1, 2, 3]
    reference = generate(ModelSession(target), None, prompt, config, fixed_block_length(0), None)
    speculative = generate(
        ModelSession(target),
        ModelSession(draft),
        prompt,
        config,
        AdaptiveController(PRIOR),
        None,
    )
    batched = generate_batch(
        BatchSession(target, 2),
        BatchSession(draft, 2),
        [prompt, prompt],
        config,
        fixed_block_length(3),
        [None, None],
    )
    if speculative.tokens != reference.tokens or batched[1].tokens != reference.tokens:
        raise AssertionError("greedy outputs differ from target-only decoding")
    print("ok   greedy equality through the single and batched engines")


def check_real_models() -> None:
    cached = [
        try_to_load_from_cache(spec.repository, "model.safetensors", revision=spec.revision)
        for spec in (TARGET, DRAFT_135M)
    ]
    if not all(isinstance(path, str) for path in cached):
        print("skip real inference: weights are not downloaded. Run `specdec generate` once.")
        return
    device = best_device()
    tokenizer = load_tokenizer(TARGET)
    target = load_model(TARGET, device, torch.float32)
    draft = load_model(DRAFT_135M, device, torch.float32)
    config = GenerationConfig(
        max_new_tokens=32,
        eos_token_id=tokenizer.eos_token_id,
        sampling=SamplingConfig(temperature=0.7),
    )
    prompt = render_prompt(tokenizer, "Give the capital of France in one sentence.")
    generation = generate(
        ModelSession(target),
        ModelSession(draft),
        prompt,
        config,
        fixed_block_length(3),
        TorchRandom(0),
    )
    text = tokenizer.decode(generation.new_tokens, skip_special_tokens=True).strip()
    print(f"ok   real inference on {device}: {text!r}")


def run_smoke() -> None:
    check_exact_law()
    check_tiny_engine()
    check_real_models()
    print("smoke run passed")
