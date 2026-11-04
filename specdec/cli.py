"""Command line entry point. Run `uv run specdec --help`."""

import argparse

import torch

from specdec.generate import GenerationConfig, fixed_block_length, generate
from specdec.models import (
    DRAFT_135M,
    DRAFT_360M,
    TARGET,
    ModelSession,
    best_device,
    check_compatible,
    load_model,
    load_tokenizer,
)
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig

DRAFTS = {"135m": DRAFT_135M, "360m": DRAFT_360M}
DTYPES = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


def render_prompt(tokenizer, user_message: str) -> list[int]:
    """Apply the one prompt format used everywhere in this project."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_message}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer(text, add_special_tokens=False).input_ids


def summarize(generation) -> str:
    drafted = sum(block.drafted for block in generation.blocks)
    accepted = sum(block.accepted for block in generation.blocks)
    delivered = len(generation.new_tokens)
    decoding_s = generation.wall_s - generation.prefill_s
    passes = len(generation.blocks)
    acceptance = f"{accepted / drafted:.2f}" if drafted else "n/a"
    return (
        f"delivered {delivered} tokens in {generation.wall_s:.2f} s "
        f"(prefill {generation.prefill_s:.2f} s, "
        f"{delivered / decoding_s:.1f} tokens/s after prefill)\n"
        f"target passes {passes}, tokens per pass {delivered / passes:.2f}, "
        f"draft acceptance {acceptance} ({accepted} of {drafted}), "
        f"finished by {generation.finish_reason}"
    )


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--greedy", action="store_true", help="argmax decoding, a separate mode")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=best_device())
    parser.add_argument("--dtype", choices=DTYPES, default="float16")
    parser.add_argument("--draft", choices=DRAFTS, default="135m")


def run_generate(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(TARGET)
    check_compatible(tokenizer, load_tokenizer(DRAFTS[args.draft]))
    dtype = DTYPES[args.dtype]
    target = load_model(TARGET, args.device, dtype)
    draft = load_model(DRAFTS[args.draft], args.device, dtype) if args.block_length > 0 else None

    config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        sampling=SamplingConfig(args.temperature, args.top_k, args.top_p),
        greedy=args.greedy,
    )
    generation = generate(
        ModelSession(target),
        ModelSession(draft) if draft is not None else None,
        render_prompt(tokenizer, args.prompt),
        config,
        fixed_block_length(args.block_length),
        TorchRandom(args.seed),
    )
    print(tokenizer.decode(generation.new_tokens, skip_special_tokens=True))
    print("---")
    print(summarize(generation))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="specdec", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    generate_parser = commands.add_parser("generate", help="generate text from one prompt")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--block-length", type=int, default=4, help="0 is target-only")
    generate_parser.add_argument("--max-new-tokens", type=int, default=64)
    generate_parser.add_argument("--seed", type=int, default=0)
    add_sampling_arguments(generate_parser)
    add_model_arguments(generate_parser)
    generate_parser.set_defaults(run=run_generate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
