"""Command line entry point. Run `uv run specdec --help`."""

import argparse
import json
from pathlib import Path

import torch

from specdec.bench import Arm, run_arm
from specdec.data import file_sha256, prepare_data
from specdec.distill import TrainConfig, record_teacher_sequences, train_draft
from specdec.experiments import run_final, run_validation
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
    render_prompt,
)
from specdec.randomness import TorchRandom
from specdec.sampling import SamplingConfig
from specdec.smoke import run_smoke

DRAFTS = {"135m": DRAFT_135M, "360m": DRAFT_360M}
DTYPES = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


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
    parser.add_argument("--draft-device", default=None, help="defaults to --device")
    # float32 is the default on purpose: see decision 2 in docs/DECISIONS.md.
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--draft", choices=DRAFTS, default="135m")


def run_generate(args: argparse.Namespace) -> None:
    tokenizer = load_tokenizer(TARGET)
    check_compatible(tokenizer, load_tokenizer(DRAFTS[args.draft]))
    dtype = DTYPES[args.dtype]
    target = load_model(TARGET, args.device, dtype)
    draft = None
    if args.block_length > 0:
        draft = load_model(DRAFTS[args.draft], args.draft_device or args.device, dtype)

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


def run_prepare_data(args: argparse.Namespace) -> None:
    for path in prepare_data():
        print(f"{path} {file_sha256(path)}")


def run_teacher(args: argparse.Namespace) -> None:
    cost = record_teacher_sequences(
        args.out, args.device, args.temperature, args.max_new_tokens, args.batch_size, args.seed
    )
    args.cost_out.parent.mkdir(parents=True, exist_ok=True)
    args.cost_out.write_text(json.dumps(cost, indent=2) + "\n")
    print(cost)


def run_distill(args: argparse.Namespace) -> None:
    config = TrainConfig(layers=args.layers, steps=args.steps, temperature=args.temperature)
    summary = train_draft(args.teacher, args.out, config, args.device)
    print(summary["train_s"], summary["parameters"], summary["log"][-1])


def run_one_arm(args: argparse.Namespace) -> None:
    arm = Arm(**json.loads(args.arm))
    run_arm(arm, args.workload, args.repeat, args.out, args.device, args.limit, args.cold)


def run_validate(args: argparse.Namespace) -> None:
    drafts = [json.loads(draft) for draft in args.draft_arm]
    selection = run_validation(args.out, drafts, args.temperature, args.batch_size)
    print(json.dumps({key: value["best_fixed"] for key, value in selection["choices"].items()}))


def run_frozen(args: argparse.Namespace) -> None:
    run_final(args.frozen, args.out)


def run_smoke_command(args: argparse.Namespace) -> None:
    run_smoke()


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

    arm_parser = commands.add_parser("arm", help="run one benchmark arm in this process")
    arm_parser.add_argument("--arm", required=True, help="the arm as a JSON object")
    arm_parser.add_argument("--workload", required=True)
    arm_parser.add_argument("--repeat", type=int, default=0)
    arm_parser.add_argument("--out", type=Path, required=True)
    arm_parser.add_argument("--device", default=best_device())
    arm_parser.add_argument("--limit", type=int)
    arm_parser.add_argument("--cold", action="store_true")
    arm_parser.set_defaults(run=run_one_arm)

    validate_parser = commands.add_parser("validate", help="sweep settings on validation prompts")
    validate_parser.add_argument("--out", type=Path, required=True)
    validate_parser.add_argument(
        "--draft-arm", action="append", required=True, help='for example {"draft": "135m"}'
    )
    validate_parser.add_argument("--temperature", type=float, default=0.7)
    validate_parser.add_argument("--batch-size", type=int, action="append", default=None)
    validate_parser.set_defaults(run=run_validate)

    final_parser = commands.add_parser("final", help="run the frozen plan on test prompts")
    final_parser.add_argument("--frozen", type=Path, default=Path("config/frozen.json"))
    final_parser.add_argument("--out", type=Path, required=True)
    final_parser.set_defaults(run=run_frozen)

    smoke_parser = commands.add_parser("smoke", help="check every layer quickly")
    smoke_parser.set_defaults(run=run_smoke_command)

    data_parser = commands.add_parser("prepare-data", help="write every workload file")
    data_parser.set_defaults(run=run_prepare_data)

    teacher_parser = commands.add_parser("teacher", help="record teacher sequences")
    teacher_parser.add_argument("--out", type=Path, default=Path("artifacts/teacher/train.jsonl"))
    teacher_parser.add_argument(
        "--cost-out", type=Path, default=Path("runs/distill/teacher_cost.json")
    )
    teacher_parser.add_argument("--device", default=best_device())
    teacher_parser.add_argument("--temperature", type=float, default=0.7)
    teacher_parser.add_argument("--max-new-tokens", type=int, default=128)
    teacher_parser.add_argument("--batch-size", type=int, default=16)
    teacher_parser.add_argument("--seed", type=int, default=0)
    teacher_parser.set_defaults(run=run_teacher)

    distill_parser = commands.add_parser("distill", help="train a draft from teacher sequences")
    distill_parser.add_argument("--layers", type=int, required=True)
    distill_parser.add_argument("--steps", type=int, default=1500)
    distill_parser.add_argument("--temperature", type=float, default=0.7)
    distill_parser.add_argument(
        "--teacher", type=Path, default=Path("artifacts/teacher/train.jsonl")
    )
    distill_parser.add_argument("--out", type=Path, required=True)
    distill_parser.add_argument("--device", default=best_device())
    distill_parser.set_defaults(run=run_distill)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
