"""Download the pinned base or run one streamed LoRA step."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpha track on the pinned open base")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("root", type=Path)
    fetch.add_argument(
        "--allow-research-licence",
        action="store_true",
        help="fall back to the research-licensed 3B pin when the 4B download fails",
    )
    smoke = commands.add_parser("smoke")
    smoke.add_argument("root", type=Path)
    smoke.add_argument("out", type=Path)
    train = commands.add_parser("train")
    train.add_argument("root", type=Path)
    train.add_argument("texts", type=Path)
    train.add_argument("out", type=Path)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--sequences", type=int, default=4)
    train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.command == "fetch":
        from internal_ai.training.alpha.fetch import fetch_primary_or_fallback

        base = fetch_primary_or_fallback(
            args.root, allow_research_licence=args.allow_research_licence
        )
        print(f"{base.repo_id} {base.revision}", flush=True)
        return
    if args.command == "train":
        from internal_ai.training.alpha.pin import available_base
        from internal_ai.training.alpha.train import train_prefix

        texts = [
            line for line in args.texts.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        print(
            "ungoverned: this writes no run row, no reproducibility record and no audit "
            "event; use services.alpha.train_alpha for anything you keep",
            flush=True,
        )
        result = train_prefix(
            available_base(args.root),
            args.root,
            texts,
            args.out,
            layers=args.layers,
            sequences=args.sequences,
            seed=args.seed,
        )
        print(
            f"{result['base_repo_id']} layers={result['layers']} "
            f"loss {result['loss_first']:.3f} -> {result['loss_last']:.3f}",
            flush=True,
        )
        return
    from internal_ai.training.alpha.smoke import smoke

    result = smoke(args.root, args.out)
    print(
        f"smoke adapter (not trained on company text): {result['base_repo_id']} "
        f"{result['adapter']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
