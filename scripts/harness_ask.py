"""Ask the harness a question and read back everything it recorded.

A tester's client for ``internal_ai.harness``: it prints the harness description,
the answer and the checks the answer applied to itself, the digest of the prompt
that was asked, what the call cost, and the audit event it left behind. With no
``--question`` it reads one question per line from stdin, so a second question
does not need a second process.

The default evidence is a two-line sample corpus, clearly not an install's data;
pass ``--hit PATH:LINE --excerpt TEXT`` together to ask against your own.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from internal_ai.context import Platform
from internal_ai.harness import (
    Harness,
    HarnessRefused,
    HarnessRequest,
    ModelSignature,
    choose_backend,
)
from internal_ai.services.metrics import operational_metrics
from internal_ai.training.alpha.citations import AskHit, hit_from_key

DEFAULT_HITS = (AskHit("notes/tickets.txt", 3), AskHit("notes/tickets.txt", 8))
DEFAULT_EXCERPTS = (
    "Ticket 42 was closed on Monday after the on-call engineer reset the queue.",
    "The queue is drained nightly at 02:00, and a ticket left open is escalated.",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="harness_ask",
        description="Ask the harness a question and read back what it recorded.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("var_harness"),
        help="where the test ledger lives (default: var_harness, not the pilot state)",
    )
    parser.add_argument(
        "--backend",
        default="scripted",
        help="scripted (default, offline), local, auto or anthropic",
    )
    parser.add_argument("--question", action="append", default=[], metavar="TEXT")
    parser.add_argument("--hit", action="append", default=[], metavar="PATH:LINE")
    parser.add_argument("--excerpt", action="append", default=[], metavar="TEXT")
    parser.add_argument("--window", type=int, default=8192, help="the model's context window")
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--no-metrics", action="store_true", help="skip the snapshot at the end")
    return parser


def _asked(args: argparse.Namespace) -> Iterator[str]:
    if args.question:
        yield from args.question
        return
    print("one question per line, blank line to stop", file=sys.stderr)
    while True:
        print("ask> ", end="", flush=True, file=sys.stderr)
        line = sys.stdin.readline()
        if not line.strip():
            return
        yield line


def _ask(harness: Harness, question: str, args: argparse.Namespace) -> None:
    print(f"\nquestion: {question.strip()}")
    try:
        answer = harness.answer(
            HarnessRequest(
                question=question.strip(),
                hits=tuple(args.hits),
                excerpts=tuple(args.excerpts),
                max_tokens=args.max_tokens,
            )
        )
    except HarnessRefused as exc:
        print(f"refused ({type(exc).__name__}): {exc}")
        return
    print(json.dumps(answer.as_dict(), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.hit and len(args.hit) != len(args.excerpt):
        parser.error("--hit and --excerpt go together, one excerpt per hit")
    args.hits = [hit_from_key(value) for value in args.hit] or list(DEFAULT_HITS)
    args.excerpts = list(args.excerpt) or list(DEFAULT_EXCERPTS)
    try:
        backend = choose_backend(args.backend)
    except (PermissionError, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1

    with Platform.create(root=args.root, llm_kind="scripted") as platform:
        harness = Harness(
            backend,
            ModelSignature(backend.name, args.window),
            audit=platform.audit,
            budget=platform.budget,
            actor="tester",
            actor_kind="human",
        )
        print(json.dumps(harness.describe(), indent=2, sort_keys=True, default=str))
        for question in _asked(args):
            _ask(harness, question, args)

        print("\naudit (newest first):")
        for event in platform.store.list_audit(action="harness.answer", limit=5, descending=True):
            details = json.dumps(event.details, sort_keys=True)
            print(f"  {event.action}  {event.outcome}  {details}")
        if not args.no_metrics:
            snapshot = operational_metrics(platform)
            print("\nmetrics:")
            print(json.dumps(snapshot["harness"], indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
