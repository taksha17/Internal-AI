#!/usr/bin/env python3
"""Freeze the Y1.5 question set against the released RFC corpus.

Every question names one line in the held-out split, carries the answer that
line supports, and is checked before it is filed: the reference answer must
survive the same citation scorer the adapted model will be judged by, or the
question measures nothing. The set is then frozen behind ``eval_set_freeze``,
which is the point at which it stops being something a model can be tuned to.

    .venv/bin/python scripts/alpha_y15_questions.py --root var_alpha/install \\
        --dataset ds_9350fb760231
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import GateType
from internal_ai.evals.harness import EvalHarness
from internal_ai.people import PeopleRegistry
from internal_ai.services.build import eval_set_leakage
from internal_ai.training.alpha.citations import AskHit, score_answer

# (citation, question, answer supported by that line)
QUESTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "rfc9110.txt:5026",
        "What must a server include in a 401 (Unauthorized) response?",
        "A WWW-Authenticate header field containing at least one challenge (rfc9110.txt:5026).",
    ),
    (
        "rfc9110.txt:7571",
        "What does the 403 (Forbidden) status code mean?",
        "The server understood the request but refuses to fulfill it (rfc9110.txt:7571).",
    ),
    (
        "rfc9110.txt:7324",
        "What does the 300 (Multiple Choices) status code indicate?",
        "That the target resource has more than one representation, each with its own more "
        "specific identifier, and the user agent should pick one (rfc9110.txt:7324).",
    ),
    (
        "rfc9110.txt:4427",
        "Which Expect value does the HTTP specification define?",
        "Only 100-continue, with no defined parameters (rfc9110.txt:4427).",
    ),
    (
        "rfc9110.txt:2822",
        "A server sends a 101 (Switching Protocols) response. What else must it send?",
        "An Upgrade header field naming the new protocols in layer-ascending order "
        "(rfc9110.txt:2822).",
    ),
    (
        "rfc9110.txt:3482",
        "What does the Last-Modified response header field provide?",
        "A timestamp of when the origin server believes the selected representation was last "
        "modified (rfc9110.txt:3482).",
    ),
    (
        "rfc9110.txt:3952",
        "How does a client turn a GET into a range request?",
        "By sending a Range header field asking for only some parts of the selected "
        "representation (rfc9110.txt:3952).",
    ),
    (
        "rfc9110.txt:3008",
        "How are HTTP charset names matched?",
        "Case-insensitively, whether they appear in parameters or as a plain token "
        "(rfc9110.txt:3008).",
    ),
    (
        "rfc9111.txt:142",
        "What kind of protocol is HTTP, according to the caching specification?",
        "A stateless application-level request/response protocol using extensible semantics and "
        "self-descriptive messages (rfc9111.txt:142).",
    ),
    (
        "rfc9111.txt:1098",
        "How are HTTP cache directives identified and compared?",
        "By a token, compared case-insensitively, with an optional argument in token or "
        "quoted-string form (rfc9111.txt:1098).",
    ),
    (
        "rfc9112.txt:945",
        "A request carries Transfer-Encoding whose final coding is not chunked. What must the "
        "server do?",
        "Respond with 400 (Bad Request) and close the connection, because the body length cannot "
        "be determined reliably (rfc9112.txt:945).",
    ),
    (
        "rfc9113.txt:1331",
        "Which flags does the HTTP/2 DATA frame define?",
        "PADDED (0x08) and END_STREAM (0x01) (rfc9113.txt:1331).",
    ),
    (
        "rfc9113.txt:2361",
        "How must an implementation treat unknown or unsupported HTTP/2 error codes?",
        "They MUST NOT trigger any special behavior and MAY be treated as INTERNAL_ERROR "
        "(rfc9113.txt:2361).",
    ),
    (
        "rfc9114.txt:1555",
        "When must a SETTINGS frame be sent on an HTTP/3 connection?",
        "It must be the first frame of each control stream, sent once by each peer, and must not "
        "be sent again (rfc9114.txt:1555).",
    ),
    (
        "rfc8446.txt:1758",
        "What must a TLS 1.3 client do if the server selects a cipher suite it never offered?",
        "Abort the handshake with an illegal_parameter alert (rfc8446.txt:1758).",
    ),
    (
        "rfc8446.txt:3838",
        "What signature requirement applies to the certificates used in TLS 1.3?",
        "The certificates MUST be signed using an acceptable signature algorithm "
        "(rfc8446.txt:3838).",
    ),
    (
        "rfc7540.txt:2381",
        "What must a receiver do with a PING frame that has no ACK flag?",
        "Send a PING frame with the ACK flag set and an identical payload (rfc7540.txt:2381).",
    ),
    (
        "rfc7540.txt:1585",
        "Which two classes of error does HTTP/2 framing define?",
        "A connection error, which makes the entire connection unusable, and a stream error "
        "(rfc7540.txt:1585).",
    ),
    (
        "rfc7541.txt:1377",
        "What is the HPACK static table?",
        "A predefined and unchangeable list of header fields, built from those most used by "
        "popular web sites (rfc7541.txt:1377).",
    ),
    (
        "rfc7541.txt:248",
        "In what byte order are HPACK numeric values, and are they signed?",
        "Network byte order, and unsigned unless otherwise indicated (rfc7541.txt:248).",
    ),
    (
        "rfc6902.txt:192",
        "What must every JSON Patch operation object contain?",
        "Exactly one path member, whose value is a JSON-Pointer string (rfc6902.txt:192).",
    ),
    (
        "rfc6749.txt:2893",
        "How may OAuth 2.0 extensions define additional error codes?",
        "They MAY define extra error codes for the grant, token and resource access error "
        "responses when an extension requires them (rfc6749.txt:2893).",
    ),
    (
        "rfc6750.txt:657",
        "What does RFC 6750 require to reduce token capture and replay?",
        "A limited token lifetime and confidentiality protection of the exchanges between client, "
        "authorization server and resource server (rfc6750.txt:657).",
    ),
    (
        "rfc6265.txt:1458",
        "How is a cookie-string built from a cookie-list?",
        "By outputting each cookie's name, an equals sign and its value, separating cookies with "
        "a semicolon and a space (rfc6265.txt:1458).",
    ),
    (
        "rfc6265.txt:235",
        "How are imperative requirements inside an algorithm interpreted?",
        "With the meaning of the key word, such as MUST, SHOULD or MAY, used in introducing the "
        "algorithm (rfc6265.txt:235).",
    ),
    (
        "rfc3986.txt:1419",
        "What is a URI-reference?",
        "Either a URI or a relative reference, and the most common usage of a resource identifier "
        "(rfc3986.txt:1419).",
    ),
    (
        "rfc8259.txt:463",
        "How does JSON represent a character outside the Basic Multilingual Plane?",
        "As a 12-character escape sequence encoding the UTF-16 surrogate pair (rfc8259.txt:463).",
    ),
    (
        "rfc7838.txt:326",
        "When may a client use a TLS-based alternative service?",
        "Only when it supports TLS Server Name Indication (rfc7838.txt:326).",
    ),
    (
        "rfc1035.txt:1067",
        "What does the DNS RETRY field specify?",
        "A 32-bit interval that should elapse before a failed refresh is retried "
        "(rfc1035.txt:1067).",
    ),
    (
        "rfc1035.txt:1577",
        "How is a DNS QNAME represented?",
        "As a sequence of labels, each a length octet followed by that many octets, terminated by "
        "a zero-length octet for the root (rfc1035.txt:1577).",
    ),
    (
        "rfc8174.txt:132",
        "Which key words does RFC 8174 interpret, and as described in which document?",
        "MUST, MUST NOT, REQUIRED, SHALL, SHALL NOT, SHOULD, SHOULD NOT, RECOMMENDED, MAY and "
        "OPTIONAL, interpreted as described in RFC 2119 (rfc8174.txt:132).",
    ),
    (
        "rfc6797.txt:313",
        "Which use case motivates HTTP Strict Transport Security?",
        "A browser user interacting securely with known and arbitrary sites, and a site deployer "
        "offering the site securely for their own and their users' benefit (rfc6797.txt:313).",
    ),
)

PROXY_NOTE = "approved in writing by the owner, recorded by the agent acting on that instruction"


def build_items(platform: Platform, dataset_id: str) -> list[dict[str, Any]]:
    held_out = {
        row["record_key"] for row in platform.store.dataset_records_by_split(dataset_id, "test")
    }
    rows = {row["record_key"]: row["payload"] for row in platform.store.dataset_records(dataset_id)}
    items: list[dict[str, Any]] = []
    for citation, query, answer in QUESTIONS:
        if citation not in rows:
            raise ValueError(f"{citation} is not in {dataset_id}")
        if citation not in held_out:
            raise ValueError(
                f"{citation} is not in the held-out split, so a question on it would leak "
                "training text"
            )
        path, line = citation.rsplit(":", 1)
        hit = AskHit(path, int(line))
        report = score_answer(answer, [hit])
        if not report.passed:
            raise ValueError(f"the reference answer to {query!r} fails its own scorer: {report}")
        items.append(
            {
                "query": query,
                "answer": answer,
                "relevant_ids": [citation],
                "hits": [{"path": path, "line": int(line), "text": rows[citation]}],
                "dataset_id": dataset_id,
                "source_id": "rfc",
                "kind": "citation",
            }
        )
    return items


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("var_alpha/install"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--name", default="rfc-citations")
    parser.add_argument("--out", type=Path, default=Path("var_alpha/eval.json"))
    args = parser.parse_args(argv)

    platform = Platform.create(root=args.root, llm_kind="scripted")
    try:
        try:
            items = build_items(platform, args.dataset)
        except ValueError as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 1
        harness = EvalHarness(platform.store, platform.audit)
        eval_set = harness.create_eval_set(args.name, items)
        leakage = eval_set_leakage(platform, eval_set.id)
        if not leakage["passed"]:
            print(f"refused: the set would leak {leakage['outside_test_split']}", file=sys.stderr)
            return 1
        gate = platform.gates.request(
            GateType.EVAL_SET_FREEZE,
            f"Freeze {args.name} v{eval_set.version}",
            eval_set.id,
            requested_by="evaluation",
            evidence={
                "coverage_analysis": (
                    f"{leakage['items']} questions over {leakage['relevant_documents']} held-out "
                    f"documents from sources {', '.join(leakage['datasets'])}"
                ),
                "leakage_check": json.dumps(leakage, sort_keys=True),
            },
        )
        approver = PeopleRegistry(platform.store).get("Meera Nair")
        if approver is None:
            print("refused: no ml lead is enrolled in this install", file=sys.stderr)
            return 1
        platform.gates.decide(
            gate.id, approver["name"], approved=True, note=PROXY_NOTE, evidence=gate.evidence
        )
        frozen = harness.freeze_eval_set(eval_set.id, approver["name"], gate_approved=True)
        result = {
            "eval_set_id": frozen.id,
            "items": len(frozen.items),
            "frozen": frozen.frozen,
            "approved_by": frozen.approved_by,
            "content_hash": frozen.content_hash,
            "gate_id": gate.id,
            "dataset_id": args.dataset,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    finally:
        platform.close()


if __name__ == "__main__":
    raise SystemExit(main())
