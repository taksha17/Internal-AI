"""Probe a running control plane and serving API and report pass or fail.

Standalone on purpose: it uses only the standard library, so an external
tester can run it against a reachable instance without installing anything.

    INTERNAL_AI_CONTROL_KEY=... INTERNAL_AI_SERVING_KEY=... \
        .venv/bin/python scripts/probe.py

Exit code is 0 when every check passes and 1 when any check fails.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

CONTROL_URL = os.environ.get("INTERNAL_AI_CONTROL_URL", "http://127.0.0.1:8080")
SERVING_URL = os.environ.get("INTERNAL_AI_SERVING_URL", "http://127.0.0.1:8081")
CONTROL_KEY = os.environ.get("INTERNAL_AI_CONTROL_KEY", "")
SERVING_KEY = os.environ.get("INTERNAL_AI_SERVING_KEY", "")


@dataclass
class Result:
    label: str
    status: int
    expected: int
    body: str

    @property
    def passed(self) -> bool:
        return self.status == self.expected


def call(
    method: str,
    url: str,
    key: str | None = None,
    body: dict[str, Any] | None = None,
    legacy_header: bool = False,
    token: str | None = None,
) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if legacy_header:
        request.add_header("X-Forge-Key", key or "")
    elif key:
        request.add_header("X-Internal-Key", key)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except OSError as error:
        return -1, repr(error)


def check(
    label: str,
    method: str,
    url: str,
    expected: int,
    key: str | None = None,
    body: dict[str, Any] | None = None,
    legacy_header: bool = False,
    token: str | None = None,
) -> Result:
    status, text = call(method, url, key, body, legacy_header, token)
    return Result(label=label, status=status, expected=expected, body=text)


def summarise(result: Result) -> str:
    body = result.body.strip().replace("\n", " ")
    return f"{body[:160]}{'...' if len(body) > 160 else ''}"


def main() -> int:
    if not CONTROL_KEY or not SERVING_KEY:
        print(
            "INTERNAL_AI_CONTROL_KEY and INTERNAL_AI_SERVING_KEY must be set.",
            file=sys.stderr,
        )
        return 2

    results = [
        check("control /health", "GET", f"{CONTROL_URL}/health", 200),
        check("control /tasks without key", "GET", f"{CONTROL_URL}/tasks", 401),
        check("control /tasks with wrong key", "GET", f"{CONTROL_URL}/tasks", 401, "wrong"),
        check("control /tasks", "GET", f"{CONTROL_URL}/tasks", 200, CONTROL_KEY),
        check("control /gates", "GET", f"{CONTROL_URL}/gates", 200, CONTROL_KEY),
        check("control /budget", "GET", f"{CONTROL_URL}/budget", 200, CONTROL_KEY),
        check("control /audit/verify", "GET", f"{CONTROL_URL}/audit/verify", 200, CONTROL_KEY),
        check("control /policy", "GET", f"{CONTROL_URL}/policy", 200, CONTROL_KEY),
        check("control /roles", "GET", f"{CONTROL_URL}/roles", 200, CONTROL_KEY),
        check("control /alerts", "GET", f"{CONTROL_URL}/alerts", 200, CONTROL_KEY),
        check("control /report", "GET", f"{CONTROL_URL}/report", 200, CONTROL_KEY),
        check("control /audit", "GET", f"{CONTROL_URL}/audit", 200, CONTROL_KEY),
        check(
            "control POST /plan without a personal token",
            "POST",
            f"{CONTROL_URL}/plan",
            401,
            CONTROL_KEY,
            {"goal": "probe", "approve_by": "program_lead"},
        ),
        check(
            "control POST /run without a personal token",
            "POST",
            f"{CONTROL_URL}/run",
            401,
            CONTROL_KEY,
            {"approve_by": "program_lead"},
        ),
        check(
            "control POST /gates without a personal token",
            "POST",
            f"{CONTROL_URL}/gates",
            401,
            CONTROL_KEY,
            {"gate_type": "budget_increase", "title": "probe", "subject": "global:global"},
        ),
        check(
            "control POST /gates/decision without a personal token",
            "POST",
            f"{CONTROL_URL}/gates/gate_probe/decision",
            401,
            CONTROL_KEY,
            {"approver": "program_lead", "approved": True},
        ),
        check(
            "control POST /budget/limits without a personal token",
            "POST",
            f"{CONTROL_URL}/budget/limits",
            401,
            CONTROL_KEY,
            {"scope": "global", "key": "global", "limit_usd": 1, "actor": "program_lead"},
        ),
        check(
            "control POST /kill-switch without a personal token",
            "POST",
            f"{CONTROL_URL}/kill-switch",
            401,
            CONTROL_KEY,
            {"engaged": True, "reason": "probe"},
        ),
        check(
            "control POST /policy without a personal token",
            "POST",
            f"{CONTROL_URL}/policy/p1",
            401,
            CONTROL_KEY,
            {"value": "no", "requested_by": "program_lead"},
        ),
        check(
            "control POST /tasks/transition without a personal token",
            "POST",
            f"{CONTROL_URL}/tasks/tsk_probe/transition?state=approved&actor=program_lead",
            401,
            CONTROL_KEY,
        ),
        check(
            "control POST /alerts/acknowledge without a personal token",
            "POST",
            f"{CONTROL_URL}/alerts/alt_probe/acknowledge",
            401,
            CONTROL_KEY,
        ),
        check("serving /health", "GET", f"{SERVING_URL}/health", 200),
        check("serving /v1/models without key", "GET", f"{SERVING_URL}/v1/models", 401),
        check(
            "serving /v1/models with legacy X-Forge-Key",
            "GET",
            f"{SERVING_URL}/v1/models",
            401,
            SERVING_KEY,
            legacy_header=True,
        ),
        check("serving /v1/models", "GET", f"{SERVING_URL}/v1/models", 200, SERVING_KEY),
        check(
            "serving /v1/embed",
            "POST",
            f"{SERVING_URL}/v1/embed",
            200,
            SERVING_KEY,
            {"query": "how does the platform enforce gates?"},
        ),
        check(
            "serving /v1/search",
            "POST",
            f"{SERVING_URL}/v1/search",
            200,
            SERVING_KEY,
            {"query": "budget enforcement", "top_k": 3},
        ),
    ]

    failed = 0
    for result in results:
        if result.passed:
            print(f"PASS  {result.label}")
        else:
            failed += 1
            print(f"FAIL  {result.label}: got {result.status}, expected {result.expected}")
            print(f"      {summarise(result)}")

    print()
    print(f"{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
