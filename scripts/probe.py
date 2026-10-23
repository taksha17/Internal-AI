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
) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    if legacy_header:
        request.add_header("X-Forge-Key", key or "")
    elif key:
        request.add_header("X-Internal-Key", key)
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
) -> Result:
    status, text = call(method, url, key, body, legacy_header)
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
