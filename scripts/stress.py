#!/usr/bin/env python3
"""Drive the platform past its normal operating range and report what held.

Every number the README quotes about scale comes from this script. It runs the
real services in-process for the build path and over HTTP for the two servers,
so a failure here is a failure a person would meet.

    .venv/bin/python scripts/stress.py --records 5000 --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from internal_ai.brief import Metric, ProjectBrief, programme_gaps
from internal_ai.brief import load as load_brief
from internal_ai.brief import save as save_brief
from internal_ai.context import Platform
from internal_ai.data.pipeline import DataPipeline
from internal_ai.data.samples import generate_corpus
from internal_ai.domain import GateType
from internal_ai.evals.harness import EvalHarness
from internal_ai.people import PeopleRegistry
from internal_ai.services.actor import Actor
from internal_ai.services.identity import IdentityService
from internal_ai.training.trainer import TrainConfig
from internal_ai.workflows import (
    build_corpus,
    derive_eval_items,
    evaluate_embedding_model,
    promote_model,
    register_model,
    split_corpus,
    train_embedding_model,
    train_tokenizer_for,
)

CSRF_FIELD = re.compile(r'name="csrf_token" value="([^"]+)"')
PASSWORD = secrets.token_urlsafe(18)


class Clock:
    def __init__(self) -> None:
        self.stages: list[dict[str, Any]] = []
        self._start = 0.0
        self._name = ""

    def __enter__(self) -> Clock:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stages.append({"stage": self._name, "seconds": self.seconds()})

    def start(self, name: str) -> None:
        self._name = name
        self._start = time.perf_counter()

    def stop(self, **extra: Any) -> float:
        elapsed = self.seconds()
        self.stages.append({"stage": self._name, "seconds": elapsed, **extra})
        return elapsed

    def seconds(self) -> float:
        return round(time.perf_counter() - self._start, 3)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(url: str, process: subprocess.Popen[bytes], timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with {process.returncode}")
        try:
            if httpx.get(url, timeout=1.5).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.2)
    raise RuntimeError(f"{url} never became healthy")


def enrol_people(platform: Platform) -> dict[str, dict[str, Any]]:
    roster = PeopleRegistry(platform.store)
    people = {
        "ml_lead": ("Meera Nair", "ML lead", ["ml_lead", "program_lead"]),
        "data_owner": ("Sam Okoye", "Data owner", ["data_owner", "data_engineer"]),
        "infra": ("Ravi Menon", "Infrastructure lead", ["infra", "engineering_manager"]),
        "compliance": (
            "Elena Fischer",
            "Compliance",
            ["compliance", "evaluation", "domain_expert"],
        ),
        "domain_expert": (
            "Elena Fischer",
            "Compliance",
            ["compliance", "evaluation", "domain_expert"],
        ),
        "security": ("Tom Halloran", "Security", ["security", "legal", "finance"]),
    }
    enrolled: dict[str, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for role, (name, title, roles) in people.items():
        if name not in by_name:
            by_name[name] = roster.new_person(name, title, roles, ["engineering"])
        enrolled[role] = by_name[name]
    return enrolled


def write_brief(
    platform: Platform, people: dict[str, dict[str, Any]], recall_bar: float
) -> list[str]:
    brief = ProjectBrief(
        target="An internal search model over support and engineering records",
        languages="English",
        context_length=256,
        quality_bar="Reviewers rate retrieval results usable for support replies",
        metrics=[Metric(name="recall_at_10", threshold=str(recall_bar), direction="at_least")],
        budget_usd=2500.0,
        timeline="One quarter",
        deployment="On premise, inside the company network",
        data_restrictions="Internal records only; no personal or regulated data",
        latency_ms=800,
        non_goals="Open-ended chat, code generation, anything trained on third-party output",
        owners={
            "ml_lead": people["ml_lead"]["id"],
            "data_engineer": people["data_owner"]["id"],
            "infra": people["infra"]["id"],
            "evaluation": people["compliance"]["id"],
            "security": people["security"]["id"],
        },
    )
    meera = Actor.from_person(people["ml_lead"], authenticated=False)
    save_brief(platform.store, platform.audit, meera, brief)
    return programme_gaps(platform, release=False)


def request_and_approve(
    platform: Platform,
    gate_type: GateType,
    title: str,
    subject: str,
    evidence: dict[str, Any],
    people: dict[str, dict[str, Any]],
    requested_by: str,
) -> str:
    gate = platform.gates.request(
        gate_type, title, subject, requested_by=requested_by, evidence=evidence
    )
    for role in gate.required_approvers:
        person = people[role]
        platform.gates.decide(
            gate.id, person["name"], approved=True, note="stress run", evidence=evidence
        )
    decided = platform.store.get_gate(gate.id)
    if decided is None or str(decided.status) != "approved":
        raise RuntimeError(f"gate {gate.id} did not reach approved: {decided and decided.status}")
    return gate.id


def race_gate(
    platform: Platform, people: dict[str, dict[str, Any]], rounds: int = 10
) -> dict[str, Any]:
    """Eight threads decide one gate as the same person, repeated `rounds` times."""
    approver = people["ml_lead"]["name"]
    reports: list[dict[str, Any]] = []
    for index in range(rounds):
        gate = platform.gates.request(
            GateType.EVAL_SET_FREEZE,
            f"Race an eval freeze {index}",
            f"stress-eval-race-{index}",
            requested_by="data_engineer",
            evidence={"coverage_analysis": "n/a", "leakage_check": "n/a"},
        )
        outcomes: list[str] = []
        lock = threading.Lock()

        def decide(
            gate_id: str = gate.id,
            results: list[str] = outcomes,
            guard: threading.Lock = lock,
        ) -> None:
            try:
                platform.gates.decide(gate_id, approver, approved=True, note="race")
                result = "accepted"
            except Exception as exc:
                result = type(exc).__name__
            with guard:
                results.append(result)

        threads = [threading.Thread(target=decide) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        decisions = platform.store.gate_decisions(gate.id)
        rows = [item for item in decisions if item.approver == approver]
        reports.append(
            {
                "accepted": sum(1 for item in outcomes if item == "accepted"),
                "refusals": sum(1 for item in outcomes if item != "accepted"),
                "rows_for_one_person": len(rows),
                "status": str(platform.store.get_gate(gate.id).status),
            }
        )
    return {
        "rounds": rounds,
        "accepted_total": sum(item["accepted"] for item in reports),
        "rounds_with_duplicate_rows": sum(1 for item in reports if item["rows_for_one_person"] > 1),
        "max_rows_for_one_person": max(item["rows_for_one_person"] for item in reports),
        "statuses": sorted({item["status"] for item in reports}),
    }


def race_budget(
    platform: Platform, cap_usd: float, threads: int, each_usd: float
) -> dict[str, Any]:
    """Threads check the agent cap and record spend, the way a spend path does."""
    from internal_ai.budget import BudgetScope

    agent = "ml_engineer"
    platform.budget.set_limit(BudgetScope.AGENT, agent, cap_usd)
    allowed = 0
    refused = 0
    lock = threading.Lock()

    def spend() -> None:
        nonlocal allowed, refused
        if platform.budget.check(agent).allowed:
            platform.budget.record_spend(each_usd, agent, notes="stress")
            with lock:
                allowed += 1
        else:
            with lock:
                refused += 1

    workers = [threading.Thread(target=spend) for _ in range(threads)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    status = platform.budget.status(BudgetScope.AGENT, agent)
    return {
        "cap_usd": cap_usd,
        "attempts": threads,
        "allowed": allowed,
        "refused": refused,
        "spent_usd": round(status.spent_usd, 4) if status else None,
        "over_cap_usd": round(status.spent_usd - cap_usd, 4) if status else None,
    }


def serving_phase(
    root: Path, links: dict[str, Any], requests: int, concurrency: int
) -> dict[str, Any]:
    port = free_port()
    keys = "key-alpha:alice:engineering|support,key-bravo:bob:finance"
    env = {**os.environ, "INTERNAL_AI_SERVING_KEYS": keys}
    log = (root / "serve-model.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "internal_ai.cli",
            "--root",
            str(root),
            "serve-model",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--rate-limit",
            "600",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(f"{base}/health", process)
        search_body = {
            "query": "how do I fix a ledgerline mismatch in invoice reconciliation",
            "top_k": 5,
        }
        headers = {"X-Internal-Key": "key-alpha"}

        with httpx.Client(timeout=60.0) as client:
            warm = client.post(f"{base}/v1/search", json=search_body, headers=headers)
        payload = warm.json() if warm.status_code == 200 else {}

        latencies: list[float] = []
        statuses: dict[int, int] = {}
        lock = threading.Lock()

        def one(_index: int) -> None:
            with httpx.Client(timeout=120.0) as client:
                started = time.perf_counter()
                response = client.post(f"{base}/v1/search", json=search_body, headers=headers)
                elapsed = (time.perf_counter() - started) * 1000
            with lock:
                latencies.append(elapsed)
                statuses[response.status_code] = statuses.get(response.status_code, 0) + 1

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(one, range(requests)))
        wall = time.perf_counter() - started

        def percentile(values: list[float], fraction: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
            return round(ordered[index], 1)

        with httpx.Client(timeout=20.0) as client:
            unauthorised = client.post(f"{base}/v1/search", json=search_body).status_code
            wrong_key = client.post(
                f"{base}/v1/search", json=search_body, headers={"X-Internal-Key": "nope"}
            ).status_code
            denied_group = client.post(
                f"{base}/v1/search",
                json={**search_body, "dataset_id": links["dataset_id"]},
                headers={"X-Internal-Key": "key-bravo"},
            )

        return {
            "requests": requests,
            "concurrency": concurrency,
            "wall_seconds": round(wall, 2),
            "throughput_per_second": round(requests / wall, 2),
            "status_counts": {str(key): value for key, value in sorted(statuses.items())},
            "latency_p50_ms": percentile(latencies, 0.50),
            "latency_p95_ms": percentile(latencies, 0.95),
            "latency_max_ms": round(max(latencies), 1) if latencies else 0.0,
            "hits_in_first_response": len(payload.get("hits", [])),
            "model_id": payload.get("model_id"),
            "unauthenticated_status": unauthorised,
            "wrong_key_status": wrong_key,
            "other_group_status": denied_group.status_code,
            "other_group_hits": len(denied_group.json().get("hits", []))
            if denied_group.status_code == 200
            else None,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def rate_limit_phase(root: Path, limit: int = 5, requests: int = 20) -> dict[str, Any]:
    """Send more requests than the limit allows and count what comes back."""
    port = free_port()
    keys = "key-alpha:alice:engineering|support"
    env = {**os.environ, "INTERNAL_AI_SERVING_KEYS": keys}
    log = (root / "serve-ratelimit.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "internal_ai.cli",
            "--root",
            str(root),
            "serve-model",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--rate-limit",
            str(limit),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        wait_for_health(f"{base}/health", process)
        with httpx.Client(timeout=60.0) as client:
            codes = [
                client.post(
                    f"{base}/v1/embed",
                    json={"query": "hello"},
                    headers={"X-Internal-Key": "key-alpha"},
                ).status_code
                for _ in range(requests)
            ]
            detail = client.post(
                f"{base}/v1/embed",
                json={"query": "hello"},
                headers={"X-Internal-Key": "key-alpha"},
            ).json()
        return {
            "limit": limit,
            "requests": requests,
            "allowed": codes.count(200),
            "refused": codes.count(429),
            "detail": detail,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def console_phase(root: Path, person: str) -> dict[str, Any]:
    port = free_port()
    log = (root / "serve-console.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "internal_ai.cli",
            "--root",
            str(root),
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    attempts: list[int] = []
    lock = threading.Lock()

    def try_sign_in(password: str) -> None:
        with httpx.Client(base_url=base, timeout=30.0, follow_redirects=False) as client:
            page = client.get("/console/signin")
            match = CSRF_FIELD.search(page.text)
            token = match.group(1) if match else ""
            response = client.post(
                "/console/signin",
                data={"person": person, "password": password, "csrf_token": token},
            )
        with lock:
            attempts.append(response.status_code)

    try:
        wait_for_health(f"{base}/health", process)
        with httpx.Client(base_url=base, timeout=30.0) as client:
            started = time.perf_counter()
            page = client.get("/console/signin")
            signin_status = page.status_code
            signin_ms = round((time.perf_counter() - started) * 1000, 1)

        for index in range(14):
            try_sign_in(f"wrong-password-{index}")
        for _ in range(4):
            try_sign_in(PASSWORD)
        throttled = sum(1 for code in attempts if code == 429)
        return {
            "signin_page_ms": signin_ms,
            "signin_status": signin_status,
            "bad_attempts": 14,
            "status_counts": {str(code): attempts.count(code) for code in sorted(set(attempts))},
            "throttled": throttled,
        }
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="var_stress")
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--requests", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--pair-mode", choices=("paraphrase", "title"), default="paraphrase")
    parser.add_argument(
        "--brief-recall",
        type=float,
        default=0.05,
        help="the recall@10 bar the harness records in its brief; a company sets its own",
    )
    parser.add_argument("--skip-serving", action="store_true")
    parser.add_argument("--keep", action="store_true", help="reuse an existing root")
    args = parser.parse_args()

    root = Path(args.root)
    if not args.keep and root.exists():
        import shutil

        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {"records_requested": args.records, "steps": args.steps}
    clock = Clock()

    platform = Platform.create(root=root, llm_kind="scripted")
    try:
        clock.start("corpus")
        corpus = root / "corpus" / "corpus.jsonl"
        generated = generate_corpus(corpus, records=args.records, seed=7)
        clock.stop(rows=generated["records"])

        people = enrol_people(platform)
        gaps = write_brief(platform, people, args.brief_recall)
        report["brief_gaps"] = gaps
        report["brief_recall_bar"] = args.brief_recall

        clock.start("data.build")
        build = build_corpus(platform, [corpus], name="stress", approved=True)
        steps = {step["step"]: step for step in build.stats["steps"]}
        clock.stop(
            ingested=build.records,
            tokens=build.tokens,
            dedup=steps["deduplicate"]["exact_duplicates_removed"],
            redactions=steps["pii_scrub"]["total_redactions"],
        )
        dataset_id = build.dataset_id
        report["dataset"] = {
            "id": dataset_id,
            "records": build.records,
            "tokens": build.tokens,
            "content_hash": build.content_hash,
            "duplicates_removed": steps["deduplicate"]["exact_duplicates_removed"],
            "near_duplicates_removed": steps["deduplicate"]["near_duplicates_removed"],
            "pii_redactions": steps["pii_scrub"]["total_redactions"],
            "excluded": steps["sensitivity_exclusion"]["input"]
            - steps["sensitivity_exclusion"]["output"],
            "dropped_by_quality": steps["quality_filter"]["dropped_by_reason"],
        }

        clock.start("data.split")
        splits = split_corpus(platform, dataset_id)
        counts = splits.stats.get("counts", {})
        clock.stop(counts=counts)
        report["splits"] = counts

        clock.start("tokenizer")
        tokenizer, coverage = train_tokenizer_for(platform, dataset_id, vocab_size=8000)
        clock.stop(vocab=tokenizer.vocab_size, tokens=coverage.get("tokens"))
        report["tokenizer"] = {
            "vocab": tokenizer.vocab_size,
            "tokens": coverage.get("tokens"),
            "trained_on_documents": coverage.get("trained_on_documents"),
        }

        clock.start("release")
        release_gate = request_and_approve(
            platform,
            GateType.DATASET_RELEASE,
            "Release the stress corpus",
            dataset_id,
            {"quality_report": "stress", "pii_audit_sample": "full", "lineage": "recorded"},
            people,
            "data_engineer",
        )
        DataPipeline(platform.store, platform.audit, platform.settings.root).release(
            dataset_id, "ml_lead", gate_approved=True
        )
        clock.stop(gate=release_gate)

        clock.start("train")
        config = TrainConfig(
            name="stress-embedding",
            steps=args.steps,
            batch_size=16,
            seq_len=128,
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_ff=512,
            checkpoint_interval=50,
            log_interval=10,
            eval_interval=25,
            learning_rate=3e-4,
        )
        outcome = train_embedding_model(
            platform,
            dataset_id,
            tokenizer,
            config,
            actor="ml_engineer",
            pair_mode=args.pair_mode,
        )
        train_seconds = clock.stop(
            tokens_seen=outcome.tokens_seen,
            final_loss=round(outcome.final_loss, 4),
            cost_usd=round(outcome.cost_usd, 4),
        )
        report["train"] = {
            "run_id": outcome.run_id,
            "steps": outcome.steps,
            "tokens_seen": outcome.tokens_seen,
            "tokens_per_second": round(outcome.tokens_seen / train_seconds, 1),
            "final_loss": round(outcome.final_loss, 4),
            "cost_usd": round(outcome.cost_usd, 4),
            "checkpoint": outcome.checkpoint_path,
            "alerts": outcome.alerts,
            "warnings": outcome.warnings,
            "status": outcome.status,
            "model_path": outcome.model_path,
        }

        harness = EvalHarness(platform.store, platform.audit)
        clock.start("eval.derive")
        items = derive_eval_items(platform, dataset_id, count=60, mode="paraphrase")
        for item in items:
            item["dataset_id"] = dataset_id
        eval_set = harness.create_eval_set("stress-eval", items)
        clock.stop(items=len(items))
        request_and_approve(
            platform,
            GateType.EVAL_SET_FREEZE,
            "Freeze the stress eval",
            eval_set.id,
            {"coverage_analysis": "60 derived queries", "leakage_check": "split held out"},
            people,
            "evaluation",
        )
        harness.freeze_eval_set(eval_set.id, people["ml_lead"]["name"], gate_approved=True)

        model = register_model(
            platform,
            "stress-embedding",
            Path(outcome.model_path),
            outcome,
            dataset_id,
            actor="ml_engineer",
        )

        clock.start("eval.run")
        evaluation = evaluate_embedding_model(
            platform,
            Path(outcome.model_path),
            eval_set.id,
            tokenizer,
            dataset_id,
            margin=0.0,
            actor="evaluation",
            model_id=model.id,
            judged_by=people["compliance"]["name"],
        )
        clock.stop(passed=evaluation["passed"])
        report["eval"] = {
            "eval_set": eval_set.id,
            "items": len(eval_set.items),
            "passed": evaluation["passed"],
            "metrics": {key: round(value, 4) for key, value in evaluation["metrics"].items()},
            "baseline": {
                key: round(value, 4) for key, value in evaluation["baseline_metrics"].items()
            },
            "checks": {
                key: value.get("passed") if isinstance(value, dict) else value
                for key, value in evaluation["checks"].items()
            },
            "notes": evaluation["notes"],
        }

        report["promotion_blocked"] = None if evaluation["passed"] else evaluation["notes"]
        promoted = False
        if evaluation["passed"]:
            request_and_approve(
                platform,
                GateType.MODEL_PROMOTION,
                "Promote the stress model",
                model.id,
                {
                    "eval_results_vs_baseline": evaluation["metrics"],
                    "model_card": model.id,
                    "red_team_results": "n/a for an embedding model",
                },
                people,
                "evaluation",
            )
            model = promote_model(
                platform, model.id, gate_approved=True, approver=people["ml_lead"]["name"]
            )
            promoted = True
        report["model"] = {"id": model.id, "status": str(model.status), "promoted": promoted}

        report["gate_race"] = race_gate(platform, people)
        report["budget_race"] = race_budget(platform, cap_usd=0.6, threads=24, each_usd=0.1)
        report["audit"] = {
            "ok": platform.audit.verify().ok,
            "events": platform.audit.verify().checked,
        }

        links = {"dataset_id": dataset_id}
        if not args.skip_serving and promoted:
            clock.start("serving")
            report["serving"] = serving_phase(root, links, args.requests, args.concurrency)
            clock.stop()
            clock.start("rate_limit")
            report["rate_limit"] = rate_limit_phase(root)
            clock.stop()

        identity = IdentityService(platform)
        admin = Actor.from_person(people["ml_lead"], authenticated=False)
        one_time = identity.create_account(people["compliance"]["id"], created_by=admin)
        identity.change_password(people["compliance"]["id"], one_time, PASSWORD)
        clock.start("console")
        report["console"] = console_phase(root, people["compliance"]["name"])
        clock.stop()

        report["stages"] = clock.stages
        report["brief_after"] = load_brief(platform.store).missing
    finally:
        platform.close()

    out = root / "stress-report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
