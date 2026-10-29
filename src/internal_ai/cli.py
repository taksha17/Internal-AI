"""Command line interface for Internal-AI."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any

from internal_ai.context import Platform
from internal_ai.domain import BudgetScope, GateType
from internal_ai.engine import PlanError
from internal_ai.gates import GATE_CATALOG, GateError
from internal_ai.llm import ScriptedAdapter
from internal_ai.planner import TemplatePlanner
from internal_ai.services import governance
from internal_ai.services.actor import Actor
from internal_ai.services.identity import AuthenticationError, IdentityService
from internal_ai.services.permissions import Action, PermissionDenied, require
from internal_ai.store import TransitionError

DEMO_APPROVER = "program_lead"
DEMO_ML_LEAD = "ml_lead"
DEMO_COMPLIANCE = "compliance"
DEMO_DOMAIN_EXPERT = "domain_expert"
ASSERTED_NOTE = (
    "note: this root has no console accounts, so '{name}' is an asserted name, not an "
    "authenticated identity"
)


def _print(payload: Any) -> None:
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(payload)


def _actor(
    platform: Platform, args: argparse.Namespace, asserted: str | None, fallback: str
) -> Actor:
    """The person acting on a governance command.

    In a root with accounts the person must prove who they are with a personal
    token or their password, and a name flag cannot point at someone else. A
    root without accounts keeps the asserted name, so the first administrator
    can be set up, and says so.
    """
    as_person = getattr(args, "as_person", None)
    identity = IdentityService(platform)
    if not identity.has_accounts():
        name = as_person or asserted or fallback
        if not name:
            raise AuthenticationError("say who is acting with --by <person> or --as <person>")
        print(ASSERTED_NOTE.format(name=name), file=sys.stderr)
        return Actor.asserted(platform.store, name)
    if not as_person:
        raise AuthenticationError(
            "this root has console accounts; pass --as <person> with INTERNAL_AI_TOKEN or "
            "INTERNAL_AI_PASSWORD set, or sign in to the console"
        )
    person = identity.person(as_person)
    if person is None:
        raise AuthenticationError(f"unknown person: {as_person}")
    if asserted and asserted not in (as_person, person["id"], person["name"]):
        raise AuthenticationError(
            f"you are signed in as {person['name']}; a name flag cannot act as {asserted}"
        )
    token = os.environ.get("INTERNAL_AI_TOKEN", "")
    if token:
        actor = identity.token_actor(token)
        if actor is None or actor.id != person["id"]:
            raise AuthenticationError("INTERNAL_AI_TOKEN is not a valid token for this person")
        return actor
    password = os.environ.get("INTERNAL_AI_PASSWORD", "")
    if not password:
        if not sys.stdin.isatty():
            raise AuthenticationError("set INTERNAL_AI_TOKEN or INTERNAL_AI_PASSWORD")
        password = getpass.getpass(f"password for {person['name']}: ")
    actor = identity.authenticate(person["id"], password, client="cli")
    account = identity.account(actor.id) or {}
    if account.get("must_change_password"):
        raise AuthenticationError(
            "this account still has its one-time password; change it with "
            "'internal-ai console change-password --as <person>' or in the console"
        )
    return actor


def _gate_ok(platform: Platform, gate_id: str | None, gate_type: GateType, subject: str) -> bool:
    """An approved gate of the right type for this subject, not just any approved gate."""
    if gate_id is None:
        return False
    gate = platform.store.get_gate(gate_id)
    if gate is None or str(gate.gate_type) != str(gate_type):
        return False
    if gate.subject != subject:
        print(
            f"error: gate {gate_id} covers {gate.subject}, not {subject}",
            file=sys.stderr,
        )
        return False
    return platform.gates.is_approved(gate_id)


def cmd_init(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.actor, "program_lead")
        governance.set_budget_limit(platform, actor, BudgetScope.GLOBAL, "global", args.budget)
        governance.set_budget_limit(platform, actor, BudgetScope.PROJECT, "default", args.budget)
        platform.audit.record(
            actor=actor.id,
            actor_kind="human",
            action="platform.init",
            subject="internal-ai",
            details={"root": str(platform.settings.root), "budget_usd": args.budget},
        )
        print(f"initialized Internal-AI at {platform.settings.home}")
        print(f"database: {platform.settings.db_path}")
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.approve_by, "program_lead")
        tasks = governance.submit_plan(
            platform,
            actor,
            args.goal,
            milestone=args.milestone,
            planner=args.planner,
            approve=bool(args.approve_by),
        )
        _print(
            {
                "planned": len(tasks),
                "tasks": [
                    {
                        "id": task.id,
                        "title": task.title,
                        "owner": task.owner_role,
                        "dependencies": task.dependencies,
                        "state": str(task.state),
                        "cost_estimate_usd": task.cost_estimate_usd,
                        "gate": task.gate_id,
                    }
                    for task in tasks
                ],
            }
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.approve_by, "program_lead")
        summary = governance.run_plan(
            platform, actor, max_iterations=args.max_iterations, approve=bool(args.approve_by)
        )
        _print(summary)
    return 0


def cmd_board(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        tasks = platform.store.list_tasks()
        if args.state:
            tasks = [task for task in tasks if str(task.state) == args.state]
        if args.json:
            _print([task.to_dict() for task in tasks])
            return 0
        for task in tasks:
            deps = ",".join(task.dependencies) or "-"
            print(
                f"{task.id:<24} {task.state!s:<12} {task.owner_role:<18} "
                f"deps={deps:<20} ${task.actual_cost_usd:.2f}  {task.title}"
            )
        print(f"\n{len(tasks)} tasks")
    return 0


def cmd_gates_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        gates = platform.store.list_gates()
        if args.status:
            gates = [gate for gate in gates if str(gate.status) == args.status]
        for gate in gates:
            print(
                f"{gate.id:<20} {gate.status!s:<10} {gate.gate_type!s:<28} "
                f"{gate.subject:<20} approvers={','.join(gate.required_approvers)}"
            )
        print(f"\n{len(gates)} gates")
    return 0


def cmd_gates_request(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "program_lead")
        gate_type = GateType(args.gate_type)
        entry = GATE_CATALOG[gate_type]
        evidence = dict.fromkeys(entry.evidence, args.note or f"evidence for {args.gate_type}")
        gate = governance.request_gate(
            platform, actor, gate_type, title=args.title, subject=args.subject, evidence=evidence
        )
        print(f"gate {gate.id} opened; approvers: {', '.join(gate.required_approvers)}")
    return 0


def _decide(args: argparse.Namespace, approved: bool) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "")
        gate = governance.decide_gate(platform, actor, args.gate_id, approved, note=args.note)
        print(f"gate {gate.id} is now {gate.status}")
    return 0


def cmd_gates_approve(args: argparse.Namespace) -> int:
    return _decide(args, approved=True)


def cmd_gates_reject(args: argparse.Namespace) -> int:
    return _decide(args, approved=False)


def cmd_audit_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        events = platform.audit.recent(limit=args.limit, action=args.action, actor=args.actor)
        if args.json:
            _print([event.__dict__ for event in events])
            return 0
        for event in events:
            print(
                f"{event.seq:>5} {event.created_at} {event.actor:<18} {event.action:<28} "
                f"{event.subject:<20} {event.outcome}"
            )
        print(f"\n{len(events)} events")
    return 0


def cmd_audit_verify(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        result = platform.audit.verify()
        _print(result.to_dict())
        return 0 if result.ok else 1


def cmd_budget_status(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        state = governance.kill_switch_state(platform)
        print(f"kill switch: {'engaged' if state['engaged'] else 'released'}")
        if state["engaged"] and state["reason"]:
            print(f"reason: {state['reason']}")
        statuses = platform.budget.all_status()
        _print([status.to_dict() for status in statuses])
    return 0


def cmd_budget_set(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.actor, "program_lead")
        scope = BudgetScope(args.scope)
        governance.set_budget_limit(
            platform, actor, scope, args.key, args.limit, gate_id=args.gate_id
        )
        print(f"{scope}:{args.key} limit set to ${args.limit:.2f}")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.actor, "program_lead")
        if args.release:
            governance.release_kill_switch(platform, actor, reason=args.reason)
        else:
            governance.engage_kill_switch(platform, actor, reason=args.reason)
        print("kill switch released" if args.release else "kill switch engaged")
    return 0


def cmd_policy_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        _print(platform.policy.status_summary())
    return 0


def cmd_policy_decide(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "")
        _decision, gate_id = governance.propose_policy(
            platform, actor, args.id, args.value, rationale=args.rationale
        )
        print(f"policy {args.id} proposed as '{args.value}'; gate {gate_id} opened")
        print("approve it with: internal_ai gates approve --gate-id <id> --by <approver>")
    return 0


def cmd_policy_confirm(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "")
        decision = governance.confirm_policy(platform, actor, args.id, gate_id=args.gate_id)
        print(f"policy {args.id} decided: {decision.current_value}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        report = platform.orchestrator.status_report()
        _print(
            {
                "generated_at": report.generated_at,
                "tasks_by_state": report.tasks_by_state,
                "blocked_tasks": report.blocked_tasks,
                "spend_by_agent": report.spend_by_agent,
                "spend_total_usd": report.spend_total_usd,
                "budget_usd": report.budget_usd,
                "open_gates": report.open_gates,
                "risks": report.risks,
            }
        )
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind="scripted") as platform:
        if IdentityService(platform).has_accounts():
            print(
                "error: the demo simulates human approvals, so it only runs in a root with no "
                "console accounts; use a fresh --root",
                file=sys.stderr,
            )
            return 1
        platform.runtime.llm = ScriptedAdapter(nominal_cost_usd=args.cost_per_run)
        platform.budget.set_limit(BudgetScope.GLOBAL, "global", args.budget)
        platform.budget.set_limit(BudgetScope.PROJECT, "default", args.budget)
        platform.audit.record(
            actor=DEMO_APPROVER,
            actor_kind="human",
            action="platform.init",
            subject="internal-ai",
            details={"mode": "demo", "budget_usd": args.budget},
        )
        drafts = TemplatePlanner().plan(
            "Deliver the first from-scratch embedding model in 8 weeks", {}
        )
        tasks = platform.orchestrator.submit_plan(drafts)
        platform.orchestrator.approve_plan(DEMO_APPROVER)
        print(f"plan submitted: {len(tasks)} tasks, approved by {DEMO_APPROVER}\n")

        approvals = {str(entry.gate_type): entry.approvers for entry in GATE_CATALOG.values()}
        summary: dict[str, Any] = {}
        for _ in range(10):
            summary = platform.orchestrator.run(max_iterations=50)
            pending = platform.gates.pending()
            if not pending:
                break
            for gate in pending:
                pair = approvals.get(str(gate.gate_type), (DEMO_APPROVER, DEMO_COMPLIANCE))
                for approver in pair:
                    try:
                        platform.gates.decide(
                            gate.id,
                            approver,
                            approved=True,
                            approver_kind="human",
                            note="demo simulation",
                        )
                    except GateError:
                        continue
                platform.audit.record(
                    actor=pair[0],
                    actor_kind="human",
                    action="gate.approved",
                    subject=gate.id,
                    details={"gate_type": str(gate.gate_type), "mode": "demo"},
                )
        summary = platform.orchestrator.run(max_iterations=50)
        summary.pop("outcomes", None)
        print("run summary:")
        _print(summary)
        print("\ntask board:")
        for task in platform.store.list_tasks():
            print(f"  {task.id:<24} {task.state!s:<12} {task.owner_role:<18} {task.title}")
        report = platform.orchestrator.status_report()
        print("\nstatus report:")
        _print(
            {
                "tasks_by_state": report.tasks_by_state,
                "spend_total_usd": report.spend_total_usd,
                "budget_usd": report.budget_usd,
                "open_gates": len(report.open_gates),
                "risks": report.risks,
            }
        )
        verification = platform.audit.verify()
        print(f"\naudit chain verified: {verification.ok} ({verification.checked} events)")
        if not verification.ok:
            print(f"broken at seq {verification.broken_at_seq}: {verification.reason}")
            return 1
        return 0


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from internal_ai.api import create_app

    tls = bool(args.ssl_certfile and args.ssl_keyfile)
    if args.host not in LOOPBACK_HOSTS and not (args.allow_remote and tls):
        print(
            "error: the control plane and console bind to loopback only; serving on "
            f"{args.host} needs --allow-remote together with --ssl-certfile and --ssl-keyfile",
            file=sys.stderr,
        )
        return 1
    app = create_app(root=args.root, llm_kind=args.llm, secure_cookies=tls)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile,
    )
    return 0


def cmd_console_create_account(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        identity = IdentityService(platform)
        admin = _actor(platform, args, None, "program_lead") if identity.has_accounts() else None
        person = identity.person(args.person)
        if person is None:
            print(f"error: unknown person: {args.person}", file=sys.stderr)
            return 1
        one_time = identity.create_account(person["id"], created_by=admin)
        print(f"account created for {person['name']} ({person['id']})")
        print(f"one-time password: {one_time}")
        print("it must be changed at first sign-in; it is not shown again")
    return 0


def cmd_console_reset_password(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        identity = IdentityService(platform)
        admin = _actor(platform, args, None, "program_lead")
        person = identity.person(args.person)
        if person is None:
            print(f"error: unknown person: {args.person}", file=sys.stderr)
            return 1
        one_time = identity.reset_password(person["id"], admin)
        print(f"one-time password for {person['name']}: {one_time}")
    return 0


def cmd_console_change_password(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        identity = IdentityService(platform)
        person = identity.person(args.as_person or "")
        if person is None:
            print("error: pass --as <person>", file=sys.stderr)
            return 1
        current = os.environ.get("INTERNAL_AI_PASSWORD") or getpass.getpass("current password: ")
        new = os.environ.get("INTERNAL_AI_NEW_PASSWORD") or getpass.getpass("new password: ")
        identity.change_password(person["id"], current, new)
        print(f"password changed for {person['name']}")
    return 0


def cmd_console_create_token(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        identity = IdentityService(platform)
        actor = _actor(platform, args, None, "program_lead")
        token = identity.create_token(actor, args.label)
        print(f"personal token for {actor.name}: {token}")
        print("it acts as you on the API; it is not shown again")
    return 0


def cmd_alerts_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.alerts import AlertEngine

        engine = AlertEngine(platform.store, platform.audit, platform.budget)
        if args.check:
            engine.check_all()
        alerts = engine.list_alerts(unacknowledged_only=not args.all)
        for alert in alerts:
            print(
                f"{alert.created_at} {alert.severity:<8} {alert.kind:<26} "
                f"{alert.subject:<24} {alert.message}"
            )
        print(f"\n{len(alerts)} alerts")
    return 0


def cmd_alerts_ack(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "program_lead")
        alert = governance.acknowledge_alert(platform, actor, args.id)
        print(f"alert {alert.id} acknowledged")
    return 0


def cmd_data_release(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.pipeline import DataPipeline

        actor = _actor(platform, args, args.by, "ml_lead")
        require(platform.audit, actor, Action.RELEASE_DATASET, args.dataset_id)
        pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
        gate_approved = _gate_ok(platform, args.gate_id, GateType.DATASET_RELEASE, args.dataset_id)
        try:
            dataset = pipeline.release(args.dataset_id, actor.id, gate_approved)
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print(
            {
                "dataset_id": dataset.id,
                "released": True,
                "pii_audit": dataset.stats.get("pii_audit"),
            }
        )
    return 0


def cmd_data_pii_audit(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.pipeline import DataPipeline

        pipeline = DataPipeline(platform.store, platform.audit, platform.settings.root)
        result = pipeline.run_pii_audit(args.dataset_id, sample_size=args.sample_size)
        _print(result)
        return 0 if result["passed"] else 1


def cmd_data_withdraw(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.governance import RetentionManager

        actor = _actor(platform, args, args.by, "data_owner")
        require(platform.audit, actor, Action.WITHDRAW_SOURCE, args.source_id)
        manager = RetentionManager(platform.store, platform.audit)
        result = manager.withdraw_source(args.source_id, args.reason, actor.id)
        if args.purge:
            removed = manager.purge_source_records(args.source_id, actor.id, args.reason)
            result["records_purged"] = removed
        _print(result)
    return 0


def cmd_classifier_train(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.training.classifier import ClassifierConfig
        from internal_ai.workflows import train_classifier_model

        tokenizer = Tokenizer.load(Path(args.tokenizer))
        config = ClassifierConfig(
            name=args.name, epochs=args.epochs, cost_cap_usd=args.cost_cap, seed=args.seed
        )
        outcome = train_classifier_model(
            platform, args.dataset_id, Path(args.encoder), tokenizer, config, actor=args.actor
        )
        _print(outcome.to_dict())
        return 0 if outcome.status == "completed" else 1


def cmd_classifier_eval(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.workflows import evaluate_classifier_model

        tokenizer = Tokenizer.load(Path(args.tokenizer))
        result = evaluate_classifier_model(
            platform, Path(args.model_path), tokenizer, args.dataset_id, split=args.split
        )
        _print(result)
        return 0 if result["passed"] else 1


def cmd_job_submit(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.compute.scheduler import JobScheduler, JobSpec
        from internal_ai.sandbox import Sandbox

        scheduler = JobScheduler(
            platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
        )
        job = scheduler.submit(
            JobSpec(
                name=args.name,
                command=shlex.split(args.command),
                agent=args.agent,
                tokens=args.tokens,
                parameters=args.parameters,
                priority=args.priority,
                cost_cap_usd=args.cost_cap,
            ),
            actor=args.agent,
        )
        _print(job.to_dict())
        return 0 if str(job.state) != "rejected" else 1


def cmd_job_run(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.compute.scheduler import JobScheduler, JobSpec
        from internal_ai.sandbox import Sandbox

        scheduler = JobScheduler(
            platform.audit, platform.budget, Sandbox(platform.settings.sandbox_dir)
        )
        job = scheduler.submit(
            JobSpec(
                name=args.name,
                command=shlex.split(args.command),
                agent=args.agent,
                tokens=args.tokens,
                parameters=args.parameters,
                cost_cap_usd=args.cost_cap,
            ),
            actor=args.agent,
        )
        if str(job.state) == "rejected":
            _print(job.to_dict())
            return 1
        job = scheduler.run(job, actor=args.agent)
        _print(job.to_dict())
        return 0 if job.exit_code == 0 else 1


def cmd_inventory(args: argparse.Namespace) -> int:
    from internal_ai.data.inventory import inventory, render_inventory

    report = inventory(Path(args.root)).to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        _print(report)
    else:
        print(render_inventory(report))
    return 0


def cmd_data_build_tree(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.tree import TreeFile, TreeIndex, build_tree_index, index_hashes
        from internal_ai.workflows import build_tree_corpus

        roots = [Path(path) for path in args.tree]
        files: list[TreeFile] = []
        stats: list[dict[str, Any]] = []
        for root in roots:
            index = build_tree_index(root, source_id=root.name)
            files.extend(index.files)
            stats.append(index.stats())
        merged = TreeIndex(
            root=roots[0],
            source_id=args.source_id or "+".join(root.name for root in roots),
            files=files,
            skipped=sum(item["skipped"] for item in stats),
            truncated=sum(item["truncated"] for item in stats),
        )
        result = build_tree_corpus(
            platform,
            merged,
            name=args.name,
            version=args.version,
            acl=args.acl.split(",") if args.acl else ["engineering", "support"],
        )
        if args.hashes_out:
            Path(args.hashes_out).write_text(
                json.dumps(index_hashes(merged), indent=2, sort_keys=True), encoding="utf-8"
            )
        _print(
            {
                "dataset_id": result.dataset_id,
                "records": result.records,
                "content_hash": result.content_hash,
                "trees": stats,
            }
        )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.agentic import AgenticSearcher
        from internal_ai.serving.service import Principal

        principal = Principal(
            user_id=args.user,
            groups=frozenset(args.groups.split(",")) if args.groups else frozenset({"engineering"}),
        )
        searcher = AgenticSearcher(
            platform.store,
            platform.audit,
            platform.budget,
            embedder=None,
            llm=platform.runtime.llm if args.planner == "model" else None,
        )
        try:
            result = searcher.search(
                args.question,
                args.dataset_id,
                principal,
                max_steps=args.max_steps,
                top_k=args.top_k,
                cost_cap_usd=args.cost_cap,
                actor=args.actor,
            )
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            _print(result.to_dict())
        else:
            print(result.answer)
            print(f"\nsteps used: {result.steps_used}")
            for step in result.steps:
                print(f"  {step.index}. [{step.strategy}] {step.query} -> {step.decision}")
        return 0 if result.sufficient_evidence else 1


def cmd_people_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.people import PeopleRegistry

        registry = PeopleRegistry(platform.store)
        people = registry.list_people()
        if args.json:
            _print(people)
            return 0
        for person in people:
            marker = " (placeholder)" if person["placeholder"] else ""
            print(
                f"{person['name']:<26} {person['title']:<32} {', '.join(person['roles'])}{marker}"
            )
        print(f"\n{len(people)} people")
    return 0


def cmd_people_replace(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        actor = _actor(platform, args, args.by, "program_lead")
        record = governance.replace_person(
            platform,
            actor,
            args.id,
            args.name,
            args.title,
            args.roles.split(","),
            groups=args.groups.split(",") if args.groups is not None else None,
        )
        _print(record)
    return 0


def cmd_people_approvers(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.people import PeopleRegistry

        registry = PeopleRegistry(platform.store)
        _print(registry.approvers_for(args.gate_type))
    return 0


def cmd_feedback_list(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.feedback import failed_citations

        for row in failed_citations(platform):
            print(f"{row['time']}  {row['model']}  hits {row['hits']}  {row['latency_ms']} ms")
    return 0


def cmd_feedback_file(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.feedback import file_feedback

        actor = _actor(platform, args, args.by, "")
        try:
            row = file_feedback(platform, actor, args.question, args.file, args.line, args.dataset)
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(row["id"])
    return 0


def cmd_feedback_export(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.feedback import export_feedback

        actor = _actor(platform, args, args.by, "")
        try:
            dataset = export_feedback(platform, actor)
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(dataset.id)
    return 0


def cmd_serve_report(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.serving.report import serve_report

        report = serve_report(platform, apply=args.apply)
        print(
            f"p50 {report['p50_ms']} ms  p95 {report['p95_ms']} ms  "
            f"target {report['latency_target_ms']} ms  "
            f"within_brief {str(report['within_brief']).lower()}"
        )
        print(f"cost_per_1000 {report['cost_per_1000']}")
        verb = "dropped" if args.apply else "would drop"
        print(f"{verb} {report['past_retention']} serving event(s) older than the brief timeline")
    return 0


def cmd_serve_model(args: argparse.Namespace) -> int:
    import uvicorn

    from internal_ai.serving.app import create_serving_app

    app = create_serving_app(root=args.root, rate_limit_per_minute=args.rate_limit)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    from internal_ai.data.samples import generate_corpus, generate_eval_pairs

    base = args.out
    corpus = generate_corpus(base / "corpus.jsonl", records=args.records)
    pairs = generate_eval_pairs(base / "eval_pairs.json")
    _print({"corpus": corpus, "eval_pairs": pairs})
    return 0


def cmd_data_build(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import build_corpus

        result = build_corpus(
            platform,
            [Path(path) for path in args.source],
            name=args.name,
            version=args.version,
            approved=not args.unapproved,
        )
        _print(
            {
                "dataset_id": result.dataset_id,
                "content_hash": result.content_hash,
                "records": result.records,
                "tokens": result.tokens,
                "stats": result.stats,
            }
        )
    return 0


def cmd_data_split(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import split_corpus

        result = split_corpus(platform, args.dataset_id, seed=args.seed)
        _print(result.stats)
    return 0


def cmd_dataset_trace(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.pipeline import DataPipeline

        dataset = platform.store.get_dataset(args.dataset_id)
        if dataset is None:
            print(f"unknown dataset: {args.dataset_id}", file=sys.stderr)
            return 1
        report = DataPipeline(platform.store, platform.audit, platform.settings.root).trace_sample(
            args.dataset_id, limit=args.sample, actor=args.by or "cli"
        )
        steps = ",".join(
            str(step.get("step", ""))
            for step in dataset.lineage.get("steps", [])
            if step.get("step")
        )
        commit = str(dataset.lineage.get("code_commit", "unknown"))
        rows = sorted(
            platform.store.dataset_records(args.dataset_id), key=lambda row: row["record_key"]
        )[: args.sample]
        for row in rows:
            print(f"{row['record_key']}\t{row['source_id']}\t{steps}\t{commit}")
        if not steps or commit == "unknown" or not report["passed"]:
            return 1
    return 0


def cmd_data_lineage(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        dataset = platform.store.get_dataset(args.dataset_id)
        if dataset is None:
            print(f"unknown dataset: {args.dataset_id}", file=sys.stderr)
            return 1
        _print(dataset.lineage)
    return 0


def cmd_tokenizer_train(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import train_tokenizer_for

        jargon = args.jargon.split(",") if args.jargon else None
        _, coverage = train_tokenizer_for(
            platform, args.dataset_id, vocab_size=args.vocab_size, jargon=jargon
        )
        _print(coverage)
    return 0


def cmd_train_run(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.training.trainer import TrainConfig
        from internal_ai.workflows import train_embedding_model

        tokenizer_path = Path(args.tokenizer)
        tokenizer = Tokenizer.load(tokenizer_path)
        config = TrainConfig(
            name=args.name,
            objective=args.objective,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            cost_cap_usd=args.cost_cap,
            seed=args.seed,
            learning_rate=args.learning_rate,
        )
        outcome = train_embedding_model(
            platform, args.dataset_id, tokenizer, config, actor=args.actor, pair_mode=args.pair_mode
        )
        _print(outcome.to_dict())
        return 0 if outcome.status == "completed" else 1


def cmd_eval_create(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.evals.harness import EvalHarness

        items = json.loads(Path(args.file).read_text(encoding="utf-8"))
        harness = EvalHarness(platform.store, platform.audit)
        eval_set = harness.create_eval_set(args.name, items, version=args.version)
        _print({"eval_set_id": eval_set.id, "items": len(items), "frozen": eval_set.frozen})
    return 0


def cmd_eval_freeze(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.evals.harness import EvalHarness

        actor = _actor(platform, args, args.by, "")
        require(platform.audit, actor, Action.FREEZE_EVAL_SET, args.id)
        harness = EvalHarness(platform.store, platform.audit)
        gate_approved = _gate_ok(platform, args.gate_id, GateType.EVAL_SET_FREEZE, args.id)
        try:
            eval_set = harness.freeze_eval_set(args.id, actor.id, gate_approved)
        except PermissionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print(
            {
                "eval_set_id": eval_set.id,
                "frozen": eval_set.frozen,
                "approved_by": eval_set.approved_by,
            }
        )
    return 0


def cmd_eval_derive(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.evals.harness import EvalHarness
        from internal_ai.workflows import derive_eval_items

        items = derive_eval_items(platform, args.dataset_id, count=args.count, mode=args.mode)
        harness = EvalHarness(platform.store, platform.audit)
        eval_set = harness.create_eval_set(args.name, items, version=args.version)
        _print({"eval_set_id": eval_set.id, "items": len(items), "frozen": eval_set.frozen})
    return 0


def cmd_eval_run(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.data.tokenizer import Tokenizer
        from internal_ai.workflows import evaluate_embedding_model

        tokenizer = Tokenizer.load(Path(args.tokenizer))
        result = evaluate_embedding_model(
            platform,
            Path(args.model_path),
            args.eval_set,
            tokenizer,
            args.dataset_id,
            margin=args.margin,
            model_id=args.model_id,
        )
        _print(result)
        return 0 if result["passed"] else 1


def cmd_model_register(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.training.trainer import TrainingOutcome
        from internal_ai.workflows import register_model

        reproducibility: dict[str, Any] = {}
        run_id = args.run_id
        if run_id == "manual":
            candidate = Path(args.model_path).stem
            if candidate.startswith("run_"):
                run_id = candidate
        run_dir = platform.settings.artifacts_dir / "runs" / run_id
        reproducibility_file = run_dir / "reproducibility.json"
        if reproducibility_file.exists():
            reproducibility = json.loads(reproducibility_file.read_text(encoding="utf-8"))
        elif args.config:
            reproducibility = {"config": json.loads(args.config)}
        outcome = TrainingOutcome(
            run_id=run_id,
            status="completed",
            steps=args.steps,
            tokens_seen=0,
            final_loss=0.0,
            cost_usd=0.0,
            checkpoint_path="",
            metrics_path="",
            model_path=args.model_path,
            reproducibility=reproducibility,
        )
        model = register_model(platform, args.name, Path(args.model_path), outcome, args.dataset_id)
        _print({"model_id": model.id, "name": model.name, "version": model.version})
    return 0


def cmd_model_promote(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import promote_model

        actor = _actor(platform, args, args.by, "ml_lead")
        require(platform.audit, actor, Action.PROMOTE_MODEL, args.id)
        gate_approved = _gate_ok(platform, args.gate_id, GateType.MODEL_PROMOTION, args.id)
        try:
            model = promote_model(platform, args.id, gate_approved, approver=actor.id)
        except (PermissionError, KeyError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print({"model_id": model.id, "status": model.status})
    return 0


def cmd_models_demote(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import demote_model

        actor = _actor(platform, args, args.by, "ml_lead")
        require(platform.audit, actor, Action.PROMOTE_MODEL, args.id)
        try:
            model = demote_model(platform, args.id, actor.id)
        except (PermissionError, KeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        _print({"model_id": model.id, "status": model.status, "path": model.path})
    return 0


def cmd_model_card(args: argparse.Namespace) -> int:
    with Platform.create(root=args.root, llm_kind=args.llm) as platform:
        from internal_ai.workflows import model_card_markdown

        model = platform.store.get_model(args.id)
        if model is None:
            print(f"unknown model: {args.id}", file=sys.stderr)
            return 1
        print(model_card_markdown(model))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="internal_ai", description="Internal-AI control plane")
    parser.add_argument("--root", type=Path, default=None, help="project root")
    parser.add_argument("--llm", default="auto", choices=["auto", "scripted", "anthropic", "local"])
    parser.add_argument(
        "--as",
        dest="as_person",
        default=None,
        help="act as this person (needs INTERNAL_AI_TOKEN or INTERNAL_AI_PASSWORD)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="initialize the platform state")
    init.add_argument("--budget", type=float, default=500.0)
    init.add_argument("--actor", default=None)
    init.set_defaults(func=cmd_init)

    plan = sub.add_parser("plan", help="decompose a goal into a task graph")
    plan.add_argument("--goal", required=True)
    plan.add_argument("--milestone", default=None)
    plan.add_argument("--planner", default="template", choices=["template", "llm"])
    plan.add_argument("--approve-by", default=None)
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run the orchestrator loop")
    run.add_argument("--max-iterations", type=int, default=100)
    run.add_argument("--approve-by", default=None)
    run.set_defaults(func=cmd_run)

    board = sub.add_parser("board", help="show the task board")
    board.add_argument("--state", default=None)
    board.add_argument("--json", action="store_true")
    board.set_defaults(func=cmd_board)

    gates = sub.add_parser("gates", help="approval gates")
    gates_sub = gates.add_subparsers(dest="gates_command", required=True)
    gates_list = gates_sub.add_parser("list")
    gates_list.add_argument("--status", default=None)
    gates_list.set_defaults(func=cmd_gates_list)
    gates_approve = gates_sub.add_parser("approve")
    gates_approve.add_argument("--gate-id", required=True)
    gates_approve.add_argument("--by", default=None)
    gates_approve.add_argument("--note", default="")
    gates_approve.set_defaults(func=cmd_gates_approve)
    gates_reject = gates_sub.add_parser("reject")
    gates_reject.add_argument("--gate-id", required=True)
    gates_reject.add_argument("--by", default=None)
    gates_reject.add_argument("--note", default="")
    gates_reject.set_defaults(func=cmd_gates_reject)
    gates_request = gates_sub.add_parser("request")
    gates_request.add_argument(
        "--gate-type", required=True, choices=[str(item) for item in GateType]
    )
    gates_request.add_argument("--title", required=True)
    gates_request.add_argument("--subject", required=True)
    gates_request.add_argument("--by", default=None)
    gates_request.add_argument("--note", default="")
    gates_request.set_defaults(func=cmd_gates_request)

    audit = sub.add_parser("audit", help="audit log")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    audit_list = audit_sub.add_parser("list")
    audit_list.add_argument("--limit", type=int, default=50)
    audit_list.add_argument("--action", default=None)
    audit_list.add_argument("--actor", default=None)
    audit_list.add_argument("--json", action="store_true")
    audit_list.set_defaults(func=cmd_audit_list)
    audit_verify = audit_sub.add_parser("verify")
    audit_verify.set_defaults(func=cmd_audit_verify)

    budget = sub.add_parser("budget", help="budgets and kill switches")
    budget_sub = budget.add_subparsers(dest="budget_command", required=True)
    budget_status = budget_sub.add_parser("status")
    budget_status.set_defaults(func=cmd_budget_status)
    budget_set = budget_sub.add_parser("set")
    budget_set.add_argument("--scope", required=True, choices=[str(item) for item in BudgetScope])
    budget_set.add_argument("--key", required=True)
    budget_set.add_argument("--limit", type=float, required=True)
    budget_set.add_argument("--actor", default=None)
    budget_set.add_argument("--gate-id", default=None, help="approved budget_increase gate")
    budget_set.set_defaults(func=cmd_budget_set)
    kill = budget_sub.add_parser("kill")
    kill.add_argument("--actor", default=None)
    kill.add_argument("--release", action="store_true")
    kill.add_argument("--reason", default="")
    kill.set_defaults(func=cmd_kill)

    policy = sub.add_parser("policy", help="policy decisions P1-P5")
    policy_sub = policy.add_subparsers(dest="policy_command", required=True)
    policy_list = policy_sub.add_parser("list")
    policy_list.set_defaults(func=cmd_policy_list)
    policy_decide = policy_sub.add_parser("decide")
    policy_decide.add_argument("--id", required=True)
    policy_decide.add_argument("--value", required=True)
    policy_decide.add_argument("--by", default=None)
    policy_decide.add_argument("--rationale", default="")
    policy_decide.set_defaults(func=cmd_policy_decide)
    policy_confirm = policy_sub.add_parser("confirm")
    policy_confirm.add_argument("--id", required=True)
    policy_confirm.add_argument("--by", default=None)
    policy_confirm.add_argument("--gate-id", default=None)
    policy_confirm.set_defaults(func=cmd_policy_confirm)

    report = sub.add_parser("report", help="status and risk report")
    report.set_defaults(func=cmd_report)

    demo = sub.add_parser("demo", help="run the Phase 0 no-op loop end to end")
    demo.add_argument("--budget", type=float, default=500.0)
    demo.add_argument("--cost-per-run", type=float, default=0.25)
    demo.set_defaults(func=cmd_demo)

    serve = sub.add_parser("serve", help="run the control plane API and the browser console")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("--allow-remote", action="store_true")
    serve.add_argument("--ssl-certfile", default=None)
    serve.add_argument("--ssl-keyfile", default=None)
    serve.set_defaults(func=cmd_serve)
    serve_sub = serve.add_subparsers(dest="serve_command")
    serve_report = serve_sub.add_parser("report", help="p50, p95 and cost from serving.request")
    serve_report.add_argument("--apply", action="store_true")
    serve_report.set_defaults(func=cmd_serve_report)

    console = sub.add_parser("console", help="browser console accounts and personal tokens")
    console_sub = console.add_subparsers(dest="console_command", required=True)
    console_create = console_sub.add_parser(
        "create-account", help="create an account; the first one needs no sign-in"
    )
    console_create.add_argument("--person", required=True, help="person id or exact name")
    console_create.set_defaults(func=cmd_console_create_account)
    console_reset = console_sub.add_parser("reset-password")
    console_reset.add_argument("--person", required=True)
    console_reset.set_defaults(func=cmd_console_reset_password)
    console_change = console_sub.add_parser("change-password")
    console_change.set_defaults(func=cmd_console_change_password)
    console_token = console_sub.add_parser("create-token")
    console_token.add_argument("--label", default="cli")
    console_token.set_defaults(func=cmd_console_create_token)

    serve_model = sub.add_parser("serve-model", help="run the inference and retrieval API")
    serve_model.add_argument("--host", default="127.0.0.1")
    serve_model.add_argument("--port", type=int, default=8081)
    serve_model.add_argument("--rate-limit", type=int, default=60)
    serve_model.set_defaults(func=cmd_serve_model)

    classifier = sub.add_parser("classifier", help="task classifier training and evaluation")
    classifier_sub = classifier.add_subparsers(dest="classifier_command", required=True)
    classifier_train = classifier_sub.add_parser("train")
    classifier_train.add_argument("--dataset-id", required=True)
    classifier_train.add_argument("--encoder", required=True)
    classifier_train.add_argument("--tokenizer", required=True)
    classifier_train.add_argument("--name", default="router-v1")
    classifier_train.add_argument("--epochs", type=int, default=40)
    classifier_train.add_argument("--cost-cap", type=float, default=2.0)
    classifier_train.add_argument("--seed", type=int, default=23)
    classifier_train.add_argument("--actor", default="ml_engineer")
    classifier_train.set_defaults(func=cmd_classifier_train)
    classifier_eval = classifier_sub.add_parser("eval")
    classifier_eval.add_argument("--model-path", required=True)
    classifier_eval.add_argument("--tokenizer", required=True)
    classifier_eval.add_argument("--dataset-id", required=True)
    classifier_eval.add_argument("--split", default="test")
    classifier_eval.set_defaults(func=cmd_classifier_eval)

    job = sub.add_parser("job", help="compute plane job submission")
    job_sub = job.add_subparsers(dest="job_command", required=True)
    for name, handler in (("submit", cmd_job_submit), ("run", cmd_job_run)):
        job_parser = job_sub.add_parser(name)
        job_parser.add_argument("--name", required=True)
        job_parser.add_argument("--command", required=True)
        job_parser.add_argument("--agent", default="infra_engineer")
        job_parser.add_argument("--tokens", type=int, default=1_000_000)
        job_parser.add_argument("--parameters", type=int, default=10_000_000)
        job_parser.add_argument("--priority", type=int, default=5)
        job_parser.add_argument("--cost-cap", type=float, default=5.0)
        job_parser.set_defaults(func=handler)

    inventory_cmd = sub.add_parser("inventory", help="catalogue candidate data sources")
    inventory_cmd.add_argument("--root", default=".")
    inventory_cmd.add_argument("--out", default=None)
    inventory_cmd.add_argument("--json", action="store_true")
    inventory_cmd.set_defaults(func=cmd_inventory)

    ask = sub.add_parser("ask", help="long-horizon agentic search over a corpus")
    ask.add_argument("question")
    ask.add_argument("--dataset-id", required=True)
    ask.add_argument("--max-steps", type=int, default=6)
    ask.add_argument("--top-k", type=int, default=8)
    ask.add_argument("--cost-cap", type=float, default=1.0)
    ask.add_argument("--user", default="developer")
    ask.add_argument("--groups", default="engineering")
    ask.add_argument("--actor", default="ai_engineer")
    ask.add_argument("--planner", default="heuristic", choices=["heuristic", "model"])
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    people = sub.add_parser("people", help="human roster and approvers")
    people_sub = people.add_subparsers(dest="people_command", required=True)
    people_list = people_sub.add_parser("list")
    people_list.add_argument("--json", action="store_true")
    people_list.set_defaults(func=cmd_people_list)
    people_replace = people_sub.add_parser("replace")
    people_replace.add_argument("--id", required=True)
    people_replace.add_argument("--name", required=True)
    people_replace.add_argument("--title", required=True)
    people_replace.add_argument("--roles", required=True)
    people_replace.add_argument("--groups", default=None)
    people_replace.add_argument("--by", default=None)
    people_replace.set_defaults(func=cmd_people_replace)
    people_approvers = people_sub.add_parser("approvers")
    people_approvers.add_argument("--gate-type", required=True)
    people_approvers.set_defaults(func=cmd_people_approvers)

    alerts = sub.add_parser("alerts", help="spend, task and data-access alerts")
    alerts_sub = alerts.add_subparsers(dest="alerts_command", required=True)
    alerts_list = alerts_sub.add_parser("list")
    alerts_list.add_argument("--all", action="store_true")
    alerts_list.add_argument("--check", action="store_true")
    alerts_list.set_defaults(func=cmd_alerts_list)
    alerts_ack = alerts_sub.add_parser("ack")
    alerts_ack.add_argument("--id", required=True)
    alerts_ack.add_argument("--by", default=None)
    alerts_ack.set_defaults(func=cmd_alerts_ack)

    sample = sub.add_parser("sample", help="generate a synthetic corpus for local runs")
    sample.add_argument("--out", type=Path, default=Path("var/samples"))
    sample.add_argument("--records", type=int, default=120)
    sample.set_defaults(func=cmd_sample)

    data = sub.add_parser("data", help="data plane operations")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_build = data_sub.add_parser("build")
    data_build.add_argument("--source", action="append", required=True)
    data_build.add_argument("--name", default="corpus")
    data_build.add_argument("--version", type=int, default=1)
    data_build.add_argument("--unapproved", action="store_true")
    data_build.set_defaults(func=cmd_data_build)
    data_split = data_sub.add_parser("split")
    data_split.add_argument("--dataset-id", required=True)
    data_split.add_argument("--seed", default="forge-v1")
    data_split.set_defaults(func=cmd_data_split)
    data_lineage = data_sub.add_parser("lineage")
    data_lineage.add_argument("--dataset-id", required=True)
    data_lineage.set_defaults(func=cmd_data_lineage)
    dataset_cmd = sub.add_parser("dataset", help="dataset lineage sample")
    dataset_sub = dataset_cmd.add_subparsers(dest="dataset_command", required=True)
    dataset_trace = dataset_sub.add_parser("trace")
    dataset_trace.add_argument("dataset_id")
    dataset_trace.add_argument("--sample", type=int, default=20)
    dataset_trace.add_argument("--by", default="cli")
    dataset_trace.set_defaults(func=cmd_dataset_trace)
    data_release = data_sub.add_parser("release")
    data_release.add_argument("--dataset-id", required=True)
    data_release.add_argument("--by", default=None)
    data_release.add_argument("--gate-id", default=None)
    data_release.set_defaults(func=cmd_data_release)
    data_audit = data_sub.add_parser("pii-audit")
    data_audit.add_argument("--dataset-id", required=True)
    data_audit.add_argument("--sample-size", type=int, default=50)
    data_audit.set_defaults(func=cmd_data_pii_audit)
    data_tree = data_sub.add_parser("build-tree")
    data_tree.add_argument("--tree", action="append", required=True)
    data_tree.add_argument("--source-id", default=None)
    data_tree.add_argument("--name", default="code-corpus")
    data_tree.add_argument("--version", type=int, default=1)
    data_tree.add_argument("--acl", default="engineering,support")
    data_tree.add_argument("--hashes-out", default=None)
    data_tree.set_defaults(func=cmd_data_build_tree)
    data_withdraw = data_sub.add_parser("withdraw")
    data_withdraw.add_argument("--source-id", required=True)
    data_withdraw.add_argument("--reason", required=True)
    data_withdraw.add_argument("--by", default=None)
    data_withdraw.add_argument("--purge", action="store_true")
    data_withdraw.set_defaults(func=cmd_data_withdraw)

    tokenizer = sub.add_parser("tokenizer", help="tokenizer training")
    tokenizer_sub = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    tokenizer_train = tokenizer_sub.add_parser("train")
    tokenizer_train.add_argument("--dataset-id", required=True)
    tokenizer_train.add_argument("--vocab-size", type=int, default=4000)
    tokenizer_train.add_argument("--jargon", default=None)
    tokenizer_train.set_defaults(func=cmd_tokenizer_train)

    train = sub.add_parser("train", help="train a from-scratch model")
    train_sub = train.add_subparsers(dest="train_command", required=True)
    train_run = train_sub.add_parser("run")
    train_run.add_argument("--dataset-id", required=True)
    train_run.add_argument("--tokenizer", required=True)
    train_run.add_argument("--name", default="embedding-v1")
    train_run.add_argument(
        "--objective", default="contrastive", choices=["contrastive", "mlm", "causal"]
    )
    train_run.add_argument("--steps", type=int, default=60)
    train_run.add_argument("--batch-size", type=int, default=8)
    train_run.add_argument("--seq-len", type=int, default=128)
    train_run.add_argument("--cost-cap", type=float, default=5.0)
    train_run.add_argument("--seed", type=int, default=17)
    train_run.add_argument("--learning-rate", type=float, default=3e-4)
    train_run.add_argument("--actor", default="ml_engineer")
    train_run.add_argument("--pair-mode", default="paraphrase", choices=["paraphrase", "title"])
    train_run.set_defaults(func=cmd_train_run)

    evaluate = sub.add_parser("eval", help="evaluation harness")
    evaluate_sub = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_create = evaluate_sub.add_parser("create")
    eval_create.add_argument("--name", required=True)
    eval_create.add_argument("--file", required=True)
    eval_create.add_argument("--version", type=int, default=1)
    eval_create.set_defaults(func=cmd_eval_create)
    eval_freeze = evaluate_sub.add_parser("freeze")
    eval_freeze.add_argument("--id", required=True)
    eval_freeze.add_argument("--by", default=None)
    eval_freeze.add_argument("--gate-id", default=None)
    eval_freeze.set_defaults(func=cmd_eval_freeze)
    eval_derive = evaluate_sub.add_parser("derive")
    eval_derive.add_argument("--dataset-id", required=True)
    eval_derive.add_argument("--name", default="retrieval-v1")
    eval_derive.add_argument("--count", type=int, default=24)
    eval_derive.add_argument("--mode", default="paraphrase", choices=["paraphrase", "title"])
    eval_derive.add_argument("--version", type=int, default=1)
    eval_derive.set_defaults(func=cmd_eval_derive)
    eval_run = evaluate_sub.add_parser("run")
    eval_run.add_argument("--model-path", required=True)
    eval_run.add_argument("--eval-set", required=True)
    eval_run.add_argument("--tokenizer", required=True)
    eval_run.add_argument("--dataset-id", required=True)
    eval_run.add_argument("--margin", type=float, default=0.0)
    eval_run.add_argument("--model-id", default=None)
    eval_run.set_defaults(func=cmd_eval_run)

    model = sub.add_parser("model", help="model registry")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_register = model_sub.add_parser("register")
    model_register.add_argument("--name", required=True)
    model_register.add_argument("--model-path", required=True)
    model_register.add_argument("--dataset-id", required=True)
    model_register.add_argument("--run-id", default="manual")
    model_register.add_argument("--steps", type=int, default=0)
    model_register.add_argument("--config", default=None)
    model_register.set_defaults(func=cmd_model_register)
    model_promote = model_sub.add_parser("promote")
    model_promote.add_argument("--id", required=True)
    model_promote.add_argument("--by", default=None)
    model_promote.add_argument("--gate-id", default=None)
    model_promote.set_defaults(func=cmd_model_promote)
    model_card = model_sub.add_parser("card")
    model_card.add_argument("--id", required=True)
    model_card.set_defaults(func=cmd_model_card)
    models = sub.add_parser("models", help="serving decisions for registered models")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    models_demote = models_sub.add_parser("demote")
    models_demote.add_argument("id")
    models_demote.add_argument("--by", default=None)
    models_demote.set_defaults(func=cmd_models_demote)

    feedback = sub.add_parser("feedback", help="ask failures a person files by hand")
    feedback_sub = feedback.add_subparsers(dest="feedback_command", required=True)
    feedback_list = feedback_sub.add_parser("list")
    feedback_list.set_defaults(func=cmd_feedback_list)
    feedback_file = feedback_sub.add_parser("file")
    feedback_file.add_argument("--question", required=True)
    feedback_file.add_argument("--file", required=True)
    feedback_file.add_argument("--line", type=int, required=True)
    feedback_file.add_argument("--dataset", required=True)
    feedback_file.add_argument("--by", default=None)
    feedback_file.set_defaults(func=cmd_feedback_file)
    feedback_export = feedback_sub.add_parser("export")
    feedback_export.add_argument("--by", default=None)
    feedback_export.set_defaults(func=cmd_feedback_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    try:
        return int(args.func(args))
    except (
        PlanError,
        GateError,
        PermissionDenied,
        AuthenticationError,
        TransitionError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyError as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
