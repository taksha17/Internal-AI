# Tester guide

Everything below runs against the repository as it stands on `main`. Commands
were executed before this guide was written; the expected outputs are what the
platform actually produced.

## The two APIs

| API | Address | Header | Backed by |
| --- | --- | --- | --- |
| Control plane | `http://127.0.0.1:8080` | `X-Internal-Key: cp-test-key-2026` | pilot state in `var/` |
| Serving and retrieval | `http://127.0.0.1:8081` | `X-Internal-Key: serve-test-key` | `var_e2e/var/` |

The serving key maps to user `alice` in groups `engineering` and
`platform-admins`. Start them yourself with:

```bash
INTERNAL_AI_API_KEY='cp-test-key-2026' \
  .venv/bin/python -m internal_ai.cli serve --host 127.0.0.1 --port 8080

INTERNAL_AI_HOME=var_e2e/var \
INTERNAL_AI_SERVING_KEYS='serve-test-key:alice:engineering|platform-admins' \
  .venv/bin/python -m internal_ai.cli serve-model --host 127.0.0.1 --port 8081
```

Both bind to loopback only. Stop them with `pkill -f 'internal_ai.cli serve'`
and `pkill -f 'internal_ai.cli serve-model'`.

The old `X-Forge-Key` header is dead: it returns 401 on both APIs.

## 0. One-command check

Before reading further, run the whole endpoint sweep. It uses only the
standard library and prints PASS or FAIL per check, exiting non-zero if
anything fails:

```bash
INTERNAL_AI_CONTROL_KEY=cp-test-key-2026 \
INTERNAL_AI_SERVING_KEY=serve-test-key \
  .venv/bin/python scripts/probe.py
```

Expected: `27/27 checks passed` and exit code 0. It covers every read
endpoint below, the three authentication negatives (no key, wrong key,
legacy header), and nine governance writes that must be refused when the
caller has only the shared key and no personal token. The rest of this
guide explains what each check means and what else to exercise by hand.

## 1. Control plane

```bash
curl -s http://127.0.0.1:8080/health
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/gates
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/budget
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/audit/verify
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/policy
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/roles
curl -s -H 'X-Internal-Key: cp-test-key-2026' http://127.0.0.1:8080/report
```

Expected: `/health` reports `kill_switch: false` and the global budget at
`limit_usd 500.0`, `spent_usd 0.0189`. `/audit/verify` reports
`{"ok": true, "checked": 82}` on the pilot state; the count grows with every
audited action, and `broken_at_seq` stays `null`. `/tasks` is empty because
the pilot ran its tasks in a separate root; `/gates` shows 11 approved gates
and `/policy` shows P1 to P5 decided.

Requests without the key get `401 {"detail": "invalid or missing
X-Internal-Key"}`. `/health` is deliberately readable without a key so a
process can check liveness.

## 2. Serving and retrieval

```bash
curl -s -H 'X-Internal-Key: serve-test-key' http://127.0.0.1:8081/v1/models
curl -s -H 'X-Internal-Key: serve-test-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "how does the platform enforce gates?"}' \
  http://127.0.0.1:8081/v1/embed
curl -s -H 'X-Internal-Key: serve-test-key' \
  -H 'Content-Type: application/json' \
  -d '{"query": "budget enforcement", "top_k": 3}' \
  http://127.0.0.1:8081/v1/search
```

`/v1/embed` takes `query`, not `input`. `/v1/models` lists the candidate
version 1 and the promoted version 2, with `active: true` on the promoted
one, because that is the version the router actually serves. A missing or
wrong key gets `401 {"detail": "invalid api key"}`.

The served model is the small pilot embedding model trained on the synthetic
corpus. It proves the plumbing, not the retrieval quality. On the real corpus
the model scores 0.06 MRR against a BM25 baseline of 0.18, so promotion was
correctly refused and nothing was registered on the pilot state.

## 3. Retrieval over the real corpus

The pilot state holds the app-builder tree as `ds_4ae89b3f496c`, a versioned
copy of a Chrome extension codebase. Agentic search answers questions with
cited `file:line` evidence and does not wait on a promotable model.

```bash
.venv/bin/python -m internal_ai.cli ask \
  'ashby autofill handler' --dataset-id ds_4ae89b3f496c \
  --user alice --groups engineering --max-steps 3
```

Expected answer cites `content/page_bridge.js` and a second location.

Permissions are enforced on this path as well:

```bash

.venv/bin/python -m internal_ai.cli ask \
  'ashby autofill handler' --dataset-id ds_4ae89b3f496c \
  --user mallory --groups legal --max-steps 3
```

Expected: `error: no records in dataset ds_4ae89b3f496c are readable by
mallory`, exit code 1, and an `agentic_search.denied` entry in the audit log.
The records are scoped to `engineering`, so a reader in another group gets a
denial rather than the corpus.

## 4. Governance checks that must fail

Use a scratch root so the pilot state stays untouched:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test demo
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test gates list
```

The demo ends with `audit chain verified: True (279 events)` and 13 accepted
tasks. Task ids are derived from the plan, so a root holds one plan. Running
the demo a second time in the same root is refused, with exit code 1 and no
traceback: `error: plan already submitted: 13 of 13 tasks already exist (first
tsk_corpus); use a fresh root or a different goal`. Use a new root for a
second run.

Request a gate, then try the forbidden approvals:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test gates request \
  --gate-type model_promotion --title 'Test gate' --subject tsk_promote --by ml_lead
```

That prints `gate gate_b320bed391dc opened; approvers: ml_lead, domain_expert`.
Use the id it gives you in the commands below.

- The requester approving their own gate is refused, exit code 1:
  `error: dual control violation: ml_lead requested gate gate_b320bed391dc and
  cannot approve it`.
- An approver whose roles do not cover the required ones leaves the gate
  pending: the CLI prints `gate gate_b320bed391dc is now pending` and the gate
  does not pass.
- A gate only passes once the union of approver roles covers every required
  approver. `model_promotion` needs `ml_lead` and `domain_expert`, so
  `gates approve --gate-id <id> --by "Marcus Delgado"` (ml_lead) leaves it
  pending, and `gates approve --gate-id <id> --by "Aisha Okonkwo"`
  (domain_expert) completes it: `gate ... is now approved`.

The kill switch stops work and says so. The demo root has no work left, so
use a second scratch root with a plan that has not run yet:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill init --budget 500
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill plan \
  --goal 'Deliver the first from-scratch embedding model in 8 weeks'
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill budget kill
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill budget status
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill run --approve-by program_lead
```

`budget status` prints `kill switch: engaged`. The run summary reports
`"kill_switch": "engaged"`, `"dispatched": 0` and `"spend_usd": 0.0`. The
tasks that were ready move to `blocked` on the board, and each one gets a
`budget.kill_switch_blocked` entry with outcome `denied` in the audit log.
Releasing the switch puts them back in the queue:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill budget kill --release
.venv/bin/python -m internal_ai.cli --root /tmp/guide_kill run
```

The next run reports `"kill_switch": "released"` and dispatches the work that
was held.

Other refusals worth checking:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test budget set \
  --scope global --key global --limit 1
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test audit verify
```

`budget set` caps spend: spending exactly the cap is allowed and going over it
is denied. `audit verify` reports `"ok": true`; the event count grows with
every audited action.

## 5. Trust model, stated plainly

The control-plane key is a shared operator credential for reads. It cannot
name the person on a governance write. Plan submission and approval, runs,
gate requests and decisions, budget limits, the kill switch, policy changes,
task transitions and alert acknowledgement all return 401 unless the request
also carries `Authorization: Bearer <personal token>`. The actor is the
person that token belongs to. Fields such as `approve_by`, `approver` and
`actor` in the body are ignored.

The command line is the same rule with one exception. On a root that has no
accounts yet, `--by` and `--actor` are asserted names and the CLI says so on
stderr. The checks in section 4 use that bootstrap mode. The moment an
account exists, those commands require `--as <person>` plus
`INTERNAL_AI_TOKEN` or `INTERNAL_AI_PASSWORD`, and a one-time password must
be changed before it can act.

The browser console never asserts a name. The session cookie is the person,
every form carries that session's CSRF token, and the pages bind to loopback
unless you pass `--allow-remote` together with a TLS certificate and key.

Inside that boundary the older rules still hold: agents cannot decide gates,
the requester cannot approve their own gate, a gate passes only when the
approvers' roles cover every required role, and the person who engaged the
kill switch cannot release it. Do not expose port 8080 beyond the host.

The serving API has a different, per-user model: each key maps to a user and
a set of groups, and those groups decide which records a search may return.

## 6. Known limits

- No model is promotable on the real corpus, so the pilot state has no
  registered model and `/v1/search` there would return 404. Use the serving
  instance above for the API path and the `ask` command for real retrieval.
- Jobs run in-process. Sandbox container isolation and the compute backend
  decision are open in `STATUS.md` section 10. Do not feed untrusted code to
  the compute plane.
- Retrieval is keyword level until the corpus grows and the compute decision
  in `STATUS.md` section 10 is made.
- The audit chain covers what ran. `audit verify` checks the hash chain, not
  the completeness of what should have run.
- Company single sign-on is not wired up. Accounts are local, and a
  placeholder person from the shipped roster cannot sign in until someone
  replaces them with a real name.

## 7. The browser console

This is the path for someone who does not want to use the command line after
the first account exists. Use a fresh root.

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/console_guide people replace \
  --id ppl_priya_raghunathan --name "Ada Admin" --title "Program lead" \
  --roles program_lead,finance --groups platform-admins
.venv/bin/python -m internal_ai.cli --root /tmp/console_guide console create-account \
  --person "Ada Admin"
.venv/bin/python -m internal_ai.cli --root /tmp/console_guide serve \
  --host 127.0.0.1 --port 8080
```

The create-account command prints a one-time password once. Open
`http://127.0.0.1:8080/console`, sign in as Ada Admin, and set a password of
at least 12 characters. From there:

1. Home names the step you are on, the budget, open alerts and the kill
   switch. The switch is on every page. Engaging it needs a typed reason.
   Releasing it needs a different signed-in person.
2. Data → generate sample data, inspect the path, and register it. That opens
   a `new_data_source_access` gate. Ada cannot approve her own request.
   Administration → add a second person who holds `data_owner` and
   `compliance` (or two people who hold them between them), create their
   account, and approve from their session. A folder of model weights, a
   path outside the approved folders, or a folder that contains secrets is
   refused before the gate is opened.
3. Build the dataset, then on its page split it, request release (the PII
   audit runs first), and after the `dataset_release` gate is approved,
   release it. Train a tokenizer on the same page and create an eval set.
4. Eval sets shows the leakage check. Freeze only after an `eval_set_freeze`
   gate. The edit control refuses, and a frozen set stays frozen.
5. Models trains from random initialisation on the released dataset. A cost
   cap above $10 needs a `training_run_above_threshold` gate, and a cap
   larger than the remaining budget is refused. Register the finished run,
   evaluate it on the frozen set, and read the model next to the baseline.
   Promotion needs a passing report, red-team notes, and an approved
   `model_promotion` gate from someone other than the requester.
6. Ask answers from released datasets only, cites the records, and hides
   records your groups cannot read.
7. Audit lists events and says whether the hash chain is intact. Plan is
   where a program lead submits a goal; task moves offer only the states the
   board allows.

A personal API token, for scripts, is created under the signed-in person's
account page and sent as `Authorization: Bearer iai_...` alongside
`X-Internal-Key` when a key is configured. Revoke it there when the script
is done. Five wrong passwords lock the account for 15 minutes, and a session
ends after 30 minutes idle or 8 hours absolute.

