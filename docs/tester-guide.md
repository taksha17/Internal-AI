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
`{"ok": true, "checked": 77}` on the pilot state. `/tasks` is empty because
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
tasks.

Request a gate, then try the forbidden approvals:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test gates request \
  --gate-type model_promotion --title 'Test gate' --subject tsk_promote --by ml_lead
```

- The requester approving their own gate is refused:
  `error: dual control violation: ml_lead requested gate ... and cannot approve it`.
- An approver whose roles do not cover the required ones leaves the gate
  pending: the CLI prints `gate ... is now pending` and the gate does not pass.
- A gate only passes once the union of approver roles covers every required
  approver.

Other refusals worth checking:

```bash
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test budget kill
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test budget kill --release
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test budget status
.venv/bin/python -m internal_ai.cli --root /tmp/guide_test audit verify
```

With the kill switch engaged, spend is denied first and the reason names the
kill switch. `--release` restores the allowance. `budget set --scope global
--key global --limit N` sets a cap; spending exactly the cap is allowed and
going over it is denied.

## 5. Trust model, stated plainly

The control-plane key is a trusted operator credential. It is one shared
secret, not a per-human identity, and the approver name in a gate decision is
asserted by the caller rather than authenticated. The platform still enforces
the rules that matter inside that trust boundary: agents cannot decide gates
through the agent paths, the requester cannot approve their own gate, and a
gate only passes when the required approver roles are covered. Do not expose
port 8080 beyond the host, and treat the key as a personal credential.

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

