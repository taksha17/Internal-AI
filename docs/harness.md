# The harness

`src/internal_ai/harness/` is the layer between a caller and a model. It is not
a scorer (that is `internal_ai/evals/harness.py`) and it does not decide whether
work may proceed (that is a gate). It does four things, and it reports all four.

| Layer | File | What it settles |
| --- | --- | --- |
| Conditioning | `conditioning.py` | which measured prompt is asked, and whether the context fits this model's window |
| Repair | `repair.py` | one bounded retry per rejection, never an open loop |
| Backends | `backends.py` | which engine answers, and the boundary refusal when it would leave the machine |
| Tools | `mcp.py` | which MCP server may offer tools, and to whom |

## An answer is its own record

`HarnessAnswer` carries the text, whether it passed its own check, the model that
answered, the prompt variant, the backend, the attempt count, the prompt digest,
the latency and the cost. `Harness.describe()` returns the harness version, the
model signature, the variant and where its text lives, the backend, the deadline
and the admitted tool server — the lines a report has to print beside a score.

The digest is `sha256` of the prompt the model was asked. The text itself is
never stored: the same reasoning that keeps a raw question out of a trace keeps
it out of an audit event.

## What it refuses, and why the refusal is the answer

- **An unknown variant.** A second prompt is a frozen run, not a setting.
- **A prompt that does not fit.** `PromptTooLong` states how far over it is;
  trimming would answer a different question.
- **A candidate list longer than the one that was measured** (`CANDIDATE_LIMIT`).
- **A document posed as a question** (`MAX_QUESTION_CHARS`), an empty question,
  more history turns than it carries (`MAX_HISTORY_TURNS`), or no room for an
  answer (`max_tokens <= 0`).
- **A cap or the kill switch.** `BudgetRefused` is raised before the call, and a
  denial is written to the audit log.
- **A deadline.** `HarnessDeadlineExceeded` is checked before each generation;
  it bounds the number of calls, not the duration of a hung one.
- **An external backend** for records (`BoundaryRefusal`), and any backend while
  the install is offline.
- **A tool server without an approved gate**, a tool the server does not offer,
  and a role without the capability the server needs.

## Tools the model may ask for

The admitted server's tools are listed in the client context with one sentence
saying how to ask for one: a line reading `TOOL: <name> {json}` and nothing else.
The harness parses that line, checks the gate again, runs the tool, and restarts
the answer with `Tool <name> returned: …` as context. `MAX_TOOL_ROUNDS = 2` is
the whole allowance; a third request is treated as the answer and fails its own
check rather than looping. A tool the server does not offer is answered back as
"there is no tool named …" — a model error is context, not an exception. A gate
withdrawn mid-answer raises `McpDenied`, because a withdrawn approval is not
something to retry. Every round is one `harness.tool` event beside the server's
own `mcp.tool` event.

## Accounting

Given a `BudgetManager`, the harness asks `check` before generating and records
the adapter's actual cost with `record_spend` after each completion, tagged
`harness:<variant>:<model>`. Given an `AuditLog`, it records one `harness.answer`
event per call, with the digest, the latency, the cost and the failure count —
success, failure or denial. Both can also be supplied per call, but the
constructor is the place that makes forgetting impossible.

## Concurrency

One MCP server is one stdio pipe, so `McpToolkit` serialises listing and calling
under a re-entrant lock: two callers cannot take each other's replies. The gate is
re-read inside that lock on every call, so a withdrawn approval stops a toolkit
that is already running.

## Where an operator reads it

`GET /metrics?window_hours=24` on the control API returns one snapshot derived
from the audit log and the spend ledger, not from memory: event volume, denials
and failures, harness answers with pass/exhaust/cost/tool-call counts and p50 and
p95 latency, tool calls, spend limits and the kill-switch state. It is read-only
and key-gated. A window that filled the event cap says `truncated: true` instead
of pretending the count is the whole story.

The console carries the same client at `/console/harness`: a signed-in person
types a question and one `path:line | excerpt` per evidence line, and the page
renders the answer beside its digest, attempts, tool calls, latency and cost,
then the recent `harness.answer` events. It asks with the install's own engine
and the same budget, so the kill switch stops it before a call is paid for, and
an evidence line without its excerpt is refused by its line number rather than
guessed at. `scripts/harness_ask.py` is the terminal client of the same layer,
for a machine with no console open.

An approved gate is withdrawn from `/console/approvals`: open the card, type a
reason, submit. One person is enough to stop what two approved, and the gate
becomes `revoked` in the same transaction that records `gate.revoked`.

## Known limits

- stdio servers only; a remote endpoint would need `NET_EGRESS`, an allowlist and
  a redaction plan.
- The deadline is a wall-clock check between calls. A single adapter call that
  never returns is not interrupted.
- Serving answers through the harness only when `INTERNAL_AI_HARNESS=1`; the
  default path is still `cited_answer`, so an eval report measures what it always
  measured. With the switch on, a served model's window is assumed to be 8192
  tokens unless `INTERNAL_AI_HARNESS_WINDOW` says otherwise.
