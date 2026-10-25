# AGENTS.md

Conventions for agents and humans working in this repository.

## Purpose

Internal-AI is the control plane for a multi-agent engineering organisation that
builds from-scratch models on company data. Changes here change how other
agents behave, so the bar is higher than for ordinary application code.
The repository is licensed under the Polychrome License in `LICENSE`. Do not
restore the Apache License.

## Non-negotiables

1. **Never load pretrained weights.** There is no code path that reads
   pretrained parameters, and none may be added. A model is trained from
   random initialisation on approved internal data only.
2. **Never train on third-party model output.** Distillation is a stated
   non-goal. Task data must come from internal documents, tickets and
   resolved questions.
3. **Never send raw records to an external API.** Agents may send schemas,
   aggregate statistics, redacted samples, code and logs. See PRD 6.3.
4. **Never write to a frozen eval set.** Frozen sets are write-protected from
   every agent. A change means a new version and a new human approval.
5. **Never decide a gate as an agent.** Gates are human-only, and the
   requester cannot approve their own gate.
6. **Never write to `audit_events`.** The table is append-only, enforced by
   triggers, and chained by hash. Use `AuditLog.record`.

## Change rules

- Anything an agent can trigger is capability checked in
  `internal_ai/roles.py` and audited in `internal_ai/agent.py` and
  `internal_ai/tools.py`. A new capability is a governance change: add it to the
  role that needs it, nowhere else.
- Task states are enforced in `internal_ai/store.py`. A new state needs a legal
  transition map entry, and the map is the specification.
- Budget caps exist at global, project, agent, job and weekly scope. New spend
  paths must call `BudgetManager.record_spend` or `check`; spending that
  bypasses the manager is a bug.
- Dataset versions are immutable once released. A change means a new version
  with a new content hash and a lineage entry, not an edit.
- Model promotion requires a passing eval report against a frozen set and an
  approved `model_promotion` gate. Both checks live in
  `internal_ai/workflows.py:promote_model`.

## Console

The browser console is a third client of the same services as the CLI and the
JSON API. It is mounted at `/console`. Pages are server-rendered. htmx is
vendored under `src/internal_ai/console/static/`. There is no CDN, no inline
script, and no inline style. How the console is built and which safety checks
it keeps are in `docs/phase-2-console.md`.

Sign-in posts must keep the session cookie and the form token, both
`SameSite=Strict`. A missing `Origin` or `Origin: null` is accepted: some
browsers send `null` for a same-site form, and the cookie is what proves the
post. `Sec-Fetch-Site: cross-site` and a named foreign origin are refused.
Treat a regression here as a safety bug. A person is found by id or by exact
name, including case. The failure text is the same for an unknown account and
a wrong password.

Brand marks:

- Originals, SVG and PNG, stay in `Logo/`. The package does not ship that
  folder. The three SVGs that the app serves are copied to
  `src/internal_ai/console/static/brand/`.
- Favicon on every page: `internal-ai-icon.svg`.
- Signed-in header: the icon at about 28px, then the wordmark in HTML
  (`internal` in ink, `-ai` in sage). Do not put the lockup SVG in the bar.
  Its solid card does not fit there.
- Sign-in: the full lockup. Light and dark are two images. The dark one is
  hidden until `prefers-color-scheme: dark`. The tagline stays inside the
  artwork. No theme toggle and no extra script.

Color comes from those marks. Light paper is `#F7F8F6`, ink `#202A31`. Dark
field is `#171D22`, ink `#F0F3F1`. Sage is the accent: button fill with ink
text. The lockup's lighter sage fails as text on paper, so light-mode text
and links use `#3F5C4E`. Body text stays at or above roughly 4.5:1. Ok,
danger, and pending stay distinct from sage. The gate stripe is a 1px line.
Type names Inter first, then the system UI face. Do not download a font.

The kill switch stays on every signed-in page, with a typed reason. Engaging
is immediate. Releasing needs a different person.

## Style

- Python 3.12, `ruff` for lint and format, `pytest` for tests.
- No comments in code unless the reasoning is not obvious from the code. The
  docstrings explain why, not what.
- Type hints on public functions. Dataclasses for domain objects, `StrEnum`
  for closed sets.
- Determinism matters: seeded randomness, sorted iteration where order is
  visible, and content hashes over the fields that define an artifact.

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Every new capability, gate type or state needs a test that shows it is denied
to the roles that must not hold it, not just a test that shows it works.

## Git

- Every change is committed, and every commit is pushed to `origin/main`.
  Leave neither uncommitted work nor unpushed commits.
- Agent commits use the agent identity so machine work is distinguishable from
  human work in the history:

  ```bash
  git -c user.name='Cline Agent' -c user.email='cline.agent@internal-ai.local' commit -m '...'
  ```

  The repository's default identity stays the human owner's, so human commits
  are never misattributed to the agent.
- A commit leaves `pytest` and `ruff` green (see Tests).

## Review rules

- A builder never reviews its own work. Reviewer pairing is in
  `internal_ai/roles.py`.
- Changes to `gates.py`, `budget.py`, `audit.py`, `policy.py` or `roles.py`
  need a second reviewer because they define the guardrails themselves.
