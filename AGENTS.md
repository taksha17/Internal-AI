# AGENTS.md

Conventions for agents and humans working in this repository.

## Purpose

Internal-AI is the control plane for a multi-agent engineering organisation that
builds from-scratch models on company data. Changes here change how other
agents behave, so the bar is higher than for ordinary application code.
The repository is licensed under the PolyForm Noncommercial License 1.0.0
in `LICENSE`. Do not restore the Apache License or a permissive grant. A
company needs a separate license from the owners for commercial use.

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
- Signed-in header: the icon at about 28px on a pale sage chip, then the
  wordmark in HTML (`internal` in rail ink, `-ai` in rail sage). The chip is
  there because the mark's own ink is dark and the rail is charcoal. Do not put
  the lockup SVG in the bar. Its solid card does not fit there.
- Sign-in: the full lockup. Light and dark are two images. The dark one is
  hidden until `prefers-color-scheme: dark`. The tagline stays inside the
  artwork. No theme toggle and no extra script.

Color comes from those marks, warmed. Light paper is `#F4F3ED`, ink `#252B27`,
cards white. The rail is one warm charcoal `#202522` carrying `#F1F4EF` text in
both schemes: only the workspace follows `prefers-color-scheme`, where the field
is `#141917` and ink `#ECF0EC`.

Sage is the accent but not as text in every place. `#3D5A48` carries light-mode
text, links and focus; `#A9C3B0` does it in dark mode and on the rail. The
lighter `#829A87` is a fill only, under `#222B25` ink, because it measures 2.73
as text on paper. The primary button is deep forest `#315B40` with ivory text.
Approve is `#2F6B45`, reject and the kill switch `#9C3A3A`, pending `#F6E7C4`
under `#6B4A0E`; those three never become sage.

Body text stays at or above 4.5:1 and a test holds it:
`tests/test_console_palette.py`. Colours live only in the token block at the top
of `console.css`, so a rule carrying its own hex is a bug, and a new state picks
a tone rather than inventing a green. The gate stripe is a 1px line. Type names
Inter first, then the system UI face. Do not download a font.

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
