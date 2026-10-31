# Contributing

Internal-AI is the control plane a company runs on its own machines to build,
evaluate and serve a model it owns. Contributions are welcome. This file is what
a change has to satisfy before it can be merged.

## Who merges

Anyone may open a pull request. Only the owner, GitHub account `taksha17`, writes
to `main` — merges, force-pushes and branch deletions are accepted from that
account alone. You do not need write access to this repository to contribute.

## Setup

Python 3.12 and SQLite 3.35 or newer.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[dev,train]"
```

The `alpha` extra (`safetensors`, `tokenizers`) is only needed for the adapter
track. Without it those tests skip, and CI runs without it on purpose: the suite
that guards the guardrails must stay runnable without a two-gigabyte wheel.

Run the data plane from a git checkout, not an exported copy:
`DataPipeline.release` stamps `git rev-parse HEAD` and refuses a tree whose
commit is unknown.

## Checks

All three must be green before a pull request is ready. CI runs the same three.

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

If a check fails, fix the cause. Do not skip the test, weaken the assertion,
swallow the error, or add a suppression to make it pass.

## Branches and pull requests

1. Branch from the owner's `main`: `change/<slug>` or `fix/<slug>`.
2. One change per branch and one pull request per change. Further work on that
   change is another commit on the same branch, not a second pull request.
3. Open the pull request against `main` and keep pushing to the branch until it
   is merged. Describe why the change exists; the diff says what it does.

```bash
git fetch upstream
git checkout -b change/your-slug upstream/main
git push -u origin change/your-slug
```

### What a change must carry

- **Tests for new behaviour.** A new capability, gate type, task state or
  restricted transition needs two tests, not one: that the roles meant to hold
  it can, and that the roles meant to be denied are **denied**. A restricted
  state also needs a test that the state machine refuses the illegal move.
- **A guardrail change stays in its guardrail file.** `gates.py`, `budget.py`,
  `audit.py`, `policy.py` and `roles.py` define the rules themselves. A change to
  any of them needs a second reviewer.
- **Determinism where it is visible.** Seeded randomness, sorted iteration, and
  content hashes over the fields that define an artifact.
- **No new dependency without a reason.** The platform is installed inside a
  company's boundary; every dependency is something they have to vet.

### Style

Python 3.12. Dataclasses for domain objects, `StrEnum` for closed sets, type
hints on public functions. Docstrings explain why, not what. No comment unless
the reasoning is not obvious from the code.

The console has its own rules, because it is a safety surface: no CDN, no inline
script, no inline style, colours only in the token block at the top of
`console.css`, body text at or above 4.5:1 in both colour schemes. Tests hold
those rules; a reviewer should not have to.

## Documentation

Documentation that ships lives in `docs/`. It describes the platform as built:
architecture, the runbook, the tester guide, the console and its design.

Working notes stay out of git: product requirements, phase plans, design
proposals, status hand-offs, acceptance transcripts and per-change records are
kept on the author's disk and referenced from the pull request. A reviewer reads
the pull request, not a journal. If a change makes a shipped document wrong,
update that document in the same pull request.

## Commits

- Present tense, and say why the change exists. The diff already says what it
  contains.
- Keep the human and machine histories separable. The repository's default git
  identity stays the owner's; agent commits override it explicitly:

```bash
git -c user.name='Cline Agent' -c user.email='cline.agent@internal-ai.local' commit -m '...'
```

## Security

- Never commit a key, a token, a password or a record. Configuration comes from
  the environment.
- `scripts/secret_scan.py` runs on every pull request. A hit fails the build.
- Report a vulnerability privately through the repository's security advisories,
  not in a public issue, and do not open a pull request that demonstrates it
  against real data.
- Hooks on `audit_events` reject `UPDATE` and `DELETE`. Do not add a write path
  that bypasses `AuditLog.record`, and do not add a spend path that bypasses
  `BudgetManager`.

## Licence

The project is source-available under the
[PolyForm Noncommercial License 1.0.0](LICENSE), not an open source licence. By
contributing you agree that your contribution is licensed under the same terms,
and that the copyright holder may license the project, including your
contribution, commercially. A company that wants to use the platform in its
business needs a separate licence; open an issue to start that conversation.

## Contributors

Human and agent work is attributed separately in git, and
[CONTRIBUTORS.md](CONTRIBUTORS.md) says which identity is which. Agents never
review their own work and never decide a gate.

## Getting help

Open an issue for a question, a bug or a proposal. Describe the behaviour you
expected, what happened, and the smallest command that shows it.
