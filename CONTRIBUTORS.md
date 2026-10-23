# Contributors

Internal-AI is built by humans and agents together. The git history is the
record of who changed what, so machine work and human work are attributed
separately.

| Contributor | Identity in git | Role |
|---|---|---|
| Taksha | `taksha17 <taksha.27@gmail.com>` | Product owner; program lead, ML lead, data owner and gate approver. Human-only roles: every gate decision is taken through the platform's gated flow, which is recorded in the audit chain, not in git. |
| Cline (AI coding agent) | `Cline Agent <cline.agent@internal-ai.local>` | Builder for application code, tests, docs and pilot work. Commits are attributed to the agent identity so agent contributions are visible and reviewable on their own. |

## Agent commit convention

The repository's default git identity stays the human owner's, so a human
commit is never misattributed to the agent. Agent commits override the
identity explicitly:

```bash
git -c user.name='Cline Agent' -c user.email='cline.agent@internal-ai.local' commit -m '...'
```

`cline.agent@internal-ai.local` is not linked to a GitHub account, so agent
commits appear in the repository history with a plain author name. Linking
that address to a GitHub account, or giving the agent its own account, gives
the commits a verified contributor entry.

## Rules

- Every change is committed, and every commit is pushed to `origin/main`.
- A commit leaves `pytest` and `ruff` green; the test commands are in
  `AGENTS.md`.
- An agent never reviews its own work and never approves a gate. Reviews and
  gate decisions are human acts, recorded in the platform's audit chain.
