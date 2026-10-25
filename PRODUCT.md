# PRODUCT.md

## What this is

Internal-AI is the control plane for a multi-agent engineering organisation that
builds language models from scratch on company data. Platform web.

The console is one client of the platform services, alongside the CLI and the
JSON API. It is mounted at `/console`, server-rendered, and shares the same
governance as every other client.

## Who uses it

A person moving a model from company data through human gates. They register
sources, release datasets, request gates, read eval reports, promote models, and
watch jobs and spend.

A gate is decided by a person other than the one who requested it. Agents can
request and prepare; they cannot approve.

## Stack

Existing FastAPI application, Jinja templates, and a vendored htmx under
`src/internal_ai/console/static/`. No CDN. No inline script and no inline style,
enforced by the console tests and the content security policy.

## Binding rules

The kill switch, the audit log, and the rules in `AGENTS.md` stay binding in
every redesign:

- No pretrained weights. Models train from random initialisation on approved
  internal data only.
- No training on third-party model output.
- No raw records leave for an external API.
- Frozen eval sets are write-protected; a change is a new version.
- Gates are human-only, and the requester cannot decide their own gate.
- `audit_events` is append-only and hash-chained; writes go through
  `AuditLog.record`.
- Spend goes through `BudgetManager` at every scope.
- Releasing the kill switch needs a different person from the one who engaged
  it.

## Brand commitments

- The name is `internal-ai`.
- The supplied marks in `Logo/` are the brand; the three SVGs the app serves are
  copied to `src/internal_ai/console/static/brand/`.
- The palette is the one drawn in those marks: paper `#F7F8F6`, ink `#202A31`,
  dark field `#171D22`, dark ink `#F0F3F1`, sage `#8AA99B` as the accent, and
  `#3F5C4E` for sage text on paper.
- Type names Inter first, then the system UI face. No font is downloaded.
- Appearance follows `prefers-color-scheme`. There is no theme toggle.
- Session and form cookies are `SameSite=Strict`.

## Vocabulary the interface keeps

Task states, gate types, roles, budget scopes, dataset versions and job states
are the domain's words. The console prints them; it does not invent synonyms.
