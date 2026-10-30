# PRODUCT.md

## What it is

The platform a company uses to build and own a language model on its own
machines: register sources, release datasets, train, evaluate, gate and serve —
with records and prompts that never leave the building.

The browser console is one client of those services, alongside the CLI and the
JSON API. It is mounted at `/console` and shares their governance.

## Who uses it

A person moving a model from company data to production: registering sources,
releasing datasets, requesting gates, reading eval reports, promoting models,
watching jobs and spend. They need no AI background, and they should always be
able to see what the platform is waiting on and who can unblock it.

Agents request and prepare. They never approve: a gate is decided by a person,
and never by the person who asked for it.

## Stack

One FastAPI process serving Jinja templates and the JSON API, with htmx
vendored under `src/internal_ai/console/static/`. No CDN, no Node build, no
frontend framework, no inline script and no inline style — enforced by tests
and by the content security policy.

## Binding rules

`AGENTS.md` holds the full set. These stay true through every redesign:

- The default install trains from random initialisation on approved internal
  data only, and no path reads pretrained parameters outside
  `internal_ai/training/alpha/`.
- No training on third-party model output.
- No raw records leave for an external API.
- Frozen eval sets are write-protected; a change is a new version.
- Gates are human-only, and the requester cannot decide their own.
- `audit_events` is append-only and hash-chained; writes go through
  `AuditLog.record`.
- Spend goes through `BudgetManager` at every scope.
- Engaging the kill switch is immediate; releasing it needs a different person.

## Brand

- The name is `internal-ai`. The marks in `Logo/` are the brand; the three SVGs
  the app serves are copied to `console/static/brand/`. The favicon is
  `internal-ai-icon.svg`. The signed-in rail carries the icon with the wordmark
  set in HTML, never the full lockup; sign-in carries the lockup, light and
  dark, in two images.
- Colour is three notes around the marks. Paper `#F2F7F5`, ink `#0F1A1E`,
  cards white; rail `#0B1220` with `#EEF3F2` text in both schemes; dark field
  `#080E15`, dark ink `#E8F1EE`.
- Aqua Jade is the accent. `#0E6B5C` carries light-mode text, `#6FE6CB` dark
  mode, `#5FE0C5` on the rail; `#2FB89E` is a fill only, under `#06231D` ink.
  The primary button is `#0D5C50` with ivory text. Approve `#17694E`, reject and
  the kill switch `#8E2F3C`, pending `#F6E7C4` under `#6B4A0E`; those three are
  never jade.
- Baby pink `#F3C6D3` is an accent, never a surface: selection, the gate
  stripe, the kicker, a faint aura at the foot of the rail.
- Type names Inter first, then the system UI face. No font is downloaded.
  Appearance follows `prefers-color-scheme`; there is no theme toggle.
- Session and form cookies are `SameSite=Strict`.

## Vocabulary

Task states, gate types, roles, budget scopes, dataset versions and job states
are the domain's words. The interface prints them and never invents synonyms for
them.
