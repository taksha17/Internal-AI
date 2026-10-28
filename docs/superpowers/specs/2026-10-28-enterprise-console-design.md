# Enterprise console upgrade

The console looks good and reads clearly. What it does not yet feel like is a
tool a company runs its AI programme on. This is the audit, the direction, and
the order of work. Nothing here changes a capability, a gate, a task state or
what the guardrails allow.

## What "enterprise-grade" has to mean here

Five tests a buyer or an operator applies in the first ten minutes, in the order
they apply them:

1. **Can I find the thing I came for?** Not "is there a nav item called
   Datasets" — can I go from a name, an id or a person to the artifact, and can
   I narrow a list to the six rows I care about.
2. **Does it tell me when and who?** Every row: when it happened, who did it,
   in a form I can read at a glance and cite exactly when I need to.
3. **Is it dense enough to work in?** An operator scanning 200 audit events, or
   a lead checking six gates, should see them without scrolling theatre.
4. **Does it refuse to lie?** No invented numbers, no chart without a source, no
   "success" that hides a degraded result, no dead end where the next step is
   implied but not offered.
5. **Can I hand it to an auditor?** Evidence I can extract: who approved what,
   against which hash, when, and a way to take it out of the tool.

Today the console scores well on 4 and 5 in substance and badly on the surface
of all five.

## What it does today

Evidence, not impressions.

**Nothing can be found.** Twenty-three routes, reachable only through the rail.
There is no search route, no filter except Audit's actor/action
(`app.py:599`), no sort anywhere, and no paging: Jobs lists `limit=100`
(`app.py:1417`), Audit `limit=100` (`app.py:603`), and Datasets, Models, People
and Evals render every row. Ten templates draw a table; three wrap it in
`.table-wrap`, so the rest overflow on a narrow window (papered over by
`main { overflow-x: auto }` at the 900px breakpoint).

**Time and identity are raw.** `created_at` is ISO-8601 with seconds
(`ids.py:18`) and templates print it directly — `2026-10-28T09:14:03+00:00`
in a job row. Nothing is relative, nothing is a `<time>` element, and nothing
gives the exact value on hover. Jobs say who started them; Datasets, Models and
Eval sets do not say who released or trained them in the list.

**The shell is built for a landing page, not a desk.** `h1` is
`clamp(2rem, 3.4vw, 2.9rem)`; every page carries an eyebrow and a lede
paragraph; body gradient is two radial washes; controls are 2.65rem tall and
rows are 0.72rem padded. Two header shapes coexist: Home, Jobs and Approvals use
`.page-head`, while Datasets, Evals, Models, People, Budget, Audit and Plan open
with a bare `<h1>` and a `<p class="muted">`.

**Two things overstate themselves.** Home shows four sparklines whose polylines
are hardcoded in the template (`home.html`) — decorative data drawn over real
figures, which is exactly the kind of thing an evaluating buyer notices. And the
kill switch's typed-reason input sits in the top bar of every page, where it
reads as a search field that does nothing.

**Details go nowhere.** A 404 inside the console returns FastAPI's JSON
(`app.py:1103`, `:1424`), not a page. Breadcrumbs are `Console / Crumb` with no
link back up. Job rows link to a job page whose whole content is one status
block. Badges, buttons and tables are consistent; there is no `.toolbar`, no
subsection nav, no `aria-live` on the polled job status, no skip link, and no
`scope="col"` on any header cell.

## The direction

**Operator console: make the work findable, dense and honest.** Keep the brand,
the palette and the tokens. Spend the effort on utility — search, filters, sort,
timestamps, real data, an inbox, export — and let the shell get quieter so the
data is the loudest thing on the page.

I recommend this one because the product's differentiator is that it is
*governed*: the artifacts already carry the evidence an operator or auditor
needs (gate decisions with reasons, approval actors, hashes, spend). What is
missing is the interface that surfaces it fast. Effort spent there pays off in
the first five minutes of any evaluation; effort spent on visual novelty does
not.

Rejected, for the record:

- **Audit-ready dossier first** — provenance blocks, printable gate records,
  retention and classification surfaced, chain verification as a panel. Real
  value, and Phase 4 below, but on its own it polishes pages few people open
  daily while search and filters stay absent. Better as the second half.
- **Command-driven console** — a command palette, keyboard-first actions, saved
  views. It reads as "modern" but it needs a vendored client script alongside
  htmx, and it would sit in front of lists that cannot yet be filtered or
  sorted. Deferred until Phase 2 has given the palette something to search.

## Phase 1 — the shell and the components (no new features)

Stop every page hand-rolling a header, a table and a timestamp.

- Density tokens in the one token block: body 0.9375rem, control height 2.25rem
  (restored to 2.65rem under 900px so touch targets stay compliant), card
  padding 1.1rem, row padding 0.45rem 0.75rem. Flat paper: the two body
  gradients go, the sage accent stays.
- One page header in `_macros.html`: eyebrow (optional), `h1` at 1.4rem,
  subtitle, actions on the right. Every template uses it; the bare-`h1` variant
  disappears. Breadcrumbs become real links on detail pages.
- One table macro: wrapper, `<th scope="col">`, right-aligned numerics, an
  actions cell, and a `m.when()` macro rendering
  `<time datetime="…" title="exact">2 min ago</time>`.
- One empty-state macro with a title, a hint and the next action, replacing both
  the `colspan` "No jobs yet." rows and the one-off `.empty` div.
- Kill switch: top bar keeps a compact armed/ready chip and the release or
  engage form opens from it, still on every signed-in page, still requiring a
  typed reason, still immediate.
- Console-level 404 and 403 pages instead of JSON, with a crumb back to Home.
- Accessibility pass: skip-to-content link, `<main id="main">`,
  `aria-live="polite"` on the polled job status, `aria-sort` on headers that
  will sort, focus-visible kept everywhere.
- Home loses the sparklines. Where real history exists (audit events per day,
  jobs by state) show it; where it does not, show the figure alone.

**Done when:** every signed-in page renders through the same header and table
macros, no template draws a table outside the wrapper, `preview_snapshot` of
Home, Jobs and Audit shows no fake series, and the palette, safety and e2e
tests pass unchanged.

## Phase 2 — finding things

- `/console/search?q=` — one route over sources, datasets, eval sets, models,
  jobs, gates, people and audit subjects, grouped by kind. Results are filtered
  through the same access rules the pages use, so search can never surface an
  artifact the actor cannot open. This is the single largest jump in "enterprise
  feel".
- Filters, sort and paging on Audit, Jobs, Datasets, Models, Eval sets, Sources
  and People: GET forms (no script), sortable headers, a page-size control, and
  the count of what is being shown out of what exists.
- An Inbox page: gates you can decide, alerts, failed jobs, and steps blocked on
  someone else. The rail's Approvals count becomes the Inbox count.

**Done when:** a dataset id typed into search lands on its page; Audit filtered
to one actor and sorted by recency shows both, and the URL is shareable; a
person with only `data_owner` searching for a model id gets the refusal, not the
row.

## Phase 3 — operating it

- Job page depth: attempts, log tail, cost against cap, the gate that allowed
  it, and a live region that says "updated 4s ago" instead of silently
  swapping.
- Dataset page depth: split sizes, record counts per split, history, and the
  source it came from with its gate.
- Model page depth: evaluation history, lineage as a list (dataset → tokenizer →
  run → model → promotion), and the frozen set it was scored against.
- CSV export of Audit, Datasets and Jobs, server-side, audited like any other
  read of the same rows.

**Done when:** a job can be diagnosed from its own page without the audit log,
and an export of the audit log verifies against the chain.

## Phase 4 — evidence

Provenance dossier on dataset, model and eval-set pages: who approved what and
when, content hashes, retention and classification, and a printable gate record.
Chain verification as a visible panel, not one line of prose.

## Guards to add

- No `<table`, `<th` or `created_at` printed outside the macros: a template lint
  test, in the spirit of `test_console_palette.py`.
- Search never returns an artifact the actor cannot read — a denial test, as the
  repo requires for anything new.
- Empty states must offer an action; locked steps keep naming the role.
- The kill switch is present, with a required reason, on every signed-in route.

## Constraints

Tokens live only in the top block of `console.css`, no rule carries its own
hex, no inline script or style, no CDN, no downloaded font, no new capability,
gate type or task state. The rail keeps one charcoal in both schemes. Contrast
pairs stay above 4.5:1 in both.

## Open

- Density default: operator-dense as above, or keep today's airiness and make
  compact a choice?
- Whether the Phase 2 search lives in the top bar on every page or on its own
  route linked from it.
