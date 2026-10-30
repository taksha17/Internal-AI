## What was wrong

An audit of the running console against the product's own narrative found five
seams:

- The breadcrumb in the topbar said "Overview" on the data-heavy pages. The
  `crumb` block in `base.html` defaults to Overview, and eight templates never
  overrode it, so Evaluations, Models, Ask, Pipelines, Budget, People, Policy
  and Request a gate all claimed to be the overview.
- The overview printed its two counters as `%02d`, so an empty queue read `00`
  rather than `0`.
- Labels were set in tiny tracked capitals: the page kicker (0.68rem, 0.16em,
  uppercase), stat labels (0.72rem, 0.12em, uppercase) and table heads
  (0.68rem, 0.05em, uppercase). The rest of the interface is sentence case and
  speaks plainly, so the shouty micro-type read as a different product.
- The Models page alternated a full-width card, two half-width `stack` cards
  and another full-width card, so the page's right edge was ragged and its
  headline idea looked the least finished screen in the product.
- A `.callout` inside a `.card` (the project brief, and the service and
  regression notes on a model) drew a filled, bordered box inside another box.

## Change

- The eight templates name their own crumb, matching the rail's group
  (Evaluations, Pipelines, Models, Ask, Budget, People, Policy, Request a gate).
- The overview prints the counters bare.
- The kicker, stat labels and table heads use the body face one step smaller,
  weight 600, no tracking and no case transform, in the same muted and clay
  tones. Colour, size and weight carry the hierarchy instead of capitals.
- The train-model and train-classifier forms use a new `.field-grid`: an
  auto-fit grid of 15rem columns inside one full-width card, so the whole
  Models page shares one card width and the fields read in columns on a wide
  screen. The `stack` max-width no longer applies to these two forms.
- `.callout` becomes a left-rule note (2px sage rule, no fill, no radius, no
  shadow), so it is legible inside a card and on the page without nesting one
  surface in another.

No token changes; the palette test still passes and no rule gained a hex. The
edits are confined to `console.css` and the templates; no Python behaviour
changed.

## Proof

- `.venv/bin/python -m pytest` → 454 passed.
- `.venv/bin/ruff check .` and `ruff format --check .` clean.
- The console was run against a real root and every page captured at 1440x900
  in both schemes: the breadcrumb matches the rail on each page, the counters
  read `0`, the labels are sentence case, and the Models page's cards share one
  width. A fixture with the real stylesheet confirmed the field grid uses four
  columns at 1440 and stays legible when the grid falls back to one.

## Out of scope

- A fresh root is still empty by design; an evaluator who wants the populated
  console runs the documented demo and sample path first.
- The rail's group label ("Training") and the page's title ("Jobs") remain
  different words, as they were; the crumb follows the rail.
