# Logo replacement — short design

## Intent and why

Replace the generic shield mark with something ownable. The current shield +
circuit-node motif reads as stock "secure tech" and says nothing specific
about internal-ai. The replacement must feel creative and confident without
sliding into generic-AI aesthetics, and it must keep working everywhere the
current mark works (favicon, rail chip, sign-in lockups, light + dark).

## Goals

- A distinctive mark with a product-true story (hash-chained audit log).
- Works at 28px on the pale sage rail chip and at lockup size on sign-in.
- Light and dark variants; palette unchanged so the console theme is untouched.
- Same served filenames, so safety tests and templates keep passing.

## Non-goals

- No rename, no tagline change, no theme token changes.
- No PNG re-exports by hand (builder regenerates if tooling exists, else SVGs
  are the source of truth).

## Anti-goals

- Must not read as another shield, padlock, or glowing AI orb.
- Must not break `test_console_safety.py` brand assertions or the CSP
  (no inline style/script; plain SVG files only).

## Constraints

- `AGENTS.md` brand rules: favicon on every page, icon-on-chip + HTML
  wordmark in the rail (no lockup SVG in the bar), full lockup on sign-in
  with light/dark swap, originals live in `Logo/`, served copies in
  `src/internal_ai/console/static/brand/`.
- Palette: charcoal ink, sage, clay. Contrast rules in
  `tests/test_console_palette.py` still hold (marks are not text, but the
  lockup text colors stay as-is).

## Approaches

- **A · Growth rings**: concentric broken rings; "grown, not downloaded".
  Warm and organic, but reads decorative at small sizes and says little
  about governance. Not picked.
- **B · The chain (picked)**: three interlocked rounded links, one per
  palette color. The audit log is hash-chained; every decision links to the
  one before. Geometric, confident, legible at favicon size. Picked by the
  user from the visual mockup.
- **C · Strata "i"**: layered-data monogram, funkiest, but risks
  reading as a generic letterform at small sizes. Not picked.

## Decided

- Direction B, vertical three-link chain: charcoal, sage, clay top to bottom.
- Filenames unchanged: `internal-ai-icon.svg`, `internal-ai-lockup.svg`,
  `internal-ai-lockup-dark.svg`.
- Dark variants swap the charcoal link for rail ink; sage and clay stay.
- Lockup layout unchanged (mark left, `internal-ai` + `PRIVATE BY DESIGN`).

## Left to the builder

- Final geometry (link proportions, overlap, stroke weight at 28px).
- Exact lockup spacing and PNG exports.
- Port into `Logo/` and `src/internal_ai/console/static/brand/`, run the
  console safety + palette tests, and confirm the rail chip and both
  lockups in a browser.
