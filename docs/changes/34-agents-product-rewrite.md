# 34: Rewrite AGENTS.md and PRODUCT.md

## Why

Both files had grown by accretion. `AGENTS.md` described the console brand in
prose that `PRODUCT.md` also described, in different words and with different
hex values, so a reader could not tell which was current. It also asserted two things
that are no longer true: that every commit is pushed to `origin/main`, which
`CONTRIBUTING.md` restricts to the owner, and colour values that no longer
matched the CSS the console ships. `PRODUCT.md` opened with a line about
"platform web" and spent its first screen on stack detail instead of the
product.

The cost is not cosmetic. These files are the instructions every agent reads
before touching a guardrail. A rule buried in three paragraphs of brand prose,
or contradicted in the neighbouring file, is a rule that gets broken by an agent
that read carefully.

## What changed

- Both files are rewritten from scratch, organised by what a reader needs in
  order: what this is, what must never happen, where each rule lives, how to
  check, how to commit.
- `AGENTS.md` gains a table mapping each guardrail to its file and to what a
  change there needs. It replaces prose that named files inconsistently.
- The palette appears once, in `PRODUCT.md`, with the values the console now
  actually ships. `AGENTS.md` keeps only the two invariants a test can hold:
  hex lives in the token block, and body text stays at or above 4.5:1.
- The git rule now matches `CONTRIBUTING.md`: agents commit on a branch, push
  that branch, open or update one pull request, and never push to `main`.
- Every non-negotiable and every binding rule from the previous versions is
  retained. Nothing was relaxed.
- Removed: the duplicated brand prose, the stale push instruction, and
  references to files that do not exist.

## Verification

- Every path, symbol and command named in either file was checked against the
  tree: `ALLOWED_TRANSITIONS` in `store.py`, `promote_model` in `workflows.py`,
  `reviewer_for` in `roles.py`, `GateType` in `domain.py`, the three SVGs in
  `console/static/brand/`, `tests/test_console_palette.py`.
- `.venv/bin/python -m pytest tests/test_console_palette.py -q` passes, so the
  palette values now documented are the ones the test enforces.
- `.venv/bin/ruff check .` and `ruff format --check .` remain clean.

## Not changed

The rules themselves. The from-scratch and alpha tracks, the gates, the budget
scopes, the audit chain, the console's sign-in invariants and the kill switch
keep exactly the behaviour they had before this rewrite.
