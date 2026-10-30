## What changed

The console now carries the three-note palette the owner asked for: Aqua Jade,
Midnight Abyss and a faint Baby pink.

- Light paper is a cool jade-white `#F2F7F5`, ink `#0F1A1E`, cards white. The
  panel is a pale jade `#E4EFEA`, and the lines are jade-grey rather than warm.
- The rail is Midnight Abyss `#0B1220` in both schemes, with `#EEF3F2` ink,
  `#5FE0C5` for its accent, and the same navy family for active and hover rows.
- Dark mode keeps the rail and puts the workspace on `#080E15` with `#E8F1EE`
  ink, cards `#111A27` and a jade `#6FE6CB` accent.
- Aqua Jade replaces sage everywhere it carried meaning: links, focus, selected
  tabs, checkboxes, the meter, counts and avatars. `#2FB89E` stays a fill only,
  under `#06231D` ink.
- The primary button is abyss-jade `#0D5C50`; approve is `#17694E`, reject and
  the kill switch `#8E2F3C`, pending keeps the amber cream.
- Baby pink is an essence, never a surface. `--blush` `#F3C6D3` draws the 1px
  gate stripe; `--blush-wash` is the text selection; `--blush-text` (rose on
  paper, pink on midnight) is the page kicker; and a 10% pink radial sits at the
  foot of the rail and the sign-in panel. `--blush-ink` covers text placed on a
  pink fill.
- The amber clay stays for warnings and the figures that pop; pink never means
  "caution", so approve/reject/pending keep their own tones.

The marks are unchanged. The only mark edit is the background plate of the two
lockups the app serves: `#F7F8F6` and `#171D22` become the new paper, because
against the new field the old plates showed as a faint rectangle behind the
wordmark. `Logo/` still holds the untouched originals.

`AGENTS.md` and `PRODUCT.md` record the new palette, and
`tests/test_console_palette.py` gains the blush pairs, including text on the
pink wash.

## Proof

- `.venv/bin/python -m pytest` → 454 passed.
- `.venv/bin/ruff check .` and `ruff format --check .` clean.
- Every declared pair clears 4.5:1 in both schemes; the tightest is reject text
  on `#A83E4A` at 5.37. `test_no_rule_carries_a_colour_of_its_own` still holds,
  so the theme remains the token block.
- Captured at 1440x900 in light and dark, and at 390x844 in light, across the
  overview, pipelines, datasets, training, evaluations, approvals (with an open
  gate), audit, inbox, data, models, brief, ask, people, budget, policy and
  sign-in. The gate stripe, the kicker and the selection read as pink; nothing
  else does.

## Out of scope

- The marks' own sage and clay strokes are retained; recolouring them would
  diverge from the originals in `Logo/` and was not requested.
- The mobile rail still scrolls horizontally through its destinations; that is
  a layout question, not a colour one, and predates this change.
