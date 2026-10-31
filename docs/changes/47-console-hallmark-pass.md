# 47 — The console, checked by the rules that were written down

A Hallmark polish pass over `/console`: the design the project already had,
read against a floor of rules, and the defects that floor found.

## What was wrong

The console looked finished and was not. The stylesheet carried two elements
fighting for the same edge of the screen: the rail and the topbar were both
`position: sticky` at `top: 0`, so on a long page one of them had to lose. A
field's border was `--line` on `--paper`, 1.26:1, which is a line you can only
infer. The sign-in brand panel drew its focus ring in the workspace sage on a
midnight ground, 2.78:1. The skip link animated `top`. The kill switch panel
was anchored to a chip that sits at the left of a wrapped bar, so opening it on
a phone put 195px of it off the screen, hidden only by the page's clip. A
suggestion carried a 3px side stripe for a state it already said in words.

The copy was slop in the same way. Every page opened with an eyebrow, a line of
small capitals that said what the breadcrumb had already said. Forty-four
middle dots joined facts that a comma or a colon joins better. Seven dashes
did the work a colon or a full stop does. A pager said `1–50 of 120`. A
progress row said `recall@10 0.412 of 0.70 — 0.288 short`. The word
"unlocks" appeared where the sentence could have said what it does.

## What changed

**The shell.** The rail is pinned (`position: fixed`) with the column offset by
`18rem`, so the topbar is the only element that owns the top edge. The page
carries `overflow-x: clip` on `html` and `body`, and a wide table scrolls
inside its own region. The skip link moves by `transform`. A field's edge takes
`--muted`, which clears 3:1 on paper, on a card and on the panel. The sign-in
panel's focus ring takes the rail's jade, because the workspace sage cannot
clear 3:1 on midnight. The suggestion's side stripe is gone, and the switch
panel is anchored to the bar on a phone, with `left` and `right` at `0.75rem`
and `width: auto`, so it fits whatever the chip's own width is. A closed
switch leaves no panel in the layout at all.

**The copy.** The eyebrow is gone from all twenty-three pages that opened with
one, and from the `page_head` macro that drew it. Meta lines are separated by
commas and colons. The pager says "1 to 50 of 120". A progress row says
`recall@10 0.412 of 0.70, 0.288 short`. Two middle dots remain, in the
`<title>` and in the live-state pill, where a dot separates two short names
rather than two facts.

**The record.** `DESIGN.md` described a warm paper, a sticky rail and an
eyebrow that the stylesheet no longer had. It is rewritten against the sheet
as it stands, with the measured contrast for every pair the palette test
checks.

**The rules.** A rule that lives in someone's head comes back. The shell's
shape is now in `tests/test_console_hallmark.py`, beside the palette test, the
markup test and the phone-band test: one sticky top, a clipped page, a field
edge that can be seen, a code face that is a token, four type weights, no
eyebrow, no middle-dot meta line, a phone panel that fits the screen.

## What was deliberately not changed

The state badge stays inside the page head. Hallmark gate 38iii notes an
accented word in a heading, and a badge beside the title is one; but the badge
names the state of the thing the heading names, it is the same mark on every
page that has one, and moving it below the title would separate the state from
its subject. It is recorded rather than rebuilt.

The base model, the gates, the audit chain and every other guardrail in
`AGENTS.md` are untouched. This pass changed how the console looks and how it
reads, and nothing about what it is allowed to do.

## Tests

`tests/test_console_hallmark.py` — the stamp is on the first line of the
sheet, the page never scrolls sideways, the skip link moves without animating
`top`, a code face is a token, a display head can wrap, a control answers a
press, a field edge is drawn in `--muted`, only the topbar sticks to the top,
the phone kill panel fits the screen, no page opens with an eyebrow, no meta
line is joined with a middle dot, and the type scale has not drifted past four
weights.

`tests/test_console_find.py` and `tests/test_project_brief.py` follow the copy:
the pager and the progress rows are asserted in the words the console now
prints.

The Hallmark sweep (`sloplint`, `--scope app`, static and render tiers) reads
**0 FAIL, 0 WARN** on the stylesheet and on every rendered page: sign-in,
overview, brief, datasets, evaluations, approvals, inbox, jobs, models, plan,
search, sources, failed answers, account, budget, policy and the error page.
Two findings stand, both noted rather than rebuilt: the badge in the head
(38iii) and the dark surface that inherits `--ink` (41), which the palette
test measures in both schemes.
