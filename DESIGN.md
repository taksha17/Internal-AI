# DESIGN.md

The console look, as built. Every value here is read from
`src/internal_ai/console/static/console.css` and the two templates that carry
the shell, `base.html` and `signin.html`. One stylesheet styles every page; no
template carries an inline style, and the content security policy would refuse
one.

## Shape

The console is one instrument, not a set of pages. A signed-in page is a pinned
rail and one content column.

- The rail is `16.5rem` wide, `position: sticky`, `height: 100vh`, on the one
  charcoal `--rail` with a `--rail-line` hairline on its right. The brand block
  is the shield at `30px` on a pale sage chip and the wordmark in HTML, above a
  hairline. Each destination is a `2.35rem` row: a `1.05rem` stroke icon from
  the sprite, the label, and the approvals count as a sage pill. The active row
  is `--rail-active` with a `--rail-sage` icon, `aria-current="page"`, and the
  same fill on hover so it does not flicker. The person sits at the bottom
  behind a hairline: an initial disc, the name, the roles, and Sign out. The
  rail does not change with the colour scheme; only the workspace does.
- The content column holds the kill strip and then `main`, capped at `74rem`
  and padded `2.75rem` by side.
- Below `900px` the grid collapses to one column. The rail becomes a top band:
  mark on the first line with the person at the top right, then the same links
  in one horizontally scrolling row (`overflow-x: auto`, scrollbar hidden),
  each keeping its icon. No menu, no script, no destination hidden.
- On narrow widths `main` is itself horizontally scrollable, so a wide table
  scrolls inside the page instead of pushing the shell sideways.

Sign-in is the one page without the rail: a `25rem` charcoal brand panel with
the mark, "Private by design." and three product facts, beside an ivory form
panel that holds the lockup and the form at `22.5rem`. Below `900px` the panel
is dropped and the form stands alone. Both lockups stay in the markup and the
scheme still chooses between them.

## Colour

Warm ivory workspace, one charcoal rail, muted sage accent. The token block at
the top of `console.css` is the whole theme: rules read names, never hex, and
`tests/test_console_palette.py` fails the build if one carries a colour of its
own.

| Token | Light | Dark |
| --- | --- | --- |
| `--paper` | `#f4f3ed` | `#141917` |
| `--panel` | `#ecebe2` | `#1f2622` |
| `--card` / `--field` | `#ffffff` | `#1a211d` |
| `--ink` | `#252b27` | `#ecf0ec` |
| `--muted` | `#5c665f` | `#a4aea6` |
| `--line` / `--line-strong` | `#dfe1d9` / `#c6cbbc` | `#2e3631` / `#3d463f` |
| `--sage` | `#3d5a48` | `#a9c3b0` |
| `--sage-fill` / `--on-sage` | `#829a87` / `#222b25` | `#829a87` / `#222b25` |
| `--sage-wash` | `#dce8de` | `#26312b` |
| `--forest` / `--forest-hover` / `--on-forest` | `#315b40` / `#27492f` / `#f4f3ed` | `#3c6b4c` / `#336042` / `#f4f3ed` |
| `--ok` / `--ok-hover` / `--on-ok` | `#2f6b45` / `#27593a` / `#f4f3ed` | `#6ec99a` / `#5fba8d` / `#16201a` |
| `--danger-fill` / `--danger-hover` / `--on-danger` | `#9c3a3a` / `#832e2e` / `#fbeeec` | `#8f3b3b` / `#7a3131` / `#fbeeec` |
| `--tone-ok-bg` / `-ink` | `#d8f0e2` / `#145c38` | `#1c3328` / `#a9dcc0` |
| `--tone-warn-bg` / `-ink` | `#f6e7c4` / `#6b4a0e` | `#3a3220` / `#f3e6c0` |
| `--tone-bad-bg` / `-ink` | `#fbdad6` / `#8a1d16` | `#3a2422` / `#f2bdb6` |
| `--tone-idle-bg` / `-ink` | `#e7e6dc` / `#4e5751` | `#262c28` / `#b6c0b8` |

Rail tokens are declared once, outside the scheme block, because the rail is
the same charcoal in both: `--rail #202522`, `--rail-ink #f1f4ef`,
`--rail-item #cfd8d1`, `--rail-sage #a9c3b0`, `--rail-muted #9ba69d`,
`--rail-line #3a423b`, `--rail-active #2c3a31`, `--rail-hover #262c27`,
`--rail-chip #dce8de`. `--rail-item` is the nav text: `10.67` on the charcoal
and `9.78` on the hover fill, where plain `--rail-ink` would glare.

The card lift is a token too, `--lift`, built from `--shade` with `color-mix`
so a card has an edge without a hex in a rule: `0 1px 2px` at 5% plus
`0 2px 10px` at 4% in light, one `0 1px 2px` at 45% in dark.

Dark values live in one `@media (prefers-color-scheme: dark)` block on `:root`,
with `color-scheme` set in both. There is no toggle.

Sage separates into two jobs. `--sage` is text, links, focus rings, the active
nav mark and the gate stripe. `--sage-fill` is a surface only — progress,
the nav count pill, the done step — always under `--on-sage` ink. The lighter
sage is not used as text anywhere: it measures 2.73 on ivory, 3.03 on white.
The primary button is `--forest` with `--on-forest` ivory text; ok, reject and
the kill bar keep their own reds and greens and never become sage.

States are tones, not colours. `.badge-*`, `.flash-*`, `.refusal` and `tr.warn`
all read `--tone-*-bg` / `--tone-*-ink`, so there is one rule per tone instead of
a light and a dark duplicate, and a state added later has to pick a tone.
Tone borders are `color-mix` of the tone ink, which keeps edges out of the token
block without adding eight more literals.

Measured contrast, light then dark: ink on paper 13.00 / 15.44, muted on paper
5.37 / 7.77, sage on paper 6.86 / 9.41, ink on the sage fill 4.76 in both,
ivory on forest 6.99 / 5.61, ivory on ok 5.71, dark ink on the dark ok 8.35,
tone ink on tone wash 6.68 / 8.81 ok, 6.57 / 10.19 warn, 7.10 / 8.75 bad,
5.97 / 7.62 idle, and the same inks on a card 8.02 / 10.70, 8.04 / 13.20,
9.26 / 9.95, 7.48 / 8.77; muted on the table-header wash 4.98 / 6.76; rail
ink 14.03, rail item 10.67, rail ink on the active fill 10.77, rail sage 8.24,
rail muted 6.18. The muted
grey-green `#747c75` that a warmer palette suggests measures 3.87 on ivory, so
the muted tone is `#5c665f`; it clears 4.5:1 on paper, on `--panel` and on the
sage wash.

## Type

`Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`. No font is
downloaded. Body `0.95rem`, line-height `1.55`.

h1 `1.6rem`/630, h2 `1.05rem`/620, h3 `0.94rem`/620, letter-spacing `-0.022em`,
with more air above than below; a heading at the top of a page, card or callout
loses its top margin. A page's `h1` is followed by its muted subtitle, which
keeps the `0.95rem` size and gains `1.7rem` of air below, so every page starts
the same way whether or not it uses `.page-head`. Prose and lists cap at `68ch`;
tables, grids and step lists do not. Money and counts use
`font-variant-numeric: tabular-nums`, on tables, tiles and the nav count.

## Surfaces

One radius (`--radius: 12px`), one hairline (`--line`) and one padding
(`1.35rem 1.5rem`) are shared by `.card` and `.callout`, with `--lift` under
both. `.flash` and `.refusal` take the radius and their own tint. Controls sit
on a tighter `--control-radius: 9px` so a button inside a card is not as round
as the card.

- `.card-head` is the row a card opens with: title and subtitle left, a badge or
  a `.quiet-link` right, wrapping on narrow widths.
- Header rows sit on the `--panel` wash in `0.7rem` uppercase muted caps; body
  rows separate with a hairline; the last row drops it; row hover is
  `--sage-wash`; `tr.warn` keeps the amber pair through hover. Corner radii are
  repeated on the corner cells so a rounded table actually looks rounded.
- A table inside a card loses its own border and corner radius: the card is the
  box, the table is a pane in it.
- Lists (`.plain`) separate items with a hairline and drop the last one; a
  `.spacer` pushes a row's action to the right.
- `.tabs` are underline tabs across the full width, with any `.button` in the
  row pushed to the right edge.
- Code and `pre` sit on `--panel` with the same hairline and radius.
- `.empty` is a dashed hairline, a muted icon and a short muted line, used
  instead of a bare "nothing here" paragraph.

## Header, tiles and icons

`.page-head` is the page's opening row: an `.eyebrow` in `0.68rem` sage caps
over the `h1`, a muted `.lede` under it, and any primary action right-aligned
and vertically bottom-aligned with the text. `home.html` uses one; every other
page gets the same rhythm from `h1` plus its subtitle.

`.tiles` is a responsive row of at most four `.tile`s. A tile is a card with a
`0.72rem` uppercase muted label, the figure at `1.85rem`/630 in tabular
figures, and a `.tile-foot` line that carries a `5px` dot and a short state:
`is-ok`, `is-warn`, `is-bad` or `is-live`, read from the tone inks. The dot
never carries the meaning alone; the words do.

Icons are one inline sprite per page, `_macros.html:sprite()`, referenced with
`<use href="#i-name">` through `icon()`. There is no icon font, no extra request
and no script; the sprite is `position: absolute` at zero size. Icons are
`1.1rem`, `fill: none`, `stroke: currentColor` at `1.6`, so they inherit the
colour of whatever they sit in, including the rail.

## Controls

`--control: 2.5rem`. Inputs, selects and buttons share that height, the radius,
the border and the focus treatment: a 2px sage outline at `2px` offset for
links and buttons, and a sage border with a 3px `color-mix` sage halo for
fields, which would otherwise show two rings. Control text is `0.92rem` at
line-height `1.4`; labels are `0.86rem`.

A button in a stacked form is content width (`align-self: flex-start`); the
sign-in button stretches. Inline forms keep fields on one line and wrap only
when they must; the gate decision form puts its reason field on its own row
below `860px`, with Approve and Reject together under it. Buttons in the kill
strip are `1.95rem` and `0.8rem`, on a `7px` radius.

Buttons are ink on card, primary is `--on-forest` ivory on `--forest` (with
`--forest-hover` rather than a lighter tint), `ok` is `--on-ok` on `--ok`,
`danger` is `--on-danger` on `--danger-fill`, and `.link` is a sage underlined
button with no fill, which turns `--rail-sage` inside the rail so it stays
readable on charcoal. Disabled controls sit at `0.55` opacity.

## Work in progress

Anything that takes longer than a page load reports itself instead of holding the
browser. Source inspection is the model: the Data page submits the path with
htmx, the scan runs in a worker, and `.inspect-live` sits under the Inspect
button with a 6px `progress` bar on `--line` filled with `--sage-fill`, a
tabular percentage, and the counts (`2,640 of 6,000 files read`). It polls
every 700ms, replaces itself in place, and swaps the finished `#inspect-result`
out of band; the page never navigates, so the tab never shows a spinner. Starting
a scan clears the previous result.

## Kill strip

Every signed-in page carries it under the rail, above `main`, with the same
fields and the same engage/release actions.

Idle it is `2.9rem`: `--panel`, a hairline underneath, muted `0.82rem` copy
behind a sage shield icon, a `1.95rem` reason field that grows to fill, and a
`danger` button.

Engaged it is the loud bar: `--danger-fill` with `--on-danger` text, extra
vertical padding (`93px` on desktop when the copy and form both fit), and its
controls inverted to a `color-mix` ivory outline so they read on the red.

## Home

`home.html` is the reference page. It opens with `.page-head` — eyebrow,
"Welcome, {name}", the lede, and a primary Review button when the inbox is not
empty — then four tiles: active jobs, pending approvals for this person, open
alerts, and steps complete. `active_jobs`, `steps_done` and `step_total` are
counted in the route so the template holds no arithmetic.

The step card carries the same `<ol>` and the same words. A `.meter` above it
is a native `<progress>` at `6px` on `--line` filled with `--sage-fill`, which
needs no inline width and therefore no inline style. CSS draws the path itself:
a `counter` in a `1.7rem` circle with a hairline border and muted number, a
`1px` connector from each circle to the next, no connector after the last, and
a `.done` step that swaps the number for a check (`\2713`) on the sage fill
while its link turns sage at weight 550.

The budget table shows the shared table treatment with right-aligned tabular
money and its own empty row; `.plain` carries recent jobs and alerts; and "No
jobs yet." / "No open alerts." are `.empty` blocks with a muted icon.

## Motion

One transition, `150ms ease`, on colour, background and border for links,
buttons and fields. Nothing animates on load, and
`prefers-reduced-motion: reduce` turns the transition off.

## Not reopened

The marks in `Logo/` and the three served SVGs, sage as accent, the charcoal
rail against an ivory workspace, Inter then the system face with no download,
`prefers-color-scheme` with no toggle, `SameSite=Strict` cookies, a CSP with no
inline script or style, and the governance in `AGENTS.md`.
