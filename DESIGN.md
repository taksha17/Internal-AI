# DESIGN.md

The console look, as built. Every value here is read from
`src/internal_ai/console/static/console.css` and the two templates that carry
the shell, `base.html` and `signin.html`. One stylesheet styles every page; no
template carries an inline style, and the content security policy would refuse
one.

## Shape

The console is one instrument, not a set of pages. A signed-in page is a pinned
rail and one content column.- The rail is `18rem` wide, `position: sticky`, `height: 100vh`, on the one
  charcoal `--rail` with a `--rail-line` hairline on its right and a faint
  `--rail-active` glow at the top. The brand block is the shield at about
  `30px` on a pale sage chip, the wordmark in HTML, and a muted
  `PRIVATE BY DESIGN` micro-cap, above a hairline. Destinations are one flat
  list with no group headings: Overview, Pipelines, Datasets, Training,
  Evaluations, Approvals, Audit log, a hairline, then Inbox, Data, Models,
  Brief, Ask and Admin. Each row is `2.45rem`: a `1.05rem` stroke icon from the
  sprite, the label, and the approvals count as a sage pill. The active row is
  `--rail-active` with a `--rail-sage` icon, `aria-current="page"`, and the
  same fill on hover so it does not flicker. The bottom holds the person: an
  initial disc, the name, the roles, and Sign out. The rail does not change with the
  colour scheme; only the workspace does.
- The content column holds a sticky topbar and then `main`, capped at `76rem`
  and padded `2rem` by side. The topbar blurs the paper behind it and carries
  a breadcrumb, a `Local · no external calls` scope pill, and the kill switch
  as a pill: status label, a reason field that grows to fill, and
  Engage/Release. Idle it is card on hairline; engaged it is `--danger-fill`
  with `--on-danger` text.
- Below `900px` the grid collapses to one column. The rail becomes a top band:
  mark, then the grouped links in one horizontally scrolling row
  (`overflow-x: auto`, scrollbar hidden), each keeping its icon, then the
  person. No menu, no script, no destination hidden. The readiness note hides;
  the topbar wraps.
- On narrow widths `main` is itself horizontally scrollable, so a wide table
  scrolls inside the page instead of pushing the shell sideways.

Sign-in is the one page without the rail: a charcoal brand panel with the
mark, "Private by design.", the product facts and three audited guarantees,
beside an ivory form panel that holds the theme-aware lockup and the form at
`22.5rem`. Below `900px` the panel is dropped and the form stands alone. Both
lockups stay in the markup and the scheme still chooses between them.

## Colour

Warm workspace: paper `#F4F3ED` and warm-white cards under ink `#252B27`, sage
text `#3D5A48`, a muted sage fill `#829A87` for surfaces under `#222B25` ink,
and one warm charcoal rail `#202522` in both schemes. Clay `#96602B` keeps the
workspace from reading all-sage. The token block at the top of `console.css` is
the whole theme: rules read names, never hex, and
`tests/test_console_palette.py` fails the build if one carries a colour of its
own.

| Token | Light | Dark |
| --- | --- | --- |
| `--paper` | `#F4F3ED` | `#141917` |
| `--panel` | `#E9E7DD` | `#1f2622` |
| `--card` / `--field` | `#ffffff` | `#1a211d` |
| `--ink` | `#252B27` | `#ecf0ec` |
| `--muted` | `#5C6157` | `#a4aea6` |
| `--line` / `--line-strong` | `#DCDAD0` / `#C4C2B4` | `#2e3631` / `#3d463f` |
| `--sage` | `#3D5A48` | `#a9c3b0` |
| `--sage-fill` / `--on-sage` | `#829A87` / `#222B25` | `#829A87` / `#222B25` |
| `--sage-wash` | `#DFE6DC` | `#26312b` |
| `--forest` / `--forest-hover` / `--on-forest` | `#315B40` / `#27492F` / `#F4F3ED` | `#3c6b4c` / `#336042` / `#F4F3ED` |
| `--ok` / `--ok-hover` / `--on-ok` | `#2f6b45` / `#27593a` / `#f4f3ed` | `#6ec99a` / `#5fba8d` / `#16201a` |
| `--danger-fill` / `--danger-hover` / `--on-danger` | `#9c3a3a` / `#832e2e` / `#fbeeec` | `#8f3b3b` / `#7a3131` / `#fbeeec` |
| `--tone-ok-bg` / `-ink` | `#d8f0e2` / `#145c38` | `#1c3328` / `#a9dcc0` |
| `--tone-warn-bg` / `-ink` | `#f6e7c4` / `#6b4a0e` | `#3a3220` / `#f3e6c0` |
| `--tone-bad-bg` / `-ink` | `#fbdad6` / `#8a1d16` | `#3a2422` / `#f2bdb6` |
| `--tone-idle-bg` / `-ink` | `#E8E9E2` / `#4e5751` | `#262c28` / `#b6c0b8` |

Rail tokens are declared once, outside the scheme block, because the rail is
the same warm charcoal in both: `--rail #202522`, `--rail-ink #F1F4EF`,
`--rail-item #D2D8D1`, `--rail-sage #A9C3B0`, `--rail-muted #9AA39A`,
`--rail-line #333A34`, `--rail-active #2C332D`, `--rail-hover #272E28`,
`--rail-chip #DCE4DA`. `--rail-item` is the nav text: `10.74` on the charcoal
and `9.61` on the hover fill, where plain `--rail-ink` would glare.

The card lift is a token too, `--lift`, built from `--shade` with `color-mix`
so a card has an edge without a hex in a rule: `0 1px 2px` at 5% plus
`0 2px 10px` at 4% in light, one `0 1px 2px` at 45% in dark.

Dark values live in one `@media (prefers-color-scheme: dark)` block on `:root`,
with `color-scheme` set in both. There is no toggle.

Sage separates into two jobs. `--sage` is text, links, focus rings, the active
nav mark and the gate stripe. `--sage-fill` is a surface only — progress,
the nav count pill, the done step — always under `--on-sage` ink.The lighter sage is not used as text anywhere: it measures 2.73 on paper, 3.03 on white.
The primary button is `--forest` with `--on-forest` ivory text; ok, reject and
the kill control keep their own reds and greens and never become sage.

A second accent, clay, keeps the workspace from reading all-teal. `--clay`
(`#9a5b23` light, `#e0a35c` dark) carries eyebrows, the headline highlight,
alternate stat icons and sparklines; `--clay-strong` inks text on it and
`--clay-wash` washes behind it. Clay never marks governance: approvals,
rejections, pending states, focus and the active nav stay on their own
colours.

States are tones, not colours. `.badge-*`, `.flash-*`, `.refusal` and `tr.warn`
all read `--tone-*-bg` / `--tone-*-ink`, so there is one rule per tone instead of
a light and a dark duplicate, and a state added later has to pick a tone.
Tone borders are `color-mix` of the tone ink, which keeps edges out of the token
block without adding eight more literals.

Measured contrast, light then dark: ink on paper 13.00 / 15.44, muted on paper
5.72 / 7.77, sage on paper 6.86 / 9.41, ink on the sage fill 4.81 in both,
ivory on forest 6.99 / 5.56, ivory on ok 5.71, dark ink on the dark ok 8.35,
tone ink on tone wash 6.68 / 8.81 ok, 6.57 / 10.19 warn, 7.10 / 8.75 bad,
6.09 / 7.62 idle, and the same inks on a card 8.02 / 10.70, 8.04 / 13.20,
9.26 / 9.95, 7.65 / 8.77; muted on the table-header wash 5.13 / 6.76; rail
ink 14.03, rail item 10.74, rail ink on the active fill 11.69, rail sage 8.24,
rail muted 5.99. The muted
grey-green `#747c75` that a warmer palette suggests measures 3.87 on paper, so
the muted tone is `#5c6157`; it clears 4.5:1 on paper, on `--panel` and on the
sage wash.

## Type

`Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`. No font is
downloaded. Body `0.96rem`, line-height `1.58`, over a faint `--sage-wash`
radial glow plus a fainter `--clay-wash` glow at the top of the paper, so the
all-sage field picks up the logo's clay second voice without a new paint.

h1 `1.6rem`/700, h2 `1.05rem`/620, h3 `0.94rem`/620, letter-spacing `-0.028em`,
with the overview headline scaling to `2.9rem` under `.page-head`. A `.pop`
word inside a headline renders in clay italics.
with more air above than below; a heading at the top of a page, card or callout
loses its top margin. A page's `h1` is followed by its muted subtitle, which
keeps the `0.95rem` size and gains `1.7rem` of air below, so every page starts
the same way whether or not it uses `.page-head`. Prose and lists cap at `68ch`;
tables, grids and step lists do not. Money and counts use
`font-variant-numeric: tabular-nums`, on tables, tiles and the nav count.

## Surfaces

One radius (`--radius: 14px`), one hairline (`--line`) and one padding
(`1.35rem 1.4rem`) are shared by `.card` and `.callout`, with a two-layer
`--lift` under both. `.flash` and `.refusal` take the radius and their own
tint. Controls sit on a tighter `--control-radius: 10px` so a button inside a
card is not as round as the card. Tables that need their own scroll region sit
in `.table-wrap`, which carries the border so the table inside stays flush.

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

`.page-head` is the page's opening row: an `.eyebrow` in `0.7rem` clay caps
over the `h1`, a muted `.lede` under it, and any primary action right-aligned
and vertically bottom-aligned with the text. Every page uses one.

`.stat-grid` is a responsive row of `.stat`s, and the overview uses two of them
under the `.two` modifier. A stat is a card with a `0.72rem` uppercase muted
label, the figure at `1.8rem`/700 in tabular figures — zero-padded to two
digits where it counts runs — and a `.stat-meta` line whose `.stat-state`
carries a small tone-coloured dot beside the words: ok for work in progress,
warn for work waiting on a person, idle when there is none. The dot never
carries the meaning alone; the words do.

Icons are one inline sprite per page, `_macros.html:sprite()`, referenced with
`<use href="#i-name">` through `icon()`. There is no icon font, no extra request
and no script; the sprite is `position: absolute` at zero size. Icons are
`1.1rem`, `fill: none`, `stroke: currentColor` at `1.6`, so they inherit the
colour of whatever they sit in, including the rail.

## Controls

`--control: 2.65rem`. Inputs, selects and buttons share that height, the radius,
the border and the focus treatment: a 2px sage outline at `2px` offset for
links and buttons, and a sage border with a 3px `color-mix` sage halo for
fields, which would otherwise show two rings. Control text is `0.92rem` at
line-height `1.4`; labels are `0.86rem`.

A button in a stacked form is content width (`align-self: flex-start`); the
sign-in button stretches. Inline forms keep fields on one line and wrap only
when they must; the gate decision form (`.gate-actions`) grows its reason
field to fill the row, with Approve and Reject beside it. Buttons in the kill
pill are `2rem` and `0.8rem`, on the pill's own radius.

Buttons are ink on card, primary is `--on-forest` ivory on `--forest` (with
`--forest-hover` rather than a lighter tint), `ok` is `--on-ok` on `--ok`,
`danger` is `--on-danger` on `--danger-fill`, and `.link` is a sage underlined
button with no fill, which turns `--rail-sage` inside the rail so it stays
readable on charcoal. Disabled controls sit at `0.55` opacity.

## Work in progress

Anything that takes longer than a page load reports itself instead of holding the
browser. Source inspection is the model: the Data page submits the path with
htmx, the scan runs in a worker, and `.inspect-live` sits in a `--panel` card
beside the form, with a 6px `progress` bar on `--line` filled with
`--sage-fill`, a tabular percentage, and the counts (`2,640 of 6,000 files
read`). It polls every 700ms, replaces itself in place, and swaps the finished
`#inspect-result` out of band; the page never navigates, so the tab never shows
a spinner. Starting a scan clears the previous result.

## Kill control

Every signed-in page carries it in the topbar, with the same fields and the
same engage/release actions.

Idle it is a pill: card on a `--line-strong` hairline, muted `0.78rem` label
behind a sage shield icon, a borderless reason field that grows to fill, and a
`danger` button.

Engaged it is the loud pill: `--danger-fill` with `--on-danger` text, and its
controls inverted to a `color-mix` ivory outline so they read on the red.

## Home

`home.html` is the reference page. It opens with `.page-head` — "System
overview", the lede "Your private model engineering workspace", and a
`.head-chip` holding the gauge icon at the right edge — then two stats: active
runs (queued and running jobs, zero-padded) and pending approvals for this
person. Counts stay in the route; the template only formats them.

Under the stats sits the evaluation card. It names its state in a
`badge-unpromoted` pill (or `badge-approved` once promoted), says in one
sentence what that state blocks — "Model promotion is blocked until evaluation
criteria are met." — fills a `6px` `.meter` with the share of the report's
checks that passed, prints "Evaluation pipeline · N% complete", and ends in a
full-width forest button that opens that model. With no model registered the
same card says so and the button offers training instead; the badge, the
figure and the button all read from the latest registered model and its newest
report.

The step card carries the same `<ol>` and the same words. A `.meter` above it
is a native `<progress>` at `6px` on `--line` filled with `--sage-fill`, which
needs no inline width and therefore no inline style. CSS draws the path itself:
a `counter` in a `1.7rem` circle with a hairline border and muted number, a
`1px` connector from each circle to the next, no connector after the last, and
a `.done` step that swaps the number for a check (`\2713`) on the sage fill
while its link turns sage at weight 550.

The budget card keeps the shared table with right-aligned tabular money; a
"Needs attention" card pairs the inbox with unacknowledged alerts and their
acknowledge forms; `.plain` carries recent jobs; and empty states are `.empty`
blocks with a muted icon. Gate cards site-wide share one shape: a card with a
1px `--sage-fill` stripe, title plus tone badge plus muted meta, and
evidence in a `details` element.

## Motion

One transition, `160ms ease`, on colour, background and border for links,
buttons and fields. Nothing animates on load, and
`prefers-reduced-motion: reduce` turns the transition off.

## Not reopened

The marks in `Logo/` and the three served SVGs, sage as accent, the charcoal
rail against an ivory workspace, Inter then the system face with no download,
`prefers-color-scheme` with no toggle, `SameSite=Strict` cookies, a CSP with no
inline script or style, and the governance in `AGENTS.md`.
