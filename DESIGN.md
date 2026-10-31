# DESIGN.md

The console look, as built. Every value here is read from
`src/internal_ai/console/static/console.css` and the two templates that carry
the shell, `base.html` and `signin.html`. One stylesheet styles every page; no
template carries an inline style, and the content security policy would refuse
one.

## Shape

The console is one instrument, not a set of pages. A signed-in page is a pinned
rail and one content column.

- The rail is `18rem` wide, `position: fixed`, `height: 100vh`, on the one
  midnight `--rail` with a `--rail-line` hairline on its right and a faint
  `--rail-active` glow at the top. It is pinned rather than sticky, so the
  topbar is the only element that owns the top edge of the screen. The brand
  block is the shield on a pale sage chip, the wordmark in HTML, and a muted
  `PRIVATE BY DESIGN` micro-cap, above a hairline. Destinations are one flat
  list with no group headings: Overview, Pipelines, Datasets, Training,
  Evaluations, Approvals, Audit log, a hairline, then Inbox, Failed answers,
  Data, Models, Brief, Ask and Admin. Each row is `2.45rem`: a `1.05rem` stroke
  icon from the sprite, the label, and the approvals count as a jade pill. The
  active row is `--rail-active` with a `--rail-sage` icon, `aria-current="page"`,
  and the same fill on hover so it does not flicker. The bottom holds the
  person: an initial disc, the name, the roles, and Sign out. The rail does not
  change with the colour scheme; only the workspace does.
- The content column is `.column`, `margin-left: 18rem`, holding a sticky
  topbar and then `main`, capped at `80rem` and padded `2rem` by side. The
  topbar blurs the paper behind it and carries a breadcrumb, a search field,
  a `Local · no external calls` scope pill, and the kill switch as a chip.
- Below `900px` the grid collapses to one column. The rail becomes a top band:
  mark, then the links in one horizontally scrolling row (`overflow-x: auto`,
  the scrollbar visible and thin), each keeping its icon, then the person. The
  active destination is first in the row (`order: -1`), so the band always
  opens on where you are. No menu, no script, no destination hidden. The
  topbar wraps, and the controls grow to `2.65rem` for a finger.
- `html, body` carry `overflow-x: clip`, so a wide table scrolls inside its own
  `.table-wrap` instead of pushing the shell sideways.

Sign-in is the one page without the rail: a midnight brand panel with the
mark, "Private by design.", the product facts and three audited guarantees,
beside a paper form panel that holds the theme-aware lockup and the form at
`22.5rem`. Below `900px` the panel is dropped and the form stands alone. Both
lockups stay in the markup and the scheme still chooses between them. The
panel is midnight, so its focus ring is the rail's jade rather than the
workspace sage, which would sit under 3:1 on that ground.

## Colour

Cool workspace: paper `#F2F7F5` and white cards under ink `#0F1A1E`, muted
`#4E5C5D`, a jade accent `#0E6B5C` for text, links, focus and the active mark,
a jade fill `#2FB89E` for surfaces only, and one midnight rail `#0B1220` in
both schemes. Clay `#8A5A22` keeps the workspace from reading all-jade. The
token block at the top of `console.css` is the whole theme: rules read names,
never hex, and `tests/test_console_palette.py` fails the build if one carries a
colour of its own.

| Token | Light | Dark |
| --- | --- | --- |
| `--paper` | `#F2F7F5` | `#080E15` |
| `--panel` | `#E4EFEA` | `#0E1622` |
| `--card` / `--field` | `#ffffff` | `#111A27` |
| `--ink` | `#0F1A1E` | `#E8F1EE` |
| `--muted` | `#4E5C5D` | `#93A5A1` |
| `--line` / `--line-strong` | `#D3E0DB` / `#B9CBC5` | `#1E2A37` / `#2B3B4A` |
| `--sage` | `#0E6B5C` | `#6FE6CB` |
| `--sage-fill` / `--on-sage` | `#2FB89E` / `#06231D` | `#2FB89E` / `#06231D` |
| `--sage-wash` | `#DCEDE7` | `#0F2A26` |
| `--forest` / `--forest-hover` / `--on-forest` | `#0D5C50` / `#0A4A40` / `#F2F7F5` | `#0F6E58` / `#0C5D4A` / `#F2F7F5` |
| `--ok` / `--ok-hover` / `--on-ok` | `#17694E` / `#125A42` / `#F2F7F5` | `#63D6A8` / `#56C297` / `#06231A` |
| `--danger-fill` / `--danger-hover` / `--on-danger` | `#8E2F3C` / `#772433` / `#fbeeec` | `#A83E4A` / `#933442` / `#fbeeec` |
| `--tone-ok-bg` / `-ink` | `#d6efe4` / `#0b5a3e` | `#12332a` / `#a8e6c9` |
| `--tone-warn-bg` / `-ink` | `#f6e7c4` / `#6b4a0e` | `#3a3220` / `#f3e6c0` |
| `--tone-bad-bg` / `-ink` | `#fbd9de` / `#7e1a28` | `#3a2228` / `#f4bdc6` |
| `--tone-idle-bg` / `-ink` | `#e3ebe7` / `#45534e` | `#1b2624` / `#b4c3bf` |

Rail tokens are declared once, outside the scheme block, because the rail is
the same midnight in both: `--rail #0B1220`, `--rail-ink #EEF3F2`,
`--rail-item #C2CFCC`, `--rail-sage #5FE0C5`, `--rail-muted #8FA0A2`,
`--rail-line #1D2A38`, `--rail-active #14202E`, `--rail-hover #0F1926`,
`--rail-chip #DDF3EC`.

The card lift is a token too, `--lift`, built from `--shade` with `color-mix`
so a card has an edge without a hex in a rule: `0 1px 2px` at 5% plus
`0 2px 14px` at 5% in light, one `0 1px 2px` at 48% in dark.

Dark values live in one `@media (prefers-color-scheme: dark)` block on `:root`,
with `color-scheme` set in both. There is no toggle.

Sage separates into two jobs. `--sage` is text, links, focus rings, the active
nav mark and the callout rule. `--sage-fill` is a surface only: progress, the
nav count pill, the done step, always under `--on-sage` ink. The primary
button is `--forest` with `--on-forest`; ok, reject and the kill control keep
their own reds and greens and never become sage.

A second accent, clay, keeps the workspace from reading all-teal. `--clay`
(`#8A5A22` light, `#E8B77A` dark) carries the `.pop` word inside a headline
and the alternate stat icon; `--clay-strong` inks text on it and `--clay-wash`
washes behind it. Clay never marks governance: approvals, rejections, pending
states, focus and the active nav stay on their own colours.

States are tones, not colours. `.badge-*`, `.flash-*`, `.refusal` and `tr.warn`
all read `--tone-*-bg` / `--tone-*-ink`, so there is one rule per tone instead
of a light and a dark duplicate, and a state added later has to pick a tone.
Tone borders are `color-mix` of the tone ink, which keeps edges out of the
token block without adding eight more literals.

A field's edge is not decoration and has to be seen: inputs, selects and
textareas take `--muted` for their border, which clears 3:1 on paper, on a
card and on the panel. Buttons keep `--line-strong`, because a button on a
card already has a fill under it.

Measured contrast, light then dark: ink on paper 16.34 / 16.83, on a card
17.69 / 15.20, on the panel 15.03 / 15.78, on the sage wash 14.59 / 13.23;
muted on paper 6.44 / 7.51, on a card 6.97 / 6.78, on the panel 5.92 / 7.04;
sage on paper 5.92 / 12.78, on a card 6.41 / 11.54, on the panel 5.44 / 11.98;
dark ink on the jade fill 6.69 in both, ivory on forest 7.28 / 5.72, ivory on
ok 6.12 / 9.27, ivory on danger 7.07 / 5.37; tone ink on tone wash 6.80 / 9.67
ok, 6.57 / 10.19 warn, 7.79 / 8.99 bad, 6.65 / 8.52 idle, and the same inks on
a card 8.25 / 12.35, 8.04 / 14.07, 10.17 / 10.78, 8.08 / 9.58; rail ink 16.71,
rail item 11.67 and 11.03 on the hover fill, rail sage 11.55, rail muted 6.88,
rail ink on the active fill 14.69.

## Type

`Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`. No font is
downloaded. Body `0.9375rem`, line-height `1.55`. Four weights and no more:
`400`, `500`, `600`, `700`.

h1 `1.5rem`/700, h2 `1.05rem`/600, h3 `0.94rem`/600, letter-spacing `-0.028em`,
with more air above than below; a heading at the top of a page, card or callout
loses its top margin. Every heading carries `overflow-wrap: break-word` and
`min-width: 0`, so a long name in a narrow column wraps instead of pushing the
page sideways. A page's `h1` is followed by its muted subtitle, so every page
starts the same way whether or not it uses `.page-head`. Prose and lists cap at
`72ch`, the lede and the callout at `70ch`; tables, grids and step lists do
not. Money and counts use `font-variant-numeric: tabular-nums`, on tables,
tiles and the nav count. Code and `pre` read `--font-mono`, a token like every
other face.

## Surfaces

One radius (`--radius: 12px`), one hairline (`--line`) and one padding
(`1.05rem 1.1rem`) are shared by `.card` and `.callout`, with a two-layer
`--lift` under both. Controls sit on a tighter `--control-radius: 8px` so a
button inside a card is not as round as the card. Tables that need their own
scroll region sit in `.table-wrap`, which carries the border so the table
inside stays flush.

- `.callout` is not a box: no fill, no shadow, a `2px` jade rule down its left
  edge and a `70ch` measure. It says something in passing.
- `.card-head` is the row a card opens with: title and subtitle left, a badge or
  a `.quiet-link` right, wrapping on narrow widths.
- Header rows sit on the `--panel` wash in `0.8rem` muted semibold; body rows
  separate with a hairline; the last row drops it; row hover is `--sage-wash`;
  `tr.warn` keeps the amber pair through hover. Corner radii are repeated on
  the corner cells so a rounded table actually looks rounded.
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

`.page-head` is the page's opening row: the `h1`, a muted `.lede` under it,
and any primary action right-aligned and aligned to the text. Every page uses
one. There is no eyebrow: the breadcrumb already says where you are, and a
line of small capitals above every title was noise.

A page may carry its state in the head, as a `.badge` beside the title: the
badge names the state of the thing the heading names, and it is the same mark
on every page that has one. It stays inside the heading because that is the
one place the state belongs. (Hallmark gate 38iii notes this as an accented
word in a heading; it is deliberate and recorded rather than rebuilt.)

`.stat-grid` is a responsive row of `.stat`s, and the overview uses two of them
under the `.two` modifier, which is one column on a phone. A stat is a card
with a `0.8rem` muted label, the figure at `1.8rem`/700 in tabular figures,
and a `.stat-meta` line whose `.stat-state` carries a small tone-coloured dot
beside the words: ok for work in progress, warn for work waiting on a person,
idle when there is none. The dot never carries the meaning alone; the words
do.

Icons are one inline sprite per page, `_macros.html:sprite()`, referenced with
`<use href="#i-name">` through `icon()`. There is no icon font, no extra request
and no script; the sprite is `position: absolute` at zero size. Icons are
`1.1rem`, `fill: none`, `stroke: currentColor` at `1.6`, so they inherit the
colour of whatever they sit in, including the rail.

## Controls

`--control: 2.25rem`, and `2.65rem` on a phone. Inputs, selects and buttons
share that height, the radius, the border and the focus treatment: a 2px sage
outline at `2px` offset for links and buttons, and a sage border with a 3px
`color-mix` sage halo for fields, which would otherwise show two rings.
Control text is `0.92rem` at line-height `1.4`; labels are `0.86rem`.

A button in a stacked form is content width (`align-self: flex-start`); the
sign-in button stretches. Inline forms keep fields on one line and wrap only
when they must; the gate decision form (`.gate-actions`) grows its reason
field to fill the row, with Approve and Reject beside it. Buttons in the kill
chip are `2rem` and `0.8rem`, on the chip's own radius.

Buttons are ink on card, primary is `--on-forest` ivory on `--forest` (with
`--forest-hover` rather than a lighter tint), `ok` is `--on-ok` on `--ok`,
`danger` is `--on-danger` on `--danger-fill`, and `.link` is a sage underlined
button with no fill, which turns `--rail-sage` inside the rail so it stays
readable on midnight. Pressing a control says so once, in place: it moves a
pixel and does not light up a second colour. Disabled controls sit at `0.55`
opacity.

## Copy

The console writes plain ASCII, and it does not join facts with a middle dot
or a dash. A meta line is separated by commas and colons, a page title carries
its state as a badge, and a pager says "1 to 50 of 120" rather than a range
with a dash in it. The two middle dots that remain are the `<title>` and the
live-state pill, where a dot separates two short names rather than two facts.

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

Idle it is a chip: card on a `--line-strong` hairline, muted `0.78rem` label
behind a sage shield icon. Engaged it is the loud chip: `--danger-fill` with
`--on-danger` text, and its controls inverted to a `color-mix` ivory outline so
they read on the red.

The reasons live in a panel that opens under the chip. A closed switch leaves
no panel in the layout, so nothing sits off the screen. On a phone the chip
stops being the anchor: the panel is pinned to the bar with `left` and `right`
at `0.75rem` and `width: auto`, so it always fits the screen whatever the
chip's own width is.

## Motion

One transition, `--beat: 160ms ease`, on colour, background and border for
links, buttons and fields. Nothing animates on load, the skip link moves by
`transform` rather than by `top`, and `prefers-reduced-motion: reduce` turns
every transition off.

## Checks

The design is a rule, so it is checked rather than trusted:
`tests/test_console_palette.py` reads the token block and fails if a rule
carries a colour of its own or a pair drops under 4.5:1,
`tests/test_console_hallmark.py` holds the shell's shape (one sticky top, a
clipped page, a field edge that can be seen, a phone panel that fits),
`tests/test_console_markup.py` holds the page furniture, and
`tests/test_console_mobile_nav.py` holds the phone band.

## Not reopened

The marks in `Logo/` and the three served SVGs, sage as accent, the midnight
rail against a pale workspace, Inter then the system face with no download,
`prefers-color-scheme` with no toggle, `SameSite=Strict` cookies, a CSP with
no inline script or style, and the governance in `AGENTS.md`.
