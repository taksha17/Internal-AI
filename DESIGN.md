# DESIGN.md

The console look, as built. Every value here is read from
`src/internal_ai/console/static/console.css` and the two templates that carry
the shell, `base.html` and `signin.html`. One stylesheet styles every page; no
template carries an inline style, and the content security policy would refuse
one.

## Shape

The console is one instrument, not a set of pages. A signed-in page is a pinned
rail and one content column.

- The rail is `15.5rem` wide, `position: sticky`, `height: 100vh`, on `--panel`
  with a hairline right border. It holds the shield at `28px`, the wordmark in
  HTML, the eleven destinations, and the signed-in person at the bottom behind a
  hairline.
- The content column holds the kill strip and then `main`, capped at `72rem`
  and padded `2.5rem` by side.
- Below `860px` the grid collapses to one column. The rail becomes a top band in
  a wrapping row: mark and person on the first line, then the same links in one
  horizontally scrolling row (`overflow-x: auto`, `white-space: nowrap`). No
  menu, no script, no destination hidden.
- On narrow widths `main` is itself horizontally scrollable, so a wide table
  scrolls inside the page instead of pushing the shell sideways.

Sign-in is the one page without the rail: `body.signed-out` centres `main` at
`34rem` and the card at `30rem`.

## Colour

| Token | Light | Dark |
| --- | --- | --- |
| `--paper` | `#f7f8f6` | `#171d22` |
| `--panel` | `#eef1ee` | `#1c2328` |
| `--card` | `#ffffff` | `#1e262c` |
| `--field` | `#ffffff` | `#1b2228` |
| `--ink` | `#202a31` | `#f0f3f1` |
| `--muted` | `#5f6c72` | `#a5b0b2` |
| `--line` | `#dbe1dc` | `#2f3b41` |
| `--sage` | `#3f5c4e` | `#8aa99b` |
| `--sage-fill` | `#8aa99b` | `#8aa99b` |
| `--sage-wash` | `#e7eee9` | `#243038` |
| `--ok` | `#1f7a4d` | `#6ec99a` |
| `--danger` | `#8f2d24` | `#f0b2ab` |
| `--warn` / `--warn-ink` | `#fff4d6` / `#6b4e00` | `#3a3420` / `#f3e6c0` |
| `--alarm` / `--alarm-ink` | `#8f2d24` / `#f7f8f6` | `#7f3229` / `#f7efed` |

Dark values live in one `@media (prefers-color-scheme: dark)` block on `:root`,
with `color-scheme` set in both. There is no toggle.

Sage is the accent: active nav, focus rings, links, the primary button, the gate
stripe, and the done step. On paper, sage text is the darker `#3f5c4e` in light
mode. Primary buttons keep `#202a31` ink on the `#8aa99b` fill in both modes.

Ok, danger and pending stay their own colours and never become sage:
`#1f7a4d` / `#8f2d24` / `#fff4d6` with their ink pairs in light, `#6ec99a` /
`#f0b2ab` / `#3a3420` in dark.

Measured contrast, light then dark: body ink 13.72 / 15.22, muted 5.09 / 7.65 on
paper, sage 6.91 / 6.66 on paper, primary ink on sage 5.73, ok 4.99 / 8.49,
danger 7.67 / 9.44, alarm bar 7.67 / 7.75, table header ink 12.84 / 14.23.
Light-mode muted is `#5f6c72` rather than `#66747a` so it also clears 4.5:1 on
`--panel` and on the sage wash.

## Type

`Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`. No font is
downloaded. Body `1rem`, line-height `1.55`.

h1 `1.7rem`/650, h2 `1.12rem`/600, h3 `0.97rem`/600, letter-spacing `-0.02em`,
with more air above than below; a heading at the top of a page, card or callout
loses its top margin. Prose and lists cap at `68ch`; tables, grids and step lists
do not. Money and counts use `font-variant-numeric: tabular-nums`, on tables and
on the nav count.

## Surfaces

One radius (`--radius: 10px`), one hairline (`--line`) and one padding
(`1.15rem 1.3rem`) are shared by `.card`, `.callout`, tables, inputs and buttons.
`.flash` and `.refusal` take the radius and their own tint.

- Header rows sit on the `--panel` wash; body rows separate with a hairline; the
  last row drops it; row hover is `--sage-wash`; `tr.warn` keeps the amber pair
  through hover. Corner radii are repeated on the corner cells so a rounded
  table actually looks rounded.
- Lists (`.plain`) separate items with a hairline and drop the last one.
- `.tabs` are underline tabs: muted at rest, sage and 2px on the active one.
- Code and `pre` sit on `--panel` with the same hairline and radius.

## Controls

`--control: 2.4rem`. Inputs, selects and buttons share that height, the radius,
the border and a 2px sage focus ring at `2px` offset. Control text is `0.94rem`
at line-height `1.4`; labels are `0.88rem`.

A button in a stacked form is content width (`align-self: flex-start`); the
sign-in button stretches. Inline forms keep fields on one line and wrap only
when they must; the gate decision form puts its reason field on its own row
below `860px`, with Approve and Reject together under it. Buttons in the kill
strip are `2rem` and `0.82rem`.

Buttons are ink on card, primary is ink on sage, `ok` is `#f7f8f6` on
`#1f7a4d` (dark ink on `#6ec99a`), `danger` is `#f7f8f6` on `#8f2d24` (dark ink
on `#f0b2ab`), and `.link` is a sage underlined button with no fill. Disabled
controls sit at `0.55` opacity.

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

Idle it is `48px`: `--panel`, a hairline underneath, muted `0.85rem` copy, a
`2rem` reason field that grows to fill, and a `danger` button.

Engaged it is the loud bar: `--alarm` fill with `--alarm-ink` text, extra
vertical padding (`93px` on desktop when the copy and form both fit), and its
controls inverted to a translucent white outline so they read on the red.

## Home

`home.html` is the reference page. The step list is still the same `<ol>` with
the same words; CSS draws the path: a `counter` in a `1.75rem` circle with a
hairline border and muted number, a `1px` connector from each circle to the
next, no connector after the last, and a `.done` step that swaps the number for
a check (`\2713`) on the sage fill while its link turns sage at weight 550.

The budget table shows the shared table treatment, `.plain` carries recent jobs
and alerts, and "No jobs yet." / "No open alerts." are the empty states, set in
muted text.

## Motion

One transition, `150ms ease`, on colour, background and border for links,
buttons and fields. Nothing animates on load, and
`prefers-reduced-motion: reduce` turns the transition off.

## Not reopened

The marks in `Logo/` and the three served SVGs, sage as accent, Inter then the
system face with no download, `prefers-color-scheme` with no toggle,
`SameSite=Strict` cookies, a CSP with no inline script or style, and the
governance in `AGENTS.md`.
