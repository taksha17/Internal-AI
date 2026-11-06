#!/usr/bin/env python3
"""Hold the landing site to its own rules.

Colour is a token block, not a habit. Markup is a shape, not a suggestion.
Run from anywhere: python3 site/tools/check-site.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "styles.css").read_text(encoding="utf-8")
HTML = (ROOT / "index.html").read_text(encoding="utf-8")
START = "/* tokens:start */"
END = "/* tokens:end */"
HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
DEF = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
VAR = re.compile(r"var\((--[\w-]+)")
ROOT_BLOCK = re.compile(r":root\s*\{(.*?)\}", re.S)
DARK_BLOCK = re.compile(
    r"@media\s*\(prefers-color-scheme:\s*dark\)\s*\{\s*:root\s*\{(.*?)\}\s*\}",
    re.S,
)

PAIRS = [
    ("ink on page", "--s-ink", "--s-bg"),
    ("muted on page", "--s-ink-muted", "--s-bg"),
    ("accent on page", "--s-accent", "--s-bg"),
    ("on-accent on accent", "--s-on-accent", "--s-accent"),
    ("highlight ink on wash", "--s-highlight-ink", "--s-highlight-wash"),
    ("on-night on night", "--s-on-night", "--s-night"),
    ("muted on night", "--s-on-night-muted", "--s-night"),
    ("accent on night", "--s-accent-on-night", "--s-night"),
    ("highlight on night", "--s-highlight", "--s-night"),
]

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def fence(css: str) -> tuple[str, str]:
    """Return the token block and everything after it."""
    if START not in css or END not in css:
        return "", css
    before, after = css.split(END, 1)
    if START not in before:
        return "", css
    return before.split(START, 1)[1], after


def channel(value: float) -> float:
    return value / 12.92 if value <= 0.03928 else ((value + 0.055) / 1.055) ** 2.4


def luminance(hex_value: str) -> float:
    raw = hex_value.lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    red, green, blue = (int(raw[i : i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue)


def contrast(first: str, second: str) -> float:
    high, low = sorted((luminance(first), luminance(second)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def declarations(block: str) -> dict[str, str]:
    return {name: value.strip() for name, value in DEF.findall(block)}


def resolve(name: str, values: dict[str, str], seen: set[str] | None = None) -> str | None:
    """Follow var() references to a literal hex, or None."""
    seen = seen or set()
    if name in seen or name not in values:
        return None
    value = values[name]
    match = re.fullmatch(r"var\((--[\w-]+)\)", value)
    if match:
        return resolve(match.group(1), values, seen | {name})
    if HEX.fullmatch(value):
        return value
    return None


def check_contrast(tokens: str) -> None:
    light_block = ROOT_BLOCK.search(tokens)
    dark_block = DARK_BLOCK.search(tokens)
    check(light_block is not None, "contrast: light :root block is missing")
    check(dark_block is not None, "contrast: dark :root block is missing")
    if light_block is None or dark_block is None:
        return
    light = declarations(light_block.group(1))
    dark = dict(light)
    dark.update(declarations(dark_block.group(1)))
    for scheme, values in (("light", light), ("dark", dark)):
        for label, foreground, background in PAIRS:
            first = resolve(foreground, values)
            second = resolve(background, values)
            if first is None or second is None:
                check(False, f"{scheme}: cannot resolve {label}")
                continue
            ratio = contrast(first, second)
            check(ratio >= 4.5, f"{scheme}: {label} is {ratio:.2f}:1")


PROBLEM_COPY = [
    "Renting has a ceiling.",
    "Open weights change the maths, not the work.",
    "Ownership is a loop.",
    "Governed, or it does not ship.",
    "internal-ai is that loop, with the governance already built in.",
    "No raw record leaves the boundary.",
    "No third-party model output enters training.",
    "No gate is decided by its requester.",
    "Expert-approved frozen sets",
    "Released, scrubbed, lineaged",
    "Traces become triage cases",
    "Gates, audit, offline by default",
]
FORBIDDEN_COPY = ["beats frontier", "air-gapped", "rebase"]


def check_markup() -> None:
    problem = re.search(r"<section[^>]*id=\"problem\"[^>]*>", HTML)
    check(problem is not None, "the Problem section has no id")
    if problem is not None:
        check("problem" in problem.group(0), "the #problem id is on the wrong section")

    nav = re.search(r'<nav class="nav-links".*?</nav>', HTML, re.S)
    labels = (
        [label.strip() for label in re.findall(r">([^<>]+)</a>", nav.group(0))]
        if nav is not None
        else []
    )
    check(
        labels == ["Problem", "Platform", "Governance", "Evidence"],
        f"nav labels are {labels}",
    )

    blocks = HTML.count('class="problem-block"')
    check(blocks == 4, f"problem blocks: {blocks}, expected 4")

    loop = re.search(r'<ol class="loop-steps">(.*?)</ol>', HTML, re.S)
    steps = loop.group(1).count("<li>") if loop is not None else 0
    check(steps == 4, f"loop steps: {steps}, expected 4")

    inside = re.search(r'<div class="inside">(.*?)</div>', HTML, re.S)
    guarantees = inside.group(1).count("<li>") if inside is not None else 0
    check(guarantees == 3, f"inside guarantees: {guarantees}, expected 3")

    for phrase in PROBLEM_COPY:
        check(phrase in HTML, f"missing copy: {phrase}")

    lower = HTML.lower()
    for phrase in FORBIDDEN_COPY:
        check(phrase not in lower, f"forbidden claim: {phrase}")


def media_block(query: str) -> str | None:
    """Return the body of an @media block, braces balanced."""
    start = CSS.find(f"@media {query}")
    if start == -1:
        return None
    brace = CSS.find("{", start)
    if brace == -1:
        return None
    depth = 0
    for index in range(brace, len(CSS)):
        if CSS[index] == "{":
            depth += 1
        elif CSS[index] == "}":
            depth -= 1
            if depth == 0:
                return CSS[brace + 1 : index]
    return None


def rule_for(selector: str) -> str | None:
    match = re.search(re.escape(selector) + r"[^{}]*\{([^}]*)\}", CSS)
    return match.group(1) if match else None


def check_structure() -> None:
    skips = HTML.count('class="skip')
    check(skips == 1, f"skip links: {skips}, expected 1")

    actions = HTML.find('class="hero-actions"')
    proof = HTML.find('class="hero-proof"')
    check(
        actions != -1 and proof != -1 and actions < proof,
        "the hero actions do not sit above the proof",
    )

    for selector in (".pill", ".kill"):
        body = rule_for(selector) or ""
        check(
            "white-space: nowrap" in body,
            f"{selector} does not stay on one line",
        )

    top = rule_for(".console-top") or ""
    check("flex-wrap: wrap" in top, ".console-top does not wrap")

    phone = media_block("(max-width: 720px)")
    check(phone is not None, "the 720px media block is missing")
    if phone is not None:
        check(
            "--masthead-h: 6.75rem" in phone,
            "the phone masthead height is not set",
        )

    check(
        "scroll-margin-top: calc(var(--masthead-h)" in CSS,
        "section anchors do not clear the masthead",
    )

    hidden = re.search(r"\.nav-links[^{}]*\{[^}]*display:\s*none", CSS)
    check(hidden is None, "the nav is hidden instead of reflowed")

    wide = media_block("(min-width: 721px)")
    check(wide is not None, "the 721px media block is missing")
    if wide is not None:
        check(
            re.search(r"\.console\s*\{[^}]*min-width", wide) is not None,
            "the console has no floor on a wide screen",
        )
    without_wide = CSS.replace(wide, "") if wide is not None else CSS
    check(
        re.search(r"\.console\s*\{[^}]*min-width", without_wide) is None,
        "a console min-width sits outside the wide media block",
    )


def check_rhythm() -> None:
    eyebrows = HTML.count('class="eyebrow"')
    check(eyebrows >= 6, f"eyebrows: {eyebrows}, expected 6 or more")

    kickers = HTML.count('class="kicker"')
    check(3 <= kickers <= 4, f"kickers: {kickers}, expected 3 or 4")

    check("counter(card" in CSS, "the cards carry no index")
    check(".card::after" in CSS, "the card index has no rule")

    check(".section-head::after" not in CSS, "the section head has a second rule")

    note = rule_for(".scale-note") or ""
    check("border-left" in note, "the evidence note is not marked as a finding")


def check_access() -> None:
    for name in ("github", "context"):
        check(
            f'aria-describedby="{name}-hint"' in HTML,
            f"{name} does not point at its hint",
        )
        check(f'id="{name}-hint"' in HTML, f"{name}-hint has no id")

    check('preload="none"' in HTML, "the video preloads before anyone asks")
    check('preload="metadata"' not in HTML, "the video asks for metadata up front")

    focus = rule_for(".band-night :focus-visible") or ""
    check(
        "--s-accent-on-night" in focus,
        "a focus ring on night does not use the night accent",
    )

    hover = media_block("(hover: hover)")
    check(hover is not None, "the hover media query is missing")
    if hover is not None:
        check(
            ".card:hover" in hover and "transform" in hover,
            "the card lift is not gated on a pointing device",
        )

    for name in ("band-night", "notfound", "kicker", "btn", "btn-primary", "home"):
        check(f".{name}" in CSS, f"the result page class .{name} has no rule")

    scripts = re.findall(r"<script\b[^>]*>", HTML)
    check(len(scripts) == 2, f"scripts: {len(scripts)}, expected 2 (ld+json + landing.js)")
    check(
        bool(scripts) and 'type="application/ld+json"' in scripts[0],
        "the first script is not the structured-data block",
    )
    check(
        'src="landing.js"' in scripts[1],
        "the second script is not landing.js",
    )
    check("<style" not in HTML, "an inline style block is present")
    check(" style=" not in HTML, "an inline style attribute is present")
    check(
        re.search(r"\son[a-z]+=", HTML) is None,
        "an inline event handler is present",
    )


def main() -> int:
    starts = CSS.count(START)
    ends = CSS.count(END)
    check(
        starts == 1,
        "tokens:start fence is missing"
        if starts == 0
        else f"tokens:start fence appears {starts} times",
    )
    check(
        ends == 1,
        "tokens:end fence is missing" if ends == 0 else f"tokens:end fence appears {ends} times",
    )

    tokens, after = fence(CSS)
    if not tokens or starts != 1 or ends != 1:
        failures.append("no hex below the fence (fence missing)")
        failures.append("primitive token read below the fence (fence missing)")
        failures.append("contrast cannot be checked (fence missing)")
    else:
        stray = HEX.search(after)
        check(stray is None, f"hex below the fence: {stray.group(0) if stray else ''}")
        primitive = re.search(r"var\(--p-", after)
        check(
            primitive is None,
            "a rule below the fence reads a primitive token",
        )
        check_contrast(tokens)

    defined = {name for name, _ in DEF.findall(CSS)}
    uses = set(VAR.findall(CSS))
    for name in sorted(uses - defined):
        check(False, f"var({name}) has no definition")

    primitives = {name for name, _ in DEF.findall(tokens) if name.startswith("--p-")}
    check(bool(primitives), "no primitive tokens are defined")

    check_markup()
    check_structure()
    check_rhythm()
    check_access()

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"{len(failures)} failure(s)")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
