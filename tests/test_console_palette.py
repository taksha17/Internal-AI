"""The palette is a rule, so it is checked rather than trusted.

Every foreground/surface pair the stylesheet actually uses must clear 4.5:1 in
both schemes, and no rule may carry a colour of its own: a theme is the token
block and nothing else. A new state therefore picks a tone; it cannot invent a
green that quietly fails on paper.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CSS_PATH = Path(__file__).resolve().parents[1] / "src/internal_ai/console/static/console.css"
MIN_CONTRAST = 4.5

PAIRS = [
    ("ink", "paper"),
    ("ink", "card"),
    ("ink", "panel"),
    ("ink", "sage-wash"),
    ("muted", "paper"),
    ("muted", "card"),
    ("muted", "panel"),
    ("sage", "paper"),
    ("sage", "card"),
    ("sage", "panel"),
    ("on-sage", "sage-fill"),
    ("on-forest", "forest"),
    ("on-ok", "ok"),
    ("on-danger", "danger-fill"),
    ("tone-ok-ink", "tone-ok-bg"),
    ("tone-warn-ink", "tone-warn-bg"),
    ("tone-bad-ink", "tone-bad-bg"),
    ("tone-idle-ink", "tone-idle-bg"),
    ("tone-ok-ink", "card"),
    ("tone-warn-ink", "card"),
    ("tone-bad-ink", "card"),
    ("tone-idle-ink", "card"),
    ("ink", "blush-wash"),
    ("blush-ink", "blush"),
    ("blush-text", "paper"),
    ("blush-text", "card"),
    ("blush-text", "panel"),
    ("rail-ink", "rail"),
    ("rail-item", "rail"),
    ("rail-item", "rail-hover"),
    ("rail-sage", "rail"),
    ("rail-muted", "rail"),
    ("rail-ink", "rail-active"),
]

DECLARATION = re.compile(r"--[a-z0-9-]+\s*:\s*[^;]*;")
HEX = re.compile(r"#[0-9a-fA-F]{3,8}")


def _read() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _block(text: str, opener: str, start: int = 0) -> tuple[str, int]:
    at = text.index(opener, start)
    depth = 0
    for index in range(at, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[at + len(opener) : index], index
    raise AssertionError(f"unbalanced block after {opener}")


def _tokens(body: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for name, value in re.findall(r"--([a-z0-9-]+)\s*:\s*([^;]+);", body):
        if name in found:
            raise AssertionError(f"{name} is declared twice in one block")
        found[name] = value.strip()
    return found


def _schemes() -> dict[str, dict[str, str]]:
    text = _read()
    light = _tokens(_block(text, ":root {")[0])
    dark_body = _block(text, "@media (prefers-color-scheme: dark)", 0)[0]
    dark = {**light, **_tokens(_block(dark_body, ":root {")[0])}
    return {"light": light, "dark": dark}


def _channel(value: int) -> float:
    ratio = value / 255
    return ratio / 12.92 if ratio <= 0.04045 else ((ratio + 0.055) / 1.055) ** 2.4


def luminance(hex_value: str) -> float:
    digits = hex_value.lstrip("#")
    red, green, blue = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(red) + 0.7152 * _channel(green) + 0.0722 * _channel(blue)


def contrast(foreground: str, background: str) -> float:
    high, low = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (high + 0.05) / (low + 0.05)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_pair_used_by_the_stylesheet_is_readable(scheme: str) -> None:
    tokens = _schemes()[scheme]
    failures = []
    for foreground, background in PAIRS:
        ratio = contrast(tokens[foreground], tokens[background])
        if ratio < MIN_CONTRAST:
            failures.append(f"{foreground} on {background}: {ratio:.2f}")
    assert not failures, f"{scheme} mode: " + "; ".join(failures)


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_colour_pair_is_declared_as_a_hex(scheme: str) -> None:
    tokens = _schemes()[scheme]
    named = {name for pair in PAIRS for name in pair}
    missing = sorted(named - tokens.keys())
    assert not missing, f"undeclared tokens: {missing}"
    malformed = [
        f"{name}: {tokens[name]}" for name in sorted(named) if not HEX.fullmatch(tokens[name])
    ]
    assert not malformed, f"{scheme} mode: {malformed}"


def test_no_rule_carries_a_colour_of_its_own() -> None:
    rules = DECLARATION.sub("", _read())
    assert not HEX.findall(rules), "colours belong in the token block, not in rules: " + ", ".join(
        HEX.findall(rules)
    )


def test_the_rail_keeps_one_midnight_in_both_schemes() -> None:
    text = _read()
    dark_body = _block(text, "@media (prefers-color-scheme: dark)", 0)[0]
    redefined = [
        name for name in _tokens(_block(dark_body, ":root {")[0]) if name.startswith("rail")
    ]
    assert not redefined, f"the rail should not change with the scheme: {redefined}"
