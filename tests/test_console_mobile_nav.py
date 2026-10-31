"""The phone rail must not hide a destination, and never the current page.

Under 900px the rail collapses into one scrolling strip. A strip that scrolls
has two ways of lying to the person using it: it can clip the rest of the
destinations with no sign that they exist, and it can leave the page you are
on somewhere off-screen, so the strip looks the same on every page. The rules
below are the fix, and they are checked rather than trusted.
"""

from __future__ import annotations

import re
from pathlib import Path

CSS_PATH = Path(__file__).resolve().parents[1] / "src/internal_ai/console/static/console.css"
PHONE_QUERY = "@media (max-width: 900px)"


def _read() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _phone_rules() -> str:
    text = _read()
    blocks: list[str] = []
    start = 0
    while (at := text.find(PHONE_QUERY, start)) != -1:
        depth = 0
        for index in range(at, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(text[at:index])
                    break
        start = at + len(PHONE_QUERY)
    return "\n".join(blocks)


def test_the_current_destination_leads_the_phone_strip() -> None:
    body = _phone_rules()
    assert re.search(r"\.rail-nav a\.active\s*\{[^}]*order:\s*-1", body), (
        "the phone strip must put the page you are on first, or it looks the same on every page"
    )


def test_the_phone_strip_shows_that_it_scrolls() -> None:
    body = _phone_rules()
    assert "scrollbar-width: none" not in body, (
        "a hidden scrollbar clips the destinations in silence"
    )
    assert not re.search(r"::-webkit-scrollbar\s*\{[^}]*display:\s*none", body), (
        "the phone strip must not hide its own scrollbar"
    )
    assert re.search(r"\.rail-nav\s*\{[^}]*scrollbar-width:\s*thin", body), (
        "the phone strip must show a scrollbar"
    )


def test_the_phone_strip_fades_into_the_rail() -> None:
    body = _phone_rules()
    assert re.search(r"background:\s*linear-gradient\([^)]*var\(--rail\)", body), (
        "the edge of the phone strip must fade into the rail, so a cut word reads as more to come"
    )
