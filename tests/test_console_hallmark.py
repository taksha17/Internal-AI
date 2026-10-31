"""The console's own design rules are checked rather than trusted.

A Hallmark pass found real defects in this shell: two elements fighting for
the same sticky edge, a field border nobody could see, a skip link that
animated `top`, and a phone kill panel that opened off the screen. A rule that
is only in someone's head is a rule that comes back, so each one below is
written down and checked.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS_PATH = ROOT / "src/internal_ai/console/static/console.css"
TEMPLATES = ROOT / "src/internal_ai/console/templates"

PHONE_QUERY = "@media (max-width: 900px)"


def _css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def _rules(text: str) -> list[tuple[str, str]]:
    """Every block in the sheet as (header, body), nested ones included."""
    found: list[tuple[str, str]] = []
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    index = 0
    while True:
        brace = text.find("{", index)
        if brace == -1:
            return found
        header = text[index:brace].strip()
        depth = 0
        for end in range(brace, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    break
        body = text[brace + 1 : end]
        found.append((header, body))
        found.extend(_rules(body))
        index = end + 1


def _rule(text: str, selector: str) -> str:
    """The declaration block for a selector, or an empty string."""
    wanted = " ".join(selector.split())
    for header, body in _rules(text):
        if " ".join(header.split()) == wanted:
            return body
    return ""


def test_the_stylesheet_says_what_it_is() -> None:
    first = next(line for line in _css().splitlines() if line.strip())
    assert first.startswith("/* Hallmark"), "the stylesheet must carry its stamp on the first line"


def test_the_page_never_scrolls_sideways() -> None:
    body = _rule(_css(), "html, body")
    assert "overflow-x: clip" in body, "a wide table must scroll inside itself, not push the shell"


def test_the_skip_link_moves_without_animating_top() -> None:
    skip = _rule(_css(), ".skip")
    assert "transform" in skip, "the skip link is moved by transform, not by top"
    assert "transition: top" not in skip, "animating top is a floor finding"
    reduced = _rule(_css(), "a, button, .button, .btn, input, select, textarea, .skip")
    assert "transition: none" in reduced, "reduced motion turns the skip link's transition off"


def test_a_code_face_is_a_token() -> None:
    text = _css()
    assert "font-family: var(--font-mono)" in text, "code reads the mono token"
    outside = re.sub(r"(?s):root\s*\{.*?\}", "", text)
    literals = [
        line for line in outside.splitlines() if "font-family" in line and "var(" not in line
    ]
    assert not literals, "a font-family literal belongs in the token block: " + ", ".join(literals)


def test_a_display_head_can_wrap() -> None:
    heads = _rule(_css(), "h1, h2, h3")
    assert "overflow-wrap: break-word" in heads
    assert "min-width: 0" in heads


def test_a_control_answers_a_press() -> None:
    assert ":active" in _css(), "a control needs its pressed state"


def test_a_field_edge_can_be_seen() -> None:
    fields = _rule(_css(), "input, select, textarea")
    assert "border: 1px solid var(--muted)" in fields, (
        "a field's edge has to clear 3:1 on paper, on a card and on the panel"
    )


def test_only_the_topbar_sticks_to_the_top() -> None:
    text = _css()
    sticky = [
        selector
        for selector, body in re.findall(r"(?m)^([^{}@]+)\{([^}]*)\}", text)
        if "position: sticky" in body and re.search(r"(?m)^\s*top:\s*0(px|rem)?\s*;", body)
    ]
    assert len(sticky) == 1, f"one element may own the top edge, not {sticky}"
    rail = _rule(text, ".rail")
    assert "position: fixed" in rail, "the rail is pinned beside the work"


def test_a_phone_kill_panel_fits_the_screen() -> None:
    phone = _rule(_css(), PHONE_QUERY)
    panel = _rule(phone, ".kill-panel")
    assert "left: 0.75rem" in panel and "right: 0.75rem" in panel, (
        "on a phone the panel is anchored to the bar, so it always fits the screen"
    )
    assert "100vw" not in panel, "a 100vw width counts the scrollbar and spills"
    assert "position: static" in _rule(phone, ".kill"), (
        "the chip stops being the anchor, or the panel opens off the screen"
    )
    assert "display: none" in _rule(_css(), ".kill:not([open]) .kill-panel"), (
        "a closed switch leaves no panel in the layout"
    )


def test_a_page_does_not_open_with_an_eyebrow() -> None:
    offenders = [
        path.name
        for path in sorted(TEMPLATES.glob("*.html"))
        if 'class="eyebrow"' in path.read_text()
    ]
    assert not offenders, f"the breadcrumb already says where you are: {offenders}"
    assert "eyebrow" not in _rule(
        TEMPLATES.joinpath("_macros.html").read_text(encoding="utf-8"), "page_head"
    )


def test_meta_lines_are_not_joined_with_middle_dots() -> None:
    allowed = {
        "base.html": ("<title>", "Local · no external calls"),
    }
    offenders = []
    for path in sorted(TEMPLATES.glob("*.html")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "·" not in line:
                continue
            if any(mark in line for mark in allowed.get(path.name, ())):
                continue
            offenders.append(f"{path.name}:{number}")
    assert not offenders, "a middle dot is not a separator: " + ", ".join(offenders)


def test_the_type_scale_uses_four_weights() -> None:
    weights = set(re.findall(r"font-weight: (\d+)", _css()))
    assert weights <= {"400", "500", "600", "700"}, f"the scale has drifted: {sorted(weights)}"
