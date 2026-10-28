"""Filtering, sorting and paging for the console's lists.

A list is a URL: the filter, the order and the page are all in the query
string, so a reload keeps them and a colleague can be sent the same view. None
of this decides access; a route passes in the rows it already loaded, which are
the rows the page was already allowed to show.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

PER_PAGE = 25
AUDIT_PER_PAGE = 50


def matches(term: str, *values: Any) -> bool:
    needle = term.strip().casefold()
    if not needle:
        return True
    return any(needle in str(value).casefold() for value in values if value is not None)


def since(span: str) -> str | None:
    """The ISO cut-off for a named window, or None for "everything"."""
    spans = {"24h": 1, "7d": 7, "30d": 30}
    days = spans.get(span)
    if days is None:
        return None
    return (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")


@dataclass(slots=True)
class Table:
    """One page of a list, with the links that move it.

    ``rows`` is what the template renders. ``total`` counts everything that
    matched the filter, not just this page, so "12 of 340" can be said honestly.
    """

    path: str
    rows: list[Any]
    total: int
    page: int
    pages: int
    per_page: int
    sort: str
    direction: str
    filters: dict[str, str] = field(default_factory=dict)

    def link(self, **changes: Any) -> str:
        values: dict[str, str] = {
            **self.filters,
            "sort": self.sort,
            "dir": self.direction,
            "page": str(self.page),
        }
        for key, value in changes.items():
            values[key] = "" if value is None else str(value)
        clean = {key: value for key, value in values.items() if value}
        if clean.get("page") == "1":
            clean.pop("page")
        if clean.get("dir") == "asc":
            clean.pop("dir")
        return f"{self.path}?{urlencode(sorted(clean.items()))}" if clean else self.path

    def sort_link(self, key: str, first: str = "asc") -> str:
        if self.sort == key:
            flipped = "desc" if self.direction == "asc" else "asc"
            return self.link(sort=key, dir=flipped, page="")
        return self.link(sort=key, dir=first, page="")

    def aria_sort(self, key: str) -> str:
        if self.sort != key:
            return ""
        return "ascending" if self.direction == "asc" else "descending"

    def is_first(self) -> int:
        return 0 if not self.rows else (self.page - 1) * self.per_page + 1

    def is_last(self) -> int:
        return (self.page - 1) * self.per_page + len(self.rows)


def _window(rows: Sequence[Any], page: int, per_page: int) -> tuple[list[Any], int, int]:
    total = len(rows)
    pages = max(1, (total + per_page - 1) // per_page)
    chosen = min(max(page, 1), pages)
    start = (chosen - 1) * per_page
    return list(rows[start : start + per_page]), chosen, pages


def build(
    rows: Iterable[Any],
    *,
    path: str,
    sorters: dict[str, Callable[[Any], Any]],
    default_sort: str,
    sort: str = "",
    direction: str = "",
    page: int = 1,
    per_page: int = PER_PAGE,
    filters: dict[str, str] | None = None,
    first_dirs: dict[str, str] | None = None,
) -> Table:
    """Sort and slice rows that were loaded whole."""
    chosen = sort if sort in sorters else default_sort
    firsts = first_dirs or {}
    way = direction if direction in ("asc", "desc") else firsts.get(chosen, "asc")
    ordered = sorted(rows, key=sorters[chosen], reverse=way == "desc")
    window, current, pages = _window(ordered, page, per_page)
    return Table(
        path=path,
        rows=window,
        total=len(ordered),
        page=current,
        pages=pages,
        per_page=per_page,
        sort=chosen,
        direction=way,
        filters={key: value for key, value in (filters or {}).items() if value},
    )


def window(
    rows: Sequence[Any],
    *,
    total: int,
    path: str,
    sort: str,
    direction: str,
    page: int = 1,
    per_page: int = AUDIT_PER_PAGE,
    filters: dict[str, str] | None = None,
) -> Table:
    """Wrap rows a store already paged and counted."""
    pages = max(1, (total + per_page - 1) // per_page)
    chosen = min(max(page, 1), pages)
    return Table(
        path=path,
        rows=list(rows),
        total=total,
        page=chosen,
        pages=pages,
        per_page=per_page,
        sort=sort,
        direction=direction,
        filters={key: value for key, value in (filters or {}).items() if value},
    )
