"""Market filters for the bias scanner.

Only the filters the scanner actually uses live here. Historical helpers
for category/keyword/geographic filtering were removed when the scanner
pivoted to bias detection — they're recoverable from git history if
needed.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from polymarket_agent.data_fetching.models import Market

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of applying filters to markets."""

    markets: list[Market]
    total_before: int
    total_after: int
    filters_applied: list[str] = field(default_factory=list)

    @property
    def filter_rate(self) -> float:
        """Fraction of markets that passed filters (0.0–1.0)."""
        if self.total_before == 0:
            return 0.0
        return self.total_after / self.total_before


class MarketFilter:
    """Applies volume, liquidity, and expiry filters to markets.

    Markets matching any of ``always_include_keywords`` bypass the
    filters entirely and are included unconditionally.
    """

    def __init__(
        self,
        *,
        min_volume: float,
        min_liquidity: float,
        max_days_to_expiry: int,
        always_include_keywords: list[str] | None = None,
    ):
        self.min_volume = min_volume
        self.min_liquidity = min_liquidity
        self.max_days_to_expiry = max_days_to_expiry
        self.always_include_keywords = always_include_keywords or []

    def apply(self, markets: list[Market]) -> FilterResult:
        """Filter ``markets`` and return a :class:`FilterResult`."""
        total_before = len(markets)
        always_included, remaining = self._split_always_included(markets)

        filtered = remaining
        filters_applied: list[str] = []
        for name, func in (
            ("volume", lambda ms: filter_by_volume(ms, min_volume=self.min_volume)),
            ("liquidity", lambda ms: filter_by_liquidity(ms, min_liquidity=self.min_liquidity)),
            ("expiry", lambda ms: filter_by_expiry(ms, max_days=self.max_days_to_expiry)),
        ):
            before = len(filtered)
            filtered = func(filtered)
            filters_applied.append(f"{name}: {before} -> {len(filtered)}")
            logger.debug(filters_applied[-1])

        final = always_included + filtered
        return FilterResult(
            markets=final,
            total_before=total_before,
            total_after=len(final),
            filters_applied=filters_applied,
        )

    def __call__(self, markets: list[Market]) -> list[Market]:
        return self.apply(markets).markets

    def _split_always_included(
        self, markets: list[Market]
    ) -> tuple[list[Market], list[Market]]:
        if not self.always_include_keywords:
            return [], markets
        pattern = re.compile(
            "|".join(re.escape(kw) for kw in self.always_include_keywords),
            re.IGNORECASE,
        )
        always, other = [], []
        for m in markets:
            searchable = " ".join(filter(None, [
                m.question, m.description, m.event_title, " ".join(m.tags),
            ]))
            (always if pattern.search(searchable) else other).append(m)
        return always, other


def filter_by_volume(
    markets: list[Market],
    min_volume: float = 0,
    max_volume: float | None = None,
) -> list[Market]:
    """Keep markets whose volume is within ``[min_volume, max_volume]``."""
    return [
        m for m in markets
        if m.volume >= min_volume and (max_volume is None or m.volume <= max_volume)
    ]


def filter_by_liquidity(
    markets: list[Market],
    min_liquidity: float = 0,
    max_liquidity: float | None = None,
) -> list[Market]:
    """Keep markets whose liquidity is within ``[min_liquidity, max_liquidity]``."""
    return [
        m for m in markets
        if m.liquidity >= min_liquidity
        and (max_liquidity is None or m.liquidity <= max_liquidity)
    ]


def filter_by_expiry(
    markets: list[Market],
    max_days: int,
    reference_date: datetime | None = None,
) -> list[Market]:
    """Keep markets that resolve within ``max_days`` of ``reference_date``.

    Markets without an end date are kept. A non-positive ``max_days`` disables
    the filter entirely.
    """
    if max_days <= 0:
        return list(markets)
    reference = (reference_date or datetime.now()).date()
    result = []
    for m in markets:
        if m.end_date is None:
            result.append(m)
            continue
        delta = (m.end_date.date() - reference).days
        if 0 <= delta <= max_days:
            result.append(m)
    return result
