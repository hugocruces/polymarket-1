"""
Tests for spread/slippage analysis.
"""

import pytest

from polymarket_agent.data_fetching.models import OrderbookDepth, OrderLevel
from polymarket_agent.analysis.spread_analysis import (
    calculate_slippage,
    analyze_spread,
    SlippageEstimate,
    SpreadAnalysis,
)


def make_orderbook(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    token_id: str = "tok-1",
) -> OrderbookDepth:
    """Helper to create an OrderbookDepth from (price, size) tuples."""
    return OrderbookDepth(
        token_id=token_id,
        bids=[OrderLevel(price=p, size=s) for p, s in bids],
        asks=[OrderLevel(price=p, size=s) for p, s in asks],
    )


class TestCalculateSlippage:
    """Tests for slippage calculation with synthetic orderbooks."""

    def test_basic_buy_slippage(self):
        """Buy into a book with multiple ask levels."""
        book = make_orderbook(
            bids=[(0.48, 500), (0.47, 500)],
            asks=[(0.52, 500), (0.53, 500)],
        )
        result = calculate_slippage(book, position_size_usd=100, side="buy")

        assert result is not None
        assert result.side == "buy"
        assert result.position_size_usd == 100
        # Mid price should be (0.48 + 0.52) / 2 = 0.50
        assert abs(result.mid_price - 0.50) < 0.01
        # Buying at 0.52 (best ask), so avg_fill ~0.52
        assert result.avg_fill_price >= 0.52
        assert result.slippage_pct > 0

    def test_basic_sell_slippage(self):
        """Sell into a book with multiple bid levels."""
        book = make_orderbook(
            bids=[(0.48, 500), (0.47, 500)],
            asks=[(0.52, 500), (0.53, 500)],
        )
        result = calculate_slippage(book, position_size_usd=100, side="sell")

        assert result is not None
        assert result.side == "sell"
        assert result.avg_fill_price <= 0.48

    def test_thick_book_low_slippage(self):
        """Thick book should have low slippage for small orders."""
        book = make_orderbook(
            bids=[(0.49, 10000), (0.48, 10000)],
            asks=[(0.51, 10000), (0.52, 10000)],
        )
        result = calculate_slippage(book, position_size_usd=100, side="buy")

        assert result is not None
        # Small order in thick book: slippage should be minimal
        assert result.slippage_pct < 0.05

    def test_thin_book_high_slippage(self):
        """Thin book should have high slippage for larger orders."""
        book = make_orderbook(
            bids=[(0.49, 10), (0.40, 10)],
            asks=[(0.51, 10), (0.60, 10)],
        )
        result = calculate_slippage(book, position_size_usd=100, side="buy")

        assert result is not None
        # Large order relative to thin book: high slippage
        assert result.slippage_pct > 0.05

    def test_empty_asks_returns_none(self):
        """Empty ask side should return None for buy."""
        book = make_orderbook(
            bids=[(0.48, 500)],
            asks=[],
        )
        result = calculate_slippage(book, position_size_usd=100, side="buy")
        assert result is None

    def test_empty_book_returns_none(self):
        """Completely empty book should return None."""
        book = make_orderbook(bids=[], asks=[])
        result = calculate_slippage(book, position_size_usd=100, side="buy")
        assert result is None


class TestAnalyzeSpread:
    """Tests for full spread analysis."""

    def test_positive_net_edge(self):
        """Net edge should be positive when mispricing exceeds spread costs."""
        book = make_orderbook(
            bids=[(0.49, 5000), (0.48, 5000)],
            asks=[(0.51, 5000), (0.52, 5000)],
        )
        result = analyze_spread(book, "market-1", mispricing_magnitude=0.15)

        assert result.market_id == "market-1"
        assert result.bid_ask_spread is not None
        assert abs(result.bid_ask_spread - 0.02) < 0.001
        assert result.net_edge > 0
        assert result.is_tradeable is True

    def test_negative_net_edge(self):
        """Net edge should be negative when spread exceeds mispricing."""
        book = make_orderbook(
            bids=[(0.40, 100), (0.30, 100)],
            asks=[(0.60, 100), (0.70, 100)],
        )
        result = analyze_spread(book, "market-2", mispricing_magnitude=0.05)

        assert result.net_edge < 0
        assert result.is_tradeable is False
        assert any("negative" in n.lower() for n in result.analysis_notes)

    def test_empty_orderbook(self):
        """Empty orderbook should produce incomplete analysis."""
        book = make_orderbook(bids=[], asks=[])
        result = analyze_spread(book, "market-3", mispricing_magnitude=0.10)

        assert result.bid_ask_spread is None
        assert result.mid_price is None
        assert result.is_tradeable is False
        assert len(result.analysis_notes) > 0

    def test_default_position_sizes(self):
        """Should generate slippage estimates for default position sizes."""
        book = make_orderbook(
            bids=[(0.49, 5000)],
            asks=[(0.51, 5000)],
        )
        result = analyze_spread(book, "market-4", mispricing_magnitude=0.10)

        # Default sizes: $100, $500, $1000 x 2 sides = up to 6 estimates
        assert len(result.slippage_estimates) > 0

    def test_to_dict(self):
        """SpreadAnalysis should serialize to dict properly."""
        book = make_orderbook(
            bids=[(0.49, 5000)],
            asks=[(0.51, 5000)],
        )
        result = analyze_spread(book, "market-5", mispricing_magnitude=0.10)
        d = result.to_dict()

        assert "token_id" in d
        assert "net_edge" in d
        assert "is_tradeable" in d
        assert isinstance(d["slippage_estimates"], list)

    def test_very_thin_liquidity_not_tradeable(self):
        """Markets with very thin liquidity should be marked not tradeable."""
        book = make_orderbook(
            bids=[(0.49, 1)],  # Only $0.49 of liquidity
            asks=[(0.51, 1)],  # Only $0.51 of liquidity
        )
        result = analyze_spread(book, "market-6", mispricing_magnitude=0.20)

        assert result.is_tradeable is False
        assert any("thin" in n.lower() for n in result.analysis_notes)
