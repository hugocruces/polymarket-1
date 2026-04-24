"""Tests for the filtering module."""

from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import Market, Outcome
from polymarket_agent.filtering.filters import (
    MarketFilter,
    filter_by_expiry,
    filter_by_liquidity,
    filter_by_volume,
)


def create_test_market(
    id: str = "test-1",
    question: str = "Will this happen?",
    category: str = "politics",
    tags: list[str] | None = None,
    volume: float = 10000,
    liquidity: float = 5000,
    days_to_expiry: int = 30,
    outcomes: list[tuple[str, float]] | None = None,
    event_id: str | None = None,
) -> Market:
    """Helper to create test Market objects."""
    if tags is None:
        tags = [category]
    if outcomes is None:
        outcomes = [("Yes", 0.6), ("No", 0.4)]

    end_date = datetime.now() + timedelta(days=days_to_expiry)

    return Market(
        id=id,
        slug=id,
        question=question,
        description=f"Description for {question}",
        outcomes=[
            Outcome(name=name, token_id=f"token-{id}-{i}", price=price)
            for i, (name, price) in enumerate(outcomes)
        ],
        category=category,
        tags=tags,
        volume=volume,
        liquidity=liquidity,
        end_date=end_date,
        created_at=datetime.now() - timedelta(days=7),
        active=True,
        closed=False,
        event_id=event_id,
    )


class TestFilterByVolume:
    def test_min_volume(self):
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=10000),
            create_test_market(id="3", volume=500),
        ]
        result = filter_by_volume(markets, min_volume=10000)
        assert {m.id for m in result} == {"1", "2"}

    def test_max_volume(self):
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=10000),
        ]
        result = filter_by_volume(markets, max_volume=25000)
        assert {m.id for m in result} == {"2"}

    def test_volume_range(self):
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=20000),
            create_test_market(id="3", volume=5000),
        ]
        result = filter_by_volume(markets, min_volume=10000, max_volume=30000)
        assert {m.id for m in result} == {"2"}


class TestFilterByLiquidity:
    def test_min_liquidity(self):
        markets = [
            create_test_market(id="1", liquidity=25000),
            create_test_market(id="2", liquidity=5000),
            create_test_market(id="3", liquidity=500),
        ]
        result = filter_by_liquidity(markets, min_liquidity=5000)
        assert {m.id for m in result} == {"1", "2"}


class TestFilterByExpiry:
    def test_max_days_to_expiry(self):
        markets = [
            create_test_market(id="1", days_to_expiry=10),
            create_test_market(id="2", days_to_expiry=30),
            create_test_market(id="3", days_to_expiry=60),
        ]
        result = filter_by_expiry(markets, max_days=30)
        assert {m.id for m in result} == {"1", "2"}

    def test_max_days_zero_disables(self):
        """max_days <= 0 should return every market untouched."""
        markets = [
            create_test_market(id="1", days_to_expiry=10),
            create_test_market(id="2", days_to_expiry=10_000),
        ]
        assert len(filter_by_expiry(markets, max_days=0)) == 2

    def test_missing_end_date_kept(self):
        """Markets with no end_date should be kept."""
        m = create_test_market(id="1")
        m.end_date = None
        assert filter_by_expiry([m], max_days=30) == [m]


class TestMarketFilter:
    def test_combined_filters(self):
        mf = MarketFilter(
            min_volume=5000,
            min_liquidity=2000,
            max_days_to_expiry=45,
        )
        markets = [
            create_test_market(id="1", volume=10000, liquidity=5000, days_to_expiry=30),
            create_test_market(id="2", volume=1000, liquidity=5000, days_to_expiry=30),
            create_test_market(id="3", volume=10000, liquidity=500, days_to_expiry=30),
            create_test_market(id="4", volume=10000, liquidity=5000, days_to_expiry=60),
        ]
        result = mf.apply(markets)
        assert {m.id for m in result.markets} == {"1"}
        assert result.total_before == 4
        assert result.total_after == 1

    def test_callable_interface(self):
        mf = MarketFilter(min_volume=5000, min_liquidity=2000, max_days_to_expiry=90)
        markets = [
            create_test_market(id="1", volume=10000),
            create_test_market(id="2", volume=100),
        ]
        result = mf(markets)
        assert {m.id for m in result} == {"1"}

    def test_always_include_bypasses_filters(self):
        """Always-included markets skip volume/liquidity/expiry."""
        mf = MarketFilter(
            min_volume=1_000_000,
            min_liquidity=1_000_000,
            max_days_to_expiry=1,
            always_include_keywords=["UFO"],
        )
        markets = [
            create_test_market(
                id="keep",
                question="Will UFOs be real?",
                volume=1,
                liquidity=1,
                days_to_expiry=5000,
            ),
            create_test_market(
                id="drop",
                question="Will the Fed cut rates?",
                volume=1,
                liquidity=1,
            ),
        ]
        result = mf.apply(markets)
        assert {m.id for m in result.markets} == {"keep"}

    def test_always_include_matches_tags(self):
        mf = MarketFilter(
            min_volume=1_000_000,
            min_liquidity=1_000_000,
            max_days_to_expiry=1,
            always_include_keywords=["UFO"],
        )
        m = create_test_market(
            id="tagged",
            question="Plain question",
            tags=["politics", "UFO"],
            volume=1,
            liquidity=1,
            days_to_expiry=5000,
        )
        result = mf.apply([m])
        assert result.markets == [m]

    def test_filter_result_rate(self):
        mf = MarketFilter(min_volume=10000, min_liquidity=0, max_days_to_expiry=365)
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=1000),
        ]
        result = mf.apply(markets)
        assert result.filter_rate == 0.5

    def test_empty_input(self):
        mf = MarketFilter(min_volume=0, min_liquidity=0, max_days_to_expiry=0)
        result = mf.apply([])
        assert result.total_before == 0
        assert result.total_after == 0
        assert result.filter_rate == 0.0
