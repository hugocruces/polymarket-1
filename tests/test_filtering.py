"""
Tests for the filtering module.

These tests verify that market filters work correctly.
"""

import pytest
from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import Market, Outcome
from polymarket_agent.filtering.filters import (
    MarketFilter,
    filter_by_category,
    filter_by_keywords,
    filter_by_volume,
    filter_by_liquidity,
    filter_by_expiry,
    filter_by_geography,
    filter_by_outcome_count,
    filter_by_price_range,
)
from polymarket_agent.config import FilterConfig


def create_test_market(
    id: str = "test-1",
    question: str = "Will this happen?",
    category: str = "politics",
    tags: list[str] = None,
    volume: float = 10000,
    liquidity: float = 5000,
    days_to_expiry: int = 30,
    outcomes: list[tuple[str, float]] = None,
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
    )


class TestFilterByCategory:
    """Tests for category filtering."""
    
    def test_single_category_match(self):
        """Test filtering by a single category."""
        markets = [
            create_test_market(id="1", category="politics"),
            create_test_market(id="2", category="crypto"),
            create_test_market(id="3", category="sports"),
        ]
        
        result = filter_by_category(markets, ["politics"])
        
        assert len(result) == 1
        assert result[0].id == "1"
    
    def test_multiple_categories(self):
        """Test filtering by multiple categories."""
        markets = [
            create_test_market(id="1", category="politics"),
            create_test_market(id="2", category="crypto"),
            create_test_market(id="3", category="sports"),
        ]
        
        result = filter_by_category(markets, ["politics", "crypto"])
        
        assert len(result) == 2
        assert {m.id for m in result} == {"1", "2"}
    
    def test_case_insensitive(self):
        """Test that category matching is case-insensitive."""
        markets = [
            create_test_market(id="1", category="Politics"),
            create_test_market(id="2", category="CRYPTO"),
        ]
        
        result = filter_by_category(markets, ["politics", "crypto"])
        
        assert len(result) == 2
    
    def test_tag_matching(self):
        """Test filtering matches tags as well as category."""
        markets = [
            create_test_market(id="1", category="general", tags=["politics", "us"]),
            create_test_market(id="2", category="crypto"),
        ]
        
        result = filter_by_category(markets, ["politics"])
        
        assert len(result) == 1
        assert result[0].id == "1"
    
    def test_empty_categories(self):
        """Test that empty categories list returns all markets."""
        markets = [
            create_test_market(id="1", category="politics"),
            create_test_market(id="2", category="crypto"),
        ]
        
        result = filter_by_category(markets, [])
        
        assert len(result) == 2


class TestFilterByKeywords:
    """Tests for keyword filtering."""
    
    def test_keyword_in_question(self):
        """Test matching keywords in question."""
        markets = [
            create_test_market(id="1", question="Will Bitcoin reach $100k?"),
            create_test_market(id="2", question="Will Ethereum pass Bitcoin?"),
            create_test_market(id="3", question="Will Democrats win?"),
        ]
        
        result = filter_by_keywords(markets, ["bitcoin"], include=True)
        
        assert len(result) == 2
    
    def test_exclude_keywords(self):
        """Test excluding markets with certain keywords."""
        markets = [
            create_test_market(id="1", question="Will Bitcoin reach $100k?"),
            create_test_market(id="2", question="Will Democrats win?"),
        ]
        
        result = filter_by_keywords(markets, ["bitcoin"], include=False)
        
        assert len(result) == 1
        assert result[0].id == "2"
    
    def test_case_insensitive(self):
        """Test case-insensitive keyword matching."""
        markets = [
            create_test_market(id="1", question="Will BITCOIN moon?"),
        ]
        
        result = filter_by_keywords(markets, ["bitcoin"], include=True)
        
        assert len(result) == 1


class TestFilterByVolume:
    """Tests for volume filtering."""
    
    def test_min_volume(self):
        """Test minimum volume filter."""
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=10000),
            create_test_market(id="3", volume=500),
        ]
        
        result = filter_by_volume(markets, min_volume=10000)
        
        assert len(result) == 2
        assert {m.id for m in result} == {"1", "2"}
    
    def test_max_volume(self):
        """Test maximum volume filter."""
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=10000),
        ]
        
        result = filter_by_volume(markets, max_volume=25000)
        
        assert len(result) == 1
        assert result[0].id == "2"
    
    def test_volume_range(self):
        """Test min and max volume together."""
        markets = [
            create_test_market(id="1", volume=50000),
            create_test_market(id="2", volume=20000),
            create_test_market(id="3", volume=5000),
        ]
        
        result = filter_by_volume(markets, min_volume=10000, max_volume=30000)
        
        assert len(result) == 1
        assert result[0].id == "2"


class TestFilterByLiquidity:
    """Tests for liquidity filtering."""
    
    def test_min_liquidity(self):
        """Test minimum liquidity filter."""
        markets = [
            create_test_market(id="1", liquidity=25000),
            create_test_market(id="2", liquidity=5000),
            create_test_market(id="3", liquidity=500),
        ]
        
        result = filter_by_liquidity(markets, min_liquidity=5000)
        
        assert len(result) == 2


class TestFilterByExpiry:
    """Tests for expiry filtering."""
    
    def test_max_days_to_expiry(self):
        """Test maximum days to expiry."""
        markets = [
            create_test_market(id="1", days_to_expiry=10),
            create_test_market(id="2", days_to_expiry=30),
            create_test_market(id="3", days_to_expiry=60),
        ]
        
        result = filter_by_expiry(markets, max_days=30)
        
        assert len(result) == 2
        assert {m.id for m in result} == {"1", "2"}
    
    def test_min_days_to_expiry(self):
        """Test minimum days to expiry."""
        markets = [
            create_test_market(id="1", days_to_expiry=5),
            create_test_market(id="2", days_to_expiry=30),
        ]
        
        result = filter_by_expiry(markets, min_days=10)
        
        assert len(result) == 1
        assert result[0].id == "2"


class TestFilterByGeography:
    """Tests for geographic filtering."""
    
    def test_us_markets(self):
        """Test filtering for US-related markets."""
        markets = [
            create_test_market(id="1", question="Will Biden win reelection?"),
            create_test_market(id="2", question="Will UK pass new law?"),
            create_test_market(id="3", question="Will Trump return?"),
        ]
        
        result = filter_by_geography(markets, ["US"])
        
        assert len(result) == 2
        assert {m.id for m in result} == {"1", "3"}
    
    def test_crypto_markets(self):
        """Test filtering for crypto-related markets."""
        markets = [
            create_test_market(id="1", question="Will Bitcoin reach $100k?"),
            create_test_market(id="2", question="Will Democrats win?"),
        ]
        
        result = filter_by_geography(markets, ["CRYPTO"])
        
        assert len(result) == 1
        assert result[0].id == "1"


class TestFilterByOutcomeCount:
    """Tests for outcome count filtering."""
    
    def test_min_outcomes(self):
        """Test minimum outcome count."""
        markets = [
            create_test_market(id="1", outcomes=[("Yes", 0.5), ("No", 0.5)]),
            create_test_market(id="2", outcomes=[("A", 0.3), ("B", 0.3), ("C", 0.4)]),
        ]
        
        result = filter_by_outcome_count(markets, min_outcomes=3)
        
        assert len(result) == 1
        assert result[0].id == "2"
    
    def test_max_outcomes(self):
        """Test maximum outcome count."""
        markets = [
            create_test_market(id="1", outcomes=[("Yes", 0.5), ("No", 0.5)]),
            create_test_market(id="2", outcomes=[("A", 0.25)] * 4),
        ]
        
        result = filter_by_outcome_count(markets, max_outcomes=2)
        
        assert len(result) == 1
        assert result[0].id == "1"


class TestFilterByPriceRange:
    """Tests for price range filtering."""
    
    def test_price_range(self):
        """Test filtering by outcome price range."""
        markets = [
            create_test_market(id="1", outcomes=[("Yes", 0.75), ("No", 0.25)]),
            create_test_market(id="2", outcomes=[("Yes", 0.30), ("No", 0.70)]),
            create_test_market(id="3", outcomes=[("Yes", 0.50), ("No", 0.50)]),
        ]
        
        # Find markets with Yes between 25% and 40%
        result = filter_by_price_range(markets, 0.25, 0.40, outcome_name="Yes")
        
        assert len(result) == 1
        assert result[0].id == "2"


class TestMarketFilter:
    """Tests for the composite MarketFilter class."""
    
    def test_combined_filters(self):
        """Test combining multiple filters."""
        config = FilterConfig(
            categories=["politics"],
            min_volume=5000,
            max_days_to_expiry=45,
        )
        
        markets = [
            create_test_market(id="1", category="politics", volume=10000, days_to_expiry=30),
            create_test_market(id="2", category="politics", volume=1000, days_to_expiry=30),  # Low volume
            create_test_market(id="3", category="crypto", volume=10000, days_to_expiry=30),  # Wrong category
            create_test_market(id="4", category="politics", volume=10000, days_to_expiry=60),  # Too far
        ]
        
        market_filter = MarketFilter(config)
        result = market_filter.apply(markets)
        
        assert len(result.markets) == 1
        assert result.markets[0].id == "1"
        assert result.total_before == 4
        assert result.total_after == 1
    
    def test_callable_interface(self):
        """Test using filter as a callable."""
        config = FilterConfig(categories=["politics"])
        
        markets = [
            create_test_market(id="1", category="politics"),
            create_test_market(id="2", category="crypto"),
        ]
        
        market_filter = MarketFilter(config)
        result = market_filter(markets)  # Use as callable
        
        assert len(result) == 1
