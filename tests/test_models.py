"""
Tests for data models.

These tests verify that data models work correctly.
"""

import pytest
from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import (
    Outcome,
    Market,
    Event,
)


class TestOutcome:
    """Tests for Outcome model."""
    
    def test_outcome_creation(self):
        """Test basic outcome creation."""
        outcome = Outcome(name="Yes", token_id="abc123", price=0.65)
        
        assert outcome.name == "Yes"
        assert outcome.token_id == "abc123"
        assert outcome.price == 0.65
    
    def test_price_normalization(self):
        """Test that percentage prices are normalized."""
        outcome = Outcome(name="Yes", token_id="abc", price=65)  # Given as percentage
        
        assert outcome.price == 0.65
    
    def test_price_clamping(self):
        """Test that invalid prices are clamped."""
        outcome = Outcome(name="Yes", token_id="abc", price=1.5)
        
        assert outcome.price == 1.0


class TestMarket:
    """Tests for Market model."""
    
    def test_market_creation(self):
        """Test basic market creation."""
        outcomes = [
            Outcome(name="Yes", token_id="yes1", price=0.60),
            Outcome(name="No", token_id="no1", price=0.40),
        ]
        
        market = Market(
            id="test-market-1",
            slug="test-market-1",
            question="Will this test pass?",
            description="A test market for unit testing.",
            outcomes=outcomes,
            category="testing",
            volume=10000,
            liquidity=5000,
        )
        
        assert market.id == "test-market-1"
        assert market.question == "Will this test pass?"
        assert len(market.outcomes) == 2
    
    def test_outcome_prices_property(self):
        """Test outcome_prices property."""
        outcomes = [
            Outcome(name="Yes", token_id="y", price=0.60),
            Outcome(name="No", token_id="n", price=0.40),
        ]
        
        market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=outcomes,
        )
        
        assert market.outcome_prices == {"Yes": 0.60, "No": 0.40}
    
    def test_token_ids_property(self):
        """Test token_ids property."""
        outcomes = [
            Outcome(name="Yes", token_id="token-yes", price=0.60),
            Outcome(name="No", token_id="token-no", price=0.40),
        ]
        
        market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=outcomes,
        )
        
        assert market.token_ids == ["token-yes", "token-no"]
    
    def test_days_to_expiry(self):
        """Test days_to_expiry calculation."""
        future_date = datetime.now() + timedelta(days=15)
        
        market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=[],
            end_date=future_date,
        )
        
        assert market.days_to_expiry == 15
    
    def test_days_to_expiry_none(self):
        """Test days_to_expiry when no end date."""
        market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=[],
            end_date=None,
        )
        
        assert market.days_to_expiry is None
    
    def test_is_binary(self):
        """Test is_binary property."""
        binary_market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=[
                Outcome(name="Yes", token_id="y", price=0.6),
                Outcome(name="No", token_id="n", price=0.4),
            ],
        )
        
        multi_market = Market(
            id="2", slug="2", question="Q", description="D",
            outcomes=[
                Outcome(name="A", token_id="a", price=0.3),
                Outcome(name="B", token_id="b", price=0.3),
                Outcome(name="C", token_id="c", price=0.4),
            ],
        )
        
        assert binary_market.is_binary is True
        assert multi_market.is_binary is False
    
    def test_get_outcome_by_name(self):
        """Test getting outcome by name."""
        market = Market(
            id="1", slug="1", question="Q", description="D",
            outcomes=[
                Outcome(name="Yes", token_id="y", price=0.6),
                Outcome(name="No", token_id="n", price=0.4),
            ],
        )
        
        yes = market.get_outcome_by_name("Yes")
        no = market.get_outcome_by_name("no")  # Case insensitive
        missing = market.get_outcome_by_name("Maybe")
        
        assert yes is not None
        assert yes.name == "Yes"
        assert no is not None
        assert no.name == "No"
        assert missing is None
    
    def test_to_dict(self):
        """Test serialization to dict."""
        market = Market(
            id="1", slug="test-slug", question="Test?", description="Desc",
            outcomes=[Outcome(name="Yes", token_id="y", price=0.6)],
            category="test",
            volume=10000,
        )
        
        d = market.to_dict()
        
        assert d["id"] == "1"
        assert d["slug"] == "test-slug"
        assert d["question"] == "Test?"
        assert len(d["outcomes"]) == 1


class TestEvent:
    """Tests for Event model."""
    
    def test_event_creation(self):
        """Test basic event creation."""
        market = Market(
            id="m1", slug="m1", question="Q", description="D",
            outcomes=[],
            volume=5000,
        )
        
        event = Event(
            id="e1",
            slug="test-event",
            title="Test Event",
            markets=[market],
        )
        
        assert event.id == "e1"
        assert event.market_count == 1
    
    def test_total_volume(self):
        """Test total volume calculation."""
        markets = [
            Market(id=str(i), slug=str(i), question="Q", description="D",
                   outcomes=[], volume=1000 * (i + 1))
            for i in range(3)
        ]
        
        event = Event(id="e1", slug="e1", title="E", markets=markets)
        
        assert event.total_volume == 6000  # 1000 + 2000 + 3000


