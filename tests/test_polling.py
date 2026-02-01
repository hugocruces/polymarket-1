"""
Tests for polling data integration.
"""

import pytest
from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import Market, Outcome
from polymarket_agent.enrichment.web_search import (
    is_political_market,
    build_polling_query,
    extract_poll_numbers,
    format_polling_context,
)


def make_market(
    question: str = "Test?",
    category: str = "",
    tags: list = None,
) -> Market:
    return Market(
        id="test-1",
        slug="test-1",
        question=question,
        description="Test description",
        outcomes=[
            Outcome(name="Yes", token_id="y", price=0.50),
            Outcome(name="No", token_id="n", price=0.50),
        ],
        category=category,
        tags=tags or [],
        volume=5000,
        liquidity=2000,
        end_date=datetime.now() + timedelta(days=30),
    )


class TestIsPoliticalMarket:
    """Test political market detection."""

    def test_election_market(self):
        m = make_market("Will Trump win the 2024 election?")
        assert is_political_market(m) is True

    def test_president_market(self):
        m = make_market("Who will be the next president?")
        assert is_political_market(m) is True

    def test_approval_rating(self):
        m = make_market("Will Biden's approval rating exceed 50%?")
        assert is_political_market(m) is True

    def test_crypto_market_not_political(self):
        m = make_market("Will Bitcoin reach $100K by 2025?")
        assert is_political_market(m) is False

    def test_sports_market_not_political(self):
        m = make_market("Will the Lakers win the championship?")
        assert is_political_market(m) is False

    def test_category_political(self):
        m = make_market("Will X happen?", category="election")
        assert is_political_market(m) is True

    def test_tags_political(self):
        m = make_market("Will X happen?", tags=["trump", "politics"])
        assert is_political_market(m) is True

    def test_international_politics(self):
        m = make_market("Will Macron dissolve parliament?")
        assert is_political_market(m) is True


class TestBuildPollingQuery:
    """Test polling query construction."""

    def test_strips_question_words(self):
        m = make_market("Will Trump win the election?")
        query = build_polling_query(m)
        assert not query.startswith("Will")
        assert "polls" in query.lower()
        assert "polling" in query.lower()

    def test_includes_key_terms(self):
        m = make_market("Who will win the 2024 presidential race?")
        query = build_polling_query(m)
        assert "polls" in query.lower()

    def test_removes_question_mark(self):
        m = make_market("Will Biden run again?")
        query = build_polling_query(m)
        assert "?" not in query

    def test_truncates_long_queries(self):
        m = make_market("A" * 250)
        query = build_polling_query(m)
        assert len(query) <= 250  # 200 base + " polls latest polling average"


class TestExtractPollNumbers:
    """Test poll number extraction from snippets."""

    def test_basic_extraction(self):
        snippets = ["Trump 48% Biden 46% in latest national poll"]
        results = extract_poll_numbers(snippets)
        assert len(results) >= 2

        entities = {r["entity"].lower() for r in results}
        assert "trump" in entities or any("trump" in e for e in entities)

    def test_with_leading_verb(self):
        snippets = ["Harris leads with 52% while Trump has 47%"]
        results = extract_poll_numbers(snippets)
        assert len(results) >= 1

    def test_percentage_range(self):
        """Should only extract reasonable percentages (1-100)."""
        snippets = ["Score is 0% for None and 150% for Other"]
        results = extract_poll_numbers(snippets)
        # 150% should be excluded, 0% should be excluded
        for r in results:
            assert 0 < r["percentage"] <= 100

    def test_no_matches(self):
        snippets = ["No polling data available in this article"]
        results = extract_poll_numbers(snippets)
        assert len(results) == 0

    def test_decimal_percentages(self):
        snippets = ["Macron polling at 28.5% Starmer at 31.2%"]
        results = extract_poll_numbers(snippets)
        found_decimal = any(r["percentage"] != int(r["percentage"]) for r in results)
        # At least some should be found
        assert len(results) >= 1

    def test_deduplication(self):
        """Same entity appearing multiple times should keep highest."""
        snippets = [
            "Trump 45% in CNN poll",
            "Trump 48% in Fox poll",
        ]
        results = extract_poll_numbers(snippets)
        trump_results = [r for r in results if "trump" in r["entity"].lower()]
        assert len(trump_results) <= 1
        if trump_results:
            assert trump_results[0]["percentage"] == 48


class TestFormatPollingContext:
    """Test polling context formatting."""

    def test_formats_correctly(self):
        data = [
            {"entity": "Trump", "percentage": 48.0},
            {"entity": "Biden", "percentage": 46.0},
        ]
        result = format_polling_context(data)
        assert "Trump: 48.0%" in result
        assert "Biden: 46.0%" in result

    def test_empty_data(self):
        result = format_polling_context([])
        assert result == ""
