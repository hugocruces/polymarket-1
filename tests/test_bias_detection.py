"""
Tests for bias detection data models.

These tests verify that bias detection models work correctly.
"""

import json

import pytest

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
)
from polymarket_agent.data_fetching.models import Market, Outcome


class TestBiasCategory:
    """Tests for BiasCategory enum."""

    def test_political_value(self):
        """Test POLITICAL has correct value."""
        assert BiasCategory.POLITICAL.value == "political"

    def test_progressive_social_value(self):
        """Test PROGRESSIVE_SOCIAL has correct value."""
        assert BiasCategory.PROGRESSIVE_SOCIAL.value == "progressive_social"

    def test_crypto_optimism_value(self):
        """Test CRYPTO_OPTIMISM has correct value."""
        assert BiasCategory.CRYPTO_OPTIMISM.value == "crypto_optimism"

    def test_all_categories_present(self):
        """Test all expected categories are present."""
        expected = {"POLITICAL", "PROGRESSIVE_SOCIAL", "CRYPTO_OPTIMISM"}
        actual = {member.name for member in BiasCategory}
        assert actual == expected


class TestBiasClassification:
    """Tests for BiasClassification dataclass."""

    def test_creation_with_all_fields(self):
        """Test BiasClassification can be created with all fields."""
        classification = BiasClassification(
            market_id="test-market-123",
            dominated_by_bias=True,
            categories=[BiasCategory.POLITICAL, BiasCategory.CRYPTO_OPTIMISM],
            bias_score=75,
            mispricing_direction="overpriced",
            european=False,
            spain=False,
            reasoning="This market shows strong political bias from US-centric demographics.",
        )

        assert classification.market_id == "test-market-123"
        assert classification.dominated_by_bias is True
        assert len(classification.categories) == 2
        assert BiasCategory.POLITICAL in classification.categories
        assert BiasCategory.CRYPTO_OPTIMISM in classification.categories
        assert classification.bias_score == 75
        assert classification.mispricing_direction == "overpriced"
        assert classification.european is False
        assert classification.spain is False
        assert "political bias" in classification.reasoning

    def test_creation_with_empty_categories(self):
        """Test BiasClassification works with empty categories list."""
        classification = BiasClassification(
            market_id="neutral-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            mispricing_direction="unclear",
            european=True,
            spain=False,
            reasoning="No significant bias detected.",
        )

        assert classification.market_id == "neutral-market"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 0
        assert classification.mispricing_direction == "unclear"
        assert classification.european is True
        assert classification.spain is False

    def test_underpriced_direction(self):
        """Test BiasClassification with underpriced direction."""
        classification = BiasClassification(
            market_id="crypto-market",
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=60,
            mispricing_direction="underpriced",
            european=False,
            spain=False,
            reasoning="Crypto optimism may lead to underpricing of negative outcomes.",
        )

        assert classification.mispricing_direction == "underpriced"

    def test_spain_implies_european(self):
        """Test classification with Spain set to True and European True."""
        classification = BiasClassification(
            market_id="spain-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=10,
            mispricing_direction="unclear",
            european=True,
            spain=True,
            reasoning="Market about Spanish politics.",
        )

        assert classification.spain is True
        assert classification.european is True

    def test_single_category(self):
        """Test BiasClassification with a single category."""
        classification = BiasClassification(
            market_id="progressive-market",
            dominated_by_bias=True,
            categories=[BiasCategory.PROGRESSIVE_SOCIAL],
            bias_score=80,
            mispricing_direction="overpriced",
            european=False,
            spain=False,
            reasoning="Progressive social bias detected.",
        )

        assert len(classification.categories) == 1
        assert classification.categories[0] == BiasCategory.PROGRESSIVE_SOCIAL


class TestBuildSystemPrompt:
    """Tests for build_system_prompt function."""

    def test_contains_demographic_info(self):
        """Test system prompt contains demographic information."""
        from polymarket_agent.bias_detection.classifier import build_system_prompt

        prompt = build_system_prompt()

        assert "73% male" in prompt
        assert "25-45" in prompt
        assert "right-leaning" in prompt
        assert "crypto" in prompt.lower()

    def test_contains_us_based_percentage(self):
        """Test system prompt contains US-based percentage."""
        from polymarket_agent.bias_detection.classifier import build_system_prompt

        prompt = build_system_prompt()

        assert "31% US-based" in prompt or "31%" in prompt

    def test_contains_bias_descriptions(self):
        """Test system prompt contains bias category descriptions."""
        from polymarket_agent.bias_detection.classifier import build_system_prompt

        prompt = build_system_prompt()

        assert "political" in prompt.lower()
        assert "progressive" in prompt.lower() or "social" in prompt.lower()
        assert "crypto" in prompt.lower()


class TestBuildUserPrompt:
    """Tests for build_user_prompt function."""

    def test_includes_market_question(self):
        """Test user prompt includes the market question."""
        from polymarket_agent.bias_detection.classifier import build_user_prompt

        market = Market(
            id="test-123",
            slug="test-market",
            question="Will Bitcoin reach $100k by end of 2025?",
            description="Test description",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.65),
                Outcome(name="No", token_id="token2", price=0.35),
            ],
        )

        prompt = build_user_prompt(market)

        assert "Will Bitcoin reach $100k by end of 2025?" in prompt

    def test_includes_outcome_prices(self):
        """Test user prompt includes outcome prices."""
        from polymarket_agent.bias_detection.classifier import build_user_prompt

        market = Market(
            id="test-456",
            slug="test-market-2",
            question="Will Trump win 2024?",
            description="Test description",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.52),
                Outcome(name="No", token_id="token2", price=0.48),
            ],
        )

        prompt = build_user_prompt(market)

        # Should contain outcome names and prices
        assert "Yes" in prompt
        assert "No" in prompt
        assert "0.52" in prompt or "52" in prompt
        assert "0.48" in prompt or "48" in prompt

    def test_includes_market_id(self):
        """Test user prompt includes market ID for reference."""
        from polymarket_agent.bias_detection.classifier import build_user_prompt

        market = Market(
            id="market-789",
            slug="example-slug",
            question="Test question?",
            description="Test description",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.50),
                Outcome(name="No", token_id="token2", price=0.50),
            ],
        )

        prompt = build_user_prompt(market)

        # Market ID or slug should be present for reference
        assert "market-789" in prompt or "example-slug" in prompt


class TestParseClassificationResponse:
    """Tests for parse_classification_response function."""

    def test_parses_valid_json_with_bias(self):
        """Test parsing valid JSON response with bias detected."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["political", "crypto_optimism"],
                "bias_score": 75,
                "mispricing_direction": "overpriced",
                "european": False,
                "spain": False,
                "reasoning": "Political and crypto bias detected.",
            }
        )

        classification = parse_classification_response(response, "market-123")

        assert classification.market_id == "market-123"
        assert classification.dominated_by_bias is True
        assert BiasCategory.POLITICAL in classification.categories
        assert BiasCategory.CRYPTO_OPTIMISM in classification.categories
        assert classification.bias_score == 75
        assert classification.mispricing_direction == "overpriced"
        assert classification.european is False
        assert classification.spain is False
        assert "Political and crypto bias detected" in classification.reasoning

    def test_parses_valid_json_no_bias(self):
        """Test parsing valid JSON response with no bias."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": False,
                "categories": [],
                "bias_score": 10,
                "mispricing_direction": "unclear",
                "european": True,
                "spain": False,
                "reasoning": "No significant demographic bias detected.",
            }
        )

        classification = parse_classification_response(response, "market-456")

        assert classification.market_id == "market-456"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 10
        assert classification.mispricing_direction == "unclear"
        assert classification.european is True

    def test_handles_invalid_json(self):
        """Test handling of invalid JSON returns default classification."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = "This is not valid JSON at all"

        classification = parse_classification_response(response, "market-789")

        # Should return a default/fallback classification
        assert classification.market_id == "market-789"
        assert classification.dominated_by_bias is False
        assert classification.categories == []

    def test_handles_partial_json(self):
        """Test handling of JSON with missing fields uses defaults."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["political"],
                # Missing other fields
            }
        )

        classification = parse_classification_response(response, "market-partial")

        assert classification.market_id == "market-partial"
        assert classification.dominated_by_bias is True
        assert BiasCategory.POLITICAL in classification.categories
        # Missing fields should have sensible defaults
        assert isinstance(classification.bias_score, int)
        assert isinstance(classification.mispricing_direction, str)

    def test_handles_json_in_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = """```json
{
    "dominated_by_bias": true,
    "categories": ["progressive_social"],
    "bias_score": 60,
    "mispricing_direction": "overpriced",
    "european": false,
    "spain": false,
    "reasoning": "Progressive social bias detected."
}
```"""

        classification = parse_classification_response(response, "market-md")

        assert classification.market_id == "market-md"
        assert classification.dominated_by_bias is True
        assert BiasCategory.PROGRESSIVE_SOCIAL in classification.categories
