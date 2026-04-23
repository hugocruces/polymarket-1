"""
Tests for bias detection data models.

These tests verify that bias detection models work correctly.
"""

import json

import pytest

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
    ClassificationError,
    ClassifiedMarket,
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
        expected = {"POLITICAL", "PROGRESSIVE_SOCIAL", "CRYPTO_OPTIMISM", "ALWAYS_MONITORED"}
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
            european=True,
            spain=False,
            reasoning="No significant bias detected.",
        )

        assert classification.market_id == "neutral-market"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 0
        assert classification.european is True
        assert classification.spain is False

    def test_spain_implies_european(self):
        """Test classification with Spain set to True and European True."""
        classification = BiasClassification(
            market_id="spain-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=10,
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
        assert classification.european is True

    def test_invalid_json_raises(self):
        """Invalid JSON should raise ClassificationError, not silently default."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = "This is not valid JSON at all"

        with pytest.raises(ClassificationError):
            parse_classification_response(response, "market-789")

    def test_partial_json_raises(self):
        """Missing required fields should raise ClassificationError."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["political"],
                # Missing bias_score, european, spain, reasoning
            }
        )

        with pytest.raises(ClassificationError):
            parse_classification_response(response, "market-partial")

    def test_unknown_category_raises(self):
        """Unknown category values should raise ClassificationError."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["made_up_category"],
                "bias_score": 50,
                "european": False,
                "spain": False,
                "reasoning": "n/a",
            }
        )

        with pytest.raises(ClassificationError):
            parse_classification_response(response, "market-bad-cat")

    def test_out_of_range_score_raises(self):
        """bias_score outside 0-100 should raise ClassificationError."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["political"],
                "bias_score": 150,
                "european": False,
                "spain": False,
                "reasoning": "n/a",
            }
        )

        with pytest.raises(ClassificationError):
            parse_classification_response(response, "market-bad-score")

    def test_ignores_extra_direction_field(self):
        """LLM returning a direction field shouldn't break parsing — it's ignored."""
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
        )

        response = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["political"],
                "bias_score": 60,
                "mispricing_direction": "overpriced",  # Extra — model must ignore.
                "european": False,
                "spain": False,
                "reasoning": "n/a",
            }
        )

        classification = parse_classification_response(response, "market-extra")
        assert classification.bias_score == 60
        assert not hasattr(classification, "mispricing_direction")

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
    "european": false,
    "spain": false,
    "reasoning": "Progressive social bias detected."
}
```"""

        classification = parse_classification_response(response, "market-md")

        assert classification.market_id == "market-md"
        assert classification.dominated_by_bias is True
        assert BiasCategory.PROGRESSIVE_SOCIAL in classification.categories


class TestClassifyMarket:
    """Tests for classify_market async function."""

    @pytest.mark.asyncio
    async def test_classify_market_returns_classification(self):
        """Test classify_market calls LLM and returns BiasClassification."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polymarket_agent.bias_detection.classifier import classify_market
        from polymarket_agent.llm_assessment.providers import LLMResponse

        # Create a mock market
        market = Market(
            id="test-market-async",
            slug="test-async-slug",
            question="Will Bitcoin reach $150k by end of 2025?",
            description="Test description",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.70),
                Outcome(name="No", token_id="token2", price=0.30),
            ],
        )

        # Create mock LLM response
        mock_response_content = json.dumps(
            {
                "dominated_by_bias": True,
                "categories": ["crypto_optimism"],
                "bias_score": 80,
                "european": False,
                "spain": False,
                "reasoning": "Crypto enthusiast bias likely overpricing Yes outcome.",
            }
        )
        mock_response = LLMResponse(
            content=mock_response_content,
            model="claude-haiku-4-5",
            provider="Anthropic",
            input_tokens=100,
            output_tokens=50,
        )

        # Mock the LLM client
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        with patch(
            "polymarket_agent.bias_detection.classifier.get_llm_client",
            return_value=mock_client,
        ):
            result = await classify_market(market)

        # Verify the result
        assert isinstance(result, BiasClassification)
        assert result.market_id == "test-market-async"
        assert result.dominated_by_bias is True
        assert BiasCategory.CRYPTO_OPTIMISM in result.categories
        assert result.bias_score == 80
        assert "Crypto enthusiast bias" in result.reasoning

        # Verify the LLM client was called correctly
        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert "max_tokens" in call_kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_classify_market_uses_custom_model(self):
        """Test classify_market accepts custom model parameter."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from polymarket_agent.bias_detection.classifier import classify_market
        from polymarket_agent.llm_assessment.providers import LLMResponse

        market = Market(
            id="test-market-model",
            slug="test-model-slug",
            question="Will Trump win 2028?",
            description="Test description",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.55),
                Outcome(name="No", token_id="token2", price=0.45),
            ],
        )

        mock_response = LLMResponse(
            content=json.dumps(
                {
                    "dominated_by_bias": True,
                    "categories": ["political"],
                    "bias_score": 70,
                    "european": False,
                    "spain": False,
                    "reasoning": "Political bias detected.",
                }
            ),
            model="claude-sonnet-4-5",
            provider="Anthropic",
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        with patch(
            "polymarket_agent.bias_detection.classifier.get_llm_client",
            return_value=mock_client,
        ) as mock_get_client:
            result = await classify_market(market, model="claude-sonnet-4-5")

        # Verify the custom model was passed to get_llm_client
        mock_get_client.assert_called_once_with("claude-sonnet-4-5")
        assert result.dominated_by_bias is True
        assert BiasCategory.POLITICAL in result.categories


class TestClassifiedMarket:
    """Tests for ClassifiedMarket dataclass."""

    def test_creation_and_access(self):
        """Test ClassifiedMarket can be created and fields accessed."""
        # Create a market
        market = Market(
            id="classified-test-123",
            slug="classified-test-slug",
            question="Will AI pass the Turing test by 2030?",
            description="Test description for classified market",
            outcomes=[
                Outcome(name="Yes", token_id="token1", price=0.40),
                Outcome(name="No", token_id="token2", price=0.60),
            ],
            volume=500000.0,
            liquidity=75000.0,
        )

        # Create a bias classification
        classification = BiasClassification(
            market_id="classified-test-123",
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=65,
            european=False,
            spain=False,
            reasoning="Tech-optimism bias may affect this market.",
        )

        # Create ClassifiedMarket
        classified_market = ClassifiedMarket(
            market=market,
            classification=classification,
        )

        # Verify market.question is accessible
        assert classified_market.market.question == "Will AI pass the Turing test by 2030?"

        # Verify classification.bias_score is accessible
        assert classified_market.classification.bias_score == 65

        # Verify volume property works
        assert classified_market.volume == 500000.0

        # Verify liquidity property works
        assert classified_market.liquidity == 75000.0
