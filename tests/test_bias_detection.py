"""
Tests for bias detection data models.

These tests verify that bias detection models work correctly.
"""

import json

import pytest

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
    ClassifiedMarket,
)
from polymarket_agent.data_fetching.models import Market, Outcome


class TestBiasCategory:
    """Tests for BiasCategory enum."""

    def test_political_value(self):
        assert BiasCategory.POLITICAL.value == "political"

    def test_progressive_social_value(self):
        assert BiasCategory.PROGRESSIVE_SOCIAL.value == "progressive_social"

    def test_crypto_optimism_value(self):
        assert BiasCategory.CRYPTO_OPTIMISM.value == "crypto_optimism"

    def test_all_categories_present(self):
        expected = {"POLITICAL", "PROGRESSIVE_SOCIAL", "CRYPTO_OPTIMISM"}
        actual = {member.name for member in BiasCategory}
        assert actual == expected


class TestBiasClassification:
    """Tests for BiasClassification dataclass."""

    def test_creation_with_all_fields(self):
        classification = BiasClassification(
            market_id="test-market-123",
            dominated_by_bias=True,
            categories=[BiasCategory.POLITICAL, BiasCategory.CRYPTO_OPTIMISM],
            bias_score=75,
            reasoning="This market shows strong political bias from US-centric demographics.",
        )

        assert classification.market_id == "test-market-123"
        assert classification.dominated_by_bias is True
        assert len(classification.categories) == 2
        assert BiasCategory.POLITICAL in classification.categories
        assert BiasCategory.CRYPTO_OPTIMISM in classification.categories
        assert classification.bias_score == 75
        assert "political bias" in classification.reasoning

    def test_creation_with_empty_categories(self):
        classification = BiasClassification(
            market_id="neutral-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            reasoning="No significant bias detected.",
        )

        assert classification.market_id == "neutral-market"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 0

    def test_single_category(self):
        classification = BiasClassification(
            market_id="progressive-market",
            dominated_by_bias=True,
            categories=[BiasCategory.PROGRESSIVE_SOCIAL],
            bias_score=80,
            reasoning="Progressive social bias detected.",
        )

        assert len(classification.categories) == 1
        assert classification.categories[0] == BiasCategory.PROGRESSIVE_SOCIAL


class TestSystemPrompt:
    """Tests for SYSTEM_PROMPT content."""

    def test_contains_demographic_info(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT

        assert "73% male" in SYSTEM_PROMPT
        assert "25-45" in SYSTEM_PROMPT
        assert "right-leaning" in SYSTEM_PROMPT
        assert "crypto" in SYSTEM_PROMPT.lower()

    def test_contains_us_based_percentage(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT

        assert "31%" in SYSTEM_PROMPT

    def test_contains_bias_descriptions(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT

        assert "political" in SYSTEM_PROMPT.lower()
        assert "progressive" in SYSTEM_PROMPT.lower() or "social" in SYSTEM_PROMPT.lower()
        assert "crypto" in SYSTEM_PROMPT.lower()


class TestBuildUserPrompt:
    """Tests for build_user_prompt function."""

    def test_includes_market_question(self):
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

        assert "Yes" in prompt
        assert "No" in prompt
        assert "0.52" in prompt or "52" in prompt
        assert "0.48" in prompt or "48" in prompt

    def test_includes_market_id(self):
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

        assert "market-789" in prompt or "example-slug" in prompt


class TestParseClassificationResponse:
    """Tests for parse_classification_response function."""

    def test_parses_valid_json_with_bias(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({
            "dominated_by_bias": True,
            "categories": ["political", "crypto_optimism"],
            "bias_score": 75,
            "reasoning": "Political and crypto bias detected.",
        })

        classification = parse_classification_response(response, "market-123")

        assert classification.market_id == "market-123"
        assert classification.dominated_by_bias is True
        assert BiasCategory.POLITICAL in classification.categories
        assert BiasCategory.CRYPTO_OPTIMISM in classification.categories
        assert classification.bias_score == 75
        assert "Political and crypto bias detected" in classification.reasoning

    def test_parses_valid_json_no_bias(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({
            "dominated_by_bias": False,
            "categories": [],
            "bias_score": 10,
            "reasoning": "No significant demographic bias detected.",
        })

        classification = parse_classification_response(response, "market-456")

        assert classification.market_id == "market-456"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 10

    def test_handles_invalid_json(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        classification = parse_classification_response("This is not valid JSON at all", "market-789")

        assert classification.market_id == "market-789"
        assert classification.dominated_by_bias is False
        assert classification.categories == []

    def test_handles_partial_json(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({
            "dominated_by_bias": True,
            "categories": ["political"],
        })

        classification = parse_classification_response(response, "market-partial")

        assert classification.market_id == "market-partial"
        assert classification.dominated_by_bias is True
        assert BiasCategory.POLITICAL in classification.categories
        assert isinstance(classification.bias_score, int)

    def test_handles_json_in_markdown(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = """```json
{
    "dominated_by_bias": true,
    "categories": ["progressive_social"],
    "bias_score": 60,
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
        from unittest.mock import AsyncMock, MagicMock, patch

        from polymarket_agent.bias_detection.classifier import classify_market
        from polymarket_agent.llm_assessment.providers import LLMResponse

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

        mock_response_content = json.dumps({
            "dominated_by_bias": True,
            "categories": ["crypto_optimism"],
            "bias_score": 80,
            "reasoning": "Crypto enthusiast bias likely overpricing Yes outcome.",
        })
        mock_response = LLMResponse(
            content=mock_response_content,
            model="claude-haiku-4-5",
            provider="Anthropic",
            input_tokens=100,
            output_tokens=50,
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        with patch(
            "polymarket_agent.bias_detection.classifier.get_llm_client",
            return_value=mock_client,
        ):
            result = await classify_market(market)

        assert isinstance(result, BiasClassification)
        assert result.market_id == "test-market-async"
        assert result.dominated_by_bias is True
        assert BiasCategory.CRYPTO_OPTIMISM in result.categories
        assert result.bias_score == 80
        assert "Crypto enthusiast bias" in result.reasoning

        mock_client.complete.assert_called_once()
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_classify_market_uses_custom_model(self):
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
            content=json.dumps({
                "dominated_by_bias": True,
                "categories": ["political"],
                "bias_score": 70,
                "reasoning": "Political bias detected.",
            }),
            model="claude-sonnet-4-6",
            provider="Anthropic",
        )

        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        with patch(
            "polymarket_agent.bias_detection.classifier.get_llm_client",
            return_value=mock_client,
        ) as mock_get_client:
            result = await classify_market(market, model="claude-sonnet-4-6")

        mock_get_client.assert_called_once_with("claude-sonnet-4-6")
        assert result.dominated_by_bias is True
        assert BiasCategory.POLITICAL in result.categories


class TestClassifiedMarket:
    """Tests for ClassifiedMarket dataclass."""

    def test_creation_and_access(self):
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

        classification = BiasClassification(
            market_id="classified-test-123",
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=65,
            reasoning="Tech-optimism bias may affect this market.",
        )

        classified_market = ClassifiedMarket(market=market, classification=classification)

        assert classified_market.market.question == "Will AI pass the Turing test by 2030?"
        assert classified_market.classification.bias_score == 65
        assert classified_market.volume == 500000.0
        assert classified_market.liquidity == 75000.0
