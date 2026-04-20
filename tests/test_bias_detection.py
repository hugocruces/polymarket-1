"""
Tests for bias detection data models.
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
    def test_political_value(self):
        assert BiasCategory.POLITICAL.value == "political"

    def test_progressive_social_value(self):
        assert BiasCategory.PROGRESSIVE_SOCIAL.value == "progressive_social"

    def test_crypto_optimism_value(self):
        assert BiasCategory.CRYPTO_OPTIMISM.value == "crypto_optimism"

    def test_all_categories_present(self):
        expected = {"POLITICAL", "PROGRESSIVE_SOCIAL", "CRYPTO_OPTIMISM"}
        assert {m.name for m in BiasCategory} == expected


class TestBiasClassification:
    def test_creation(self):
        c = BiasClassification(
            market_id="test-123",
            dominated_by_bias=True,
            categories=[BiasCategory.POLITICAL, BiasCategory.CRYPTO_OPTIMISM],
            bias_score=75,
            reasoning="Strong political bias.",
        )
        assert c.market_id == "test-123"
        assert c.dominated_by_bias is True
        assert len(c.categories) == 2
        assert c.bias_score == 75

    def test_empty_categories(self):
        c = BiasClassification(
            market_id="neutral",
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            reasoning="No bias.",
        )
        assert c.categories == []
        assert c.dominated_by_bias is False


class TestSystemPrompt:
    def test_contains_demographic_info(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "73% male" in SYSTEM_PROMPT
        assert "25-45" in SYSTEM_PROMPT
        assert "right-leaning" in SYSTEM_PROMPT

    def test_contains_bias_categories(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "Political Bias" in SYSTEM_PROMPT
        assert "Progressive Social Bias" in SYSTEM_PROMPT
        assert "Crypto Optimism" in SYSTEM_PROMPT

    def test_contains_score_calibration(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "0–20" in SYSTEM_PROMPT or "0-20" in SYSTEM_PROMPT
        assert "70–100" in SYSTEM_PROMPT or "70-100" in SYSTEM_PROMPT

    def test_contains_guardrail(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "Do NOT flag" in SYSTEM_PROMPT or "do not flag" in SYSTEM_PROMPT.lower()

    def test_does_not_ask_for_dominated_by_bias(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "dominated_by_bias" not in SYSTEM_PROMPT

    def test_does_not_mention_mispricing_direction(self):
        from polymarket_agent.bias_detection.classifier import SYSTEM_PROMPT
        assert "mispricing_direction" not in SYSTEM_PROMPT


class TestBuildUserPrompt:
    def _make_market(self, **kwargs):
        defaults = dict(
            id="test-123",
            slug="test-market",
            question="Will Bitcoin reach $100k?",
            description="Resolves YES if BTC closes above $100,000 on any day before end of 2025.",
            outcomes=[
                Outcome(name="Yes", token_id="t1", price=0.65),
                Outcome(name="No", token_id="t2", price=0.35),
            ],
        )
        defaults.update(kwargs)
        return Market(**defaults)

    def test_includes_question(self):
        from polymarket_agent.bias_detection.classifier import build_user_prompt
        prompt = build_user_prompt(self._make_market())
        assert "Will Bitcoin reach $100k?" in prompt

    def test_includes_description(self):
        from polymarket_agent.bias_detection.classifier import build_user_prompt
        prompt = build_user_prompt(self._make_market())
        assert "closes above $100,000" in prompt

    def test_includes_outcome_prices(self):
        from polymarket_agent.bias_detection.classifier import build_user_prompt
        prompt = build_user_prompt(self._make_market())
        assert "Yes" in prompt and "No" in prompt
        assert "0.65" in prompt or "65" in prompt

    def test_no_market_id(self):
        from polymarket_agent.bias_detection.classifier import build_user_prompt
        prompt = build_user_prompt(self._make_market())
        assert "test-123" not in prompt

    def test_missing_description_shows_na(self):
        from polymarket_agent.bias_detection.classifier import build_user_prompt
        market = self._make_market(description="")
        prompt = build_user_prompt(market)
        assert "N/A" in prompt


class TestParseClassificationResponse:
    def test_parses_valid_json(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({
            "categories": ["political", "crypto_optimism"],
            "bias_score": 75,
            "reasoning": "Political and crypto bias detected.",
        })
        c = parse_classification_response(response, "m-1")

        assert c.market_id == "m-1"
        assert c.dominated_by_bias is True  # 75 >= 50
        assert BiasCategory.POLITICAL in c.categories
        assert BiasCategory.CRYPTO_OPTIMISM in c.categories
        assert c.bias_score == 75

    def test_dominated_by_bias_derived_from_score(self):
        from polymarket_agent.bias_detection.classifier import (
            parse_classification_response,
            BIAS_SCORE_THRESHOLD,
        )
        below = parse_classification_response(
            json.dumps({"categories": [], "bias_score": BIAS_SCORE_THRESHOLD - 1, "reasoning": "x"}), "m"
        )
        assert below.dominated_by_bias is False

        at = parse_classification_response(
            json.dumps({"categories": [], "bias_score": BIAS_SCORE_THRESHOLD, "reasoning": "x"}), "m"
        )
        assert at.dominated_by_bias is True

    def test_no_bias_low_score(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({"categories": [], "bias_score": 10, "reasoning": "No bias."})
        c = parse_classification_response(response, "m-2")

        assert c.dominated_by_bias is False
        assert c.categories == []
        assert c.bias_score == 10

    def test_handles_invalid_json(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        c = parse_classification_response("not json", "m-3")
        assert c.dominated_by_bias is False
        assert c.categories == []
        assert c.bias_score == 0

    def test_handles_partial_json(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = json.dumps({"categories": ["political"]})
        c = parse_classification_response(response, "m-4")
        assert BiasCategory.POLITICAL in c.categories
        assert isinstance(c.bias_score, int)

    def test_handles_json_in_markdown(self):
        from polymarket_agent.bias_detection.classifier import parse_classification_response

        response = '```json\n{"categories": ["progressive_social"], "bias_score": 60, "reasoning": "test"}\n```'
        c = parse_classification_response(response, "m-5")
        assert c.dominated_by_bias is True
        assert BiasCategory.PROGRESSIVE_SOCIAL in c.categories


class TestClassifyMarket:
    @pytest.mark.asyncio
    async def test_classify_market_returns_classification(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from polymarket_agent.bias_detection.classifier import classify_market
        from polymarket_agent.llm_assessment.providers import LLMResponse

        market = Market(
            id="test-async",
            slug="test-async-slug",
            question="Will Bitcoin reach $150k by end of 2025?",
            description="Resolves YES if BTC hits $150k.",
            outcomes=[
                Outcome(name="Yes", token_id="t1", price=0.70),
                Outcome(name="No", token_id="t2", price=0.30),
            ],
        )
        mock_response = LLMResponse(
            content=json.dumps({
                "categories": ["crypto_optimism"],
                "bias_score": 80,
                "reasoning": "Crypto enthusiast bias likely inflating Yes.",
            }),
            model="claude-haiku-4-5",
            provider="Anthropic",
            input_tokens=100,
            output_tokens=50,
        )
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=mock_response)

        with patch("polymarket_agent.bias_detection.classifier.get_llm_client", return_value=mock_client):
            result = await classify_market(market)

        assert result.dominated_by_bias is True  # 80 >= 50
        assert BiasCategory.CRYPTO_OPTIMISM in result.categories
        assert result.bias_score == 80
        call_kwargs = mock_client.complete.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500
        assert call_kwargs["temperature"] == 0.1

    @pytest.mark.asyncio
    async def test_classify_market_uses_custom_model(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from polymarket_agent.bias_detection.classifier import classify_market
        from polymarket_agent.llm_assessment.providers import LLMResponse

        market = Market(
            id="test-model",
            slug="test-model-slug",
            question="Will Trump win 2028?",
            description="",
            outcomes=[
                Outcome(name="Yes", token_id="t1", price=0.55),
                Outcome(name="No", token_id="t2", price=0.45),
            ],
        )
        mock_client = MagicMock()
        mock_client.complete = AsyncMock(return_value=LLMResponse(
            content=json.dumps({"categories": ["political"], "bias_score": 70, "reasoning": "Political bias."}),
            model="claude-sonnet-4-6",
            provider="Anthropic",
        ))

        with patch("polymarket_agent.bias_detection.classifier.get_llm_client", return_value=mock_client) as mock_get:
            result = await classify_market(market, model="claude-sonnet-4-6")

        mock_get.assert_called_once_with("claude-sonnet-4-6")
        assert BiasCategory.POLITICAL in result.categories


class TestClassifiedMarket:
    def test_creation_and_access(self):
        market = Market(
            id="cm-1",
            slug="cm-slug",
            question="Will AI pass the Turing test by 2030?",
            description="",
            outcomes=[
                Outcome(name="Yes", token_id="t1", price=0.40),
                Outcome(name="No", token_id="t2", price=0.60),
            ],
            volume=500000.0,
            liquidity=75000.0,
        )
        classification = BiasClassification(
            market_id="cm-1",
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=65,
            reasoning="Tech-optimism bias.",
        )
        cm = ClassifiedMarket(market=market, classification=classification)
        assert cm.market.question == "Will AI pass the Turing test by 2030?"
        assert cm.classification.bias_score == 65
        assert cm.volume == 500000.0
        assert cm.liquidity == 75000.0
