"""
Tests for LLM-based bias analysis refinement.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import Market, Outcome
from polymarket_agent.analysis.demographic_bias import analyze_bias_with_llm


def make_market(question: str, description: str = "") -> Market:
    return Market(
        id="test-1",
        slug="test-1",
        question=question,
        description=description,
        outcomes=[
            Outcome(name="Yes", token_id="y", price=0.55),
            Outcome(name="No", token_id="n", price=0.45),
        ],
        category="crypto",
        volume=10000,
        liquidity=5000,
        raw_data={
            "_detected_biases": ["crypto_optimism"],
            "_bias_direction": "overestimate",
            "_blind_spot_score": 30,
        },
    )


def make_mock_client(response_json: dict):
    """Create a mock LLM client that returns a specific JSON response."""
    client = AsyncMock()
    response = MagicMock()
    response.content = json.dumps(response_json)
    client.complete.return_value = response
    return client


class TestAnalyzeBiasWithLLM:
    """Tests for LLM-based bias direction correction."""

    @pytest.mark.asyncio
    async def test_corrects_crypto_pessimistic_question(self):
        """Should override direction for crypto-pessimistic question framing."""
        market = make_market("Will Bitcoin fall below $30K by end of 2025?")
        keyword_analysis = market.raw_data

        client = make_mock_client({
            "corrected_direction": "underestimate",
            "confidence": 0.8,
            "reasoning": "Users won't believe BTC will drop, so negative outcome is underpriced",
        })

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is not None
        assert result["_bias_direction"] == "underestimate"
        assert result["_bias_llm_refined"] is True
        assert result["_bias_llm_confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_confirms_crypto_bullish_question(self):
        """Should confirm direction for clearly bullish crypto question."""
        market = make_market("Will Bitcoin reach $200K by 2026?")
        keyword_analysis = market.raw_data

        client = make_mock_client({
            "corrected_direction": "overestimate",
            "confidence": 0.9,
            "reasoning": "Users are bullish, overestimating positive crypto outcomes",
        })

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is not None
        assert result["_bias_direction"] == "overestimate"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """Should return None on LLM failure (caller falls back to keyword analysis)."""
        market = make_market("Will Bitcoin crash?")
        keyword_analysis = market.raw_data

        client = AsyncMock()
        client.complete.side_effect = Exception("API error")

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is None

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self):
        """Should return None on invalid JSON response."""
        market = make_market("Will Bitcoin crash?")
        keyword_analysis = market.raw_data

        client = AsyncMock()
        response = MagicMock()
        response.content = "This is not valid JSON"
        client.complete.return_value = response

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_biases_returns_none(self):
        """Should return None if no detected biases."""
        market = make_market("Will it rain tomorrow?")
        keyword_analysis = {"_detected_biases": []}

        client = make_mock_client({"corrected_direction": "neutral"})

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_direction_uses_current(self):
        """Invalid corrected_direction should fall back to current direction."""
        market = make_market("Will Bitcoin moon?")
        keyword_analysis = market.raw_data

        client = make_mock_client({
            "corrected_direction": "invalid_value",
            "confidence": 0.5,
            "reasoning": "test",
        })

        result = await analyze_bias_with_llm(market, keyword_analysis, client)

        assert result is not None
        # Should use the original direction since "invalid_value" is rejected
        assert result["_bias_direction"] == "overestimate"

    @pytest.mark.asyncio
    async def test_prompt_contains_market_info(self):
        """Verify the LLM is called with relevant market info."""
        market = make_market(
            "Will Trump win the 2024 election?",
            description="Resolves YES if Trump wins",
        )
        market.raw_data = {
            "_detected_biases": ["political_right_bias"],
            "_bias_direction": "overestimate",
        }

        client = make_mock_client({
            "corrected_direction": "overestimate",
            "confidence": 0.85,
            "reasoning": "Right-leaning bias confirmed for pro-Trump question",
        })

        await analyze_bias_with_llm(market, market.raw_data, client)

        # Check that the client was called with a prompt containing the question
        call_kwargs = client.complete.call_args
        prompt = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
        assert "Trump" in prompt
        assert "political_right_bias" in prompt
