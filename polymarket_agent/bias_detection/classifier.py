"""
LLM-based bias classifier for Polymarket markets.

This module provides functions to build prompts for bias classification
and parse LLM responses into BiasClassification objects.
"""

import json
import re
from typing import Any

from polymarket_agent.bias_detection.models import BiasCategory, BiasClassification
from polymarket_agent.data_fetching.models import Market
from polymarket_agent.llm_assessment.providers import get_llm_client


SYSTEM_PROMPT = """You are an expert analyst specializing in prediction market bias detection.

Your task is to analyze Polymarket prediction markets for demographic bias that could lead to mispricing.

## Polymarket Demographics

The Polymarket user base has the following characteristics:
- ~73% male, ages 25-45
- ~31% US-based
- Right-leaning politically
- Crypto/tech enthusiasts

## Bias Categories

1. **Political Bias**: Markets may be affected by the right-leaning political views of the user base. This can lead to overpricing of conservative outcomes and underpricing of progressive outcomes.

2. **Progressive Social Bias**: The young, male, tech-oriented demographic may underweight progressive social movements or changes. Markets related to social issues, diversity initiatives, or progressive policies may be mispriced.

3. **Crypto Optimism**: As crypto/tech enthusiasts, users may be overly optimistic about cryptocurrency adoption, prices, and blockchain technology outcomes.

## Your Task

Analyze markets to identify if they are susceptible to these biases.

Respond with a JSON object containing:
- dominated_by_bias: boolean indicating if bias significantly affects this market
- categories: list of bias categories (political, progressive_social, crypto_optimism)
- bias_score: 0-100 indicating strength of bias effect
- reasoning: explanation of your analysis
"""

USER_PROMPT_TEMPLATE = """Analyze the following Polymarket market for demographic bias:

## Market Information

**Market ID**: {market_id}
**Question**: {question}

**Current Prices**:
{outcomes}

## Instructions

Based on the Polymarket demographics and bias categories described in the system prompt, analyze whether this market is susceptible to demographic biases that could lead to mispricing.

Provide your analysis as a JSON object with the following fields:
- dominated_by_bias (boolean)
- categories (list of strings: "political", "progressive_social", "crypto_optimism")
- bias_score (integer 0-100)
- reasoning (string)
"""


def build_user_prompt(market: Market) -> str:
    """Build a user prompt for bias classification from market data."""
    outcomes_text = "\n".join(
        f"- {outcome.name}: {outcome.price:.2f} ({outcome.price * 100:.0f}%)"
        for outcome in market.outcomes
    )

    return USER_PROMPT_TEMPLATE.format(
        market_id=market.id,
        question=market.question,
        outcomes=outcomes_text,
    )


def _extract_json_from_response(response: str) -> str:
    """Extract JSON from a response that may contain markdown code blocks."""
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response


def _parse_categories(categories: list[Any]) -> list[BiasCategory]:
    """Parse category strings into BiasCategory enum values."""
    result = []
    for cat in categories:
        if isinstance(cat, str):
            cat_lower = cat.lower().strip()
            if cat_lower == "political":
                result.append(BiasCategory.POLITICAL)
            elif cat_lower in ("progressive_social", "progressive social"):
                result.append(BiasCategory.PROGRESSIVE_SOCIAL)
            elif cat_lower in ("crypto_optimism", "crypto optimism"):
                result.append(BiasCategory.CRYPTO_OPTIMISM)
    return result


def parse_classification_response(response: str, market_id: str) -> BiasClassification:
    """Parse an LLM response into a BiasClassification object.

    Returns a default no-bias classification if parsing fails.
    """
    try:
        json_str = _extract_json_from_response(response)
        data = json.loads(json_str)

        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=bool(data.get("dominated_by_bias", False)),
            categories=_parse_categories(data.get("categories", [])),
            bias_score=int(data.get("bias_score", 0)),
            reasoning=str(data.get("reasoning", "Unable to determine bias.")),
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            reasoning="Failed to parse LLM response.",
        )


async def classify_market(
    market: Market,
    model: str = "claude-haiku-4-5",
) -> BiasClassification:
    """Classify a market for demographic bias potential using LLM."""
    client = get_llm_client(model)

    response = await client.complete(
        prompt=build_user_prompt(market),
        system_prompt=SYSTEM_PROMPT,
        max_tokens=500,
        temperature=0.1,
    )

    return parse_classification_response(response.content, market.id)
