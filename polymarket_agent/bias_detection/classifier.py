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

Analyze markets to identify if they are susceptible to these biases and estimate the direction of potential mispricing.

Respond with a JSON object containing:
- dominated_by_bias: boolean indicating if bias significantly affects this market
- categories: list of bias categories (political, progressive_social, crypto_optimism)
- bias_score: 0-100 indicating strength of bias effect
- mispricing_direction: "overpriced", "underpriced", or "unclear"
- european: boolean indicating if this is a European-focused topic
- spain: boolean indicating if this is Spain-specific
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
- mispricing_direction ("overpriced", "underpriced", or "unclear")
- european (boolean)
- spain (boolean)
- reasoning (string)
"""


def build_system_prompt() -> str:
    """Return the system prompt for bias classification.

    Returns:
        The system prompt string containing demographic information
        and bias category descriptions.
    """
    return SYSTEM_PROMPT


def build_user_prompt(market: Market) -> str:
    """Build a user prompt for bias classification from market data.

    Args:
        market: The Market object to analyze.

    Returns:
        A formatted user prompt string with market details.
    """
    # Format outcomes with names and prices
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
    """Extract JSON from a response that may contain markdown code blocks.

    Args:
        response: The raw response string.

    Returns:
        The extracted JSON string.
    """
    # Try to find JSON in markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # Try to find raw JSON object
    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response


def _parse_categories(categories: list[Any]) -> list[BiasCategory]:
    """Parse category strings into BiasCategory enum values.

    Args:
        categories: List of category strings.

    Returns:
        List of BiasCategory enum values.
    """
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

    Args:
        response: The raw LLM response string (may contain JSON or markdown).
        market_id: The market ID to associate with the classification.

    Returns:
        A BiasClassification object. Returns a default classification
        with no bias detected if parsing fails.
    """
    try:
        json_str = _extract_json_from_response(response)
        data = json.loads(json_str)

        # Parse categories
        raw_categories = data.get("categories", [])
        categories = _parse_categories(raw_categories)

        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=bool(data.get("dominated_by_bias", False)),
            categories=categories,
            bias_score=int(data.get("bias_score", 0)),
            mispricing_direction=str(data.get("mispricing_direction", "unclear")),
            european=bool(data.get("european", False)),
            spain=bool(data.get("spain", False)),
            reasoning=str(data.get("reasoning", "Unable to determine bias.")),
        )

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Return default classification if parsing fails
        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            mispricing_direction="unclear",
            european=False,
            spain=False,
            reasoning="Failed to parse LLM response.",
        )


async def classify_market(
    market: Market,
    model: str = "claude-haiku-4-5",
) -> BiasClassification:
    """
    Classify a market for demographic bias potential using LLM.

    This function sends the market details to an LLM to analyze whether
    the market is susceptible to demographic biases that could lead to
    mispricing based on Polymarket's user demographics.

    Args:
        market: The Market object to analyze for bias.
        model: The LLM model to use for classification.
            Defaults to "claude-haiku-4-5".

    Returns:
        A BiasClassification object containing the analysis results.
    """
    client = get_llm_client(model)
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(market)

    response = await client.complete(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.1,
    )

    return parse_classification_response(response.content, market.id)
