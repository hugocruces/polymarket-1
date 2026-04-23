"""
LLM-based bias classifier for Polymarket markets.

This module provides functions to build prompts for bias classification
and parse LLM responses into BiasClassification objects.
"""

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
    ClassificationError,
)
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

Analyze markets to identify whether they are susceptible to these biases.
Do NOT try to call the direction of any mispricing — that judgment is made
later by a human reviewer. Just identify whether bias could be present.

Respond with a JSON object containing:
- dominated_by_bias: boolean indicating if bias significantly affects this market
- categories: list of bias categories (political, progressive_social, crypto_optimism)
- bias_score: 0-100 indicating strength of bias effect
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
- european (boolean)
- spain (boolean)
- reasoning (string)
"""


_CATEGORY_ALIASES = {
    "political": BiasCategory.POLITICAL,
    "progressive_social": BiasCategory.PROGRESSIVE_SOCIAL,
    "progressive social": BiasCategory.PROGRESSIVE_SOCIAL,
    "crypto_optimism": BiasCategory.CRYPTO_OPTIMISM,
    "crypto optimism": BiasCategory.CRYPTO_OPTIMISM,
}


class _LLMBiasResponse(BaseModel):
    """Pydantic schema for validating the raw LLM JSON payload.

    Kept internal to the classifier — callers receive a BiasClassification
    dataclass built from this after validation.
    """

    dominated_by_bias: bool
    categories: list[BiasCategory] = Field(default_factory=list)
    bias_score: int = Field(ge=0, le=100)
    european: bool
    spain: bool
    reasoning: str

    @field_validator("categories", mode="before")
    @classmethod
    def _normalize_categories(cls, value: object) -> list[BiasCategory]:
        if not isinstance(value, list):
            raise ValueError("categories must be a list")
        out: list[BiasCategory] = []
        for item in value:
            if isinstance(item, BiasCategory):
                out.append(item)
                continue
            if not isinstance(item, str):
                raise ValueError(f"category must be a string, got {type(item).__name__}")
            key = item.lower().strip()
            if key not in _CATEGORY_ALIASES:
                raise ValueError(f"unknown category: {item!r}")
            out.append(_CATEGORY_ALIASES[key])
        return out


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
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    json_match = re.search(r"\{.*\}", response, re.DOTALL)
    if json_match:
        return json_match.group(0)

    return response


def parse_classification_response(response: str, market_id: str) -> BiasClassification:
    """Parse an LLM response into a BiasClassification object.

    Args:
        response: The raw LLM response string (may contain JSON or markdown).
        market_id: The market ID to associate with the classification.

    Returns:
        A BiasClassification object.

    Raises:
        ClassificationError: If the response is not valid JSON, is missing
            required fields, or contains values outside the allowed ranges.
    """
    json_str = _extract_json_from_response(response)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ClassificationError(f"LLM response is not valid JSON: {e}") from e

    try:
        parsed = _LLMBiasResponse.model_validate(data)
    except ValidationError as e:
        raise ClassificationError(f"LLM response failed schema validation: {e}") from e

    return BiasClassification(
        market_id=market_id,
        dominated_by_bias=parsed.dominated_by_bias,
        categories=parsed.categories,
        bias_score=parsed.bias_score,
        european=parsed.european,
        spain=parsed.spain,
        reasoning=parsed.reasoning,
    )


async def classify_market(
    market: Market,
    model: str = "claude-sonnet-4-6",
) -> BiasClassification:
    """
    Classify a market for demographic bias potential using LLM.

    Args:
        market: The Market object to analyze for bias.
        model: The LLM model to use for classification.

    Returns:
        A BiasClassification object containing the analysis results.

    Raises:
        ClassificationError: If the LLM response cannot be parsed or
            validated. Transport errors from the provider propagate
            unchanged.
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
