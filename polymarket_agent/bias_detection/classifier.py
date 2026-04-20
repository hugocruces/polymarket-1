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

# Markets with bias_score >= this threshold are flagged as dominated_by_bias
BIAS_SCORE_THRESHOLD = 50

SYSTEM_PROMPT = """You are an expert analyst specializing in prediction market bias detection.

Your task is to identify Polymarket markets where the platform's user demographics would cause the crowd to bet systematically differently from a demographically neutral group — regardless of the actual underlying probability.

## Polymarket Demographics

- ~73% male, ages 25-45
- ~31% US-based, right-leaning politically
- Crypto/tech enthusiasts

## Bias Categories

**Political Bias** — The right-leaning userbase systematically favors conservative/Republican outcomes. Target markets: US elections, legislation, regulatory appointments, policy decisions where one side is clearly conservative and the other progressive. Do not flag merely because a topic is political — only flag if there is a clear directional demographic lean.

**Progressive Social Bias** — The young male tech demographic structurally underweights progressive social outcomes. Target markets: gender policy, abortion rights, climate action, DEI initiatives, social safety net expansions, labor rights. These outcomes tend to be underpriced relative to a neutral crowd.

**Crypto Optimism** — Crypto enthusiasts are systematically over-optimistic. Target markets: price targets, ETF approvals, adoption milestones, regulatory outcomes for specific coins or platforms, blockchain technology adoption.

## Bias Score

Rate 0–100:
- 0–20: No meaningful bias — topic does not map to a demographic blind spot
- 20–40: Mild susceptibility — some lean possible but weak signal
- 40–70: Moderate — demographic would clearly favor one side
- 70–100: Strong — price likely reflects userbase skew more than objective probability

## Guardrail

Do NOT flag a market just because it touches a political or social topic. Only flag it if the Polymarket demographic would systematically lean one way. A politically charged topic with no clear directional bias should score 0–20.

Return a JSON object with:
- categories: list of applicable bias categories ("political", "progressive_social", "crypto_optimism") — empty list if none apply
- bias_score: integer 0–100 per the calibration above
- reasoning: 1–2 sentences on why this market is or is not susceptible
"""

USER_PROMPT_TEMPLATE = """Analyze this Polymarket market for demographic bias:

**Question**: {question}
**Description**: {description}

**Current Prices**:
{outcomes}

Return JSON with fields: categories, bias_score, reasoning.
"""


def build_user_prompt(market: Market) -> str:
    """Build a user prompt for bias classification from market data."""
    outcomes_text = "\n".join(
        f"- {outcome.name}: {outcome.price:.2f} ({outcome.price * 100:.0f}%)"
        for outcome in market.outcomes
    )
    description = (market.description or "").strip() or "N/A"

    return USER_PROMPT_TEMPLATE.format(
        question=market.question,
        description=description,
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

    dominated_by_bias is derived from bias_score >= BIAS_SCORE_THRESHOLD,
    not from the LLM output, to keep the threshold explicit and consistent.

    Returns a default no-bias classification if parsing fails.
    """
    try:
        json_str = _extract_json_from_response(response)
        data = json.loads(json_str)

        bias_score = int(data.get("bias_score", 0))
        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=bias_score >= BIAS_SCORE_THRESHOLD,
            categories=_parse_categories(data.get("categories", [])),
            bias_score=bias_score,
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
    model: str = "claude-sonnet-4-6",
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
