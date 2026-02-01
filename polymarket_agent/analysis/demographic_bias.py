"""
Polymarket User Demographics & Bias Analysis

This module captures research on Polymarket user demographics and provides
tools to identify markets where user biases may lead to mispricing.

Demographics Research Summary (as of late 2024/early 2025):
- Gender: ~73% male, ~27% female
- Age: Concentrated 25-45, largest group 25-34
- Geography: US (~31%), Germany (~7%), Canada (~5%), Vietnam (~4%), UK (~3%)
- Crypto affinity: High overlap with cryptocurrency users
- Political lean: Right-leaning bias on political markets (documented in academic research)

Sources:
- SimilarWeb traffic analysis
- Academic paper: "Political Betting Leaning Score" (PBLS) methodology
- Polymarket trading pattern analysis
"""

import asyncio
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

from polymarket_agent.data_fetching.models import Market

logger = logging.getLogger(__name__)


@dataclass
class DemographicProfile:
    """Polymarket user demographic profile based on research."""

    # Gender distribution
    male_pct: float = 73.0
    female_pct: float = 27.0

    # Age distribution (approximate percentages)
    age_18_24_pct: float = 15.0
    age_25_34_pct: float = 35.0
    age_35_44_pct: float = 25.0
    age_45_54_pct: float = 15.0
    age_55_plus_pct: float = 10.0

    # Geographic distribution (top countries by traffic)
    geography: dict = field(default_factory=lambda: {
        "US": 31.0,
        "Germany": 7.0,
        "Canada": 5.0,
        "Vietnam": 4.0,
        "UK": 3.5,
        "France": 3.0,
        "Netherlands": 2.5,
        "Australia": 2.0,
        "India": 2.0,
        "Other": 40.0,
    })

    # Affinity groups (high overlap populations)
    affinities: list = field(default_factory=lambda: [
        "cryptocurrency",
        "tech",
        "finance",
        "sports_betting",
        "online_gambling",
    ])

    # Political lean indicators
    political_lean: str = "right_leaning"  # On average for political markets
    political_lean_confidence: float = 0.7  # Moderate-high confidence


# Known biases based on demographic research
DEMOGRAPHIC_BIASES = {
    # Geographic blind spots - regions underrepresented in user base
    "geographic_blind_spots": {
        "regions": ["LATAM", "AFRICA", "MIDDLE_EAST", "ASIA"],
        "description": "Markets about these regions may be mispriced due to low user familiarity",
        "bias_direction": "uncertain",  # Could be over or under-estimated
        "confidence": 0.6,
    },

    # Gender-related topics
    "gender_topics": {
        "keywords": [
            "women", "female", "maternal", "pregnancy", "abortion",
            "reproductive", "gender gap", "metoo", "feminism",
        ],
        "description": "Markets on women-centric topics may reflect male-dominated perspective",
        "bias_direction": "uncertain",
        "confidence": 0.5,
    },

    # Age-related blind spots
    "age_blind_spots": {
        "keywords": [
            "gen z", "generation z", "tiktok", "youth", "teenager",
            "retirement", "medicare", "social security", "pension",
            "boomer", "elderly", "nursing home",
        ],
        "description": "Markets about very young or old demographics may be mispriced",
        "bias_direction": "uncertain",
        "confidence": 0.4,
    },

    # Political bias (documented in academic research)
    "political_right_bias": {
        "keywords": [
            "trump", "republican", "gop", "conservative", "maga",
            "desantis", "fox news", "red state",
        ],
        "description": "Right-leaning outcomes may be systematically overpriced",
        "bias_direction": "overestimate",  # User base tends to overestimate
        "confidence": 0.7,
    },

    "political_left_bias": {
        "keywords": [
            "biden", "democrat", "liberal", "progressive",
            "aoc", "bernie", "msnbc", "blue state",
        ],
        "description": "Left-leaning outcomes may be systematically underpriced",
        "bias_direction": "underestimate",
        "confidence": 0.6,
    },

    # Crypto enthusiasm bias
    "crypto_optimism": {
        "keywords": [
            "bitcoin", "btc", "ethereum", "eth", "crypto",
            "blockchain", "defi", "nft", "web3", "solana",
        ],
        "description": "Crypto-positive outcomes may be overpriced due to enthusiast bias",
        "bias_direction": "overestimate",
        "confidence": 0.7,
    },

    # Tech industry familiarity
    "tech_familiarity": {
        "keywords": [
            "apple", "google", "meta", "microsoft", "amazon",
            "nvidia", "tesla", "spacex", "ai", "artificial intelligence",
            "openai", "chatgpt", "silicon valley",
        ],
        "description": "Tech topics well-understood; less likely to be mispriced",
        "bias_direction": "neutral",
        "confidence": 0.6,
    },

    # Sports betting crossover
    "sports_familiarity": {
        "keywords": [
            "nfl", "nba", "mlb", "nhl", "premier league",
            "champions league", "super bowl", "world series",
            "playoff", "championship",
        ],
        "description": "Sports markets likely well-priced due to betting expertise overlap",
        "bias_direction": "neutral",
        "confidence": 0.7,
    },

    # Non-US politics (potential blind spot)
    "non_us_politics": {
        "keywords": [
            "parliament", "prime minister", "eu election", "brexit",
            "macron", "scholz", "trudeau", "modi", "xi jinping",
            "labour party", "tory", "bundestag", "european commission",
            "spain", "spanish", "sanchez", "psoe", "partido popular",
        ],
        "description": "Non-US political markets may be mispriced due to US-centric user base",
        "bias_direction": "uncertain",
        "confidence": 0.5,
    },
}


@dataclass
class BiasAnalysis:
    """Analysis of potential demographic biases affecting a market."""
    market_id: str

    # Identified biases
    detected_biases: list[str]  # List of bias category names
    bias_descriptions: list[str]  # Human-readable descriptions

    # Overall assessment
    mispricing_likelihood: str  # "high", "medium", "low"
    likely_bias_direction: str  # "overestimate", "underestimate", "uncertain", "neutral"

    # Scores
    blind_spot_score: float  # 0-100, higher = more likely to be a blind spot
    familiarity_score: float  # 0-100, higher = more familiar to users

    # Confidence and notes
    confidence: float  # 0-1
    analysis_notes: str

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "detected_biases": self.detected_biases,
            "bias_descriptions": self.bias_descriptions,
            "mispricing_likelihood": self.mispricing_likelihood,
            "likely_bias_direction": self.likely_bias_direction,
            "blind_spot_score": self.blind_spot_score,
            "familiarity_score": self.familiarity_score,
            "confidence": self.confidence,
            "analysis_notes": self.analysis_notes,
        }


def analyze_demographic_bias(market: Market) -> BiasAnalysis:
    """
    Analyze a market for potential demographic biases.

    Args:
        market: Market to analyze

    Returns:
        BiasAnalysis with identified biases and mispricing likelihood
    """
    text = f"{market.question} {market.description or ''}".lower()

    detected_biases = []
    bias_descriptions = []
    bias_directions = []
    confidences = []

    blind_spot_score = 0
    familiarity_score = 50  # Start at neutral

    # Check each bias category
    for bias_name, bias_config in DEMOGRAPHIC_BIASES.items():
        matched = False

        # Check keywords
        if "keywords" in bias_config:
            for keyword in bias_config["keywords"]:
                pattern = rf"\b{re.escape(keyword)}\b"
                if re.search(pattern, text, re.IGNORECASE):
                    matched = True
                    break

        # Check regions (for geographic blind spots)
        if "regions" in bias_config and hasattr(market, 'geo_regions'):
            if market.geo_regions:
                for region in bias_config["regions"]:
                    if region in market.geo_regions:
                        matched = True
                        break

        if matched:
            detected_biases.append(bias_name)
            bias_descriptions.append(bias_config["description"])
            bias_directions.append(bias_config["bias_direction"])
            confidences.append(bias_config["confidence"])

            # Adjust scores based on bias type
            if "blind_spot" in bias_name or bias_config["bias_direction"] == "uncertain":
                blind_spot_score += 20
            elif bias_config["bias_direction"] == "neutral":
                familiarity_score += 15
            else:
                blind_spot_score += 10

    # Normalize scores
    blind_spot_score = min(100, blind_spot_score)
    familiarity_score = min(100, familiarity_score)

    # Determine overall bias direction
    direction_counts = {
        "overestimate": bias_directions.count("overestimate"),
        "underestimate": bias_directions.count("underestimate"),
        "uncertain": bias_directions.count("uncertain"),
        "neutral": bias_directions.count("neutral"),
    }
    likely_direction = max(direction_counts, key=direction_counts.get)
    if direction_counts[likely_direction] == 0:
        likely_direction = "neutral"

    # Calculate mispricing likelihood
    if blind_spot_score >= 40 and len(detected_biases) >= 2:
        mispricing_likelihood = "high"
    elif blind_spot_score >= 20 or len(detected_biases) >= 1:
        mispricing_likelihood = "medium"
    else:
        mispricing_likelihood = "low"

    # Calculate overall confidence
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.3

    # Generate analysis notes
    notes = []
    if detected_biases:
        notes.append(f"Detected {len(detected_biases)} potential bias factor(s).")
        if "political_right_bias" in detected_biases:
            notes.append("Right-leaning political content - user base tends to overestimate.")
        if "political_left_bias" in detected_biases:
            notes.append("Left-leaning political content - user base may underestimate.")
        if "crypto_optimism" in detected_biases:
            notes.append("Crypto content - enthusiast bias may inflate positive outcomes.")
        if "geographic_blind_spots" in detected_biases:
            notes.append("Geographic blind spot - low user familiarity with region.")
        if "non_us_politics" in detected_biases:
            notes.append("Non-US politics - US-centric user base may misprice.")
    else:
        notes.append("No significant demographic biases detected.")

    if mispricing_likelihood == "high":
        notes.append("HIGH mispricing potential due to demographic factors.")

    return BiasAnalysis(
        market_id=market.id,
        detected_biases=detected_biases,
        bias_descriptions=bias_descriptions,
        mispricing_likelihood=mispricing_likelihood,
        likely_bias_direction=likely_direction,
        blind_spot_score=blind_spot_score,
        familiarity_score=familiarity_score,
        confidence=overall_confidence,
        analysis_notes=" ".join(notes),
    )


def filter_by_bias_potential(
    markets: list[Market],
    min_blind_spot_score: float = 20,
    mispricing_levels: Optional[list[str]] = None,
    exclude_familiar: bool = False,
) -> list[tuple[Market, BiasAnalysis]]:
    """
    Filter markets by demographic bias potential.

    Args:
        markets: Markets to filter
        min_blind_spot_score: Minimum blind spot score (0-100)
        mispricing_levels: Filter by mispricing likelihood ["high", "medium", "low"]
        exclude_familiar: Exclude markets with high user familiarity

    Returns:
        List of (market, analysis) tuples
    """
    if mispricing_levels is None:
        mispricing_levels = ["high", "medium"]

    results = []

    for market in markets:
        analysis = analyze_demographic_bias(market)

        # Apply filters
        if analysis.blind_spot_score < min_blind_spot_score:
            continue
        if analysis.mispricing_likelihood not in mispricing_levels:
            continue
        if exclude_familiar and analysis.familiarity_score > 70:
            continue

        results.append((market, analysis))

    # Sort by blind spot score (descending)
    results.sort(key=lambda x: x[1].blind_spot_score, reverse=True)

    logger.info(
        f"Bias filter: {len(markets)} -> {len(results)} markets "
        f"(min_blind_spot={min_blind_spot_score}, levels={mispricing_levels})"
    )

    return results


def get_bias_summary(markets: list[Market]) -> dict:
    """
    Get summary statistics about demographic biases across markets.

    Args:
        markets: Markets to analyze

    Returns:
        Summary dict with statistics
    """
    analyses = [analyze_demographic_bias(m) for m in markets]

    # Count bias types
    bias_counts = {}
    for analysis in analyses:
        for bias in analysis.detected_biases:
            bias_counts[bias] = bias_counts.get(bias, 0) + 1

    # Count mispricing likelihood
    mispricing_counts = {
        "high": len([a for a in analyses if a.mispricing_likelihood == "high"]),
        "medium": len([a for a in analyses if a.mispricing_likelihood == "medium"]),
        "low": len([a for a in analyses if a.mispricing_likelihood == "low"]),
    }

    # Count bias directions
    direction_counts = {
        "overestimate": len([a for a in analyses if a.likely_bias_direction == "overestimate"]),
        "underestimate": len([a for a in analyses if a.likely_bias_direction == "underestimate"]),
        "uncertain": len([a for a in analyses if a.likely_bias_direction == "uncertain"]),
        "neutral": len([a for a in analyses if a.likely_bias_direction == "neutral"]),
    }

    return {
        "total_markets": len(markets),
        "bias_type_counts": bias_counts,
        "mispricing_likelihood_distribution": mispricing_counts,
        "bias_direction_distribution": direction_counts,
        "avg_blind_spot_score": sum(a.blind_spot_score for a in analyses) / len(analyses) if analyses else 0,
        "avg_familiarity_score": sum(a.familiarity_score for a in analyses) / len(analyses) if analyses else 0,
        "high_potential_count": mispricing_counts["high"],
        "high_potential_pct": mispricing_counts["high"] / len(markets) * 100 if markets else 0,
    }


async def analyze_bias_with_llm(
    market: Market,
    keyword_analysis: dict,
    llm_client,
    model: str = "claude-haiku-4-5",
) -> Optional[dict]:
    """
    Use a cheap LLM call to refine keyword-detected bias direction.

    Fixes the keyword-matching limitation where e.g. "Will Bitcoin fall below $30K?"
    gets tagged as crypto-optimistic when it's actually a bearish question.

    Only called for markets that already passed keyword pre-filter.

    Args:
        market: Market being analyzed
        keyword_analysis: Dict with '_detected_biases', '_bias_direction', etc.
        llm_client: LLMClient instance (should be cheap model)
        model: Model name for logging

    Returns:
        Updated dict with corrected bias fields, or None on failure
    """
    detected_biases = keyword_analysis.get('_detected_biases', [])
    if not detected_biases:
        return None

    bias_list = ", ".join(detected_biases)
    current_direction = keyword_analysis.get('_bias_direction', 'uncertain')

    prompt = f"""Analyze the bias direction for this prediction market question given the detected demographic biases.

Question: {market.question}
Description: {(market.description or '')[:500]}
Current market prices: {market.outcome_prices}

Detected biases (from keyword matching): {bias_list}
Current assumed direction: {current_direction}

Given the SPECIFIC FRAMING of this question, is the keyword-detected bias direction correct?

For example:
- "Will Bitcoin reach $100K?" with crypto_optimism bias -> overestimate (users bullish)
- "Will Bitcoin fall below $30K?" with crypto_optimism bias -> underestimate (users won't believe it'll drop)
- "Will Trump win?" with political_right_bias -> overestimate (users overestimate Trump)
- "Will Trump lose?" with political_right_bias -> underestimate (users won't believe he'll lose)

Respond with ONLY a JSON object:
{{
    "corrected_direction": "overestimate|underestimate|uncertain|neutral",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

    try:
        response = await llm_client.complete(
            prompt=prompt,
            max_tokens=300,
            temperature=0.1,
        )

        import json
        from polymarket_agent.llm_assessment.prompts import extract_json_from_response

        json_str = extract_json_from_response(response.content)
        data = json.loads(json_str)

        corrected_direction = data.get("corrected_direction", current_direction)
        if corrected_direction not in ("overestimate", "underestimate", "uncertain", "neutral"):
            corrected_direction = current_direction

        return {
            "_bias_direction": corrected_direction,
            "_bias_llm_refined": True,
            "_bias_llm_confidence": data.get("confidence", 0.5),
            "_bias_llm_reasoning": data.get("reasoning", ""),
        }

    except Exception as e:
        logger.debug(f"LLM bias analysis failed for {market.id}: {e}")
        return None


# Convenience function to get demographic profile
def get_demographic_profile() -> DemographicProfile:
    """Get the current demographic profile of Polymarket users."""
    return DemographicProfile()
