"""
Market Reasoning Classifier

Classifies markets by whether they favor reasoning/analysis (where LLMs might
have an edge) versus information access (where market participants likely have
an edge).

Reasoning-heavy markets:
- Complex conditional probabilities
- Multi-factor analysis required
- Base rate / statistical reasoning
- Long time horizons
- Abstract or conceptual questions

Information-heavy markets:
- Near-term events (days away)
- Simple factual outcomes
- Insider knowledge dependent
- Breaking news dependent
- Real-time data dependent
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from polymarket_agent.data_fetching.models import Market

logger = logging.getLogger(__name__)


@dataclass
class ReasoningAnalysis:
    """Result of analyzing a market's reasoning characteristics."""
    market_id: str
    reasoning_score: float  # 0-100, higher = more reasoning-heavy
    information_score: float  # 0-100, higher = more information-dependent

    # Component scores
    complexity_score: float  # Question complexity
    temporal_score: float  # Time horizon factor
    conditionality_score: float  # Conditional/hypothetical nature
    abstraction_score: float  # Abstract vs concrete

    # Flags
    is_reasoning_heavy: bool  # reasoning_score > information_score
    reasoning_factors: list[str]  # Why it's reasoning-heavy
    information_factors: list[str]  # Why it's information-heavy

    # Recommendation
    llm_edge_likelihood: str  # "high", "medium", "low", "unlikely"
    analysis_notes: str

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "reasoning_score": self.reasoning_score,
            "information_score": self.information_score,
            "complexity_score": self.complexity_score,
            "temporal_score": self.temporal_score,
            "conditionality_score": self.conditionality_score,
            "abstraction_score": self.abstraction_score,
            "is_reasoning_heavy": self.is_reasoning_heavy,
            "reasoning_factors": self.reasoning_factors,
            "information_factors": self.information_factors,
            "llm_edge_likelihood": self.llm_edge_likelihood,
            "analysis_notes": self.analysis_notes,
        }


# Patterns indicating reasoning-heavy markets
REASONING_PATTERNS = {
    # Conditional questions (high weight)
    "conditional": {
        "patterns": [
            r"\bif\b.+\bwill\b",
            r"\bwhen\b.+\bwill\b",
            r"\bassuming\b",
            r"\bgiven that\b",
            r"\bin the event\b",
            r"\bshould\b.+\bhappen\b",
            r"\bcontingent\b",
        ],
        "weight": 15,
        "description": "Conditional/hypothetical question",
    },
    # Comparative analysis
    "comparative": {
        "patterns": [
            r"\bmore than\b",
            r"\bless than\b",
            r"\bhigher than\b",
            r"\blower than\b",
            r"\bexceed\b",
            r"\boutperform\b",
            r"\bcompared to\b",
            r"\bversus\b",
            r"\bvs\.?\b",
        ],
        "weight": 10,
        "description": "Comparative analysis required",
    },
    # Quantitative thresholds
    "quantitative": {
        "patterns": [
            r"\b\d+%",  # Percentages
            r"\b\d+\s*(billion|million|thousand|trillion)\b",
            r"\bat least\b",
            r"\bat most\b",
            r"\bbetween\b.+\band\b",
            r"\bover\b\s*\d+",
            r"\bunder\b\s*\d+",
            r"\brange\b",
        ],
        "weight": 12,
        "description": "Quantitative threshold analysis",
    },
    # Multi-factor questions
    "multi_factor": {
        "patterns": [
            r"\band\b.+\band\b",  # Multiple "and"s
            r"\bboth\b.+\band\b",
            r"\beither\b.+\bor\b",
            r"\ball of\b",
            r"\bany of\b",
            r"\bnone of\b",
            r"\bmultiple\b",
        ],
        "weight": 10,
        "description": "Multi-factor evaluation",
    },
    # Abstract/conceptual
    "abstract": {
        "patterns": [
            r"\blikely\b",
            r"\bprobability\b",
            r"\bchance\b",
            r"\brisk\b",
            r"\buncertain\b",
            r"\bpossible\b",
            r"\bscenario\b",
            r"\boutcome\b",
        ],
        "weight": 8,
        "description": "Abstract/probabilistic reasoning",
    },
    # Long-term trends
    "long_term": {
        "patterns": [
            r"\bby\s+(end of\s+)?(20\d{2})\b",
            r"\bby\s+(Q[1-4]|quarter)\b",
            r"\bover the (next|coming)\b",
            r"\blong[- ]term\b",
            r"\beventually\b",
            r"\bultimately\b",
        ],
        "weight": 8,
        "description": "Long-term trend analysis",
    },
    # Causal reasoning
    "causal": {
        "patterns": [
            r"\bbecause\b",
            r"\bdue to\b",
            r"\bresult in\b",
            r"\blead to\b",
            r"\bcause\b",
            r"\beffect\b",
            r"\bimpact\b",
            r"\binfluence\b",
        ],
        "weight": 8,
        "description": "Causal reasoning involved",
    },
}

# Patterns indicating information-heavy markets
INFORMATION_PATTERNS = {
    # Near-term events
    "near_term": {
        "patterns": [
            r"\btoday\b",
            r"\btomorrow\b",
            r"\btonight\b",
            r"\bthis week\b",
            r"\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
            r"\bJanuary\s+\d{1,2}(st|nd|rd|th)?\b",  # Specific dates
            r"\bFebruary\s+\d{1,2}(st|nd|rd|th)?\b",
            r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d{1,2}\b",
        ],
        "weight": 15,
        "description": "Near-term event (insider info matters)",
    },
    # Sports outcomes (highly information-dependent)
    "sports": {
        "patterns": [
            r"\bwin\b.+\b(game|match|series|championship|cup|bowl|tournament)\b",
            r"\bscore\b.+\b(points?|goals?|runs?)\b",
            r"\bbeat\b",
            r"\bdefeat\b",
            r"\b(mvp|finals|playoff|championship)\b",
            r"\b(nfl|nba|mlb|nhl|premier league|champions league)\b",
        ],
        "weight": 12,
        "description": "Sports outcome (real-time info dependent)",
    },
    # Simple yes/no factual
    "simple_factual": {
        "patterns": [
            r"^will\s+\w+\s+(resign|die|win|lose|announce|release|launch)\b",
            r"\bconfirm(ed)?\b",
            r"\bannounce(d)?\b",
            r"\bofficial(ly)?\b",
        ],
        "weight": 10,
        "description": "Simple factual outcome",
    },
    # Real-time price dependent
    "price_dependent": {
        "patterns": [
            r"\bprice\b.+\b(above|below|reach|hit)\b",
            r"\b(btc|eth|bitcoin|ethereum)\b.+\$\d+",
            r"\bstock\b.+\b(price|value)\b",
            r"\bmarket cap\b",
            r"\ball[- ]time high\b",
        ],
        "weight": 15,
        "description": "Real-time price dependent",
    },
    # Election results (local knowledge)
    "election_results": {
        "patterns": [
            r"\bwin\b.+\b(election|primary|vote|race)\b",
            r"\belected\b",
            r"\bnominate[ds]?\b",
            r"\bcandidate\b",
            r"\bpolling\b",
        ],
        "weight": 10,
        "description": "Election outcome (local knowledge matters)",
    },
    # Breaking news dependent
    "news_dependent": {
        "patterns": [
            r"\bbreak(ing)?\s+news\b",
            r"\breport(ed|s)?\b",
            r"\bsource(s)?\b",
            r"\bleak(ed|s)?\b",
            r"\brumor(ed|s)?\b",
        ],
        "weight": 12,
        "description": "Breaking news dependent",
    },
    # Binary simple
    "binary_simple": {
        "patterns": [
            r"^will\s+.{10,50}\?$",  # Short simple questions
        ],
        "weight": 5,
        "description": "Simple binary question",
    },
}


def analyze_market_reasoning(market: Market) -> ReasoningAnalysis:
    """
    Analyze whether a market favors reasoning/analysis or information access.

    Args:
        market: Market to analyze

    Returns:
        ReasoningAnalysis with scores and factors
    """
    text = f"{market.question} {market.description or ''}".lower()

    reasoning_score = 0
    information_score = 0
    reasoning_factors = []
    information_factors = []

    # Check reasoning patterns
    for category, config in REASONING_PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                reasoning_score += config["weight"]
                if config["description"] not in reasoning_factors:
                    reasoning_factors.append(config["description"])
                break  # Only count each category once

    # Check information patterns
    for category, config in INFORMATION_PATTERNS.items():
        for pattern in config["patterns"]:
            if re.search(pattern, text, re.IGNORECASE):
                information_score += config["weight"]
                if config["description"] not in information_factors:
                    information_factors.append(config["description"])
                break  # Only count each category once

    # Temporal analysis (days to expiry affects score)
    temporal_score = 0
    if market.days_to_expiry is not None:
        if market.days_to_expiry <= 7:
            # Very near-term: information-heavy
            information_score += 20
            information_factors.append(f"Resolves in {market.days_to_expiry} days (very near-term)")
            temporal_score = -30
        elif market.days_to_expiry <= 30:
            # Near-term
            information_score += 10
            information_factors.append(f"Resolves in {market.days_to_expiry} days (near-term)")
            temporal_score = -10
        elif market.days_to_expiry >= 180:
            # Long-term: more reasoning-friendly
            reasoning_score += 15
            reasoning_factors.append(f"Long time horizon ({market.days_to_expiry} days)")
            temporal_score = 20
        elif market.days_to_expiry >= 90:
            # Medium-term
            reasoning_score += 8
            reasoning_factors.append(f"Medium time horizon ({market.days_to_expiry} days)")
            temporal_score = 10

    # Outcome complexity
    complexity_score = 0
    num_outcomes = len(market.outcomes)
    if num_outcomes > 2:
        # Multi-outcome markets often require more analysis
        reasoning_score += min(15, (num_outcomes - 2) * 5)
        reasoning_factors.append(f"Multi-outcome market ({num_outcomes} outcomes)")
        complexity_score = min(30, (num_outcomes - 2) * 10)

    # Question length as proxy for complexity
    question_words = len(market.question.split())
    if question_words > 20:
        reasoning_score += 10
        reasoning_factors.append(f"Complex question ({question_words} words)")
        complexity_score += 15
    elif question_words < 8:
        information_score += 5
        complexity_score -= 10

    # Conditionality score (from pattern matching above)
    conditionality_score = 0
    if "Conditional/hypothetical question" in reasoning_factors:
        conditionality_score = 30

    # Abstraction score
    abstraction_score = 0
    if "Abstract/probabilistic reasoning" in reasoning_factors:
        abstraction_score = 20
    if "Causal reasoning involved" in reasoning_factors:
        abstraction_score += 15

    # Normalize scores to 0-100
    reasoning_score = min(100, reasoning_score)
    information_score = min(100, information_score)

    # Determine if reasoning-heavy
    is_reasoning_heavy = reasoning_score > information_score

    # Calculate LLM edge likelihood
    score_diff = reasoning_score - information_score
    if score_diff >= 30:
        llm_edge_likelihood = "high"
    elif score_diff >= 10:
        llm_edge_likelihood = "medium"
    elif score_diff >= -10:
        llm_edge_likelihood = "low"
    else:
        llm_edge_likelihood = "unlikely"

    # Generate analysis notes
    notes = []
    if is_reasoning_heavy:
        notes.append(f"Reasoning-heavy market (score: {reasoning_score} vs {information_score}).")
        if reasoning_factors:
            notes.append(f"Key factors: {', '.join(reasoning_factors[:3])}.")
    else:
        notes.append(f"Information-heavy market (score: {information_score} vs {reasoning_score}).")
        if information_factors:
            notes.append(f"Key factors: {', '.join(information_factors[:3])}.")

    if llm_edge_likelihood in ["high", "medium"]:
        notes.append("LLM analysis may provide edge over market consensus.")
    else:
        notes.append("Market participants likely have information advantage.")

    return ReasoningAnalysis(
        market_id=market.id,
        reasoning_score=reasoning_score,
        information_score=information_score,
        complexity_score=max(0, min(100, complexity_score + 50)),  # Normalize around 50
        temporal_score=max(0, min(100, temporal_score + 50)),  # Normalize around 50
        conditionality_score=conditionality_score,
        abstraction_score=abstraction_score,
        is_reasoning_heavy=is_reasoning_heavy,
        reasoning_factors=reasoning_factors,
        information_factors=information_factors,
        llm_edge_likelihood=llm_edge_likelihood,
        analysis_notes=" ".join(notes),
    )


def filter_reasoning_heavy(
    markets: list[Market],
    min_reasoning_score: float = 30,
    max_information_score: Optional[float] = None,
    llm_edge_levels: Optional[list[str]] = None,
) -> list[tuple[Market, ReasoningAnalysis]]:
    """
    Filter markets to those that are reasoning-heavy.

    Args:
        markets: Markets to filter
        min_reasoning_score: Minimum reasoning score (0-100)
        max_information_score: Maximum information score (None = no limit)
        llm_edge_levels: Filter by LLM edge likelihood ["high", "medium", "low", "unlikely"]

    Returns:
        List of (market, analysis) tuples for reasoning-heavy markets
    """
    if llm_edge_levels is None:
        llm_edge_levels = ["high", "medium"]

    results = []

    for market in markets:
        analysis = analyze_market_reasoning(market)

        # Apply filters
        if analysis.reasoning_score < min_reasoning_score:
            continue
        if max_information_score is not None and analysis.information_score > max_information_score:
            continue
        if analysis.llm_edge_likelihood not in llm_edge_levels:
            continue

        results.append((market, analysis))

    # Sort by reasoning score (descending)
    results.sort(key=lambda x: x[1].reasoning_score, reverse=True)

    logger.info(
        f"Reasoning filter: {len(markets)} -> {len(results)} markets "
        f"(min_reasoning={min_reasoning_score}, edge_levels={llm_edge_levels})"
    )

    return results


def get_reasoning_summary(markets: list[Market]) -> dict:
    """
    Get summary statistics about reasoning characteristics of markets.

    Args:
        markets: Markets to analyze

    Returns:
        Summary dict with statistics
    """
    analyses = [analyze_market_reasoning(m) for m in markets]

    reasoning_heavy = [a for a in analyses if a.is_reasoning_heavy]
    info_heavy = [a for a in analyses if not a.is_reasoning_heavy]

    edge_counts = {
        "high": len([a for a in analyses if a.llm_edge_likelihood == "high"]),
        "medium": len([a for a in analyses if a.llm_edge_likelihood == "medium"]),
        "low": len([a for a in analyses if a.llm_edge_likelihood == "low"]),
        "unlikely": len([a for a in analyses if a.llm_edge_likelihood == "unlikely"]),
    }

    return {
        "total_markets": len(markets),
        "reasoning_heavy_count": len(reasoning_heavy),
        "information_heavy_count": len(info_heavy),
        "reasoning_heavy_pct": len(reasoning_heavy) / len(markets) * 100 if markets else 0,
        "avg_reasoning_score": sum(a.reasoning_score for a in analyses) / len(analyses) if analyses else 0,
        "avg_information_score": sum(a.information_score for a in analyses) / len(analyses) if analyses else 0,
        "llm_edge_distribution": edge_counts,
    }
