"""
Market Scorer

Implements the scoring framework for ranking potentially mispriced markets.
Combines multiple factors to produce an overall attractiveness score.

Scoring Components (default weights):
- Mispricing Magnitude (30%): How much the market deviates from estimated fair value
- Model Confidence (25%): LLM's self-reported confidence in its assessment
- Evidence Strength (20%): Quality and freshness of supporting information
- Liquidity Score (15%): Ability to enter/exit positions
- Risk Adjustment (10%): Adjusts based on user's risk tolerance

All component scores are normalized to 0-100 scale.
"""

import logging
from typing import Optional

from polymarket_agent.config import (
    AgentConfig,
    RiskTolerance,
    SCORING_WEIGHTS,
    RISK_THRESHOLDS,
)
from polymarket_agent.data_fetching.models import (
    Market,
    EnrichedMarket,
    LLMAssessment,
    ScoredMarket,
)

logger = logging.getLogger(__name__)

# Reference values for normalization
REFERENCE_VOLUME = 100000  # $100K volume for max score
REFERENCE_LIQUIDITY = 50000  # $50K liquidity for max score


def calculate_mispricing_score(
    assessment: LLMAssessment,
    market: Market,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> float:
    """
    Calculate score based on mispricing magnitude.
    
    Higher scores for larger deviations between market price and LLM estimate.
    Score is adjusted based on risk tolerance.
    
    Args:
        assessment: LLM assessment with probability estimates
        market: The market being scored
        risk_tolerance: User's risk tolerance level
        
    Returns:
        Score from 0 to 100
    """
    magnitude = assessment.mispricing_magnitude
    
    # Get threshold for this risk tolerance
    thresholds = RISK_THRESHOLDS[risk_tolerance]
    min_mispricing = thresholds["min_mispricing"]
    
    # No score if below threshold
    if magnitude < min_mispricing:
        return 0.0
    
    # Score based on how much magnitude exceeds threshold
    # Max score at 40% mispricing
    excess = magnitude - min_mispricing
    max_excess = 0.40 - min_mispricing
    
    score = min(100, (excess / max_excess) * 100)
    
    # Bonus for significant mispricings
    if magnitude >= 0.25:
        score = min(100, score * 1.2)
    
    return score


def calculate_confidence_score(
    assessment: LLMAssessment,
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE,
) -> float:
    """
    Calculate score based on model confidence.
    
    Higher confidence = higher score, but adjusted by risk tolerance.
    Conservative profiles require higher confidence.
    
    Args:
        assessment: LLM assessment with confidence
        risk_tolerance: User's risk tolerance
        
    Returns:
        Score from 0 to 100
    """
    confidence = assessment.confidence
    thresholds = RISK_THRESHOLDS[risk_tolerance]
    min_confidence = thresholds["min_confidence"]
    
    # No score if below threshold
    if confidence < min_confidence:
        return 0.0
    
    # Linear scaling above threshold
    range_above = 1.0 - min_confidence
    excess = confidence - min_confidence
    
    score = (excess / range_above) * 100
    
    return min(100, score)


def calculate_evidence_score(
    enrichment: Optional[EnrichedMarket],
    assessment: LLMAssessment,
) -> float:
    """
    Calculate score based on quality of supporting evidence.
    
    Considers:
    - Number of sources found
    - Number of key facts extracted
    - Presence of warnings about limited data
    
    Args:
        enrichment: Optional enrichment data
        assessment: LLM assessment
        
    Returns:
        Score from 0 to 100
    """
    if enrichment is None:
        return 30.0  # Base score without enrichment
    
    score = 30.0  # Base score
    
    # Add points for sources
    num_sources = len(enrichment.sources)
    score += min(30, num_sources * 6)  # Up to 30 points for 5+ sources
    
    # Add points for key facts
    num_facts = len(enrichment.key_facts)
    score += min(25, num_facts * 5)  # Up to 25 points for 5+ facts
    
    # Deduct for warnings about limited data
    limited_data_warnings = [
        w for w in assessment.warnings
        if any(term in w.lower() for term in ["limited", "insufficient", "no data", "outdated"])
    ]
    score -= len(limited_data_warnings) * 10
    
    # Ensure bounds
    return max(0, min(100, score))


def calculate_liquidity_score(
    market: Market,
    reference_liquidity: float = REFERENCE_LIQUIDITY,
) -> float:
    """
    Calculate score based on market liquidity.
    
    Higher liquidity = easier to enter/exit = higher score.
    Uses logarithmic scaling to handle wide range of values.
    
    Args:
        market: Market with liquidity data
        reference_liquidity: Reference liquidity for max score
        
    Returns:
        Score from 0 to 100
    """
    import math
    
    liquidity = market.liquidity
    volume = market.volume
    
    # Use the higher of liquidity or volume as proxy
    effective_liquidity = max(liquidity, volume / 10)
    
    if effective_liquidity <= 0:
        return 10.0  # Minimum score for markets with no data
    
    # Logarithmic scaling
    # $1K -> ~40, $10K -> ~70, $50K+ -> ~100
    log_liq = math.log10(max(1, effective_liquidity))
    log_ref = math.log10(reference_liquidity)
    
    score = (log_liq / log_ref) * 100
    
    return max(0, min(100, score))


def calculate_risk_adjusted_score(
    mispricing_score: float,
    confidence_score: float,
    assessment: LLMAssessment,
    risk_tolerance: RiskTolerance,
) -> float:
    """
    Calculate risk-adjusted score.
    
    Combines mispricing potential with risk factors.
    Aggressive profiles get bonus for high-risk/high-reward.
    Conservative profiles get bonus for low-risk situations.
    
    Args:
        mispricing_score: Mispricing component score
        confidence_score: Confidence component score
        assessment: LLM assessment
        risk_tolerance: User's risk tolerance
        
    Returns:
        Risk-adjusted score from 0 to 100
    """
    # Calculate uncertainty from probability range width
    ranges = assessment.probability_estimates.values()
    avg_uncertainty = 0
    if ranges:
        avg_uncertainty = sum(high - low for low, high in ranges) / len(list(ranges))
    
    # Base risk score (inverse of uncertainty)
    risk_base = (1 - avg_uncertainty) * 100
    
    # Number of identified risks
    num_risks = len(assessment.risks)
    risk_penalty = min(30, num_risks * 10)
    
    risk_score = max(0, risk_base - risk_penalty)
    
    # Adjust based on risk tolerance
    if risk_tolerance == RiskTolerance.CONSERVATIVE:
        # Conservative: heavily weight safety
        return risk_score * 1.2
    elif risk_tolerance == RiskTolerance.AGGRESSIVE:
        # Aggressive: bonus for high mispricing even with risk
        if mispricing_score > 50:
            return min(100, risk_score + 20)
        return risk_score * 0.8
    else:
        # Moderate: balanced
        return risk_score


def score_market(
    market: Market,
    assessment: LLMAssessment,
    enrichment: Optional[EnrichedMarket] = None,
    config: Optional[AgentConfig] = None,
    spread_analysis: Optional[dict] = None,
) -> ScoredMarket:
    """
    Calculate full scores for a market.
    
    Computes all component scores and combines them using configured weights.
    
    Args:
        market: The market to score
        assessment: LLM assessment for the market
        enrichment: Optional enrichment data
        config: Optional agent configuration
        
    Returns:
        ScoredMarket with all scores populated
    """
    if config is None:
        from polymarket_agent.config import AgentConfig
        config = AgentConfig()
    
    risk_tolerance = config.risk_tolerance
    weights = SCORING_WEIGHTS
    
    # Calculate component scores
    mispricing = calculate_mispricing_score(assessment, market, risk_tolerance)
    confidence = calculate_confidence_score(assessment, risk_tolerance)
    evidence = calculate_evidence_score(enrichment, assessment)
    liquidity = calculate_liquidity_score(market)
    risk_adj = calculate_risk_adjusted_score(mispricing, confidence, assessment, risk_tolerance)
    
    # Calculate weighted total
    total = (
        mispricing * weights["mispricing_magnitude"] +
        confidence * weights["model_confidence"] +
        evidence * weights["evidence_strength"] +
        liquidity * weights["liquidity_score"] +
        risk_adj * weights["risk_adjustment"]
    )
    
    # Apply final risk tolerance adjustment
    if risk_tolerance == RiskTolerance.CONSERVATIVE:
        # Conservative: require higher overall score
        if total < 50 or confidence < 60:
            total *= 0.5
    elif risk_tolerance == RiskTolerance.AGGRESSIVE:
        # Aggressive: boost borderline opportunities
        if mispricing > 40:
            total = min(100, total * 1.1)

    # Apply spread analysis penalty if available
    if spread_analysis:
        net_edge = spread_analysis.get("net_edge", 0)
        if net_edge <= 0:
            # Negative net edge: heavily penalize
            total *= 0.3
        elif net_edge < 0.03:
            # Thin net edge: moderate penalty
            total *= 0.7

    return ScoredMarket(
        market=market,
        enrichment=enrichment,
        assessment=assessment,
        mispricing_score=mispricing,
        confidence_score=confidence,
        evidence_score=evidence,
        liquidity_score=liquidity,
        risk_score=risk_adj,
        total_score=total,
        rank=0,  # Set during ranking
        spread_analysis=spread_analysis,
    )


def rank_markets(
    scored_markets: list[ScoredMarket],
    min_score: float = 0,
) -> list[ScoredMarket]:
    """
    Rank scored markets by total score.
    
    Args:
        scored_markets: List of scored markets
        min_score: Minimum score to include in rankings
        
    Returns:
        Sorted list with rank values assigned
    """
    # Filter by minimum score
    filtered = [m for m in scored_markets if m.total_score >= min_score]
    
    # Sort by total score descending
    sorted_markets = sorted(filtered, key=lambda m: m.total_score, reverse=True)
    
    # Assign ranks
    for i, market in enumerate(sorted_markets, 1):
        market.rank = i
    
    return sorted_markets


class MarketScorer:
    """
    Handles scoring and ranking of markets.
    
    Example:
        >>> scorer = MarketScorer(config)
        >>> scored = scorer.score_all(markets, assessments, enrichments)
        >>> ranked = scorer.rank(scored)
        >>> for market in ranked[:10]:
        ...     print(f"#{market.rank}: {market.market.question} - {market.total_score:.1f}")
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the scorer.
        
        Args:
            config: Agent configuration with risk tolerance and other settings
        """
        self.config = config
    
    def score(
        self,
        market: Market,
        assessment: LLMAssessment,
        enrichment: Optional[EnrichedMarket] = None,
    ) -> ScoredMarket:
        """Score a single market."""
        return score_market(market, assessment, enrichment, self.config)
    
    def score_all(
        self,
        markets: list[Market],
        assessments: list[LLMAssessment],
        enrichments: Optional[dict[str, EnrichedMarket]] = None,
    ) -> list[ScoredMarket]:
        """
        Score multiple markets.
        
        Args:
            markets: List of markets
            assessments: List of assessments (same order as markets)
            enrichments: Optional dict mapping market IDs to enrichments
            
        Returns:
            List of ScoredMarket objects
        """
        if enrichments is None:
            enrichments = {}
        
        scored = []
        for market, assessment in zip(markets, assessments):
            enrichment = enrichments.get(market.id)
            scored_market = self.score(market, assessment, enrichment)
            scored.append(scored_market)
        
        return scored
    
    def rank(
        self,
        scored_markets: list[ScoredMarket],
        min_score: Optional[float] = None,
    ) -> list[ScoredMarket]:
        """
        Rank scored markets.
        
        Args:
            scored_markets: List of scored markets
            min_score: Minimum score to include (uses risk-based default if None)
            
        Returns:
            Sorted and ranked list
        """
        if min_score is None:
            # Default minimum based on risk tolerance
            min_score_map = {
                RiskTolerance.CONSERVATIVE: 40,
                RiskTolerance.MODERATE: 25,
                RiskTolerance.AGGRESSIVE: 10,
            }
            min_score = min_score_map.get(self.config.risk_tolerance, 25)
        
        return rank_markets(scored_markets, min_score)
    
    def get_top_opportunities(
        self,
        scored_markets: list[ScoredMarket],
        n: int = 10,
    ) -> list[ScoredMarket]:
        """
        Get the top N opportunities.
        
        Args:
            scored_markets: List of scored markets
            n: Number of top opportunities to return
            
        Returns:
            Top N ranked markets
        """
        ranked = self.rank(scored_markets)
        return ranked[:n]
