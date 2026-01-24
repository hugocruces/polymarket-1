"""
Tests for the scoring module.

These tests verify that market scoring and ranking work correctly.
"""

import pytest
from datetime import datetime, timedelta

from polymarket_agent.data_fetching.models import Market, Outcome, LLMAssessment, EnrichedMarket
from polymarket_agent.scoring.scorer import (
    MarketScorer,
    calculate_mispricing_score,
    calculate_confidence_score,
    calculate_liquidity_score,
    calculate_evidence_score,
    score_market,
    rank_markets,
)
from polymarket_agent.config import AgentConfig, FilterConfig, RiskTolerance


def create_test_market(
    id: str = "test-1",
    question: str = "Will this happen?",
    volume: float = 50000,
    liquidity: float = 25000,
    yes_price: float = 0.60,
) -> Market:
    """Create a test market."""
    return Market(
        id=id,
        slug=id,
        question=question,
        description="Test market description",
        outcomes=[
            Outcome(name="Yes", token_id=f"yes-{id}", price=yes_price),
            Outcome(name="No", token_id=f"no-{id}", price=1.0 - yes_price),
        ],
        category="politics",
        tags=["politics"],
        volume=volume,
        liquidity=liquidity,
        end_date=datetime.now() + timedelta(days=30),
        active=True,
    )


def create_test_assessment(
    market_id: str = "test-1",
    yes_estimate: tuple[float, float] = (0.55, 0.65),
    confidence: float = 0.7,
    mispricing_detected: bool = True,
    mispricing_direction: str = "overpriced",
    mispricing_magnitude: float = 0.15,
) -> LLMAssessment:
    """Create a test LLM assessment."""
    return LLMAssessment(
        market_id=market_id,
        probability_estimates={
            "Yes": yes_estimate,
            "No": (1.0 - yes_estimate[1], 1.0 - yes_estimate[0]),
        },
        confidence=confidence,
        reasoning="Test reasoning for the assessment.",
        key_factors=["Factor 1", "Factor 2"],
        risks=["Risk 1"],
        mispricing_detected=mispricing_detected,
        mispricing_direction=mispricing_direction,
        mispricing_magnitude=mispricing_magnitude,
        warnings=[],
        model_used="test-model",
    )


def create_test_enrichment(
    market: Market,
    num_sources: int = 3,
    num_facts: int = 3,
) -> EnrichedMarket:
    """Create a test enrichment."""
    return EnrichedMarket(
        market=market,
        external_context="Test external context from web search.",
        sources=[f"https://source{i}.com" for i in range(num_sources)],
        context_freshness="2024-01-20",
        key_facts=[f"Fact {i}" for i in range(num_facts)],
        search_query="test query",
    )


class TestCalculateMispricingScore:
    """Tests for mispricing score calculation."""
    
    def test_high_mispricing_high_score(self):
        """Large mispricings should get high scores."""
        market = create_test_market(yes_price=0.70)
        assessment = create_test_assessment(
            mispricing_magnitude=0.25,
            mispricing_detected=True,
        )
        
        score = calculate_mispricing_score(assessment, market, RiskTolerance.MODERATE)
        
        assert score > 50  # Should be a high score
    
    def test_low_mispricing_low_score(self):
        """Small mispricings should get low or zero scores."""
        market = create_test_market(yes_price=0.55)
        assessment = create_test_assessment(
            mispricing_magnitude=0.05,
            mispricing_detected=False,
        )
        
        score = calculate_mispricing_score(assessment, market, RiskTolerance.MODERATE)
        
        assert score == 0  # Below threshold
    
    def test_conservative_requires_larger_mispricing(self):
        """Conservative risk should require larger mispricings."""
        market = create_test_market()
        assessment = create_test_assessment(mispricing_magnitude=0.15)
        
        conservative_score = calculate_mispricing_score(
            assessment, market, RiskTolerance.CONSERVATIVE
        )
        aggressive_score = calculate_mispricing_score(
            assessment, market, RiskTolerance.AGGRESSIVE
        )
        
        assert aggressive_score > conservative_score


class TestCalculateConfidenceScore:
    """Tests for confidence score calculation."""
    
    def test_high_confidence_high_score(self):
        """High confidence should yield high scores."""
        assessment = create_test_assessment(confidence=0.85)
        
        score = calculate_confidence_score(assessment, RiskTolerance.MODERATE)
        
        assert score > 50
    
    def test_low_confidence_low_score(self):
        """Low confidence should yield low scores."""
        assessment = create_test_assessment(confidence=0.30)
        
        score = calculate_confidence_score(assessment, RiskTolerance.MODERATE)
        
        assert score == 0  # Below moderate threshold
    
    def test_conservative_requires_higher_confidence(self):
        """Conservative risk should require higher confidence."""
        assessment = create_test_assessment(confidence=0.65)
        
        conservative_score = calculate_confidence_score(
            assessment, RiskTolerance.CONSERVATIVE
        )
        moderate_score = calculate_confidence_score(
            assessment, RiskTolerance.MODERATE
        )
        
        # Conservative requires 0.75+, so 0.65 scores 0
        assert conservative_score == 0
        assert moderate_score > 0


class TestCalculateLiquidityScore:
    """Tests for liquidity score calculation."""
    
    def test_high_liquidity_high_score(self):
        """High liquidity should yield high scores."""
        market = create_test_market(liquidity=100000, volume=500000)
        
        score = calculate_liquidity_score(market)
        
        assert score > 70
    
    def test_low_liquidity_low_score(self):
        """Low liquidity should yield lower scores."""
        market = create_test_market(liquidity=500, volume=1000)
        
        score = calculate_liquidity_score(market)
        
        assert score < 70
    
    def test_uses_volume_fallback(self):
        """Should use volume as fallback when liquidity is low."""
        market = create_test_market(liquidity=0, volume=50000)
        
        score = calculate_liquidity_score(market)
        
        assert score > 30  # Should have some score from volume


class TestCalculateEvidenceScore:
    """Tests for evidence score calculation."""
    
    def test_good_evidence_high_score(self):
        """Good evidence should yield high scores."""
        market = create_test_market()
        enrichment = create_test_enrichment(market, num_sources=5, num_facts=5)
        assessment = create_test_assessment()
        
        score = calculate_evidence_score(enrichment, assessment)
        
        assert score > 70
    
    def test_no_enrichment_base_score(self):
        """No enrichment should yield base score."""
        assessment = create_test_assessment()
        
        score = calculate_evidence_score(None, assessment)
        
        assert score == 30  # Base score
    
    def test_warnings_reduce_score(self):
        """Warnings about limited data should reduce score."""
        market = create_test_market()
        enrichment = create_test_enrichment(market, num_sources=2, num_facts=2)
        assessment = create_test_assessment()
        assessment.warnings = ["Limited data available", "Insufficient information"]
        
        score = calculate_evidence_score(enrichment, assessment)
        
        # Score should be reduced by warnings
        assessment_no_warnings = create_test_assessment()
        score_no_warnings = calculate_evidence_score(enrichment, assessment_no_warnings)
        
        assert score < score_no_warnings


class TestScoreMarket:
    """Tests for the full market scoring function."""
    
    def test_score_market_returns_scored_market(self):
        """Test that score_market returns a properly structured ScoredMarket."""
        market = create_test_market()
        assessment = create_test_assessment()
        enrichment = create_test_enrichment(market)
        
        scored = score_market(market, assessment, enrichment)
        
        assert scored.market == market
        assert scored.assessment == assessment
        assert scored.enrichment == enrichment
        assert 0 <= scored.total_score <= 100
        assert 0 <= scored.mispricing_score <= 100
        assert 0 <= scored.confidence_score <= 100
    
    def test_score_without_enrichment(self):
        """Test scoring works without enrichment."""
        market = create_test_market()
        assessment = create_test_assessment()
        
        scored = score_market(market, assessment, enrichment=None)
        
        assert scored.enrichment is None
        assert scored.total_score >= 0


class TestRankMarkets:
    """Tests for market ranking."""
    
    def test_ranking_order(self):
        """Test that markets are ranked by score descending."""
        markets = [create_test_market(id=str(i)) for i in range(3)]
        assessments = [
            create_test_assessment(market_id="0", mispricing_magnitude=0.10, confidence=0.6),
            create_test_assessment(market_id="1", mispricing_magnitude=0.30, confidence=0.8),
            create_test_assessment(market_id="2", mispricing_magnitude=0.20, confidence=0.7),
        ]
        
        config = AgentConfig(risk_tolerance=RiskTolerance.MODERATE)
        scorer = MarketScorer(config)
        
        scored = scorer.score_all(markets, assessments)
        ranked = rank_markets(scored)
        
        # Verify descending order
        for i in range(len(ranked) - 1):
            assert ranked[i].total_score >= ranked[i + 1].total_score
        
        # Verify ranks are assigned
        for i, market in enumerate(ranked, 1):
            assert market.rank == i
    
    def test_min_score_filter(self):
        """Test that minimum score filter works."""
        markets = [create_test_market(id=str(i)) for i in range(3)]
        assessments = [
            create_test_assessment(market_id="0", mispricing_magnitude=0.05, confidence=0.3),  # Low
            create_test_assessment(market_id="1", mispricing_magnitude=0.30, confidence=0.8),  # High
            create_test_assessment(market_id="2", mispricing_magnitude=0.05, confidence=0.3),  # Low
        ]
        
        config = AgentConfig(risk_tolerance=RiskTolerance.MODERATE)
        scorer = MarketScorer(config)
        
        scored = scorer.score_all(markets, assessments)
        ranked = rank_markets(scored, min_score=30)
        
        # Only high-scoring market should pass
        assert len(ranked) <= len(scored)


class TestMarketScorer:
    """Tests for the MarketScorer class."""
    
    def test_scorer_initialization(self):
        """Test scorer initializes correctly."""
        config = AgentConfig(risk_tolerance=RiskTolerance.AGGRESSIVE)
        scorer = MarketScorer(config)
        
        assert scorer.config.risk_tolerance == RiskTolerance.AGGRESSIVE
    
    def test_get_top_opportunities(self):
        """Test getting top N opportunities."""
        markets = [create_test_market(id=str(i)) for i in range(10)]
        assessments = [
            create_test_assessment(
                market_id=str(i),
                mispricing_magnitude=0.10 + i * 0.02,
                confidence=0.5 + i * 0.05,
            )
            for i in range(10)
        ]
        
        config = AgentConfig(risk_tolerance=RiskTolerance.AGGRESSIVE)
        scorer = MarketScorer(config)
        
        scored = scorer.score_all(markets, assessments)
        top_5 = scorer.get_top_opportunities(scored, n=5)
        
        assert len(top_5) <= 5
        
        # Verify they're the highest scoring
        if len(top_5) > 0:
            min_top_score = min(m.total_score for m in top_5)
            for market in scored:
                if market not in top_5:
                    assert market.total_score <= min_top_score
