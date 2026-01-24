"""
Tests for the LLM assessment module.

These tests verify prompt building and response parsing.
"""

import pytest

from polymarket_agent.data_fetching.models import Market, Outcome, EnrichedMarket
from polymarket_agent.llm_assessment.prompts import (
    build_assessment_prompt,
    format_outcome_prices,
    format_key_facts,
    extract_json_from_response,
    SYSTEM_PROMPT,
)
from polymarket_agent.llm_assessment.assessor import parse_assessment_response
from datetime import datetime, timedelta


def create_test_market() -> Market:
    """Create a test market for prompt testing."""
    return Market(
        id="test-market-1",
        slug="test-market-1",
        question="Will Bitcoin reach $100,000 by end of 2024?",
        description="This market resolves YES if Bitcoin price reaches $100,000 USD on any major exchange.",
        outcomes=[
            Outcome(name="Yes", token_id="yes-token", price=0.45),
            Outcome(name="No", token_id="no-token", price=0.55),
        ],
        category="crypto",
        tags=["crypto", "bitcoin"],
        volume=150000,
        liquidity=75000,
        end_date=datetime.now() + timedelta(days=60),
    )


def create_test_enrichment(market: Market) -> EnrichedMarket:
    """Create test enrichment data."""
    return EnrichedMarket(
        market=market,
        external_context="Recent news suggests increasing institutional adoption of Bitcoin.",
        sources=["https://example.com/news1", "https://example.com/news2"],
        key_facts=[
            "Bitcoin recently crossed $80,000",
            "Major ETF inflows continue",
            "Fed signals rate cuts ahead",
        ],
        search_query="Bitcoin $100k prediction 2024",
    )


class TestFormatOutcomePrices:
    """Tests for outcome price formatting."""
    
    def test_format_binary_outcomes(self):
        """Test formatting binary outcome prices."""
        outcomes = [
            Outcome(name="Yes", token_id="y", price=0.65),
            Outcome(name="No", token_id="n", price=0.35),
        ]
        
        result = format_outcome_prices(outcomes)
        
        assert "Yes: 65.0%" in result
        assert "No: 35.0%" in result
    
    def test_format_multiple_outcomes(self):
        """Test formatting multiple outcomes."""
        outcomes = [
            Outcome(name="Trump", token_id="t", price=0.45),
            Outcome(name="Biden", token_id="b", price=0.30),
            Outcome(name="Other", token_id="o", price=0.25),
        ]
        
        result = format_outcome_prices(outcomes)
        
        assert "Trump: 45.0%" in result
        assert "Biden: 30.0%" in result
        assert "Other: 25.0%" in result


class TestFormatKeyFacts:
    """Tests for key facts formatting."""
    
    def test_format_facts(self):
        """Test formatting key facts."""
        facts = ["Fact one", "Fact two"]
        
        result = format_key_facts(facts)
        
        assert "- Fact one" in result
        assert "- Fact two" in result
    
    def test_format_empty_facts(self):
        """Test formatting when no facts available."""
        result = format_key_facts([])
        
        assert "No key facts" in result


class TestBuildAssessmentPrompt:
    """Tests for prompt building."""
    
    def test_prompt_contains_market_info(self):
        """Test that prompt includes market information."""
        market = create_test_market()
        
        prompt = build_assessment_prompt(market)
        
        assert market.question in prompt
        assert "crypto" in prompt.lower()
        assert "$150,000" in prompt or "150,000" in prompt  # Volume
    
    def test_prompt_contains_prices(self):
        """Test that prompt includes outcome prices."""
        market = create_test_market()
        
        prompt = build_assessment_prompt(market)
        
        assert "45" in prompt  # Yes price
        assert "55" in prompt  # No price
    
    def test_prompt_with_enrichment(self):
        """Test that prompt includes enrichment data."""
        market = create_test_market()
        enrichment = create_test_enrichment(market)
        
        prompt = build_assessment_prompt(market, enrichment)
        
        assert "institutional adoption" in prompt
        assert "Bitcoin recently crossed" in prompt
    
    def test_prompt_without_enrichment(self):
        """Test prompt generation without enrichment."""
        market = create_test_market()
        
        prompt = build_assessment_prompt(market, enrichment=None)
        
        assert "No external context available" in prompt


class TestExtractJsonFromResponse:
    """Tests for JSON extraction from LLM responses."""
    
    def test_extract_from_code_block(self):
        """Test extracting JSON from markdown code block."""
        response = '''Here's my analysis:

```json
{"confidence": 0.7, "mispricing_detected": true}
```

That's my assessment.'''
        
        result = extract_json_from_response(response)
        
        assert '"confidence": 0.7' in result
    
    def test_extract_from_labeled_code_block(self):
        """Test extracting from ```json block."""
        response = '''```json
{"test": "value"}
```'''
        
        result = extract_json_from_response(response)
        
        assert '"test": "value"' in result
    
    def test_extract_raw_json(self):
        """Test extracting raw JSON without code block."""
        response = '{"confidence": 0.5, "reasoning": "test"}'
        
        result = extract_json_from_response(response)
        
        assert result == response
    
    def test_extract_embedded_json(self):
        """Test extracting JSON embedded in text."""
        response = 'Based on my analysis: {"result": true} end.'
        
        result = extract_json_from_response(response)
        
        assert '{"result": true}' in result


class TestParseAssessmentResponse:
    """Tests for parsing LLM assessment responses."""
    
    def test_parse_valid_response(self):
        """Test parsing a valid JSON response."""
        market = create_test_market()
        response = '''{
            "probability_estimates": {
                "Yes": [0.50, 0.60],
                "No": [0.40, 0.50]
            },
            "confidence": 0.75,
            "reasoning": "Based on market trends...",
            "key_factors": ["Factor 1", "Factor 2"],
            "risks": ["Risk 1"],
            "mispricing_detected": true,
            "mispricing_direction": "underpriced",
            "mispricing_magnitude": 0.12,
            "warnings": []
        }'''
        
        assessment = parse_assessment_response(response, market, "test-model")
        
        assert assessment.confidence == 0.75
        assert assessment.mispricing_detected is True
        assert assessment.mispricing_direction == "underpriced"
        assert assessment.probability_estimates["Yes"] == (0.50, 0.60)
    
    def test_parse_percentage_estimates(self):
        """Test parsing probability estimates given as percentages."""
        market = create_test_market()
        response = '''{
            "probability_estimates": {
                "Yes": [50, 60]
            },
            "confidence": 75,
            "reasoning": "test",
            "mispricing_detected": false,
            "mispricing_direction": "fair",
            "mispricing_magnitude": 0
        }'''
        
        assessment = parse_assessment_response(response, market, "test-model")
        
        # Should be normalized to 0-1 range
        assert assessment.probability_estimates["Yes"] == (0.50, 0.60)
        assert assessment.confidence == 0.75
    
    def test_parse_invalid_json_fallback(self):
        """Test that invalid JSON produces fallback assessment."""
        market = create_test_market()
        response = "This is not valid JSON at all."
        
        assessment = parse_assessment_response(response, market, "test-model")
        
        # Should return a fallback assessment
        assert assessment.market_id == market.id
        assert assessment.confidence == 0.2  # Low confidence fallback
        assert "Failed to parse" in assessment.reasoning
        assert len(assessment.warnings) > 0
    
    def test_parse_with_code_block(self):
        """Test parsing response with code block wrapping."""
        market = create_test_market()
        response = '''```json
{
    "probability_estimates": {"Yes": [0.55, 0.65]},
    "confidence": 0.8,
    "reasoning": "Analysis complete",
    "key_factors": [],
    "risks": [],
    "mispricing_detected": true,
    "mispricing_direction": "overpriced",
    "mispricing_magnitude": 0.15,
    "warnings": []
}
```'''
        
        assessment = parse_assessment_response(response, market, "test-model")
        
        assert assessment.confidence == 0.8
        assert assessment.mispricing_detected is True


class TestSystemPrompt:
    """Tests for system prompt content."""
    
    def test_system_prompt_emphasizes_uncertainty(self):
        """Test that system prompt emphasizes probabilistic thinking."""
        assert "probabilistic" in SYSTEM_PROMPT.lower()
        assert "uncertain" in SYSTEM_PROMPT.lower()
    
    def test_system_prompt_warns_about_hallucination(self):
        """Test that system prompt warns about hallucination."""
        assert "fabricate" in SYSTEM_PROMPT.lower() or "hallucin" in SYSTEM_PROMPT.lower()
    
    def test_system_prompt_not_financial_advice(self):
        """Test that system prompt disclaims financial advice."""
        assert "not" in SYSTEM_PROMPT.lower() and "advice" in SYSTEM_PROMPT.lower()
