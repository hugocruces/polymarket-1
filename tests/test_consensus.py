"""
Tests for multi-model consensus assessment.
"""

import pytest

from polymarket_agent.data_fetching.models import LLMAssessment
from polymarket_agent.llm_assessment.consensus import (
    aggregate_assessments,
    ConsensusAssessment,
)


def make_assessment(
    model: str = "model-a",
    yes_range: tuple = (0.60, 0.70),
    no_range: tuple = (0.30, 0.40),
    confidence: float = 0.7,
    mispricing_detected: bool = True,
    mispricing_direction: str = "underpriced",
    mispricing_magnitude: float = 0.10,
) -> LLMAssessment:
    return LLMAssessment(
        market_id="m1",
        probability_estimates={"Yes": yes_range, "No": no_range},
        confidence=confidence,
        reasoning=f"Reasoning from {model}",
        key_factors=[f"Factor from {model}"],
        risks=[f"Risk from {model}"],
        mispricing_detected=mispricing_detected,
        mispricing_direction=mispricing_direction,
        mispricing_magnitude=mispricing_magnitude,
        model_used=model,
    )


class TestAggregateAssessments:
    """Tests for aggregating multiple model assessments."""

    def test_single_assessment(self):
        """Single assessment should pass through."""
        a = make_assessment("model-a")
        result = aggregate_assessments([a])

        assert result.agreement_score == 1.0
        assert result.high_disagreement is False
        assert len(result.models_used) == 1
        assert result.consensus_probability_estimates == a.probability_estimates

    def test_two_agreeing_models(self):
        """Two models with similar estimates should have high agreement."""
        a1 = make_assessment("model-a", yes_range=(0.60, 0.70))
        a2 = make_assessment("model-b", yes_range=(0.62, 0.72))
        result = aggregate_assessments([a1, a2])

        assert result.agreement_score > 0.9
        assert result.high_disagreement is False
        assert len(result.models_used) == 2

        # Consensus should be average
        yes_est = result.consensus_probability_estimates["Yes"]
        assert abs(yes_est[0] - 0.61) < 0.01
        assert abs(yes_est[1] - 0.71) < 0.01

    def test_high_disagreement_flagged(self):
        """Models that disagree significantly should flag high_disagreement."""
        a1 = make_assessment("model-a", yes_range=(0.20, 0.30))
        a2 = make_assessment("model-b", yes_range=(0.70, 0.80))
        result = aggregate_assessments([a1, a2])

        assert result.agreement_score < 0.7
        assert result.high_disagreement is True

    def test_agreement_score_perfect(self):
        """Identical estimates should give agreement_score = 1.0."""
        a1 = make_assessment("model-a", yes_range=(0.60, 0.70))
        a2 = make_assessment("model-b", yes_range=(0.60, 0.70))
        result = aggregate_assessments([a1, a2])

        assert abs(result.agreement_score - 1.0) < 0.01

    def test_mispricing_majority_vote(self):
        """Mispricing consensus should follow majority."""
        a1 = make_assessment("a", mispricing_detected=True, mispricing_direction="underpriced")
        a2 = make_assessment("b", mispricing_detected=True, mispricing_direction="underpriced")
        a3 = make_assessment("c", mispricing_detected=False, mispricing_direction="fair")
        result = aggregate_assessments([a1, a2, a3])

        assert result.mispricing_consensus is True
        assert result.mispricing_direction_consensus == "underpriced"

    def test_to_llm_assessment_conversion(self):
        """ConsensusAssessment should convert to LLMAssessment."""
        a1 = make_assessment("model-a")
        a2 = make_assessment("model-b")
        result = aggregate_assessments([a1, a2])

        llm_a = result.to_llm_assessment("m1")

        assert llm_a.market_id == "m1"
        assert "consensus:" in llm_a.model_used
        assert "model-a" in llm_a.model_used
        assert "model-b" in llm_a.model_used
        assert llm_a.probability_estimates == result.consensus_probability_estimates
        assert llm_a.confidence == result.consensus_confidence

    def test_confidence_weighted_by_range_tightness(self):
        """Model with tighter range should contribute more to consensus confidence."""
        # model-a: tight range, high confidence
        a1 = make_assessment("model-a", yes_range=(0.64, 0.66), confidence=0.9)
        # model-b: wide range, low confidence
        a2 = make_assessment("model-b", yes_range=(0.40, 0.80), confidence=0.3)
        result = aggregate_assessments([a1, a2])

        # Consensus confidence should be closer to model-a (tighter range, higher weight)
        assert result.consensus_confidence > 0.5

    def test_three_models(self):
        """Three models should aggregate properly."""
        a1 = make_assessment("a", yes_range=(0.55, 0.65), confidence=0.6)
        a2 = make_assessment("b", yes_range=(0.60, 0.70), confidence=0.7)
        a3 = make_assessment("c", yes_range=(0.58, 0.68), confidence=0.8)
        result = aggregate_assessments([a1, a2, a3])

        assert len(result.models_used) == 3
        assert result.agreement_score > 0.8

    def test_single_valid_after_failures(self):
        """If only one valid assessment remains, it should still work."""
        a = make_assessment("model-a", confidence=0.8)
        result = aggregate_assessments([a])

        assert result.agreement_score == 1.0
        assert result.consensus_confidence == 0.8

    def test_empty_list_raises(self):
        """Empty assessment list should raise ValueError."""
        with pytest.raises(ValueError):
            aggregate_assessments([])
