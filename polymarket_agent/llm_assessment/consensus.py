"""
Multi-Model Consensus

Run each market through multiple LLMs concurrently, aggregate estimates,
and measure agreement. Higher agreement = higher confidence.
"""

import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

from polymarket_agent.data_fetching.models import LLMAssessment

logger = logging.getLogger(__name__)


@dataclass
class ConsensusAssessment:
    """Aggregated assessment from multiple models."""
    individual_assessments: list[LLMAssessment]
    models_used: list[str]
    consensus_probability_estimates: dict[str, tuple[float, float]]
    consensus_confidence: float
    agreement_score: float  # 0-1, higher = more agreement
    mispricing_consensus: bool
    mispricing_direction_consensus: str
    mispricing_magnitude_consensus: float
    high_disagreement: bool
    consensus_bias_adjustment: Optional[dict] = None

    def to_llm_assessment(self, market_id: str) -> LLMAssessment:
        """Convert to LLMAssessment for backward compatibility with scorer."""
        model_str = "consensus:" + ",".join(self.models_used)

        # Merge key factors and risks from all assessments
        all_factors = []
        all_risks = []
        all_warnings = []
        for a in self.individual_assessments:
            all_factors.extend(a.key_factors)
            all_risks.extend(a.risks)
            all_warnings.extend(a.warnings)

        # Deduplicate
        key_factors = list(dict.fromkeys(all_factors))[:10]
        risks = list(dict.fromkeys(all_risks))[:10]
        warnings = list(dict.fromkeys(all_warnings))

        if self.high_disagreement:
            warnings.append(
                f"High model disagreement (agreement score: {self.agreement_score:.2f})"
            )

        # Combine reasoning
        reasoning_parts = []
        for a in self.individual_assessments:
            reasoning_parts.append(f"[{a.model_used}]: {a.reasoning}")
        reasoning = "\n\n".join(reasoning_parts)

        return LLMAssessment(
            market_id=market_id,
            probability_estimates=self.consensus_probability_estimates,
            confidence=self.consensus_confidence,
            reasoning=reasoning,
            key_factors=key_factors,
            risks=risks,
            mispricing_detected=self.mispricing_consensus,
            mispricing_direction=self.mispricing_direction_consensus,
            mispricing_magnitude=self.mispricing_magnitude_consensus,
            warnings=warnings,
            model_used=model_str,
            bias_adjustment=self.consensus_bias_adjustment,
        )


def aggregate_assessments(assessments: list[LLMAssessment]) -> ConsensusAssessment:
    """
    Aggregate multiple LLM assessments into a consensus.

    Args:
        assessments: List of assessments from different models

    Returns:
        ConsensusAssessment with averaged estimates and agreement metrics
    """
    if not assessments:
        raise ValueError("Cannot aggregate empty assessment list")

    if len(assessments) == 1:
        a = assessments[0]
        return ConsensusAssessment(
            individual_assessments=assessments,
            models_used=[a.model_used],
            consensus_probability_estimates=a.probability_estimates,
            consensus_confidence=a.confidence,
            agreement_score=1.0,
            mispricing_consensus=a.mispricing_detected,
            mispricing_direction_consensus=a.mispricing_direction,
            mispricing_magnitude_consensus=a.mispricing_magnitude,
            high_disagreement=False,
            consensus_bias_adjustment=a.bias_adjustment,
        )

    models_used = [a.model_used for a in assessments]

    # Aggregate probability estimates by averaging ranges
    all_outcomes = set()
    for a in assessments:
        all_outcomes.update(a.probability_estimates.keys())

    consensus_estimates = {}
    midpoint_values = {}  # For agreement calculation

    for outcome in all_outcomes:
        lows = []
        highs = []
        for a in assessments:
            if outcome in a.probability_estimates:
                low, high = a.probability_estimates[outcome]
                lows.append(low)
                highs.append(high)

        if lows and highs:
            avg_low = sum(lows) / len(lows)
            avg_high = sum(highs) / len(highs)
            consensus_estimates[outcome] = (avg_low, avg_high)
            midpoint_values[outcome] = [(l + h) / 2 for l, h in zip(lows, highs)]

    # Agreement score: 1 - normalized std dev of midpoints
    all_stds = []
    for outcome, midpoints in midpoint_values.items():
        if len(midpoints) >= 2:
            std = statistics.stdev(midpoints)
            all_stds.append(std)

    avg_std = sum(all_stds) / len(all_stds) if all_stds else 0
    # Normalize: std of 0.5 (max disagreement) maps to agreement 0
    agreement_score = max(0, 1 - (avg_std / 0.5))

    high_disagreement = avg_std > 0.15

    # Confidence: average confidence, weighted by range tightness
    confidences = [a.confidence for a in assessments]
    # Weight by inverse range width (tighter range = more weight)
    weights = []
    for a in assessments:
        ranges = list(a.probability_estimates.values())
        if ranges:
            avg_width = sum(h - l for l, h in ranges) / len(ranges)
            weights.append(1 / (avg_width + 0.01))
        else:
            weights.append(1.0)

    total_weight = sum(weights)
    consensus_confidence = sum(c * w for c, w in zip(confidences, weights)) / total_weight

    # Mispricing: majority vote
    mispricing_votes = [a.mispricing_detected for a in assessments]
    mispricing_consensus = sum(mispricing_votes) > len(mispricing_votes) / 2

    # Direction: majority vote
    direction_counts: dict[str, int] = {}
    for a in assessments:
        d = a.mispricing_direction
        direction_counts[d] = direction_counts.get(d, 0) + 1
    direction_consensus = max(direction_counts, key=direction_counts.get)

    # Magnitude: average
    magnitude_consensus = sum(a.mispricing_magnitude for a in assessments) / len(assessments)

    # Bias adjustment: merge
    bias_adjustments = [a.bias_adjustment for a in assessments if a.bias_adjustment]
    consensus_bias = None
    if bias_adjustments:
        # Take the first one's structure, average magnitude
        consensus_bias = bias_adjustments[0].copy()
        if len(bias_adjustments) > 1:
            mags = [b.get("estimated_skew_magnitude", 0) for b in bias_adjustments]
            consensus_bias["estimated_skew_magnitude"] = sum(mags) / len(mags)

    return ConsensusAssessment(
        individual_assessments=assessments,
        models_used=models_used,
        consensus_probability_estimates=consensus_estimates,
        consensus_confidence=consensus_confidence,
        agreement_score=agreement_score,
        mispricing_consensus=mispricing_consensus,
        mispricing_direction_consensus=direction_consensus,
        mispricing_magnitude_consensus=magnitude_consensus,
        high_disagreement=high_disagreement,
        consensus_bias_adjustment=consensus_bias,
    )
