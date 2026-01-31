# Analysis module for market classification and assessment

from polymarket_agent.analysis.reasoning_classifier import (
    ReasoningAnalysis,
    analyze_market_reasoning,
    filter_reasoning_heavy,
    get_reasoning_summary,
)

from polymarket_agent.analysis.demographic_bias import (
    BiasAnalysis,
    DemographicProfile,
    analyze_demographic_bias,
    filter_by_bias_potential,
    get_bias_summary,
    get_demographic_profile,
    DEMOGRAPHIC_BIASES,
)

__all__ = [
    # Reasoning classifier
    "ReasoningAnalysis",
    "analyze_market_reasoning",
    "filter_reasoning_heavy",
    "get_reasoning_summary",
    # Demographic bias
    "BiasAnalysis",
    "DemographicProfile",
    "analyze_demographic_bias",
    "filter_by_bias_potential",
    "get_bias_summary",
    "get_demographic_profile",
    "DEMOGRAPHIC_BIASES",
]
