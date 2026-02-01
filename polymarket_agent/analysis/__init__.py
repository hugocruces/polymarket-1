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
    analyze_bias_with_llm,
    filter_by_bias_potential,
    get_bias_summary,
    get_demographic_profile,
    DEMOGRAPHIC_BIASES,
)

from polymarket_agent.analysis.spread_analysis import (
    SlippageEstimate,
    SpreadAnalysis,
    calculate_slippage,
    analyze_spread,
    analyze_spreads_batch,
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
    "analyze_bias_with_llm",
    "filter_by_bias_potential",
    "get_bias_summary",
    "get_demographic_profile",
    "DEMOGRAPHIC_BIASES",
    # Spread analysis
    "SlippageEstimate",
    "SpreadAnalysis",
    "calculate_slippage",
    "analyze_spread",
    "analyze_spreads_batch",
]
