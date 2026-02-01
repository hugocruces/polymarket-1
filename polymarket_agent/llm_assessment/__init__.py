"""
LLM Assessment Module

Provides multi-provider LLM integration for market probability assessment.
"""

from polymarket_agent.llm_assessment.providers import (
    LLMClient,
    AnthropicClient,
    OpenAIClient,
    GoogleClient,
    get_llm_client,
)
from polymarket_agent.llm_assessment.assessor import (
    MarketAssessor,
    assess_market,
    assess_markets_batch,
)
from polymarket_agent.llm_assessment.prompts import (
    SYSTEM_PROMPT,
    ASSESSMENT_PROMPT_TEMPLATE,
    build_assessment_prompt,
)
from polymarket_agent.llm_assessment.consensus import (
    ConsensusAssessment,
    aggregate_assessments,
)

__all__ = [
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "GoogleClient",
    "get_llm_client",
    "MarketAssessor",
    "assess_market",
    "assess_markets_batch",
    "SYSTEM_PROMPT",
    "ASSESSMENT_PROMPT_TEMPLATE",
    "build_assessment_prompt",
    "ConsensusAssessment",
    "aggregate_assessments",
]
