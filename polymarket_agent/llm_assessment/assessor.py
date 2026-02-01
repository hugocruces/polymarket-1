"""
Market Assessor

Orchestrates LLM-based assessment of prediction markets.
Handles prompt building, LLM calling, and response parsing.
"""

import asyncio
import json
import logging
from typing import Optional

from polymarket_agent.config import AgentConfig
from polymarket_agent.data_fetching.models import (
    Market,
    EnrichedMarket,
    LLMAssessment,
)
from polymarket_agent.llm_assessment.providers import LLMClient, get_llm_client
from polymarket_agent.llm_assessment.prompts import (
    SYSTEM_PROMPT,
    build_assessment_prompt,
    extract_json_from_response,
)

logger = logging.getLogger(__name__)


class AssessmentError(Exception):
    """Exception raised when assessment fails."""
    pass


def parse_assessment_response(
    response_text: str,
    market: Market,
    model_used: str,
) -> LLMAssessment:
    """
    Parse the LLM response into an LLMAssessment object.
    
    Handles various response formats and extracts structured data.
    
    Args:
        response_text: Raw LLM response
        market: The market being assessed
        model_used: Model identifier that generated the response
        
    Returns:
        Parsed LLMAssessment object
        
    Raises:
        AssessmentError: If response cannot be parsed
    """
    try:
        # Extract JSON from response
        json_str = extract_json_from_response(response_text)
        data = json.loads(json_str)
        
        # Parse probability estimates
        prob_estimates = {}
        raw_estimates = data.get("probability_estimates", {})
        
        for outcome_name, estimate in raw_estimates.items():
            if isinstance(estimate, list) and len(estimate) == 2:
                low, high = float(estimate[0]), float(estimate[1])
                # Normalize if given as percentages
                if low > 1 or high > 1:
                    low, high = low / 100, high / 100
                prob_estimates[outcome_name] = (low, high)
            elif isinstance(estimate, (int, float)):
                # Single value - create small range around it
                val = float(estimate)
                if val > 1:
                    val = val / 100
                prob_estimates[outcome_name] = (max(0, val - 0.05), min(1, val + 0.05))
        
        # Ensure we have estimates for all outcomes
        for outcome in market.outcomes:
            if outcome.name not in prob_estimates:
                # Default to market price with uncertainty
                price = outcome.price
                prob_estimates[outcome.name] = (
                    max(0, price - 0.1),
                    min(1, price + 0.1)
                )
        
        # Parse confidence
        confidence = float(data.get("confidence", 0.5))
        if confidence > 1:
            confidence = confidence / 100
        confidence = max(0, min(1, confidence))
        
        # Parse mispricing info
        mispricing_detected = bool(data.get("mispricing_detected", False))
        mispricing_direction = data.get("mispricing_direction", "fair")
        if mispricing_direction not in ["overpriced", "underpriced", "fair"]:
            mispricing_direction = "fair"
        
        mispricing_magnitude = float(data.get("mispricing_magnitude", 0))
        if mispricing_magnitude > 1:
            mispricing_magnitude = mispricing_magnitude / 100
        mispricing_magnitude = max(0, min(1, mispricing_magnitude))
        
        # Extract bias_adjustment if present
        bias_adjustment = data.get("bias_adjustment")
        if bias_adjustment and not isinstance(bias_adjustment, dict):
            bias_adjustment = None

        return LLMAssessment(
            market_id=market.id,
            probability_estimates=prob_estimates,
            confidence=confidence,
            reasoning=data.get("reasoning", ""),
            key_factors=data.get("key_factors", []),
            risks=data.get("risks", []),
            mispricing_detected=mispricing_detected,
            mispricing_direction=mispricing_direction,
            mispricing_magnitude=mispricing_magnitude,
            warnings=data.get("warnings", []),
            model_used=model_used,
            bias_adjustment=bias_adjustment,
        )
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from response: {e}")
        logger.debug(f"Response text: {response_text[:500]}...")
        
        # Return a default assessment with high uncertainty
        return LLMAssessment(
            market_id=market.id,
            probability_estimates={
                o.name: (max(0, o.price - 0.15), min(1, o.price + 0.15))
                for o in market.outcomes
            },
            confidence=0.2,
            reasoning="Failed to parse LLM response. Using market prices with high uncertainty.",
            key_factors=[],
            risks=["Assessment failed - response parsing error"],
            mispricing_detected=False,
            mispricing_direction="fair",
            mispricing_magnitude=0,
            warnings=["Failed to parse LLM response", "Using fallback assessment"],
            model_used=model_used,
        )
    except Exception as e:
        logger.error(f"Error parsing assessment: {e}")
        raise AssessmentError(f"Failed to parse assessment: {e}")


class MarketAssessor:
    """
    Handles LLM-based assessment of prediction markets.

    Manages the LLM client, prompt construction, and response parsing.
    Supports single-model and multi-model consensus modes.

    Example:
        >>> assessor = MarketAssessor(config)
        >>> assessment = await assessor.assess(market, enrichment)
        >>> print(f"Confidence: {assessment.confidence}")
        >>> print(f"Mispricing: {assessment.mispricing_direction}")
    """

    def __init__(
        self,
        config: AgentConfig,
        llm_client: Optional[LLMClient] = None,
    ):
        """
        Initialize the assessor.

        Args:
            config: Agent configuration
            llm_client: Optional pre-configured LLM client
        """
        self.config = config
        self.model_name = config.llm_model

        # Consensus mode: multiple models
        self.consensus_mode = bool(config.consensus_models)
        if self.consensus_mode:
            self.consensus_clients = {
                model: get_llm_client(model) for model in config.consensus_models
            }
            self.llm_client = list(self.consensus_clients.values())[0]
        else:
            self.llm_client = llm_client or get_llm_client(config.llm_model)
            self.consensus_clients = {}
    
    async def assess(
        self,
        market: Market,
        enrichment: Optional[EnrichedMarket] = None,
    ) -> LLMAssessment:
        """
        Assess a single market using the LLM.

        If consensus mode is enabled, runs multiple models concurrently
        and aggregates their assessments.

        Args:
            market: Market to assess
            enrichment: Optional enrichment with external context

        Returns:
            LLMAssessment with probability estimates and analysis
        """
        if self.consensus_mode:
            return await self._assess_consensus(market, enrichment)
        return await self._assess_single(market, enrichment, self.llm_client, self.model_name)

    async def _assess_single(
        self,
        market: Market,
        enrichment: Optional[EnrichedMarket],
        client: LLMClient,
        model_name: str,
    ) -> LLMAssessment:
        """Assess a single market with a single model."""
        prompt = build_assessment_prompt(market, enrichment)

        try:
            response = await client.complete(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                max_tokens=self.config.llm_config.get("max_tokens", 4096),
                temperature=0.3,
            )

            logger.debug(
                f"LLM response for {market.slug} ({model_name}): "
                f"{response.input_tokens} in, {response.output_tokens} out"
            )

            assessment = parse_assessment_response(
                response.content,
                market,
                model_name,
            )

            return assessment

        except Exception as e:
            logger.error(f"LLM assessment failed for {market.slug} ({model_name}): {e}")

            return LLMAssessment(
                market_id=market.id,
                probability_estimates={
                    o.name: (max(0, o.price - 0.1), min(1, o.price + 0.1))
                    for o in market.outcomes
                },
                confidence=0.1,
                reasoning=f"Assessment failed due to error: {e}",
                key_factors=[],
                risks=["Assessment failed"],
                mispricing_detected=False,
                mispricing_direction="fair",
                mispricing_magnitude=0,
                warnings=[f"LLM error: {str(e)[:100]}"],
                model_used=model_name,
            )

    async def _assess_consensus(
        self,
        market: Market,
        enrichment: Optional[EnrichedMarket],
    ) -> LLMAssessment:
        """Assess using multiple models and aggregate via consensus."""
        from polymarket_agent.llm_assessment.consensus import aggregate_assessments

        # Run all models concurrently
        tasks = []
        for model_name, client in self.consensus_clients.items():
            tasks.append(self._assess_single(market, enrichment, client, model_name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_assessments = []
        for r in results:
            if isinstance(r, LLMAssessment) and r.confidence > 0.1:
                valid_assessments.append(r)
            elif isinstance(r, Exception):
                logger.warning(f"Consensus model failed: {r}")

        if not valid_assessments:
            # All failed, return fallback
            return LLMAssessment(
                market_id=market.id,
                probability_estimates={
                    o.name: (max(0, o.price - 0.15), min(1, o.price + 0.15))
                    for o in market.outcomes
                },
                confidence=0.1,
                reasoning="All consensus models failed",
                mispricing_detected=False,
                mispricing_direction="fair",
                mispricing_magnitude=0,
                warnings=["All consensus models failed"],
                model_used="consensus:none",
            )

        consensus = aggregate_assessments(valid_assessments)
        return consensus.to_llm_assessment(market.id)
    
    async def assess_batch(
        self,
        markets: list[Market],
        enrichments: Optional[dict[str, EnrichedMarket]] = None,
        progress_callback: Optional[callable] = None,
    ) -> list[LLMAssessment]:
        """
        Assess multiple markets.
        
        Markets are assessed sequentially to avoid rate limiting.
        
        Args:
            markets: List of markets to assess
            enrichments: Optional dict mapping market IDs to enrichments
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of LLMAssessment objects
        """
        if enrichments is None:
            enrichments = {}
        
        assessments = []
        total = len(markets)
        
        for i, market in enumerate(markets):
            enrichment = enrichments.get(market.id)
            
            assessment = await self.assess(market, enrichment)
            assessments.append(assessment)
            
            if progress_callback:
                progress_callback(i + 1, total)
            
            # Small delay between requests to avoid rate limiting
            if i < total - 1:
                await asyncio.sleep(0.5)
        
        return assessments


async def assess_market(
    market: Market,
    enrichment: Optional[EnrichedMarket] = None,
    model: str = "claude-sonnet-4-5",
) -> LLMAssessment:
    """
    Convenience function to assess a single market.
    
    Args:
        market: Market to assess
        enrichment: Optional enrichment data
        model: LLM model to use
        
    Returns:
        LLMAssessment
    """
    from polymarket_agent.config import AgentConfig
    
    config = AgentConfig(llm_model=model)
    assessor = MarketAssessor(config)
    
    return await assessor.assess(market, enrichment)


async def assess_markets_batch(
    markets: list[Market],
    enrichments: Optional[dict[str, EnrichedMarket]] = None,
    model: str = "claude-sonnet-4-5",
    progress_callback: Optional[callable] = None,
) -> list[LLMAssessment]:
    """
    Convenience function to assess multiple markets.
    
    Args:
        markets: List of markets
        enrichments: Optional enrichments dict
        model: LLM model to use
        progress_callback: Optional progress callback
        
    Returns:
        List of LLMAssessment objects
    """
    from polymarket_agent.config import AgentConfig
    
    config = AgentConfig(llm_model=model)
    assessor = MarketAssessor(config)
    
    return await assessor.assess_batch(markets, enrichments, progress_callback)
