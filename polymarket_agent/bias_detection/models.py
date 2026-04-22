"""
Data models for bias detection.
"""

from dataclasses import dataclass
from enum import Enum

from polymarket_agent.data_fetching.models import Market


class BiasCategory(Enum):
    """Categories of bias that may affect Polymarket predictions."""

    POLITICAL = "political"
    PROGRESSIVE_SOCIAL = "progressive_social"
    CRYPTO_OPTIMISM = "crypto_optimism"
    ALWAYS_MONITORED = "always_monitored"


@dataclass
class BiasClassification:
    """Classification of a market's susceptibility to demographic biases.

    Attributes:
        market_id: Unique identifier for the market.
        dominated_by_bias: Whether the market is significantly affected by bias.
        categories: List of bias categories affecting this market.
        bias_score: Score from 0-100 indicating strength of bias effect.
        mispricing_direction: Expected direction of mispricing due to bias
            ("overpriced", "underpriced", or "unclear").
        european: Whether the market topic is European-focused.
        spain: Whether the market topic is Spain-specific.
        reasoning: Explanation of the bias classification.
    """

    market_id: str
    dominated_by_bias: bool
    categories: list[BiasCategory]
    bias_score: int
    mispricing_direction: str
    european: bool
    spain: bool
    reasoning: str


@dataclass
class ClassifiedMarket:
    """A market with its bias classification."""

    market: Market
    classification: BiasClassification

    @property
    def volume(self) -> float:
        """Market volume for convenience."""
        return self.market.volume

    @property
    def liquidity(self) -> float:
        """Market liquidity for convenience."""
        return self.market.liquidity
