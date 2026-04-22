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


class MispricingDirection(Enum):
    """Expected direction of mispricing due to demographic bias."""

    OVERPRICED = "overpriced"
    UNDERPRICED = "underpriced"
    UNCLEAR = "unclear"
    NOT_APPLICABLE = "n/a"


class ClassificationError(Exception):
    """Raised when an LLM response cannot be parsed into a BiasClassification."""


@dataclass
class BiasClassification:
    """Classification of a market's susceptibility to demographic biases.

    Attributes:
        market_id: Unique identifier for the market.
        dominated_by_bias: Whether the market is significantly affected by bias.
        categories: List of bias categories affecting this market.
        bias_score: Score from 0-100 indicating strength of bias effect.
        mispricing_direction: Expected direction of mispricing due to bias.
        european: Whether the market topic is European-focused.
        spain: Whether the market topic is Spain-specific.
        reasoning: Explanation of the bias classification.
    """

    market_id: str
    dominated_by_bias: bool
    categories: list[BiasCategory]
    bias_score: int
    mispricing_direction: MispricingDirection
    european: bool
    spain: bool
    reasoning: str


@dataclass
class ClassificationFailure:
    """A market whose LLM classification failed.

    Surfaced in reporting so silent drops don't masquerade as no-bias.
    """

    market: Market
    error: str


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
