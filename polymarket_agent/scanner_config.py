"""Configuration for the bias scanner."""

from dataclasses import dataclass


@dataclass
class ScannerConfig:
    """
    Configuration for the Polymarket bias scanner.

    Attributes:
        min_volume: Minimum trading volume in USD
        min_liquidity: Minimum liquidity in USD
        max_days_to_expiry: Maximum days until resolution
        llm_model: LLM model for classification
        max_markets: Maximum markets to fetch
        output_dir: Directory for output reports
        verbose: Enable verbose logging
    """
    min_volume: float = 1000
    min_liquidity: float = 500
    max_days_to_expiry: int = 90
    llm_model: str = "claude-haiku-4-5"
    max_markets: int = 500
    output_dir: str = "output"
    verbose: bool = False
