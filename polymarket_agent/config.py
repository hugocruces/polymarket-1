"""
Configuration Module

Defines all configuration classes, enums, and constants used throughout the agent.
Supports configuration via environment variables, YAML files, and programmatic setup.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class RiskTolerance(Enum):
    """
    Risk tolerance levels that affect scoring thresholds and market selection.
    
    - CONSERVATIVE: Only high-confidence, large mispricings (>20% deviation)
    - MODERATE: Balanced approach (>10% deviation with reasonable confidence)
    - AGGRESSIVE: Includes speculative opportunities (>5% deviation)
    """
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"


# ============================================================================
# API Endpoints - Polymarket Public APIs
# ============================================================================

# Gamma API - For fetching events and markets metadata
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GAMMA_EVENTS_ENDPOINT = f"{GAMMA_API_BASE}/events"
GAMMA_MARKETS_ENDPOINT = f"{GAMMA_API_BASE}/markets"

# CLOB API - For real-time prices and orderbook data
CLOB_API_BASE = "https://clob.polymarket.com"
CLOB_PRICE_ENDPOINT = f"{CLOB_API_BASE}/price"
CLOB_BOOK_ENDPOINT = f"{CLOB_API_BASE}/book"
CLOB_MARKETS_ENDPOINT = f"{CLOB_API_BASE}/markets"

# ============================================================================
# Default Configuration Values
# ============================================================================

DEFAULT_PAGE_SIZE = 100  # Pagination limit for API requests
DEFAULT_MAX_MARKETS = 500  # Maximum markets to fetch in one run
DEFAULT_MIN_VOLUME = 1000  # Minimum trading volume in USD
DEFAULT_MIN_LIQUIDITY = 500  # Minimum liquidity in USD
DEFAULT_MAX_DAYS_TO_EXPIRY = 90  # Maximum days until market resolution
DEFAULT_ENRICHMENT_LIMIT = 20  # Max markets to enrich with web search
DEFAULT_LLM_ANALYSIS_LIMIT = 15  # Max markets to send to LLM

# Scoring weights (should sum to 1.0)
SCORING_WEIGHTS = {
    "mispricing_magnitude": 0.30,
    "model_confidence": 0.25,
    "evidence_strength": 0.20,
    "liquidity_score": 0.15,
    "risk_adjustment": 0.10,
}

# Risk tolerance thresholds
RISK_THRESHOLDS = {
    RiskTolerance.CONSERVATIVE: {
        "min_mispricing": 0.20,  # 20% minimum deviation
        "min_confidence": 0.75,  # High confidence required
        "max_uncertainty": 0.15,  # Low uncertainty in estimate
    },
    RiskTolerance.MODERATE: {
        "min_mispricing": 0.10,  # 10% minimum deviation
        "min_confidence": 0.55,  # Moderate confidence
        "max_uncertainty": 0.25,
    },
    RiskTolerance.AGGRESSIVE: {
        "min_mispricing": 0.05,  # 5% minimum deviation
        "min_confidence": 0.35,  # Lower confidence acceptable
        "max_uncertainty": 0.40,  # Higher uncertainty tolerated
    },
}

# LLM Model mappings
LLM_MODELS = {
    # ==========================================================================
    # Anthropic Claude Models (Latest)
    # ==========================================================================
    "claude-sonnet-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5",
        "max_tokens": 64000,
    },
    "claude-haiku-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "max_tokens": 64000,
    },
    "claude-opus-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-opus-4-5-20251101",
        "display_name": "Claude Opus 4.5",
        "max_tokens": 64000,
    },

    # ==========================================================================
    # OpenAI Models (Latest)
    # ==========================================================================
    "gpt-5.2": {
        "provider": LLMProvider.OPENAI,
        "model_id": "gpt-5.2-2025-12-11",
        "display_name": "GPT-5.2",
        "max_tokens": 128000,
    },
    "gpt-5-mini": {
        "provider": LLMProvider.OPENAI,
        "model_id": "gpt-5-mini-2025-08-07",
        "display_name": "GPT-5 Mini",
        "max_tokens": 128000,
    },

    # ==========================================================================
    # Google Gemini Models (Latest)
    # ==========================================================================
    "gemini-3-pro-preview": {
        "provider": LLMProvider.GOOGLE,
        "model_id": "gemini-3-pro-preview",
        "display_name": "Gemini 3 Pro (Preview)",
        "max_tokens": 8192,
    },
    "gemini-3-flash-preview": {
        "provider": LLMProvider.GOOGLE,
        "model_id": "gemini-3-flash-preview",
        "display_name": "Gemini 3 Flash (Preview)",
        "max_tokens": 8192,
    },
}

# Default model
DEFAULT_LLM_MODEL = "claude-sonnet-4-5"


@dataclass
class FilterConfig:
    """
    Configuration for market filtering criteria.

    Attributes:
        categories: List of category/tag names to include (e.g., ["politics", "crypto"])
        keywords: Keywords to search for in title/description
        exclude_keywords: Keywords to exclude from results
        min_volume: Minimum trading volume in USD
        max_volume: Maximum trading volume in USD (None = no limit).
            Useful for finding potentially mispriced low-volume markets.
        min_liquidity: Minimum market liquidity in USD
        max_liquidity: Maximum market liquidity in USD (None = no limit)
        max_days_to_expiry: Maximum days until market resolution
        geographic_regions: Filter by geographic relevance. Available regions:
            US, EU, UK, ASIA, LATAM, MIDDLE_EAST, AFRICA, CRYPTO, GLOBAL
        geo_min_score: Minimum confidence score (0-100) for geographic matching.
            - 30 (default): Include if any moderate keyword matches
            - 50: Require strong indicator or multiple matches
            - 70: Require definitive indicator (e.g., "United States Congress")
        min_outcomes: Minimum number of outcomes (default 2)
        max_outcomes: Maximum number of outcomes (None = no limit)
    """
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    exclude_keywords: list[str] = field(default_factory=list)
    min_volume: float = DEFAULT_MIN_VOLUME
    max_volume: Optional[float] = None  # None = no upper limit
    min_liquidity: float = DEFAULT_MIN_LIQUIDITY
    max_liquidity: Optional[float] = None  # None = no upper limit
    max_days_to_expiry: int = DEFAULT_MAX_DAYS_TO_EXPIRY
    geographic_regions: list[str] = field(default_factory=list)
    geo_min_score: float = 30.0  # Minimum score for geographic filtering
    tag_ids: list[int] = field(default_factory=list)  # Filter by Polymarket tag IDs
    min_outcomes: int = 2
    max_outcomes: Optional[int] = None
    # Reasoning analysis filters
    reasoning_heavy_only: bool = False  # Only include reasoning-heavy markets
    min_reasoning_score: float = 20.0  # Minimum reasoning score (0-100)
    llm_edge_levels: list[str] = field(default_factory=lambda: ["high", "medium"])  # Filter by LLM edge likelihood
    # Demographic bias filters (based on Polymarket user demographics research)
    bias_filter_enabled: bool = False  # Filter for markets with demographic bias potential
    min_blind_spot_score: float = 10.0  # Minimum blind spot score (0-100), lower = more inclusive
    mispricing_levels: list[str] = field(default_factory=lambda: ["high", "medium"])  # Filter by mispricing likelihood


@dataclass
class AgentConfig:
    """
    Main configuration for the Polymarket Agent.
    
    Attributes:
        filters: Market filtering configuration
        risk_tolerance: Risk tolerance level for scoring
        llm_model: LLM model identifier to use for assessment
        max_markets_to_fetch: Maximum markets to retrieve from API
        enrichment_limit: Maximum markets to enrich with web search
        llm_analysis_limit: Maximum markets to analyze with LLM
        output_dir: Directory for output reports
        output_format: Output format(s) - "json", "csv", "markdown", or "both" (json+csv)
        verbose: Enable verbose logging
        dry_run: Skip LLM calls, only fetch and filter
    """
    filters: FilterConfig = field(default_factory=FilterConfig)
    risk_tolerance: RiskTolerance = RiskTolerance.MODERATE
    llm_model: str = DEFAULT_LLM_MODEL
    max_markets_to_fetch: int = DEFAULT_MAX_MARKETS
    enrichment_limit: int = DEFAULT_ENRICHMENT_LIMIT
    llm_analysis_limit: int = DEFAULT_LLM_ANALYSIS_LIMIT
    output_dir: str = "output"
    output_format: str = "both"  # "json", "csv", "markdown", or "both"
    verbose: bool = False
    dry_run: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.llm_model not in LLM_MODELS:
            available = ", ".join(LLM_MODELS.keys())
            raise ValueError(
                f"Unknown LLM model: {self.llm_model}. Available: {available}"
            )
        
        if isinstance(self.risk_tolerance, str):
            self.risk_tolerance = RiskTolerance(self.risk_tolerance)
    
    @property
    def llm_config(self) -> dict:
        """Get the LLM configuration for the selected model."""
        return LLM_MODELS[self.llm_model]
    
    @property
    def risk_thresholds(self) -> dict:
        """Get the risk thresholds for the current risk tolerance."""
        return RISK_THRESHOLDS[self.risk_tolerance]
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to the YAML configuration file
            
        Returns:
            AgentConfig instance
        """
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Parse filters if present
        filters_data = data.pop("filters", {})
        filters = FilterConfig(**filters_data) if filters_data else FilterConfig()
        
        # Parse risk tolerance
        if "risk_tolerance" in data:
            data["risk_tolerance"] = RiskTolerance(data["risk_tolerance"])
        
        # Handle nested llm config
        if "llm" in data:
            llm_config = data.pop("llm")
            if "model" in llm_config:
                data["llm_model"] = llm_config["model"]
        
        return cls(filters=filters, **data)


def get_api_key(provider: LLMProvider) -> Optional[str]:
    """
    Get the API key for the specified LLM provider.
    
    Args:
        provider: The LLM provider
        
    Returns:
        API key string or None if not found
    """
    key_names = {
        LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
        LLMProvider.OPENAI: "OPENAI_API_KEY",
        LLMProvider.GOOGLE: "GOOGLE_API_KEY",
    }
    return os.getenv(key_names.get(provider, ""))


def validate_api_keys(config: AgentConfig) -> list[str]:
    """
    Validate that required API keys are present.
    
    Args:
        config: Agent configuration
        
    Returns:
        List of missing key names (empty if all present)
    """
    missing = []
    
    llm_config = config.llm_config
    provider = llm_config["provider"]
    
    if not get_api_key(provider):
        key_names = {
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
        }
        missing.append(key_names[provider])
    
    return missing
