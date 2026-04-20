"""
Configuration Module

Defines all configuration classes, enums, and constants used throughout the agent.
Supports configuration via environment variables, YAML files, and programmatic setup.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

# ============================================================================
# Default Configuration Values
# ============================================================================

DEFAULT_PAGE_SIZE = 100  # Pagination limit for API requests
DEFAULT_MAX_MARKETS = 500  # Maximum markets to fetch in one run
DEFAULT_MIN_VOLUME = 1000  # Minimum trading volume in USD
DEFAULT_MIN_LIQUIDITY = 500  # Minimum liquidity in USD
DEFAULT_MAX_DAYS_TO_EXPIRY = 90  # Maximum days until market resolution

# LLM Model mappings
LLM_MODELS = {
    # ==========================================================================
    # Anthropic Claude Models
    # ==========================================================================
    "claude-sonnet-4-6": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "max_tokens": 64000,
    },
    "claude-haiku-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-haiku-4-5-20251001",
        "display_name": "Claude Haiku 4.5",
        "max_tokens": 64000,
    },
    "claude-opus-4-7": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-opus-4-7",
        "display_name": "Claude Opus 4.7",
        "max_tokens": 64000,
    },

    # ==========================================================================
    # OpenAI Models
    # ==========================================================================
    "gpt-4o": {
        "provider": LLMProvider.OPENAI,
        "model_id": "gpt-4o",
        "display_name": "GPT-4o",
        "max_tokens": 128000,
    },
    "gpt-4o-mini": {
        "provider": LLMProvider.OPENAI,
        "model_id": "gpt-4o-mini",
        "display_name": "GPT-4o Mini",
        "max_tokens": 128000,
    },

    # ==========================================================================
    # Google Gemini Models
    # ==========================================================================
    "gemini-1.5-pro": {
        "provider": LLMProvider.GOOGLE,
        "model_id": "gemini-1.5-pro",
        "display_name": "Gemini 1.5 Pro",
        "max_tokens": 8192,
    },
    "gemini-1.5-flash": {
        "provider": LLMProvider.GOOGLE,
        "model_id": "gemini-1.5-flash",
        "display_name": "Gemini 1.5 Flash",
        "max_tokens": 8192,
    },
}

# Default model
DEFAULT_LLM_MODEL = "claude-sonnet-4-6"


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
    # Multi-market event filter
    max_markets_per_event: Optional[int] = None  # Exclude events with more markets than this (e.g., 5)


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


