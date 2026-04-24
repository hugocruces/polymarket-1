"""
Configuration Module

Defines enums, constants, and the LLM model registry used throughout the agent.
Runtime-configurable values live on ScannerConfig (see scanner_config.py), not here.
"""

import os
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
# LLM Model Registry
# ============================================================================

LLM_MODELS = {
    # Anthropic Claude
    "claude-sonnet-4-6": {
        "provider": LLMProvider.ANTHROPIC,
        "model_id": "claude-sonnet-4-6",
        "display_name": "Claude Sonnet 4.6",
        "max_tokens": 64000,
    },
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

    # OpenAI
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

    # Google Gemini
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
