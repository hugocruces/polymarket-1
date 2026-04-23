"""
Polymarket Bias Scanner

Scans Polymarket for markets where demographic biases may create mispricing opportunities.
Uses LLM classification to identify markets affected by:
- Political bias (left/right leaning outcomes)
- Progressive social bias (social issues, climate, etc.)
- Crypto optimism bias (crypto/tech enthusiasm)

Usage:
    python -m polymarket_agent.scan [OPTIONS]

Example:
    python -m polymarket_agent.scan --min-volume 50000 --model claude-sonnet-4-6
"""

from polymarket_agent.scanner import BiasScanner
from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
    ClassifiedMarket,
)
from polymarket_agent.config import FilterConfig, LLM_MODELS

__version__ = "2.0.0"
__author__ = "Polymarket Bias Scanner Team"

__all__ = [
    "BiasScanner",
    "ScannerConfig",
    "BiasCategory",
    "BiasClassification",
    "ClassifiedMarket",
    "FilterConfig",
    "LLM_MODELS",
]
