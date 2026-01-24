"""
Polymarket AI Agent

An intelligent agent that monitors active markets on Polymarket, retrieves and 
filters market data, uses LLMs to assess pricing, and outputs a ranked list of 
potentially mispriced markets with explanations.

This agent is designed for one-off analysis runs and emphasizes:
- Modular architecture for easy customization
- Multi-provider LLM support
- Configurable risk tolerance
- Transparent probabilistic reasoning
"""

from polymarket_agent.config import AgentConfig, FilterConfig, RiskTolerance
from polymarket_agent.agent import PolymarketAgent

__version__ = "1.0.0"
__author__ = "Polymarket AI Agent Team"

__all__ = [
    "PolymarketAgent",
    "AgentConfig", 
    "FilterConfig",
    "RiskTolerance",
]
