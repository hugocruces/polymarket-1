"""
Data Fetching Module

Provides functionality to retrieve market data from Polymarket's public APIs.
"""

from polymarket_agent.data_fetching.gamma_api import (
    fetch_active_events,
    fetch_active_markets,
    fetch_market_details,
)
from polymarket_agent.data_fetching.models import (
    Event,
    Market,
    Outcome,
)

__all__ = [
    "fetch_active_events",
    "fetch_active_markets",
    "fetch_market_details",
    "Event",
    "Market",
    "Outcome",
]
