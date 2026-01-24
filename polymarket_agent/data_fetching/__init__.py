"""
Data Fetching Module

Provides functionality to retrieve market data from Polymarket's public APIs.
"""

from polymarket_agent.data_fetching.gamma_api import (
    fetch_active_events,
    fetch_active_markets,
    fetch_market_details,
)
from polymarket_agent.data_fetching.clob_api import (
    fetch_market_price,
    fetch_orderbook,
    fetch_orderbook_depth,
)
from polymarket_agent.data_fetching.models import (
    Event,
    Market,
    Outcome,
    OrderbookDepth,
    PriceData,
)

__all__ = [
    "fetch_active_events",
    "fetch_active_markets",
    "fetch_market_details",
    "fetch_market_price",
    "fetch_orderbook",
    "fetch_orderbook_depth",
    "Event",
    "Market",
    "Outcome",
    "OrderbookDepth",
    "PriceData",
]
