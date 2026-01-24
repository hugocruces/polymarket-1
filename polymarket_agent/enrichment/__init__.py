"""
Enrichment Module

Provides functionality to gather external context for markets via web search.
"""

from polymarket_agent.enrichment.web_search import (
    WebSearchProvider,
    DuckDuckGoSearch,
    SerperSearch,
    search_for_context,
    enrich_market,
    enrich_markets_batch,
)

__all__ = [
    "WebSearchProvider",
    "DuckDuckGoSearch",
    "SerperSearch",
    "search_for_context",
    "enrich_market",
    "enrich_markets_batch",
]
