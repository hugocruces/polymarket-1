"""
CLOB API Client

Fetches real-time price and orderbook data from Polymarket's CLOB (Central Limit Order Book) API.
This API provides more granular pricing data than the Gamma API.

API Documentation: https://docs.polymarket.com/
Base URL: https://clob.polymarket.com

Note: This API is publicly accessible for read operations.
Authentication is only required for trading.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional
import httpx

from polymarket_agent.config import (
    CLOB_PRICE_ENDPOINT,
    CLOB_BOOK_ENDPOINT,
    CLOB_MARKETS_ENDPOINT,
)
from polymarket_agent.data_fetching.models import (
    PriceData,
    OrderbookDepth,
    OrderLevel,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0
MAX_RETRIES = 3
RETRY_DELAY = 0.5


class CLOBAPIError(Exception):
    """Exception raised for CLOB API errors."""
    pass


async def _make_request(
    client: httpx.AsyncClient,
    url: str,
    params: Optional[dict] = None,
    retries: int = MAX_RETRIES,
) -> dict:
    """
    Make an HTTP request with retry logic.
    
    Args:
        client: httpx async client
        url: Request URL
        params: Query parameters
        retries: Number of retries remaining
        
    Returns:
        JSON response data
    """
    for attempt in range(retries):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            elif e.response.status_code >= 500:
                wait_time = RETRY_DELAY * (2 ** attempt)
                await asyncio.sleep(wait_time)
            elif e.response.status_code == 404:
                raise CLOBAPIError(f"Token not found: {url}")
            else:
                raise CLOBAPIError(f"HTTP error: {e.response.status_code}")
        except httpx.RequestError as e:
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                raise CLOBAPIError(f"Request failed: {e}")
    
    raise CLOBAPIError(f"Request failed after {retries} attempts")


async def fetch_market_price(
    token_id: str,
    side: str = "buy",
) -> Optional[PriceData]:
    """
    Fetch the current price for a market outcome token.
    
    The CLOB API returns prices for buying or selling a specific outcome.
    
    Args:
        token_id: The outcome token ID (from market.token_ids)
        side: "buy" or "sell" - perspective for the price
        
    Returns:
        PriceData object with current price, or None if unavailable
        
    Example:
        >>> price = await fetch_market_price("12345678...")
        >>> print(f"Current price: {price.price:.3f}")
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            params = {
                "token_id": token_id,
                "side": side.upper(),
            }
            data = await _make_request(client, CLOB_PRICE_ENDPOINT, params)
            
            # Parse price from response
            price = float(data.get("price", 0))
            
            return PriceData(
                token_id=token_id,
                price=price,
                timestamp=datetime.now(),
            )
        except CLOBAPIError as e:
            logger.warning(f"Failed to fetch price for {token_id[:20]}...: {e}")
            return None


async def fetch_orderbook(
    token_id: str,
) -> Optional[OrderbookDepth]:
    """
    Fetch the full orderbook for a market outcome token.
    
    Returns all bid and ask levels with their sizes.
    
    Args:
        token_id: The outcome token ID
        
    Returns:
        OrderbookDepth object with bid/ask levels, or None if unavailable
        
    Example:
        >>> book = await fetch_orderbook("12345678...")
        >>> print(f"Spread: {book.spread:.4f}")
        >>> print(f"Total liquidity: ${book.total_liquidity:,.2f}")
    """
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            params = {"token_id": token_id}
            data = await _make_request(client, CLOB_BOOK_ENDPOINT, params)
            
            # Parse bids and asks
            bids = []
            asks = []
            
            for bid in data.get("bids", []):
                try:
                    bids.append(OrderLevel(
                        price=float(bid.get("price", 0)),
                        size=float(bid.get("size", 0)),
                    ))
                except (ValueError, TypeError):
                    continue
            
            for ask in data.get("asks", []):
                try:
                    asks.append(OrderLevel(
                        price=float(ask.get("price", 0)),
                        size=float(ask.get("size", 0)),
                    ))
                except (ValueError, TypeError):
                    continue
            
            return OrderbookDepth(
                token_id=token_id,
                bids=sorted(bids, key=lambda x: x.price, reverse=True),
                asks=sorted(asks, key=lambda x: x.price),
            )
        except CLOBAPIError as e:
            logger.warning(f"Failed to fetch orderbook for {token_id[:20]}...: {e}")
            return None


async def fetch_orderbook_depth(
    token_id: str,
    depth: int = 5,
) -> Optional[OrderbookDepth]:
    """
    Fetch orderbook depth up to a specified number of levels.
    
    More efficient than full orderbook when only top levels are needed.
    
    Args:
        token_id: The outcome token ID
        depth: Number of price levels to fetch (default 5)
        
    Returns:
        OrderbookDepth with limited levels
    """
    book = await fetch_orderbook(token_id)
    if book is None:
        return None
    
    return OrderbookDepth(
        token_id=token_id,
        bids=book.bids[:depth],
        asks=book.asks[:depth],
    )


async def fetch_multiple_prices(
    token_ids: list[str],
    side: str = "buy",
    concurrent_limit: int = 10,
) -> dict[str, PriceData]:
    """
    Fetch prices for multiple tokens concurrently.
    
    Uses semaphore to limit concurrent requests and avoid rate limiting.
    
    Args:
        token_ids: List of outcome token IDs
        side: "buy" or "sell"
        concurrent_limit: Maximum concurrent requests
        
    Returns:
        Dictionary mapping token_id to PriceData
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def fetch_with_semaphore(token_id: str):
        async with semaphore:
            price = await fetch_market_price(token_id, side)
            if price:
                results[token_id] = price
    
    await asyncio.gather(*[
        fetch_with_semaphore(tid) for tid in token_ids
    ])
    
    return results


async def fetch_multiple_orderbooks(
    token_ids: list[str],
    depth: int = 5,
    concurrent_limit: int = 5,
) -> dict[str, OrderbookDepth]:
    """
    Fetch orderbooks for multiple tokens concurrently.
    
    Args:
        token_ids: List of outcome token IDs
        depth: Number of price levels per book
        concurrent_limit: Maximum concurrent requests
        
    Returns:
        Dictionary mapping token_id to OrderbookDepth
    """
    results = {}
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    async def fetch_with_semaphore(token_id: str):
        async with semaphore:
            book = await fetch_orderbook_depth(token_id, depth)
            if book:
                results[token_id] = book
    
    await asyncio.gather(*[
        fetch_with_semaphore(tid) for tid in token_ids
    ])
    
    return results


async def enrich_market_with_clob_data(
    market,  # Market object
    fetch_orderbooks: bool = True,
) -> None:
    """
    Enrich a Market object with CLOB data in-place.
    
    Updates outcome prices with real-time CLOB data and adds
    orderbook information for liquidity assessment.
    
    Args:
        market: Market object to enrich
        fetch_orderbooks: Whether to also fetch orderbook depth
        
    Note:
        Modifies the market object in-place.
    """
    token_ids = [o.token_id for o in market.outcomes if o.token_id]
    
    if not token_ids:
        return
    
    # Fetch prices
    prices = await fetch_multiple_prices(token_ids)
    
    # Update outcome prices
    for outcome in market.outcomes:
        if outcome.token_id in prices:
            outcome.price = prices[outcome.token_id].price
    
    # Optionally fetch orderbook data
    if fetch_orderbooks:
        books = await fetch_multiple_orderbooks(token_ids)
        
        # Calculate total liquidity from orderbooks
        total_liquidity = sum(
            book.total_liquidity for book in books.values()
        )
        
        if total_liquidity > 0:
            market.liquidity = total_liquidity


# Synchronous wrappers for non-async contexts

def fetch_market_price_sync(token_id: str, side: str = "buy") -> Optional[PriceData]:
    """Synchronous wrapper for fetch_market_price."""
    return asyncio.run(fetch_market_price(token_id, side))


def fetch_orderbook_sync(token_id: str) -> Optional[OrderbookDepth]:
    """Synchronous wrapper for fetch_orderbook."""
    return asyncio.run(fetch_orderbook(token_id))
