"""
Gamma API Client

Fetches events and markets data from Polymarket's Gamma API.
This is the primary API for retrieving market metadata, descriptions,
and basic pricing information.

API Documentation: https://docs.polymarket.com/
Base URL: https://gamma-api.polymarket.com

Note: This API is publicly accessible and does not require authentication.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator
import httpx

from polymarket_agent.config import (
    GAMMA_EVENTS_ENDPOINT,
    GAMMA_MARKETS_ENDPOINT,
    DEFAULT_PAGE_SIZE,
    DEFAULT_MAX_MARKETS,
)
from polymarket_agent.data_fetching.models import Event, Market, Outcome

logger = logging.getLogger(__name__)

# HTTP client configuration
DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 3
RETRY_DELAY = 1.0


class GammaAPIError(Exception):
    """Exception raised for Gamma API errors."""
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
        
    Raises:
        GammaAPIError: If request fails after all retries
    """
    for attempt in range(retries):
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:  # Rate limited
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            elif e.response.status_code >= 500:  # Server error
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Server error {e.response.status_code}, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise GammaAPIError(f"HTTP error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            if attempt < retries - 1:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise GammaAPIError(f"Request failed after {retries} attempts") from e
    
    raise GammaAPIError(f"Request failed after {retries} attempts")


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse datetime string from API response."""
    if not value:
        return None
    try:
        # Handle various datetime formats
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
        ]:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
    except Exception:
        return None


def _parse_market(data: dict, event_data: Optional[dict] = None) -> Market:
    """
    Parse a market from API response data.
    
    Args:
        data: Market data from API
        event_data: Optional parent event data
        
    Returns:
        Parsed Market object
    """
    import json as json_module
    
    # Parse outcomes from various possible formats
    outcomes = []
    
    # Try parsing from outcomes array
    if "outcomes" in data:
        outcome_names = data["outcomes"]
        
        # Handle case where outcomes is a JSON string
        if isinstance(outcome_names, str):
            try:
                outcome_names = json_module.loads(outcome_names)
            except:
                outcome_names = []
        
        if isinstance(outcome_names, list):
            outcome_prices = data.get("outcomePrices", [])
            
            # Handle case where outcomePrices is a string (JSON)
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json_module.loads(outcome_prices)
                except:
                    outcome_prices = []
            
            # Get token IDs if available
            clob_token_ids = data.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                try:
                    clob_token_ids = json_module.loads(clob_token_ids)
                except:
                    clob_token_ids = []
            
            for i, name in enumerate(outcome_names):
                price = float(outcome_prices[i]) if i < len(outcome_prices) else 0.5
                token_id = clob_token_ids[i] if i < len(clob_token_ids) else ""
                outcomes.append(Outcome(
                    name=str(name),
                    token_id=str(token_id),
                    price=price,
                ))
    
    # Parse tags - handle various formats
    raw_tags = data.get("tags", [])
    tags = []
    if isinstance(raw_tags, str):
        tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    elif isinstance(raw_tags, list):
        for t in raw_tags:
            if isinstance(t, str):
                tags.append(t.strip())
            elif isinstance(t, dict) and "label" in t:
                tags.append(str(t["label"]))
            elif isinstance(t, dict) and "name" in t:
                tags.append(str(t["name"]))
    
    # Determine category (use first tag or explicit category)
    category = ""
    raw_category = data.get("category", "")
    if isinstance(raw_category, str):
        category = raw_category
    elif isinstance(raw_category, dict):
        category = str(raw_category.get("label", raw_category.get("name", "")))
    elif tags:
        category = tags[0]
    elif event_data and "tags" in event_data:
        event_tags = event_data["tags"]
        if isinstance(event_tags, list) and event_tags:
            first_tag = event_tags[0]
            if isinstance(first_tag, str):
                category = first_tag
            elif isinstance(first_tag, dict):
                category = str(first_tag.get("label", first_tag.get("name", "")))
    
    # Parse volume - handle various field names
    volume = 0.0
    for vol_field in ["volume", "volumeNum", "volume24hr", "totalVolume"]:
        if vol_field in data and data[vol_field]:
            try:
                volume = float(data[vol_field])
                break
            except (ValueError, TypeError):
                continue
    
    # Parse liquidity
    liquidity = 0.0
    for liq_field in ["liquidity", "liquidityNum", "totalLiquidity"]:
        if liq_field in data and data[liq_field]:
            try:
                liquidity = float(data[liq_field])
                break
            except (ValueError, TypeError):
                continue
    
    return Market(
        id=str(data.get("id", "")),
        slug=data.get("slug", data.get("conditionId", "")),
        question=data.get("question", data.get("title", "")),
        description=data.get("description", ""),
        outcomes=outcomes,
        category=category,
        tags=tags if isinstance(tags, list) else [],
        volume=volume,
        liquidity=liquidity,
        end_date=_parse_datetime(data.get("endDate") or data.get("end_date_iso")),
        created_at=_parse_datetime(data.get("createdAt") or data.get("created_at")),
        active=data.get("active", True),
        closed=data.get("closed", False),
        resolved=data.get("resolved", False),
        event_id=(
            str(data.get("eventId", "")) if data.get("eventId")
            else str(event_data["id"]) if event_data and event_data.get("id")
            else None
        ),
        event_title=event_data.get("title") if event_data else None,
        condition_id=data.get("conditionId"),
        raw_data=data,
    )


def _parse_event(data: dict) -> Event:
    """
    Parse an event from API response data.
    
    Args:
        data: Event data from API
        
    Returns:
        Parsed Event object
    """
    # Parse markets within the event
    markets = []
    markets_data = data.get("markets", [])
    for market_data in markets_data:
        try:
            market = _parse_market(market_data, event_data=data)
            markets.append(market)
        except Exception as e:
            logger.warning(f"Failed to parse market: {e}")
            continue
    
    # Parse tags
    tags = data.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    
    return Event(
        id=str(data.get("id", "")),
        slug=data.get("slug", ""),
        title=data.get("title", ""),
        description=data.get("description", ""),
        markets=markets,
        category=tags[0] if tags else "",
        tags=tags if isinstance(tags, list) else [],
        start_date=_parse_datetime(data.get("startDate")),
        end_date=_parse_datetime(data.get("endDate")),
        active=data.get("active", True),
        closed=data.get("closed", False),
    )


async def fetch_active_events(
    limit: int = DEFAULT_MAX_MARKETS,
    offset: int = 0,
    order: str = "volume",
    ascending: bool = False,
    tag_ids: Optional[list[int]] = None,
    query: Optional[str] = None,
) -> list[Event]:
    """
    Fetch active events from the Gamma API.
    
    This endpoint returns events with their associated markets. Events are
    containers that group related markets together.
    
    Args:
        limit: Maximum number of events to fetch
        offset: Pagination offset
        order: Field to order by (volume, createdAt, endDate)
        ascending: Sort ascending if True, descending if False
        tag_ids: Filter by specific tag IDs (only first tag used)
        query: Optional search query to filter events by keyword
        
    Returns:
        List of Event objects
        
    Example:
        >>> events = await fetch_active_events(limit=50)
        >>> for event in events:
        ...     print(f"{event.title}: {event.market_count} markets")
    """
    events = []
    current_offset = offset
    page_size = min(limit, DEFAULT_PAGE_SIZE)
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        while len(events) < limit:
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": current_offset,
                "order": order,
                "ascending": str(ascending).lower(),
            }
            
            # Add tag_id filter if provided (API only supports one tag at a time)
            if tag_ids and len(tag_ids) > 0:
                params["tag_id"] = str(tag_ids[0])
            
            # Add search query if provided
            if query:
                params["q"] = query
            
            logger.debug(f"Fetching events: offset={current_offset}, limit={page_size}")
            
            try:
                data = await _make_request(client, GAMMA_EVENTS_ENDPOINT, params)
            except GammaAPIError as e:
                logger.error(f"Failed to fetch events: {e}")
                break
            
            # Handle response - could be list or dict with data key
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("data", data.get("events", []))
            else:
                break
            
            if not items:
                break
            
            for item in items:
                try:
                    event = _parse_event(item)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event: {e}")
                    continue
            
            current_offset += len(items)
            
            # Check if we've fetched all available
            if len(items) < page_size:
                break
    
    logger.info(f"Fetched {len(events)} active events")
    return events[:limit]


async def fetch_active_markets(
    limit: int = DEFAULT_MAX_MARKETS,
    offset: int = 0,
    order: str = "volume",
    ascending: bool = False,
    tag: Optional[str] = None,
    query: Optional[str] = None,
) -> list[Market]:
    """
    Fetch active markets directly from the Gamma API markets endpoint.
    
    This provides a flat list of markets without event grouping.
    
    Args:
        limit: Maximum number of markets to fetch
        offset: Pagination offset
        order: Field to order by
        ascending: Sort direction
        tag: Optional tag filter
        query: Optional search query
        
    Returns:
        List of Market objects
        
    Example:
        >>> markets = await fetch_active_markets(limit=100, tag="politics")
        >>> for market in markets:
        ...     print(f"{market.question}: {market.outcome_prices}")
    """
    markets = []
    current_offset = offset
    page_size = min(limit, DEFAULT_PAGE_SIZE)
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        while len(markets) < limit:
            params = {
                "active": "true",
                "closed": "false",
                "limit": page_size,
                "offset": current_offset,
                "order": order,
                "ascending": str(ascending).lower(),
            }
            
            if tag:
                params["tag"] = tag
            
            if query:
                params["q"] = query
            
            logger.debug(f"Fetching markets: offset={current_offset}")
            
            try:
                data = await _make_request(client, GAMMA_MARKETS_ENDPOINT, params)
            except GammaAPIError as e:
                logger.error(f"Failed to fetch markets: {e}")
                break
            
            # Handle response format
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("data", data.get("markets", []))
            else:
                break
            
            if not items:
                break
            
            for item in items:
                try:
                    market = _parse_market(item)
                    markets.append(market)
                except Exception as e:
                    logger.warning(f"Failed to parse market: {e}")
                    continue
            
            current_offset += len(items)
            
            if len(items) < page_size:
                break
    
    logger.info(f"Fetched {len(markets)} active markets")
    return markets[:limit]


async def fetch_market_details(market_id: str) -> Optional[Market]:
    """
    Fetch detailed information for a specific market.
    
    Args:
        market_id: The market ID or slug
        
    Returns:
        Market object or None if not found
    """
    url = f"{GAMMA_MARKETS_ENDPOINT}/{market_id}"
    
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
        try:
            data = await _make_request(client, url)
            return _parse_market(data)
        except GammaAPIError as e:
            logger.error(f"Failed to fetch market {market_id}: {e}")
            return None


def fetch_active_events_sync(
    limit: int = DEFAULT_MAX_MARKETS,
    **kwargs,
) -> list[Event]:
    """
    Synchronous wrapper for fetch_active_events.
    
    Convenience function for non-async contexts.
    """
    return asyncio.run(fetch_active_events(limit=limit, **kwargs))


def fetch_active_markets_sync(
    limit: int = DEFAULT_MAX_MARKETS,
    **kwargs,
) -> list[Market]:
    """
    Synchronous wrapper for fetch_active_markets.
    
    Convenience function for non-async contexts.
    """
    return asyncio.run(fetch_active_markets(limit=limit, **kwargs))
