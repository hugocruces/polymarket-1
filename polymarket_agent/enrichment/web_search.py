"""
Web Search Module

Provides web search functionality to gather external context for market analysis.
Supports multiple search providers with a fallback mechanism.

The module uses free search APIs by default (DuckDuckGo) with optional
paid APIs (Serper, etc.) for higher rate limits and better results.
"""

import asyncio
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from urllib.parse import quote_plus

import httpx

from polymarket_agent.data_fetching.models import Market, EnrichedMarket

logger = logging.getLogger(__name__)

# Configuration
DEFAULT_TIMEOUT = 15.0
MAX_SEARCH_RESULTS = 5
CONCURRENT_SEARCHES = 3


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    date: Optional[str] = None
    source: str = ""


@dataclass
class SearchResponse:
    """Response from a search query."""
    query: str
    results: list[SearchResult]
    provider: str
    timestamp: datetime


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """
        Execute a search query.
        
        Args:
            query: Search query string
            num_results: Maximum results to return
            
        Returns:
            List of SearchResult objects
        """
        pass


class DuckDuckGoSearch(WebSearchProvider):
    """
    DuckDuckGo search provider.
    
    Uses DuckDuckGo's HTML interface for searching.
    No API key required, but has rate limits.
    
    Note: This is a basic implementation. For production use,
    consider using the duckduckgo-search library.
    """
    
    @property
    def name(self) -> str:
        return "DuckDuckGo"
    
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """
        Search DuckDuckGo for the query.
        
        Uses the HTML version of DDG which doesn't require an API key.
        """
        results = []
        
        # Use DuckDuckGo HTML search
        url = "https://html.duckduckgo.com/html/"
        
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(
                    url,
                    data={"q": query},
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                    },
                )
                response.raise_for_status()
                html = response.text
                
                # Parse results from HTML (basic parsing)
                results = self._parse_html_results(html, num_results)
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
        
        return results
    
    def _parse_html_results(self, html: str, max_results: int) -> list[SearchResult]:
        """Parse search results from DDG HTML."""
        results = []
        
        # Simple regex-based parsing (for robustness, consider using BeautifulSoup)
        # Looking for result blocks
        result_pattern = re.compile(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'<a[^>]+class="result__snippet"[^>]*>([^<]+)</a>',
            re.DOTALL | re.IGNORECASE
        )
        
        # Alternative pattern for snippet
        alt_pattern = re.compile(
            r'<a[^>]+rel="nofollow"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>.*?'
            r'class="result__snippet"[^>]*>([^<]+)<',
            re.DOTALL | re.IGNORECASE
        )
        
        matches = result_pattern.findall(html)
        if not matches:
            matches = alt_pattern.findall(html)
        
        for match in matches[:max_results]:
            url, title, snippet = match
            
            # Clean up the text
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            
            # Skip ads and internal links
            if 'duckduckgo.com' in url:
                continue
            
            results.append(SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source="DuckDuckGo",
            ))
        
        return results


class SerperSearch(WebSearchProvider):
    """
    Serper.dev search provider.
    
    Higher quality results with Google search data.
    Requires SERPER_API_KEY environment variable.
    
    Pricing: https://serper.dev/pricing
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
    
    @property
    def name(self) -> str:
        return "Serper"
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search using Serper API."""
        if not self.api_key:
            logger.warning("Serper API key not configured")
            return []
        
        results = []
        
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json",
                    },
                    json={
                        "q": query,
                        "num": num_results,
                    },
                )
                response.raise_for_status()
                data = response.json()
                
                # Parse organic results
                for item in data.get("organic", [])[:num_results]:
                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        date=item.get("date"),
                        source="Serper/Google",
                    ))
                    
        except Exception as e:
            logger.warning(f"Serper search failed: {e}")
        
        return results


class NewsAPISearch(WebSearchProvider):
    """
    NewsAPI search provider for news-specific queries.
    
    Requires NEWS_API_KEY environment variable.
    Free tier: 100 requests/day
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
    
    @property
    def name(self) -> str:
        return "NewsAPI"
    
    @property
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    async def search(self, query: str, num_results: int = 5) -> list[SearchResult]:
        """Search for news articles."""
        if not self.api_key:
            return []
        
        results = []
        
        try:
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                response = await client.get(
                    "https://newsapi.org/v2/everything",
                    params={
                        "q": query,
                        "pageSize": num_results,
                        "sortBy": "relevancy",
                        "language": "en",
                    },
                    headers={"X-API-KEY": self.api_key},
                )
                response.raise_for_status()
                data = response.json()
                
                for article in data.get("articles", []):
                    results.append(SearchResult(
                        title=article.get("title", ""),
                        url=article.get("url", ""),
                        snippet=article.get("description", ""),
                        date=article.get("publishedAt"),
                        source=article.get("source", {}).get("name", "NewsAPI"),
                    ))
                    
        except Exception as e:
            logger.warning(f"NewsAPI search failed: {e}")
        
        return results


def get_search_provider() -> WebSearchProvider:
    """
    Get the best available search provider.
    
    Prefers paid APIs if configured, falls back to free options.
    
    Returns:
        WebSearchProvider instance
    """
    # Check for Serper API key first (best results)
    serper = SerperSearch()
    if serper.is_available:
        return serper
    
    # Fall back to DuckDuckGo (free, no key needed)
    return DuckDuckGoSearch()


def build_search_query(market: Market) -> str:
    """
    Build an effective search query for a market.
    
    Constructs a query that will find relevant information
    for assessing the market's probability.
    
    Args:
        market: The market to search for
        
    Returns:
        Optimized search query string
    """
    # Start with the core question
    query = market.question
    
    # Remove common question prefixes for cleaner search
    query = re.sub(r'^(will|would|does|do|is|are|can|could)\s+', '', query, flags=re.IGNORECASE)
    
    # Add temporal context if available
    if market.end_date:
        year = market.end_date.year
        month = market.end_date.strftime("%B")
        query += f" {month} {year}"
    
    # Add key category context
    if market.category:
        if market.category.lower() not in query.lower():
            query += f" {market.category}"
    
    # Truncate if too long
    if len(query) > 200:
        query = query[:200]
    
    return query.strip()


async def search_for_context(
    market: Market,
    provider: Optional[WebSearchProvider] = None,
    num_results: int = MAX_SEARCH_RESULTS,
) -> SearchResponse:
    """
    Search for external context relevant to a market.
    
    Args:
        market: The market to find context for
        provider: Search provider to use (auto-selects if None)
        num_results: Maximum number of results
        
    Returns:
        SearchResponse with results
        
    Example:
        >>> response = await search_for_context(market)
        >>> for result in response.results:
        ...     print(f"{result.title}: {result.snippet}")
    """
    if provider is None:
        provider = get_search_provider()
    
    query = build_search_query(market)
    logger.debug(f"Searching for: {query}")
    
    results = await provider.search(query, num_results)
    
    return SearchResponse(
        query=query,
        results=results,
        provider=provider.name,
        timestamp=datetime.now(),
    )


def summarize_search_results(results: list[SearchResult]) -> str:
    """
    Summarize search results into a coherent context string.
    
    Args:
        results: List of search results
        
    Returns:
        Summarized context string
    """
    if not results:
        return "No external information found."
    
    summary_parts = []
    
    for i, result in enumerate(results, 1):
        # Format each result
        date_str = f" ({result.date})" if result.date else ""
        source_str = f" - {result.source}" if result.source else ""
        
        part = f"[{i}] {result.title}{date_str}{source_str}\n{result.snippet}"
        summary_parts.append(part)
    
    return "\n\n".join(summary_parts)


def extract_key_facts(results: list[SearchResult]) -> list[str]:
    """
    Extract key facts from search results.
    
    Uses simple heuristics to identify factual statements.
    
    Args:
        results: Search results to analyze
        
    Returns:
        List of extracted key facts
    """
    facts = []
    
    # Patterns that often indicate factual statements
    fact_patterns = [
        r'\d+%',  # Percentages
        r'\$[\d,]+',  # Dollar amounts
        r'\d{4}',  # Years
        r'according to',
        r'reported that',
        r'announced',
        r'confirmed',
        r'estimated',
    ]
    
    combined_pattern = re.compile('|'.join(fact_patterns), re.IGNORECASE)
    
    for result in results:
        text = result.snippet
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check if sentence contains factual indicators
            if combined_pattern.search(sentence):
                # Clean and add
                fact = sentence.strip()
                if fact and fact not in facts:
                    facts.append(fact)
    
    return facts[:10]  # Limit to top 10 facts


async def enrich_market(
    market: Market,
    provider: Optional[WebSearchProvider] = None,
) -> EnrichedMarket:
    """
    Enrich a market with external context from web search.
    
    Args:
        market: The market to enrich
        provider: Search provider to use
        
    Returns:
        EnrichedMarket with context added
        
    Example:
        >>> enriched = await enrich_market(market)
        >>> print(enriched.external_context)
        >>> print(enriched.key_facts)
    """
    response = await search_for_context(market, provider)
    
    context = summarize_search_results(response.results)
    sources = [r.url for r in response.results]
    key_facts = extract_key_facts(response.results)
    
    # Determine freshness
    freshness = "unknown"
    for result in response.results:
        if result.date:
            try:
                # Try to parse date and determine freshness
                freshness = result.date
                break
            except:
                pass
    
    return EnrichedMarket(
        market=market,
        external_context=context,
        sources=sources,
        context_freshness=freshness,
        key_facts=key_facts,
        search_query=response.query,
    )


async def enrich_markets_batch(
    markets: list[Market],
    provider: Optional[WebSearchProvider] = None,
    concurrent_limit: int = CONCURRENT_SEARCHES,
    progress_callback: Optional[callable] = None,
) -> list[EnrichedMarket]:
    """
    Enrich multiple markets concurrently.
    
    Args:
        markets: Markets to enrich
        provider: Search provider to use
        concurrent_limit: Maximum concurrent searches
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of EnrichedMarket objects
        
    Example:
        >>> def on_progress(current, total):
        ...     print(f"Enriching {current}/{total}...")
        >>> enriched = await enrich_markets_batch(markets, progress_callback=on_progress)
    """
    if provider is None:
        provider = get_search_provider()
    
    results = []
    semaphore = asyncio.Semaphore(concurrent_limit)
    completed = 0
    total = len(markets)
    
    async def enrich_with_semaphore(market: Market) -> EnrichedMarket:
        nonlocal completed
        async with semaphore:
            # Add small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            result = await enrich_market(market, provider)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result
    
    tasks = [enrich_with_semaphore(m) for m in markets]
    results = await asyncio.gather(*tasks)
    
    logger.info(f"Enriched {len(results)} markets with web search data")
    return results
