"""
Market Filters

Implements configurable filtering logic for Polymarket markets.
Filters can be combined and applied to narrow down markets of interest.

Each filter function follows a consistent interface:
- Takes a list of Market objects and filter parameters
- Returns a filtered list of Market objects
- Is composable with other filters
"""

import re
import logging
from datetime import datetime
from typing import Callable, Optional
from dataclasses import dataclass

from polymarket_agent.data_fetching.models import Market
from polymarket_agent.config import FilterConfig

logger = logging.getLogger(__name__)


# Geographic keywords for inferring market geography
GEOGRAPHY_KEYWORDS = {
    "US": [
        "united states", "america", "american", "u.s.", "usa", "biden", "trump",
        "congress", "senate", "house of representatives", "federal reserve",
        "democrat", "republican", "gop", "white house", "supreme court",
        "california", "texas", "florida", "new york", "washington",
    ],
    "EU": [
        "european", "europe", "eu", "euro", "ecb", "european central bank",
        "germany", "france", "italy", "spain", "netherlands", "belgium",
        "brussels", "macron", "scholz", "european commission", "eurozone",
    ],
    "UK": [
        "united kingdom", "britain", "british", "uk", "england", "scotland",
        "wales", "london", "parliament", "prime minister", "labour", "tory",
        "conservative", "starmer", "sunak", "westminster",
    ],
    "ASIA": [
        "china", "chinese", "japan", "japanese", "korea", "korean",
        "india", "indian", "asia", "asian", "beijing", "tokyo", "shanghai",
        "xi jinping", "modi", "boj", "pboc",
    ],
    "CRYPTO": [
        "bitcoin", "ethereum", "crypto", "blockchain", "btc", "eth",
        "defi", "nft", "solana", "binance", "coinbase", "sec crypto",
    ],
    "GLOBAL": [
        "global", "world", "international", "un", "united nations",
        "imf", "world bank", "g7", "g20", "nato", "who",
    ],
}


@dataclass
class FilterResult:
    """Result of applying filters to markets."""
    markets: list[Market]
    total_before: int
    total_after: int
    filters_applied: list[str]
    
    @property
    def filter_rate(self) -> float:
        """Percentage of markets that passed filters."""
        if self.total_before == 0:
            return 0.0
        return self.total_after / self.total_before


class MarketFilter:
    """
    Composable market filter that applies multiple filter criteria.
    
    Usage:
        >>> filter = MarketFilter(config.filters)
        >>> filtered = filter.apply(markets)
        
    Or with custom filters:
        >>> filter = MarketFilter()
        >>> filter.add_filter(filter_by_category, categories=["politics"])
        >>> filter.add_filter(filter_by_volume, min_volume=10000)
        >>> filtered = filter.apply(markets)
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize the filter with optional configuration.
        
        Args:
            config: FilterConfig with filter parameters
        """
        self.config = config
        self._filters: list[tuple[Callable, dict]] = []
        
        if config:
            self._build_filters_from_config()
    
    def _build_filters_from_config(self):
        """Build filter chain from configuration."""
        config = self.config
        
        # Category filter
        if config.categories:
            self.add_filter(filter_by_category, categories=config.categories)
        
        # Keyword filters
        if config.keywords:
            self.add_filter(filter_by_keywords, keywords=config.keywords, include=True)
        
        if config.exclude_keywords:
            self.add_filter(filter_by_keywords, keywords=config.exclude_keywords, include=False)
        
        # Volume filter
        if config.min_volume > 0:
            self.add_filter(filter_by_volume, min_volume=config.min_volume)
        
        # Liquidity filter
        if config.min_liquidity > 0:
            self.add_filter(filter_by_liquidity, min_liquidity=config.min_liquidity)
        
        # Expiry filter
        if config.max_days_to_expiry:
            self.add_filter(filter_by_expiry, max_days=config.max_days_to_expiry)
        
        # Geography filter
        if config.geographic_regions:
            self.add_filter(filter_by_geography, regions=config.geographic_regions)
        
        # Outcome count filter
        if config.min_outcomes or config.max_outcomes:
            self.add_filter(
                filter_by_outcome_count,
                min_outcomes=config.min_outcomes,
                max_outcomes=config.max_outcomes,
            )
    
    def add_filter(self, filter_func: Callable, **kwargs):
        """
        Add a filter function to the chain.
        
        Args:
            filter_func: Filter function that takes (markets, **kwargs) -> markets
            **kwargs: Arguments to pass to the filter function
        """
        self._filters.append((filter_func, kwargs))
    
    def apply(self, markets: list[Market]) -> FilterResult:
        """
        Apply all filters to the market list.
        
        Args:
            markets: List of markets to filter
            
        Returns:
            FilterResult with filtered markets and metadata
        """
        total_before = len(markets)
        filtered = markets.copy()
        filters_applied = []
        
        for filter_func, kwargs in self._filters:
            before_count = len(filtered)
            filtered = filter_func(filtered, **kwargs)
            after_count = len(filtered)
            
            filter_name = filter_func.__name__
            filters_applied.append(
                f"{filter_name}: {before_count} -> {after_count}"
            )
            
            logger.debug(f"Applied {filter_name}: {before_count} -> {after_count}")
        
        return FilterResult(
            markets=filtered,
            total_before=total_before,
            total_after=len(filtered),
            filters_applied=filters_applied,
        )
    
    def __call__(self, markets: list[Market]) -> list[Market]:
        """Allow using the filter as a callable."""
        return self.apply(markets).markets


def filter_by_category(
    markets: list[Market],
    categories: list[str],
    match_tags: bool = True,
) -> list[Market]:
    """
    Filter markets by category or tags.
    
    Args:
        markets: Markets to filter
        categories: List of category/tag names to include (case-insensitive)
        match_tags: If True, also match against market tags
        
    Returns:
        Markets matching any of the specified categories
        
    Example:
        >>> filtered = filter_by_category(markets, ["politics", "elections"])
    """
    if not categories:
        return markets
    
    categories_lower = [c.lower().strip() for c in categories]
    filtered = []
    
    for market in markets:
        # Check primary category
        if market.category.lower().strip() in categories_lower:
            filtered.append(market)
            continue
        
        # Check tags
        if match_tags:
            market_tags_lower = [t.lower().strip() for t in market.tags]
            if any(cat in market_tags_lower for cat in categories_lower):
                filtered.append(market)
                continue
            
            # Also check for partial matches in tags
            for tag in market_tags_lower:
                if any(cat in tag or tag in cat for cat in categories_lower):
                    filtered.append(market)
                    break
    
    return filtered


def filter_by_keywords(
    markets: list[Market],
    keywords: list[str],
    include: bool = True,
    search_description: bool = True,
) -> list[Market]:
    """
    Filter markets by keyword presence in title/description.
    
    Args:
        markets: Markets to filter
        keywords: Keywords to search for
        include: If True, include matching markets. If False, exclude them.
        search_description: If True, also search in market description
        
    Returns:
        Filtered markets
        
    Example:
        >>> # Include markets mentioning bitcoin
        >>> filtered = filter_by_keywords(markets, ["bitcoin", "btc"], include=True)
        
        >>> # Exclude markets mentioning specific topics
        >>> filtered = filter_by_keywords(markets, ["nsfw", "adult"], include=False)
    """
    if not keywords:
        return markets
    
    # Build regex pattern for efficient matching
    pattern = re.compile(
        "|".join(re.escape(kw) for kw in keywords),
        re.IGNORECASE
    )
    
    filtered = []
    
    for market in markets:
        # Build text to search
        search_text = market.question
        if search_description and market.description:
            search_text += " " + market.description
        
        # Also search in tags
        search_text += " " + " ".join(market.tags)
        
        # Check for match
        has_match = bool(pattern.search(search_text))
        
        if include and has_match:
            filtered.append(market)
        elif not include and not has_match:
            filtered.append(market)
    
    return filtered


def filter_by_volume(
    markets: list[Market],
    min_volume: float = 0,
    max_volume: Optional[float] = None,
) -> list[Market]:
    """
    Filter markets by trading volume.
    
    Args:
        markets: Markets to filter
        min_volume: Minimum volume in USD
        max_volume: Optional maximum volume in USD
        
    Returns:
        Markets within the volume range
    """
    filtered = []
    
    for market in markets:
        if market.volume < min_volume:
            continue
        if max_volume is not None and market.volume > max_volume:
            continue
        filtered.append(market)
    
    return filtered


def filter_by_liquidity(
    markets: list[Market],
    min_liquidity: float = 0,
    max_liquidity: Optional[float] = None,
) -> list[Market]:
    """
    Filter markets by liquidity.
    
    Args:
        markets: Markets to filter
        min_liquidity: Minimum liquidity in USD
        max_liquidity: Optional maximum liquidity in USD
        
    Returns:
        Markets within the liquidity range
    """
    filtered = []
    
    for market in markets:
        if market.liquidity < min_liquidity:
            continue
        if max_liquidity is not None and market.liquidity > max_liquidity:
            continue
        filtered.append(market)
    
    return filtered


def filter_by_expiry(
    markets: list[Market],
    min_days: int = 0,
    max_days: Optional[int] = None,
) -> list[Market]:
    """
    Filter markets by time to expiry/resolution.
    
    Args:
        markets: Markets to filter
        min_days: Minimum days until expiry
        max_days: Maximum days until expiry
        
    Returns:
        Markets within the expiry range
    """
    filtered = []
    
    for market in markets:
        days = market.days_to_expiry
        
        # Skip markets without end date if we need expiry info
        if days is None:
            if max_days is None:
                # No max constraint, include markets without dates
                filtered.append(market)
            continue
        
        if days < min_days:
            continue
        if max_days is not None and days > max_days:
            continue
        
        filtered.append(market)
    
    return filtered


def filter_by_geography(
    markets: list[Market],
    regions: list[str],
) -> list[Market]:
    """
    Filter markets by inferred geographic relevance.
    
    Geographic relevance is inferred from keywords in the market
    title, description, and tags.
    
    Args:
        markets: Markets to filter
        regions: List of region codes (US, EU, UK, ASIA, CRYPTO, GLOBAL)
        
    Returns:
        Markets with inferred geographic relevance to specified regions
        
    Example:
        >>> us_markets = filter_by_geography(markets, ["US"])
        >>> crypto_markets = filter_by_geography(markets, ["CRYPTO"])
    """
    if not regions:
        return markets
    
    # Build combined keyword list for requested regions
    region_keywords = []
    for region in regions:
        region_upper = region.upper()
        if region_upper in GEOGRAPHY_KEYWORDS:
            region_keywords.extend(GEOGRAPHY_KEYWORDS[region_upper])
    
    if not region_keywords:
        logger.warning(f"No keywords found for regions: {regions}")
        return markets
    
    # Build regex pattern
    pattern = re.compile(
        "|".join(re.escape(kw) for kw in region_keywords),
        re.IGNORECASE
    )
    
    filtered = []
    
    for market in markets:
        search_text = (
            market.question + " " +
            market.description + " " +
            " ".join(market.tags) + " " +
            (market.event_title or "")
        )
        
        if pattern.search(search_text):
            filtered.append(market)
    
    return filtered


def filter_by_outcome_count(
    markets: list[Market],
    min_outcomes: int = 2,
    max_outcomes: Optional[int] = None,
) -> list[Market]:
    """
    Filter markets by number of outcomes.
    
    Args:
        markets: Markets to filter
        min_outcomes: Minimum number of outcomes (default 2)
        max_outcomes: Maximum number of outcomes
        
    Returns:
        Markets with outcome count in range
    """
    filtered = []
    
    for market in markets:
        count = len(market.outcomes)
        
        if count < min_outcomes:
            continue
        if max_outcomes is not None and count > max_outcomes:
            continue
        
        filtered.append(market)
    
    return filtered


def filter_by_price_range(
    markets: list[Market],
    min_price: float = 0.0,
    max_price: float = 1.0,
    outcome_name: Optional[str] = None,
) -> list[Market]:
    """
    Filter markets where an outcome's price is within a range.
    
    Useful for finding markets with extreme prices that might be mispriced.
    
    Args:
        markets: Markets to filter
        min_price: Minimum price (0.0 to 1.0)
        max_price: Maximum price (0.0 to 1.0)
        outcome_name: Specific outcome to check (e.g., "Yes"). If None, checks any outcome.
        
    Returns:
        Markets with at least one outcome in price range
        
    Example:
        >>> # Find markets with Yes outcome between 10% and 40%
        >>> filtered = filter_by_price_range(markets, 0.1, 0.4, outcome_name="Yes")
    """
    filtered = []
    
    for market in markets:
        outcomes_to_check = market.outcomes
        
        if outcome_name:
            outcome = market.get_outcome_by_name(outcome_name)
            if outcome:
                outcomes_to_check = [outcome]
            else:
                continue
        
        for outcome in outcomes_to_check:
            if min_price <= outcome.price <= max_price:
                filtered.append(market)
                break
    
    return filtered


def apply_filters(
    markets: list[Market],
    config: FilterConfig,
) -> FilterResult:
    """
    Convenience function to apply filters from a FilterConfig.
    
    Args:
        markets: Markets to filter
        config: Filter configuration
        
    Returns:
        FilterResult with filtered markets and metadata
    """
    filter_chain = MarketFilter(config)
    return filter_chain.apply(markets)
