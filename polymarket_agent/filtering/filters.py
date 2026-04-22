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
# Keywords are weighted: higher weight = stronger regional indicator
# Format: (keyword, weight) where weight is 1-10
# Weight 10: Definitive (e.g., "United States Congress")
# Weight 7-9: Strong indicator (e.g., country names, major institutions)
# Weight 4-6: Moderate indicator (e.g., politicians, cities)
# Weight 1-3: Weak indicator (e.g., terms that could be ambiguous)

GEOGRAPHY_KEYWORDS_WEIGHTED = {
    "US": [
        # Definitive indicators (10)
        ("united states", 10), ("u.s. congress", 10), ("u.s. senate", 10),
        ("u.s. house", 10), ("american president", 10), ("us federal", 10),
        # Strong indicators (7-9)
        ("america", 8), ("american", 7), ("u.s.", 8), ("usa", 9),
        ("federal reserve", 9), ("the fed", 7), ("fomc", 9),
        ("white house", 9), ("oval office", 9), ("supreme court", 8),
        ("congress", 8), ("senate", 7), ("house of representatives", 9),
        ("democrat", 7), ("republican", 7), ("gop", 8), ("dnc", 9), ("rnc", 9),
        ("pentagon", 9), ("cia", 8), ("fbi", 8), ("sec", 6), ("cftc", 8),
        ("medicare", 9), ("medicaid", 9), ("social security", 8),
        # Politicians (6-8)
        ("biden", 8), ("trump", 7), ("kamala harris", 8), ("desantis", 8),
        ("newsom", 7), ("pence", 7), ("pelosi", 8), ("mcconnell", 8),
        ("schumer", 8), ("aoc", 7), ("bernie sanders", 8), ("ted cruz", 8),
        ("ron paul", 8), ("elon musk", 5), ("jd vance", 8), ("rfk", 7),
        # States and cities (5-7)
        ("california", 6), ("texas", 6), ("florida", 6), ("new york", 5),
        ("washington dc", 8), ("washington d.c.", 8), ("wall street", 6),
        ("silicon valley", 6), ("hollywood", 5), ("las vegas", 5),
        ("chicago", 5), ("los angeles", 5), ("miami", 5), ("boston", 5),
        # Sports/Culture (4-6)
        ("nfl", 6), ("nba", 6), ("mlb", 6), ("nhl", 5), ("super bowl", 7),
        ("march madness", 7), ("world series", 6), ("stanley cup", 5),
        ("oscars", 6), ("grammy", 6), ("emmy", 6), ("tony awards", 6),
    ],
    "EU": [
        # Definitive indicators (10)
        ("european union", 10), ("european commission", 10), ("european parliament", 10),
        ("eurozone", 10), ("schengen", 9),
        # Strong indicators (7-9)
        ("europe", 7), ("european", 7), ("eu", 7), ("euro", 6),
        ("ecb", 9), ("european central bank", 10),
        ("brussels", 8), ("strasbourg", 8),
        # Countries (7-8)
        ("germany", 8), ("german", 7), ("france", 8), ("french", 6),
        ("italy", 8), ("italian", 6), ("spain", 8), ("spanish", 6),
        ("netherlands", 8), ("dutch", 7), ("belgium", 8), ("belgian", 7),
        ("austria", 8), ("poland", 8), ("portugal", 8), ("greece", 8),
        ("ireland", 7), ("sweden", 8), ("finland", 8), ("denmark", 8),
        # Politicians (6-8)
        ("macron", 8), ("scholz", 8), ("von der leyen", 9), ("draghi", 8),
        ("meloni", 8), ("sanchez", 7), ("rutte", 7), ("orban", 7),
        # Cities (5-7)
        ("paris", 5), ("berlin", 6), ("rome", 5), ("madrid", 5),
        ("amsterdam", 5), ("vienna", 5), ("frankfurt", 6), ("munich", 5),
        # Institutions (7-9)
        ("bundesbank", 9), ("bundestag", 9), ("élysée", 9), ("reichstag", 8),
    ],
    "UK": [
        # Definitive indicators (10)
        ("united kingdom", 10), ("british parliament", 10), ("house of commons", 10),
        ("house of lords", 10), ("westminster", 9), ("downing street", 10),
        # Strong indicators (7-9)
        ("britain", 9), ("british", 8), ("uk", 8), ("england", 7),
        ("scotland", 8), ("scottish", 7), ("wales", 8), ("welsh", 7),
        ("northern ireland", 9),
        ("bank of england", 10), ("boe", 8), ("nhs", 8),
        ("labour party", 9), ("conservative party", 9), ("tory", 8), ("tories", 8),
        ("lib dem", 8), ("snp", 8),
        # Politicians (6-8)
        ("starmer", 8), ("sunak", 8), ("truss", 7), ("johnson", 5),
        ("king charles", 8), ("prince william", 7), ("royal family", 7),
        ("farage", 8), ("corbyn", 7),
        # Cities (5-7)
        ("london", 6), ("manchester", 6), ("birmingham", 5), ("edinburgh", 7),
        ("glasgow", 6), ("cardiff", 6), ("belfast", 7), ("liverpool", 5),
        # Institutions/Events (6-8)
        ("premier league", 7), ("wimbledon", 7), ("the ashes", 8),
        ("ftse", 8), ("city of london", 7), ("canary wharf", 6),
    ],
    "ASIA": [
        # China (7-10)
        ("china", 9), ("chinese", 8), ("beijing", 8), ("shanghai", 7),
        ("hong kong", 8), ("taiwan", 8), ("taiwanese", 8),
        ("xi jinping", 9), ("ccp", 9), ("communist party of china", 10),
        ("pboc", 9), ("people's bank of china", 10),
        ("shenzhen", 7), ("guangzhou", 6), ("tiananmen", 8),
        # Japan (7-10)
        ("japan", 9), ("japanese", 8), ("tokyo", 7), ("osaka", 6),
        ("boj", 9), ("bank of japan", 10), ("yen", 6),
        ("kishida", 8), ("abe", 7), ("nikkei", 8),
        # Korea (7-10)
        ("south korea", 9), ("north korea", 10), ("korean", 7),
        ("seoul", 7), ("pyongyang", 9), ("kim jong", 10),
        ("samsung", 6), ("hyundai", 6), ("k-pop", 5),
        # India (7-10)
        ("india", 9), ("indian", 7), ("mumbai", 7), ("delhi", 7),
        ("modi", 8), ("bjp", 8), ("rbi", 8), ("reserve bank of india", 10),
        ("bollywood", 6), ("sensex", 8),
        # Southeast Asia (6-8)
        ("singapore", 8), ("indonesia", 8), ("vietnam", 8), ("thailand", 7),
        ("philippines", 8), ("malaysia", 8), ("asean", 9),
        # General (5-7)
        ("asia", 7), ("asian", 6), ("asia-pacific", 8), ("apac", 7),
    ],
    "LATAM": [
        # Definitive indicators (9-10)
        ("latin america", 10), ("south america", 9), ("central america", 9),
        # Countries (7-9)
        ("brazil", 9), ("brazilian", 8), ("mexico", 9), ("mexican", 8),
        ("argentina", 9), ("argentine", 8), ("colombia", 8), ("colombian", 7),
        ("chile", 8), ("chilean", 7), ("peru", 8), ("venezuela", 9),
        ("cuba", 8), ("cuban", 7), ("puerto rico", 7),
        # Politicians (7-8)
        ("lula", 8), ("bolsonaro", 8), ("amlo", 8), ("milei", 8),
        ("maduro", 9), ("bukele", 8),
        # Cities (5-7)
        ("são paulo", 7), ("sao paulo", 7), ("rio de janeiro", 7),
        ("mexico city", 7), ("buenos aires", 7), ("bogota", 6),
        ("lima", 6), ("santiago", 6), ("caracas", 7), ("havana", 7),
        # Institutions (7-8)
        ("mercosur", 9), ("petrobras", 8), ("pemex", 8),
    ],
    "MIDDLE_EAST": [
        # Definitive indicators (9-10)
        ("middle east", 10), ("mideast", 9),
        # Countries (7-9)
        ("israel", 9), ("israeli", 8), ("palestine", 9), ("palestinian", 8),
        ("iran", 9), ("iranian", 8), ("iraq", 9), ("iraqi", 8),
        ("saudi arabia", 9), ("saudi", 8), ("uae", 8), ("dubai", 7),
        ("qatar", 8), ("kuwait", 8), ("bahrain", 8), ("oman", 8),
        ("jordan", 7), ("lebanon", 8), ("lebanese", 7), ("syria", 9), ("syrian", 8),
        ("turkey", 8), ("turkish", 7), ("egypt", 8), ("egyptian", 7),
        # Politicians/Leaders (7-9)
        ("netanyahu", 9), ("khamenei", 9), ("erdogan", 8), ("mbs", 8),
        ("mohammed bin salman", 9), ("sisi", 8), ("assad", 9),
        # Cities (6-8)
        ("jerusalem", 8), ("tel aviv", 8), ("tehran", 8), ("riyadh", 8),
        ("abu dhabi", 7), ("doha", 7), ("cairo", 7), ("istanbul", 6),
        ("gaza", 9), ("west bank", 9), ("golan", 8),
        # Institutions/Conflicts (8-10)
        ("opec", 9), ("idf", 9), ("hamas", 10), ("hezbollah", 10),
        ("irgc", 9), ("mossad", 9), ("abraham accords", 9),
    ],
    "AFRICA": [
        # Definitive indicators (9-10)
        ("africa", 9), ("african", 8), ("sub-saharan", 10),
        ("african union", 10),
        # Countries (7-9)
        ("south africa", 9), ("nigeria", 9), ("nigerian", 8),
        ("kenya", 8), ("kenyan", 7), ("ethiopia", 8), ("egyptian", 7),
        ("ghana", 8), ("morocco", 8), ("algeria", 8), ("tanzania", 8),
        ("uganda", 8), ("zimbabwe", 8), ("congo", 8), ("sudan", 8),
        ("libya", 8), ("libyan", 7), ("tunisia", 8),
        # Politicians (7-8)
        ("ramaphosa", 8), ("tinubu", 8), ("ruto", 8),
        # Cities (5-7)
        ("johannesburg", 7), ("lagos", 7), ("nairobi", 7), ("cairo", 7),
        ("cape town", 7), ("addis ababa", 7), ("casablanca", 6),
        # Institutions (7-8)
        ("anc", 8), ("ecowas", 9),
    ],
    "CRYPTO": [
        # Definitive indicators (9-10)
        ("cryptocurrency", 10), ("crypto market", 10), ("blockchain", 9),
        ("decentralized finance", 10), ("defi", 9),
        # Major coins (7-9)
        ("bitcoin", 9), ("btc", 8), ("ethereum", 9), ("eth", 7),
        ("solana", 8), ("sol", 6), ("cardano", 8), ("ada", 5),
        ("ripple", 8), ("xrp", 7), ("dogecoin", 8), ("doge", 7),
        ("litecoin", 8), ("polkadot", 8), ("avalanche", 7), ("polygon", 7),
        ("chainlink", 8), ("uniswap", 8), ("aave", 8),
        # Stablecoins (7-9)
        ("tether", 8), ("usdt", 8), ("usdc", 8), ("dai", 7),
        # Exchanges/Platforms (7-9)
        ("binance", 9), ("coinbase", 9), ("kraken", 8), ("ftx", 8),
        ("opensea", 8), ("metamask", 8), ("ledger", 7),
        # Concepts (6-8)
        ("nft", 8), ("smart contract", 8), ("web3", 8), ("dao", 7),
        ("staking", 7), ("mining", 5), ("halving", 9), ("memecoin", 8),
        ("altcoin", 8), ("token", 5), ("ico", 8), ("airdrop", 7),
        # People (7-9)
        ("satoshi", 9), ("vitalik", 9), ("cz", 8), ("sbf", 8),
        ("gensler", 7), ("sec crypto", 9),
    ],
    "GLOBAL": [
        # Definitive indicators (9-10)
        ("global", 8), ("worldwide", 9), ("international", 7),
        ("united nations", 10), ("un security council", 10),
        # Organizations (8-10)
        ("imf", 9), ("international monetary fund", 10),
        ("world bank", 10), ("wto", 9), ("world trade organization", 10),
        ("who", 8), ("world health organization", 10),
        ("nato", 9), ("g7", 9), ("g20", 9), ("g8", 9),
        ("brics", 9), ("oecd", 9), ("opec", 8),
        ("world economic forum", 10), ("davos", 9),
        ("paris agreement", 9), ("climate accord", 8),
        # Concepts (6-8)
        ("geopolitical", 8), ("world war", 10), ("global economy", 9),
        ("pandemic", 7), ("climate change", 7),
    ],
}

# Simple list for backward compatibility (flattened from weighted)
GEOGRAPHY_KEYWORDS = {
    region: [kw for kw, _ in keywords]
    for region, keywords in GEOGRAPHY_KEYWORDS_WEIGHTED.items()
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

        # Multi-market event filter (drop events with too many sub-markets)
        if config.max_markets_per_event is not None:
            self.add_filter(
                filter_by_event_market_count,
                max_markets_per_event=config.max_markets_per_event,
            )

        # Volume filter
        if config.min_volume > 0 or config.max_volume is not None:
            self.add_filter(
                filter_by_volume,
                min_volume=config.min_volume,
                max_volume=getattr(config, 'max_volume', None),
            )

        # Liquidity filter
        if config.min_liquidity > 0 or getattr(config, 'max_liquidity', None) is not None:
            self.add_filter(
                filter_by_liquidity,
                min_liquidity=config.min_liquidity,
                max_liquidity=getattr(config, 'max_liquidity', None),
            )
        
        # Expiry filter
        if config.max_days_to_expiry:
            self.add_filter(filter_by_expiry, max_days=config.max_days_to_expiry)
        
        # Geography filter
        if config.geographic_regions:
            self.add_filter(
                filter_by_geography,
                regions=config.geographic_regions,
                min_score=getattr(config, 'geo_min_score', 30.0),
            )
        
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

        # Extract markets that must always be included, bypassing all other filters.
        always_included: list[Market] = []
        if self.config and self.config.always_include_keywords:
            pattern = re.compile(
                "|".join(re.escape(kw) for kw in self.config.always_include_keywords),
                re.IGNORECASE,
            )
            remaining: list[Market] = []
            for m in markets:
                searchable = " ".join(filter(None, [
                    m.question,
                    m.description,
                    m.event_title,
                    " ".join(m.tags),
                ]))
                (always_included if pattern.search(searchable) else remaining).append(m)
            markets = remaining

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
        
        final = always_included + filtered
        return FilterResult(
            markets=final,
            total_before=total_before,
            total_after=len(final),
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


def calculate_region_score(
    text: str,
    region: str,
) -> tuple[float, list[str]]:
    """
    Calculate a confidence score for how strongly text relates to a region.

    Args:
        text: Text to analyze (lowercase)
        region: Region code (US, EU, UK, ASIA, etc.)

    Returns:
        Tuple of (score, matched_keywords) where score is 0-100
    """
    region_upper = region.upper()
    if region_upper not in GEOGRAPHY_KEYWORDS_WEIGHTED:
        return 0.0, []

    keywords = GEOGRAPHY_KEYWORDS_WEIGHTED[region_upper]
    matched = []
    total_weight = 0
    max_weight = 0

    # Keywords that need strict word boundaries to avoid false positives
    # (e.g., "india" shouldn't match "indiana")
    STRICT_BOUNDARY_KEYWORDS = {
        "india", "indian", "china", "chinese", "asia", "asian",
        "korea", "korean", "japan", "japanese", "uk", "eu",
        "us", "usa", "fed", "the fed", "paris", "rome", "madrid",
        "turkey", "turkish", "jordan", "cuba", "cuban", "chile",
        "guinea", "niger", "mali", "chad", "togo", "congo",
        "georgia", "colombia", "panama", "peru", "argentina",
        "sol", "ada", "eth", "btc", "token",
    }

    for keyword, weight in keywords:
        keyword_lower = keyword.lower()

        # Determine if we need strict word boundaries
        needs_boundary = (
            len(keyword) <= 4 or  # Short keywords always need boundaries
            keyword_lower in STRICT_BOUNDARY_KEYWORDS or
            any(keyword_lower.startswith(s) or keyword_lower.endswith(s)
                for s in ["ian", "ese", "ish"])  # Demonyms
        )

        if needs_boundary:
            # Strict word boundary matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
        else:
            # Allow substring matches for longer, specific terms
            pattern = re.escape(keyword)

        if re.search(pattern, text, re.IGNORECASE):
            matched.append(keyword)
            total_weight += weight
            max_weight = max(max_weight, weight)

    if not matched:
        return 0.0, []

    # Score calculation:
    # - Base: highest weight keyword found (0-10 scaled to 0-50)
    # - Bonus: additional keywords add confidence (up to +50)
    # - Multiple strong matches boost score significantly
    base_score = (max_weight / 10) * 50

    # Bonus for multiple matches (diminishing returns)
    match_bonus = min(50, len(matched) * 10 + (total_weight - max_weight) * 2)

    final_score = min(100, base_score + match_bonus)

    return final_score, matched


def filter_by_geography(
    markets: list[Market],
    regions: list[str],
    min_score: float = 30.0,
    return_scores: bool = False,
) -> list[Market]:
    """
    Filter markets by inferred geographic relevance using weighted keyword matching.

    Geographic relevance is inferred from keywords in the market title,
    description, tags, and event title. Each keyword has a weight indicating
    how strongly it suggests regional relevance.

    Args:
        markets: Markets to filter
        regions: List of region codes:
            - US: United States
            - EU: European Union
            - UK: United Kingdom
            - ASIA: Asia-Pacific (China, Japan, Korea, India, SE Asia)
            - LATAM: Latin America
            - MIDDLE_EAST: Middle East
            - AFRICA: Africa
            - CRYPTO: Cryptocurrency/blockchain
            - GLOBAL: Global/international topics
        min_score: Minimum confidence score (0-100) to include market.
            - 30 (default): Include if any moderate keyword matches
            - 50: Require strong indicator or multiple matches
            - 70: Require definitive indicator
        return_scores: If True, adds '_geo_score' and '_geo_matches' to market.raw_data

    Returns:
        Markets with inferred geographic relevance to specified regions

    Example:
        >>> us_markets = filter_by_geography(markets, ["US"])
        >>> crypto_markets = filter_by_geography(markets, ["CRYPTO"], min_score=50)
        >>> # Multiple regions (OR logic)
        >>> western_markets = filter_by_geography(markets, ["US", "EU", "UK"])
    """
    if not regions:
        return markets

    # Validate regions
    valid_regions = []
    for region in regions:
        region_upper = region.upper()
        if region_upper in GEOGRAPHY_KEYWORDS_WEIGHTED:
            valid_regions.append(region_upper)
        else:
            logger.warning(f"Unknown region '{region}'. Valid: {list(GEOGRAPHY_KEYWORDS_WEIGHTED.keys())}")

    if not valid_regions:
        logger.warning(f"No valid regions found in: {regions}")
        return markets

    filtered = []

    for market in markets:
        # Build comprehensive text to search
        search_text = " ".join([
            market.question,
            market.description or "",
            " ".join(market.tags),
            market.event_title or "",
            market.category or "",
        ]).lower()

        # Calculate score for each requested region, take the best match
        best_score = 0.0
        best_region = None
        all_matches = []

        for region in valid_regions:
            score, matches = calculate_region_score(search_text, region)
            if score > best_score:
                best_score = score
                best_region = region
                all_matches = matches

        if best_score >= min_score:
            # Optionally store score info for debugging/analysis
            if return_scores and market.raw_data is not None:
                market.raw_data['_geo_score'] = best_score
                market.raw_data['_geo_region'] = best_region
                market.raw_data['_geo_matches'] = all_matches

            filtered.append(market)
            logger.debug(
                f"Market '{market.question[:50]}...' matched {best_region} "
                f"with score {best_score:.1f} ({len(all_matches)} keywords)"
            )

    logger.info(f"Geographic filter: {len(markets)} -> {len(filtered)} markets for regions {valid_regions}")

    return filtered


def get_market_regions(
    market: Market,
    min_score: float = 30.0,
) -> dict[str, float]:
    """
    Analyze which regions a market relates to.

    Useful for understanding market geographic relevance without filtering.

    Args:
        market: Market to analyze
        min_score: Minimum score to include a region

    Returns:
        Dict mapping region codes to confidence scores

    Example:
        >>> regions = get_market_regions(market)
        >>> print(regions)  # {'US': 85.0, 'GLOBAL': 45.0}
    """
    search_text = " ".join([
        market.question,
        market.description or "",
        " ".join(market.tags),
        market.event_title or "",
    ]).lower()

    results = {}
    for region in GEOGRAPHY_KEYWORDS_WEIGHTED.keys():
        score, _ = calculate_region_score(search_text, region)
        if score >= min_score:
            results[region] = score

    # Sort by score descending
    return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))


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


def filter_by_event_market_count(
    markets: list[Market],
    max_markets_per_event: int = 5,
) -> list[Market]:
    """
    Exclude markets belonging to events that have too many sub-markets.

    Events like "Who will be PM of Japan?" spawn one market per candidate,
    producing many low-probability, hard-to-assess markets that waste LLM calls.

    Args:
        markets: Markets to filter
        max_markets_per_event: Events with more markets than this are excluded entirely.
            Markets without an event_id are always kept.

    Returns:
        Markets whose parent event has at most max_markets_per_event sub-markets
    """
    from collections import Counter

    # Count how many markets each event_id has in the current list
    event_counts: Counter = Counter()
    for m in markets:
        if m.event_id:
            event_counts[m.event_id] += 1

    # Build set of oversized events
    oversized = {eid for eid, count in event_counts.items() if count > max_markets_per_event}

    if oversized:
        logger.info(
            f"Event market-count filter: excluding {len(oversized)} events "
            f"with >{max_markets_per_event} markets each"
        )

    filtered = [m for m in markets if m.event_id not in oversized]

    logger.info(
        f"Event market-count filter: {len(markets)} -> {len(filtered)} markets "
        f"(max_per_event={max_markets_per_event})"
    )
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
