"""
Data Models for Polymarket Markets

Defines dataclasses representing market data structures used throughout the agent.
These models provide type safety and clear documentation of data fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any


@dataclass
class Outcome:
    """
    Represents a single outcome in a prediction market.
    
    Attributes:
        name: Display name of the outcome (e.g., "Yes", "No", "Trump")
        token_id: Unique token ID for trading this outcome
        price: Current market price / implied probability (0.0 to 1.0)
        winner: Whether this outcome has been resolved as the winner
    """
    name: str
    token_id: str
    price: float
    winner: Optional[bool] = None
    
    def __post_init__(self):
        """Validate price is in valid range."""
        if not 0.0 <= self.price <= 1.0:
            # Prices sometimes come as whole-number percentages (e.g. 65 meaning 65%).
            # Values >= 2 are unambiguously percentages; values like 1.5 are just
            # slightly out-of-range probabilities that should be clamped.
            if self.price >= 2 and self.price <= 100:
                self.price = self.price / 100.0
            else:
                # Clamp to valid range
                self.price = max(0.0, min(1.0, self.price))


@dataclass
class Market:
    """
    Represents a Polymarket prediction market.
    
    Attributes:
        id: Unique market identifier
        slug: URL-friendly market identifier
        question: The market question being predicted
        description: Detailed description and resolution criteria
        outcomes: List of possible outcomes with current prices
        category: Primary category/tag
        tags: Additional tags/categories
        volume: Total trading volume in USD
        liquidity: Current market liquidity in USD
        end_date: When the market resolves
        created_at: When the market was created
        active: Whether the market is currently active
        closed: Whether the market has been closed
        resolved: Whether the market has been resolved
        event_id: Parent event ID if part of an event
        event_title: Parent event title
        condition_id: Condition ID for CLOB API
        raw_data: Original API response data
    """
    id: str
    slug: str
    question: str
    description: str
    outcomes: list[Outcome]
    category: str = ""
    tags: list[str] = field(default_factory=list)
    volume: float = 0.0
    liquidity: float = 0.0
    end_date: Optional[datetime] = None
    created_at: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    resolved: bool = False
    event_id: Optional[str] = None
    event_title: Optional[str] = None
    condition_id: Optional[str] = None
    raw_data: Optional[dict] = None
    
    @property
    def outcome_prices(self) -> dict[str, float]:
        """Get a dictionary of outcome name to price."""
        return {o.name: o.price for o in self.outcomes}
    
    @property
    def token_ids(self) -> list[str]:
        """Get list of all outcome token IDs."""
        return [o.token_id for o in self.outcomes]
    
    @property
    def days_to_expiry(self) -> Optional[int]:
        """Calculate days until market resolution."""
        if self.end_date is None:
            return None
        delta = self.end_date.date() - datetime.now().date()
        return max(0, delta.days)
    
    @property
    def is_binary(self) -> bool:
        """Check if this is a binary (Yes/No) market."""
        return len(self.outcomes) == 2
    
    def get_outcome_by_name(self, name: str) -> Optional[Outcome]:
        """Get an outcome by its name (case-insensitive)."""
        name_lower = name.lower()
        for outcome in self.outcomes:
            if outcome.name.lower() == name_lower:
                return outcome
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "slug": self.slug,
            "question": self.question,
            "description": self.description[:500] if self.description else "",
            "outcomes": [
                {"name": o.name, "token_id": o.token_id, "price": o.price}
                for o in self.outcomes
            ],
            "category": self.category,
            "tags": self.tags,
            "volume": self.volume,
            "liquidity": self.liquidity,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "days_to_expiry": self.days_to_expiry,
            "event_title": self.event_title,
        }


@dataclass
class Event:
    """
    Represents a Polymarket event containing one or more markets.
    
    Events group related markets together (e.g., "2024 US Presidential Election"
    might contain markets for each state).
    
    Attributes:
        id: Unique event identifier
        slug: URL-friendly event identifier
        title: Event title
        description: Event description
        markets: List of markets in this event
        category: Primary category
        tags: Additional tags
        start_date: Event start date
        end_date: Event end date
        active: Whether the event is active
        closed: Whether the event is closed
    """
    id: str
    slug: str
    title: str
    description: str = ""
    markets: list[Market] = field(default_factory=list)
    category: str = ""
    tags: list[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    active: bool = True
    closed: bool = False
    
    @property
    def total_volume(self) -> float:
        """Calculate total volume across all markets."""
        return sum(m.volume for m in self.markets)
    
    @property
    def market_count(self) -> int:
        """Get the number of markets in this event."""
        return len(self.markets)


@dataclass
class OrderLevel:
    """A single level in an orderbook."""
    price: float
    size: float


@dataclass
class OrderbookDepth:
    """
    Orderbook depth data for a market outcome.
    
    Attributes:
        token_id: The outcome token ID
        bids: List of bid levels (buyers)
        asks: List of ask levels (sellers)
        spread: Bid-ask spread
        mid_price: Midpoint price
        total_bid_liquidity: Total USD value of bids
        total_ask_liquidity: Total USD value of asks
    """
    token_id: str
    bids: list[OrderLevel] = field(default_factory=list)
    asks: list[OrderLevel] = field(default_factory=list)
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate the bid-ask spread."""
        if not self.bids or not self.asks:
            return None
        best_bid = max(b.price for b in self.bids)
        best_ask = min(a.price for a in self.asks)
        return best_ask - best_bid
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate the midpoint price."""
        if not self.bids or not self.asks:
            return None
        best_bid = max(b.price for b in self.bids)
        best_ask = min(a.price for a in self.asks)
        return (best_bid + best_ask) / 2
    
    @property
    def total_bid_liquidity(self) -> float:
        """Calculate total bid-side liquidity."""
        return sum(b.price * b.size for b in self.bids)
    
    @property
    def total_ask_liquidity(self) -> float:
        """Calculate total ask-side liquidity."""
        return sum(a.price * a.size for a in self.asks)
    
    @property
    def total_liquidity(self) -> float:
        """Calculate total liquidity (both sides)."""
        return self.total_bid_liquidity + self.total_ask_liquidity


@dataclass
class PriceData:
    """
    Price data from the CLOB API.
    
    Attributes:
        token_id: The outcome token ID
        price: Current price
        timestamp: When the price was fetched
    """
    token_id: str
    price: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EnrichedMarket:
    """
    A market enriched with additional context from web search.
    
    Attributes:
        market: The base market data
        external_context: Summarized external information
        sources: List of source URLs
        context_freshness: How recent the context is
        key_facts: Extracted key facts relevant to resolution
    """
    market: Market
    external_context: str = ""
    sources: list[str] = field(default_factory=list)
    context_freshness: Optional[str] = None
    key_facts: list[str] = field(default_factory=list)
    search_query: str = ""


@dataclass 
class LLMAssessment:
    """
    LLM's assessment of a market's pricing.
    
    Attributes:
        market_id: The market being assessed
        probability_estimates: Dict of outcome name to (low, high) probability range
        confidence: LLM's confidence in its assessment (0.0 to 1.0)
        reasoning: Detailed reasoning for the assessment
        key_factors: List of key factors considered
        risks: Identified risks or uncertainties
        mispricing_detected: Whether mispricing was detected
        mispricing_direction: "overpriced", "underpriced", or "fair"
        mispricing_magnitude: Estimated magnitude of mispricing
        warnings: Any warnings about the assessment (e.g., limited data)
        model_used: The LLM model that generated this assessment
    """
    market_id: str
    probability_estimates: dict[str, tuple[float, float]]
    confidence: float
    reasoning: str
    key_factors: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    mispricing_detected: bool = False
    mispricing_direction: str = "fair"
    mispricing_magnitude: float = 0.0
    warnings: list[str] = field(default_factory=list)
    model_used: str = ""
    
    @property
    def primary_outcome_estimate(self) -> tuple[float, float]:
        """Get the probability estimate for the primary outcome (usually 'Yes')."""
        for name in ["Yes", "yes", "YES"]:
            if name in self.probability_estimates:
                return self.probability_estimates[name]
        # Return first outcome if no 'Yes'
        if self.probability_estimates:
            return list(self.probability_estimates.values())[0]
        return (0.5, 0.5)


@dataclass
class ScoredMarket:
    """
    A market with scoring information for ranking.
    
    Attributes:
        market: The base market data
        enrichment: Optional enrichment data
        assessment: LLM assessment
        mispricing_score: Score for mispricing magnitude (0-100)
        confidence_score: Score for model confidence (0-100)
        evidence_score: Score for evidence quality (0-100)
        liquidity_score: Score for market liquidity (0-100)
        risk_score: Risk-adjusted score (0-100)
        total_score: Overall attractiveness score (0-100)
        rank: Final ranking position
    """
    market: Market
    enrichment: Optional[EnrichedMarket]
    assessment: LLMAssessment
    mispricing_score: float = 0.0
    confidence_score: float = 0.0
    evidence_score: float = 0.0
    liquidity_score: float = 0.0
    risk_score: float = 0.0
    total_score: float = 0.0
    rank: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "rank": self.rank,
            "market_slug": self.market.slug,
            "title": self.market.question,
            "category": self.market.category,
            "market_prices": self.market.outcome_prices,
            "llm_estimate": self.assessment.probability_estimates,
            "mispricing_detected": self.assessment.mispricing_detected,
            "mispricing_direction": self.assessment.mispricing_direction,
            "mispricing_magnitude": self.assessment.mispricing_magnitude,
            "confidence": self.assessment.confidence,
            "explanation": self.assessment.reasoning,
            "key_factors": self.assessment.key_factors,
            "risks": self.assessment.risks,
            "evidence_sources": self.enrichment.sources if self.enrichment else [],
            "volume": self.market.volume,
            "liquidity": self.market.liquidity,
            "days_to_expiry": self.market.days_to_expiry,
            "scores": {
                "mispricing": self.mispricing_score,
                "confidence": self.confidence_score,
                "evidence": self.evidence_score,
                "liquidity": self.liquidity_score,
                "risk_adjusted": self.risk_score,
                "total": self.total_score,
            },
            "warnings": self.assessment.warnings,
        }
