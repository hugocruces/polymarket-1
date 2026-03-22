"""
Data Models for Polymarket Markets

Defines dataclasses representing market data structures used throughout the agent.
These models provide type safety and clear documentation of data fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


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
            "description": self.description or "",
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


