"""
Spread/Slippage Analysis

Calculates actual trading costs from orderbook data. Subtracts from mispricing
to get "net edge". Filters out opportunities where spread exceeds the mispricing.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from polymarket_agent.data_fetching.models import OrderbookDepth, OrderLevel

logger = logging.getLogger(__name__)


@dataclass
class SlippageEstimate:
    """Estimated slippage for a given position size."""
    position_size_usd: float
    side: str  # "buy" or "sell"
    avg_fill_price: float
    mid_price: float
    slippage_pct: float
    total_cost_usd: float


@dataclass
class SpreadAnalysis:
    """Full spread analysis for a market outcome."""
    token_id: str
    market_id: str
    bid_ask_spread: Optional[float]
    mid_price: Optional[float]
    slippage_estimates: list[SlippageEstimate] = field(default_factory=list)
    effective_spread_pct: float = 0.0
    net_edge: float = 0.0
    is_tradeable: bool = False
    analysis_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "token_id": self.token_id,
            "market_id": self.market_id,
            "bid_ask_spread": self.bid_ask_spread,
            "mid_price": self.mid_price,
            "slippage_estimates": [
                {
                    "position_size_usd": s.position_size_usd,
                    "side": s.side,
                    "avg_fill_price": s.avg_fill_price,
                    "slippage_pct": s.slippage_pct,
                    "total_cost_usd": s.total_cost_usd,
                }
                for s in self.slippage_estimates
            ],
            "effective_spread_pct": self.effective_spread_pct,
            "net_edge": self.net_edge,
            "is_tradeable": self.is_tradeable,
            "analysis_notes": self.analysis_notes,
        }


def calculate_slippage(
    book: OrderbookDepth,
    position_size_usd: float,
    side: str = "buy",
) -> Optional[SlippageEstimate]:
    """
    Walk orderbook levels to compute average fill price for a given position size.

    Args:
        book: Orderbook with bid/ask levels
        position_size_usd: Size of position in USD
        side: "buy" (lift asks) or "sell" (hit bids)

    Returns:
        SlippageEstimate or None if orderbook is empty
    """
    mid = book.mid_price
    if mid is None:
        return None

    # Select the correct side of the book
    if side == "buy":
        levels = sorted(book.asks, key=lambda x: x.price)  # ascending
    else:
        levels = sorted(book.bids, key=lambda x: x.price, reverse=True)  # descending

    if not levels:
        return None

    remaining_usd = position_size_usd
    total_tokens = 0.0
    total_spent = 0.0

    for level in levels:
        if remaining_usd <= 0:
            break

        level_usd_available = level.price * level.size
        fill_usd = min(remaining_usd, level_usd_available)
        tokens_filled = fill_usd / level.price if level.price > 0 else 0

        total_tokens += tokens_filled
        total_spent += fill_usd
        remaining_usd -= fill_usd

    if total_tokens == 0:
        return None

    avg_fill = total_spent / total_tokens
    slippage_pct = abs(avg_fill - mid) / mid if mid > 0 else 0

    return SlippageEstimate(
        position_size_usd=position_size_usd,
        side=side,
        avg_fill_price=avg_fill,
        mid_price=mid,
        slippage_pct=slippage_pct,
        total_cost_usd=total_spent,
    )


def analyze_spread(
    book: OrderbookDepth,
    market_id: str,
    mispricing_magnitude: float = 0.0,
    position_sizes: Optional[list[float]] = None,
) -> SpreadAnalysis:
    """
    Full spread analysis for a single orderbook.

    Args:
        book: Orderbook depth data
        market_id: Market identifier
        mispricing_magnitude: Detected mispricing magnitude (0-1)
        position_sizes: Position sizes in USD to estimate slippage for

    Returns:
        SpreadAnalysis with net edge and tradeability assessment
    """
    if position_sizes is None:
        position_sizes = [100, 500, 1000]

    notes: list[str] = []
    spread = book.spread
    mid = book.mid_price

    if spread is None or mid is None:
        return SpreadAnalysis(
            token_id=book.token_id,
            market_id=market_id,
            bid_ask_spread=None,
            mid_price=None,
            analysis_notes=["Empty or incomplete orderbook"],
        )

    # Calculate slippage for each position size
    slippage_estimates = []
    for size in position_sizes:
        for side in ["buy", "sell"]:
            est = calculate_slippage(book, size, side)
            if est:
                slippage_estimates.append(est)

    # Effective spread as percentage of mid price
    effective_spread_pct = spread / mid if mid > 0 else 0

    # Use the $100 buy slippage as the baseline cost
    baseline_cost_pct = effective_spread_pct / 2  # half-spread
    for est in slippage_estimates:
        if est.position_size_usd == position_sizes[0] and est.side == "buy":
            baseline_cost_pct = est.slippage_pct
            break

    # Net edge = mispricing - trading cost (round-trip)
    round_trip_cost = baseline_cost_pct * 2
    net_edge = mispricing_magnitude - round_trip_cost

    is_tradeable = net_edge > 0.01  # At least 1% net edge

    if effective_spread_pct > 0.10:
        notes.append(f"Wide spread ({effective_spread_pct:.1%})")
    if net_edge <= 0:
        notes.append("Net edge is negative after trading costs")
    elif net_edge < 0.03:
        notes.append("Thin net edge after trading costs")

    total_liq = book.total_liquidity
    if total_liq < 100:
        notes.append("Very thin orderbook liquidity")
        is_tradeable = False

    return SpreadAnalysis(
        token_id=book.token_id,
        market_id=market_id,
        bid_ask_spread=spread,
        mid_price=mid,
        slippage_estimates=slippage_estimates,
        effective_spread_pct=effective_spread_pct,
        net_edge=net_edge,
        is_tradeable=is_tradeable,
        analysis_notes=notes,
    )


async def analyze_spreads_batch(
    markets: list,
    mispricing_magnitudes: dict[str, float],
    position_sizes: Optional[list[float]] = None,
) -> dict[str, SpreadAnalysis]:
    """
    Fetch orderbooks via CLOB API and analyze spreads for multiple markets.

    Args:
        markets: List of Market objects (need token_ids)
        mispricing_magnitudes: Dict of market_id -> mispricing magnitude
        position_sizes: Position sizes to estimate

    Returns:
        Dict of market_id -> SpreadAnalysis
    """
    from polymarket_agent.data_fetching.clob_api import fetch_orderbook

    results = {}

    for market in markets:
        if not market.outcomes:
            continue

        # Use the first outcome's token (typically "Yes")
        token_id = market.outcomes[0].token_id
        if not token_id:
            continue

        try:
            book = await fetch_orderbook(token_id)
            if book is None:
                results[market.id] = SpreadAnalysis(
                    token_id=token_id,
                    market_id=market.id,
                    bid_ask_spread=None,
                    mid_price=None,
                    analysis_notes=["Failed to fetch orderbook"],
                )
                continue

            magnitude = mispricing_magnitudes.get(market.id, 0.0)
            analysis = analyze_spread(book, market.id, magnitude, position_sizes)
            results[market.id] = analysis

        except Exception as e:
            logger.warning(f"Spread analysis failed for {market.id}: {e}")
            results[market.id] = SpreadAnalysis(
                token_id=token_id,
                market_id=market.id,
                bid_ask_spread=None,
                mid_price=None,
                analysis_notes=[f"Error: {str(e)[:100]}"],
            )

    return results
