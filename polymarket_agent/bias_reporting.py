"""Markdown report generator for bias scanner."""

from datetime import datetime
from pathlib import Path

from polymarket_agent.bias_detection.models import BiasCategory, ClassifiedMarket
from polymarket_agent.data_fetching.models import Market
from polymarket_agent.filtering.filters import calculate_region_score

_SPAIN_KEYWORDS = {"spain", "spanish", "españa", "sanchez", "madrid", "barcelona", "catalonia"}


def _eu_flag(market: Market) -> str:
    """Return a flag emoji if the market is EU/Spain-focused, else empty string."""
    text = " ".join([
        market.question,
        market.description or "",
        " ".join(market.tags),
        market.event_title or "",
    ]).lower()
    if any(kw in text for kw in _SPAIN_KEYWORDS):
        return "🇪🇸"
    score, _ = calculate_region_score(text, "EU")
    if score >= 30:
        return "🇪🇺"
    return ""


def format_currency(value: float) -> str:
    """Format a number as currency with K/M suffixes."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"


def generate_market_url(market_slug: str) -> str:
    """Generate a Polymarket URL for a given market slug."""
    return f"https://polymarket.com/market/{market_slug}"


def generate_bias_report(
    grouped_markets: dict[BiasCategory, list[ClassifiedMarket]],
    output_path: str | Path,
    min_volume: float = 1000,
    min_liquidity: float = 500,
) -> Path:
    """Generate a markdown report of bias-classified markets."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    lines.append("# Polymarket Bias Scanner Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    category_titles = {
        BiasCategory.POLITICAL: "Political Bias",
        BiasCategory.PROGRESSIVE_SOCIAL: "Progressive Social",
        BiasCategory.CRYPTO_OPTIMISM: "Crypto Optimism",
    }

    total_markets = 0

    for category, title in category_titles.items():
        markets = grouped_markets.get(category, [])
        if not markets:
            continue

        total_markets += len(markets)

        lines.append(f"## {title} ({len(markets)} markets)")
        lines.append("")
        lines.append("| Rank | Market | URL | Score | Volume | Liquidity | EU |")
        lines.append("|------|--------|-----|-------|--------|-----------|-----|")

        for rank, cm in enumerate(markets, 1):
            market = cm.market
            classification = cm.classification
            url_link = f"[🔗]({generate_market_url(market.slug)})"
            lines.append(
                f"| {rank} | {market.question} | {url_link} | "
                f"{classification.bias_score} | {format_currency(market.volume)} | "
                f"{format_currency(market.liquidity)} | {_eu_flag(market)} |"
            )

        lines.append("")

    if total_markets == 0:
        lines.append("*No markets with bias potential found matching the criteria.*")
        lines.append("")

    lines.append("---")
    lines.append(f"Filters applied: min_volume={format_currency(min_volume)}, min_liquidity={format_currency(min_liquidity)}")
    lines.append(f"Markets classified with bias: {total_markets}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
