"""Markdown report generator for bias scanner."""

from datetime import datetime
from pathlib import Path

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    ClassificationFailure,
    ClassifiedMarket,
)


def format_currency(value: float) -> str:
    """
    Format a number as currency with K/M suffixes.

    Args:
        value: Amount in USD.

    Returns:
        Formatted string like "$1.5M" or "$250K".
    """
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"


def generate_market_url(market_slug: str) -> str:
    """
    Generate a Polymarket URL for a given market slug.

    Args:
        market_slug: The market's slug (URL-friendly identifier).

    Returns:
        Full URL to the market on Polymarket.
    """
    return f"https://polymarket.com/market/{market_slug}"


def generate_bias_report(
    grouped_markets: dict[BiasCategory, list[ClassifiedMarket]],
    output_path: str | Path,
    min_volume: float = 1000,
    min_liquidity: float = 500,
    failures: list[ClassificationFailure] | None = None,
) -> Path:
    """
    Generate a markdown report of bias-classified markets.

    Args:
        grouped_markets: Markets grouped by bias category.
        output_path: Path to write the report.
        min_volume: Minimum volume filter used (for footer).
        min_liquidity: Minimum liquidity filter used (for footer).
        failures: Classification failures to surface in the report.

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = failures or []

    lines: list[str] = []

    # Header
    lines.append("# Polymarket Bias Scanner Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    total_markets = 0

    # Always-monitored section (no LLM classification — rendered first).
    always_monitored = grouped_markets.get(BiasCategory.ALWAYS_MONITORED, [])
    if always_monitored:
        total_markets += len(always_monitored)
        lines.append(f"## Always Monitored ({len(always_monitored)} markets)")
        lines.append("")
        lines.append("| # | Market | URL | Volume | Liquidity |")
        lines.append("|---|--------|-----|--------|-----------|")
        for rank, cm in enumerate(always_monitored, 1):
            market = cm.market
            question = market.question if len(market.question) <= 50 else market.question[:47] + "..."
            url_link = f"[🔗]({generate_market_url(market.slug)})"
            lines.append(
                f"| {rank} | {question} | {url_link} | "
                f"{format_currency(market.volume)} | {format_currency(market.liquidity)} |"
            )
        lines.append("")

    # LLM-classified bias-category sections.
    category_titles = {
        BiasCategory.POLITICAL: "Political Bias",
        BiasCategory.PROGRESSIVE_SOCIAL: "Progressive Social",
        BiasCategory.CRYPTO_OPTIMISM: "Crypto Optimism",
    }

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

            eu_flag = ""
            if classification.spain:
                eu_flag = "🇪🇸"
            elif classification.european:
                eu_flag = "🇪🇺"

            question = market.question
            if len(question) > 50:
                question = question[:47] + "..."

            url_link = f"[🔗]({generate_market_url(market.slug)})"

            lines.append(
                f"| {rank} | {question} | {url_link} | "
                f"{classification.bias_score} | {format_currency(market.volume)} | "
                f"{format_currency(market.liquidity)} | {eu_flag} |"
            )

        lines.append("")

    # Handle case where no markets found
    if total_markets == 0:
        lines.append("*No markets with bias potential found matching the criteria.*")
        lines.append("")

    # Classification failures — surface so silent drops don't look like no-bias.
    if failures:
        lines.append(f"## Classification Failures ({len(failures)} markets)")
        lines.append("")
        lines.append("| # | Market | URL | Error |")
        lines.append("|---|--------|-----|-------|")
        for rank, failure in enumerate(failures, 1):
            market = failure.market
            question = market.question if len(market.question) <= 50 else market.question[:47] + "..."
            url_link = f"[🔗]({generate_market_url(market.slug)})"
            error = failure.error if len(failure.error) <= 80 else failure.error[:77] + "..."
            lines.append(f"| {rank} | {question} | {url_link} | {error} |")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"Filters applied: min_volume={format_currency(min_volume)}, min_liquidity={format_currency(min_liquidity)}")
    lines.append(f"Markets classified with bias: {total_markets}")
    if failures:
        lines.append(f"Classification failures: {len(failures)}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
