"""Markdown report generator for bias scanner."""

from datetime import datetime
from pathlib import Path

from polymarket_agent.bias_detection.models import BiasCategory, ClassifiedMarket


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


def generate_bias_report(
    grouped_markets: dict[BiasCategory, list[ClassifiedMarket]],
    output_path: str | Path,
    min_volume: float = 1000,
    min_liquidity: float = 500,
) -> Path:
    """
    Generate a markdown report of bias-classified markets.

    Args:
        grouped_markets: Markets grouped by bias category.
        output_path: Path to write the report.
        min_volume: Minimum volume filter used (for footer).
        min_liquidity: Minimum liquidity filter used (for footer).

    Returns:
        Path to the generated report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Header
    lines.append("# Polymarket Bias Scanner Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    # Category sections
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
        lines.append("| Rank | Market | Direction | Score | Volume | Liquidity | EU |")
        lines.append("|------|--------|-----------|-------|--------|-----------|-----|")

        for rank, cm in enumerate(markets, 1):
            market = cm.market
            classification = cm.classification

            # EU flag
            eu_flag = ""
            if classification.spain:
                eu_flag = "🇪🇸"
            elif classification.european:
                eu_flag = "🇪🇺"

            # Truncate long questions
            question = market.question
            if len(question) > 50:
                question = question[:47] + "..."

            lines.append(
                f"| {rank} | {question} | {classification.mispricing_direction} | "
                f"{classification.bias_score} | {format_currency(market.volume)} | "
                f"{format_currency(market.liquidity)} | {eu_flag} |"
            )

        lines.append("")

    # Handle case where no markets found
    if total_markets == 0:
        lines.append("*No markets with bias potential found matching the criteria.*")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"Filters applied: min_volume={format_currency(min_volume)}, min_liquidity={format_currency(min_liquidity)}")
    lines.append(f"Markets classified with bias: {total_markets}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
