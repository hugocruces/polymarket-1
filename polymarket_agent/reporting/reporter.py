"""
Report Generator

Generates structured output reports in JSON, CSV, and console formats.
Provides comprehensive documentation of analysis results.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from polymarket_agent.config import AgentConfig
from polymarket_agent.data_fetching.models import ScoredMarket

logger = logging.getLogger(__name__)


def generate_json_report(
    scored_markets: list[ScoredMarket],
    output_path: str | Path,
    config: Optional[AgentConfig] = None,
    include_metadata: bool = True,
) -> Path:
    """
    Generate a JSON report of analysis results.
    
    Args:
        scored_markets: List of scored and ranked markets
        output_path: Path to write the JSON file
        config: Optional agent configuration for metadata
        include_metadata: Whether to include run metadata
        
    Returns:
        Path to the generated file
        
    Example:
        >>> path = generate_json_report(markets, "output/report.json")
        >>> print(f"Report saved to {path}")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {}
    
    # Add metadata
    if include_metadata:
        report["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "total_markets_analyzed": len(scored_markets),
            "agent_version": "1.0.0",
        }
        
        if config:
            report["metadata"]["configuration"] = {
                "llm_model": config.llm_model,
                "risk_tolerance": config.risk_tolerance.value,
                "filters": {
                    "categories": config.filters.categories,
                    "keywords": config.filters.keywords,
                    "min_volume": config.filters.min_volume,
                    "min_liquidity": config.filters.min_liquidity,
                    "max_days_to_expiry": config.filters.max_days_to_expiry,
                },
            }
    
    # Add summary statistics
    if scored_markets:
        mispriced = [m for m in scored_markets if m.assessment.mispricing_detected]
        report["summary"] = {
            "total_ranked": len(scored_markets),
            "mispricing_detected": len(mispriced),
            "average_score": sum(m.total_score for m in scored_markets) / len(scored_markets),
            "top_score": scored_markets[0].total_score if scored_markets else 0,
            "overpriced_count": len([m for m in mispriced if m.assessment.mispricing_direction == "overpriced"]),
            "underpriced_count": len([m for m in mispriced if m.assessment.mispricing_direction == "underpriced"]),
        }
    
    # Add market details
    report["markets"] = [m.to_dict() for m in scored_markets]
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"JSON report saved to {output_path}")
    return output_path


def generate_csv_report(
    scored_markets: list[ScoredMarket],
    output_path: str | Path,
) -> Path:
    """
    Generate a CSV report of analysis results.
    
    Provides a flat format suitable for spreadsheet analysis.
    
    Args:
        scored_markets: List of scored and ranked markets
        output_path: Path to write the CSV file
        
    Returns:
        Path to the generated file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Define columns
    columns = [
        "rank",
        "market_slug",
        "title",
        "category",
        "market_price_yes",
        "llm_estimate_low",
        "llm_estimate_high",
        "mispricing_detected",
        "mispricing_direction",
        "mispricing_magnitude",
        "confidence",
        "total_score",
        "mispricing_score",
        "confidence_score",
        "evidence_score",
        "liquidity_score",
        "volume",
        "liquidity",
        "days_to_expiry",
        "model_used",
        "explanation",
        "warnings",
    ]
    
    rows = []
    for market in scored_markets:
        # Get Yes outcome price and estimate
        yes_price = market.market.outcome_prices.get("Yes", 0)
        yes_estimate = market.assessment.probability_estimates.get("Yes", (0, 0))
        
        row = {
            "rank": market.rank,
            "market_slug": market.market.slug,
            "title": market.market.question[:200],  # Truncate long titles
            "category": market.market.category,
            "market_price_yes": f"{yes_price:.3f}",
            "llm_estimate_low": f"{yes_estimate[0]:.3f}",
            "llm_estimate_high": f"{yes_estimate[1]:.3f}",
            "mispricing_detected": market.assessment.mispricing_detected,
            "mispricing_direction": market.assessment.mispricing_direction,
            "mispricing_magnitude": f"{market.assessment.mispricing_magnitude:.3f}",
            "confidence": f"{market.assessment.confidence:.3f}",
            "total_score": f"{market.total_score:.1f}",
            "mispricing_score": f"{market.mispricing_score:.1f}",
            "confidence_score": f"{market.confidence_score:.1f}",
            "evidence_score": f"{market.evidence_score:.1f}",
            "liquidity_score": f"{market.liquidity_score:.1f}",
            "volume": f"{market.market.volume:.0f}",
            "liquidity": f"{market.market.liquidity:.0f}",
            "days_to_expiry": market.market.days_to_expiry or "N/A",
            "model_used": market.assessment.model_used,
            "explanation": market.assessment.reasoning[:500],  # Truncate
            "warnings": "; ".join(market.assessment.warnings),
        }
        rows.append(row)
    
    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    
    logger.info(f"CSV report saved to {output_path}")
    return output_path


def print_console_report(
    scored_markets: list[ScoredMarket],
    max_display: int = 10,
    show_details: bool = True,
) -> None:
    """
    Print a formatted report to the console.
    
    Provides a human-readable summary of top opportunities.
    
    Args:
        scored_markets: List of scored and ranked markets
        max_display: Maximum number of markets to display
        show_details: Whether to show detailed explanations
    """
    if not scored_markets:
        print("\n⚠️  No markets to display.\n")
        return
    
    # Header
    print("\n" + "═" * 70)
    print("TOP POTENTIALLY MISPRICED MARKETS")
    print("═" * 70)
    
    # Summary stats
    mispriced = [m for m in scored_markets if m.assessment.mispricing_detected]
    print(f"\n📊 Summary: {len(scored_markets)} markets ranked, "
          f"{len(mispriced)} with detected mispricing\n")
    
    # Display top markets
    for market in scored_markets[:max_display]:
        _print_market_entry(market, show_details)
    
    # Footer
    if len(scored_markets) > max_display:
        print(f"\n... and {len(scored_markets) - max_display} more markets.\n")
    
    print("═" * 70)
    print("⚠️  DISCLAIMER: This analysis is probabilistic, not financial advice.")
    print("    Always conduct your own research before trading.")
    print("═" * 70 + "\n")


def _print_market_entry(market: ScoredMarket, show_details: bool = True) -> None:
    """Print a single market entry to console."""
    # Get primary outcome (usually Yes)
    yes_price = market.market.outcome_prices.get("Yes", 0)
    yes_estimate = market.assessment.probability_estimates.get("Yes", (0, 0))
    estimate_mid = (yes_estimate[0] + yes_estimate[1]) / 2
    
    # Direction indicator
    if market.assessment.mispricing_direction == "overpriced":
        direction_icon = "📉"
        direction_text = "OVERPRICED"
    elif market.assessment.mispricing_direction == "underpriced":
        direction_icon = "📈"
        direction_text = "UNDERPRICED"
    else:
        direction_icon = "➡️"
        direction_text = "FAIR"
    
    # Confidence indicator
    conf = market.assessment.confidence
    if conf >= 0.7:
        conf_text = "High"
    elif conf >= 0.4:
        conf_text = "Medium"
    else:
        conf_text = "Low"
    
    # Print entry
    print(f"\n#{market.rank} [Score: {market.total_score:.1f}] {market.market.question[:60]}")
    print(f"   {direction_icon} Market: Yes @ {yes_price:.0%} | "
          f"LLM Estimate: {yes_estimate[0]:.0%}-{yes_estimate[1]:.0%}")
    
    if market.assessment.mispricing_detected:
        print(f"   Direction: {direction_text} by ~{market.assessment.mispricing_magnitude:.0%}")
    
    print(f"   Confidence: {conf_text} ({conf:.0%}) | "
          f"Volume: ${market.market.volume:,.0f} | "
          f"Liquidity: ${market.market.liquidity:,.0f}")
    
    if show_details and market.assessment.reasoning:
        # Truncate reasoning for display
        reasoning = market.assessment.reasoning[:200]
        if len(market.assessment.reasoning) > 200:
            reasoning += "..."
        print(f"   Rationale: {reasoning}")
    
    # Show warnings if any
    if market.assessment.warnings:
        warnings_text = "; ".join(market.assessment.warnings[:2])
        print(f"   ⚠️  Warnings: {warnings_text}")


class Reporter:
    """
    Handles report generation in multiple formats.
    
    Example:
        >>> reporter = Reporter(config)
        >>> reporter.generate(scored_markets)  # Creates JSON, CSV, and console output
    """
    
    def __init__(
        self,
        config: AgentConfig,
        output_dir: Optional[str | Path] = None,
    ):
        """
        Initialize the reporter.
        
        Args:
            config: Agent configuration
            output_dir: Output directory (defaults to config.output_dir)
        """
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
    
    def generate(
        self,
        scored_markets: list[ScoredMarket],
        filename_prefix: Optional[str] = None,
        console: bool = True,
    ) -> dict[str, Path]:
        """
        Generate reports in configured formats.
        
        Args:
            scored_markets: Scored and ranked markets
            filename_prefix: Optional prefix for output files
            console: Whether to print console report
            
        Returns:
            Dictionary of format -> output path
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        prefix = filename_prefix or f"report_{timestamp}"
        
        outputs = {}
        
        # Generate based on config
        if self.config.output_format in ["json", "both"]:
            json_path = self.output_dir / f"{prefix}.json"
            generate_json_report(scored_markets, json_path, self.config)
            outputs["json"] = json_path
        
        if self.config.output_format in ["csv", "both"]:
            csv_path = self.output_dir / f"{prefix}.csv"
            generate_csv_report(scored_markets, csv_path)
            outputs["csv"] = csv_path
        
        # Console output
        if console:
            print_console_report(scored_markets)
        
        return outputs
    
    def generate_json(
        self,
        scored_markets: list[ScoredMarket],
        filename: str = "report.json",
    ) -> Path:
        """Generate JSON report only."""
        path = self.output_dir / filename
        return generate_json_report(scored_markets, path, self.config)
    
    def generate_csv(
        self,
        scored_markets: list[ScoredMarket],
        filename: str = "report.csv",
    ) -> Path:
        """Generate CSV report only."""
        path = self.output_dir / filename
        return generate_csv_report(scored_markets, path)
