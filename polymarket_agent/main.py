"""
Polymarket Agent CLI

Command-line interface for running the Polymarket analysis agent.

Usage:
    python -m polymarket_agent.main [OPTIONS]

Examples:
    # Basic run
    python -m polymarket_agent.main
    
    # With specific model
    python -m polymarket_agent.main --model claude-opus-4-5-20250514
    
    # With filters
    python -m polymarket_agent.main --categories politics crypto --min-volume 10000
    
    # Dry run (no LLM calls)
    python -m polymarket_agent.main --dry-run
"""

import argparse
import sys
from pathlib import Path

from polymarket_agent.config import (
    AgentConfig,
    FilterConfig,
    RiskTolerance,
    LLM_MODELS,
)
from polymarket_agent.agent import PolymarketAgent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket AI Agent - Analyze prediction markets for mispricings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model claude-sonnet-4-5-20250514
  %(prog)s --categories politics --min-volume 50000
  %(prog)s --risk-tolerance aggressive --limit 20
  %(prog)s --dry-run  # Quick scan without LLM calls
  %(prog)s --config config.yaml
        """,
    )
    
    # LLM Configuration
    llm_group = parser.add_argument_group("LLM Configuration")
    llm_group.add_argument(
        "--model", "-m",
        choices=list(LLM_MODELS.keys()),
        default="claude-sonnet-4-5-20250514",
        help="LLM model to use for assessment (default: claude-sonnet-4-5-20250514)",
    )
    
    # Filtering
    filter_group = parser.add_argument_group("Market Filters")
    filter_group.add_argument(
        "--categories", "-c",
        nargs="+",
        default=[],
        help="Filter by categories/tags (e.g., politics crypto)",
    )
    filter_group.add_argument(
        "--keywords", "-k",
        nargs="+",
        default=[],
        help="Filter by keywords in title/description",
    )
    filter_group.add_argument(
        "--exclude-keywords",
        nargs="+",
        default=[],
        help="Exclude markets containing these keywords",
    )
    filter_group.add_argument(
        "--min-volume",
        type=float,
        default=1000,
        help="Minimum trading volume in USD (default: 1000)",
    )
    filter_group.add_argument(
        "--min-liquidity",
        type=float,
        default=500,
        help="Minimum liquidity in USD (default: 500)",
    )
    filter_group.add_argument(
        "--max-days",
        type=int,
        default=90,
        help="Maximum days to market resolution (default: 90)",
    )
    filter_group.add_argument(
        "--regions",
        nargs="+",
        default=[],
        help="Filter by geographic regions (US, EU, UK, ASIA, CRYPTO, GLOBAL)",
    )
    
    # Risk and Scoring
    risk_group = parser.add_argument_group("Risk Configuration")
    risk_group.add_argument(
        "--risk-tolerance", "-r",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Risk tolerance for scoring (default: moderate)",
    )
    
    # Output Configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for reports (default: output)",
    )
    output_group.add_argument(
        "--format", "-f",
        choices=["json", "csv", "both"],
        default="both",
        help="Output format (default: both)",
    )
    output_group.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum markets to analyze with LLM (default: 15)",
    )
    
    # Run modes
    mode_group = parser.add_argument_group("Run Modes")
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and filter only, skip LLM assessment",
    )
    mode_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    mode_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AgentConfig:
    """Build AgentConfig from command line arguments."""
    # If config file provided, load it and override with CLI args
    if args.config:
        config = AgentConfig.from_yaml(args.config)
        
        # Override with CLI args if specified
        if args.model != "claude-sonnet-4-5-20250514":
            config.llm_model = args.model
        if args.categories:
            config.filters.categories = args.categories
        if args.keywords:
            config.filters.keywords = args.keywords
        if args.risk_tolerance != "moderate":
            config.risk_tolerance = RiskTolerance(args.risk_tolerance)
        
        return config
    
    # Build from CLI arguments
    filters = FilterConfig(
        categories=args.categories,
        keywords=args.keywords,
        exclude_keywords=args.exclude_keywords,
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
        max_days_to_expiry=args.max_days,
        geographic_regions=args.regions,
    )
    
    return AgentConfig(
        filters=filters,
        risk_tolerance=RiskTolerance(args.risk_tolerance),
        llm_model=args.model,
        llm_analysis_limit=args.limit,
        enrichment_limit=args.limit + 5,  # Enrich a few more than we analyze
        output_dir=args.output,
        output_format=args.format,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        config = build_config(args)
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
    
    # Print configuration summary
    print("\n" + "=" * 60)
    print("POLYMARKET AI AGENT")
    print("=" * 60)
    print(f"Model: {config.llm_model}")
    print(f"Risk Tolerance: {config.risk_tolerance.value}")
    print(f"Analysis Limit: {config.llm_analysis_limit} markets")
    
    if config.filters.categories:
        print(f"Categories: {', '.join(config.filters.categories)}")
    if config.filters.keywords:
        print(f"Keywords: {', '.join(config.filters.keywords)}")
    
    print(f"Min Volume: ${config.filters.min_volume:,.0f}")
    print(f"Dry Run: {config.dry_run}")
    print("=" * 60)
    
    # Run the agent
    try:
        agent = PolymarketAgent(config)
        result = agent.run()
        
        # Print summary
        print("\n" + "=" * 60)
        print("RUN SUMMARY")
        print("=" * 60)
        print(f"Markets Fetched: {result.markets_fetched}")
        print(f"Markets Filtered: {result.markets_filtered}")
        print(f"Markets Enriched: {result.markets_enriched}")
        print(f"Markets Assessed: {result.markets_assessed}")
        print(f"Markets Ranked: {len(result.ranked_markets)}")
        print(f"Run Time: {result.run_time_seconds:.1f}s")
        
        if result.errors:
            print(f"\n⚠️ Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"   - {error}")
        
        if result.output_paths:
            print(f"\n📁 Output Files:")
            for fmt, path in result.output_paths.items():
                print(f"   {fmt}: {path}")
        
        print("=" * 60 + "\n")
        
        # Exit with error if no results
        if not result.ranked_markets:
            sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
