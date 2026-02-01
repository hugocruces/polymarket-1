"""
Polymarket Agent CLI

Command-line interface for running the Polymarket analysis agent.

Usage:
    python -m polymarket_agent.main [OPTIONS]

Examples:
    # Basic run
    python -m polymarket_agent.main

    # With specific model
    python -m polymarket_agent.main --model claude-opus-4-5

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
    DEFAULT_LLM_MODEL,
)
from polymarket_agent.agent import PolymarketAgent


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket AI Agent - Analyze prediction markets for mispricings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model claude-sonnet-4-5
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
        default=DEFAULT_LLM_MODEL,
        help=f"LLM model to use for assessment (default: {DEFAULT_LLM_MODEL})",
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
        "--max-volume",
        type=float,
        default=None,
        help="Maximum trading volume in USD (default: no limit). "
             "Use to find potentially mispriced low-volume markets.",
    )
    filter_group.add_argument(
        "--min-liquidity",
        type=float,
        default=500,
        help="Minimum liquidity in USD (default: 500)",
    )
    filter_group.add_argument(
        "--max-liquidity",
        type=float,
        default=None,
        help="Maximum liquidity in USD (default: no limit)",
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
        help="Filter by geographic regions (US, EU, UK, ASIA, LATAM, MIDDLE_EAST, AFRICA, CRYPTO, GLOBAL)",
    )
    filter_group.add_argument(
        "--reasoning-heavy",
        action="store_true",
        help="Only include reasoning-heavy markets (where LLM analysis may have edge)",
    )
    filter_group.add_argument(
        "--min-reasoning-score",
        type=float,
        default=20,
        help="Minimum reasoning score for --reasoning-heavy filter (default: 20)",
    )
    filter_group.add_argument(
        "--llm-edge",
        nargs="+",
        choices=["high", "medium", "low", "unlikely"],
        default=["high", "medium"],
        help="Filter by LLM edge likelihood (default: high medium)",
    )
    filter_group.add_argument(
        "--min-blind-spot-score",
        type=float,
        default=10,
        help="Minimum blind spot score for bias filter (default: 10)",
    )
    filter_group.add_argument(
        "--mispricing-levels",
        nargs="+",
        choices=["high", "medium", "low"],
        default=["high", "medium"],
        help="Filter by mispricing likelihood (default: high medium)",
    )
    filter_group.add_argument(
        "--max-markets-per-event",
        type=int,
        default=None,
        help="Exclude events with more than N sub-markets (e.g., 5). "
             "Filters out 'one market per candidate' events like elections.",
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
        choices=["json", "csv", "markdown", "both"],
        default="both",
        help="Output format (default: both)",
    )
    output_group.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Maximum markets to analyze with LLM (default: 15)",
    )
    
    # Database
    db_group = parser.add_argument_group("Database")
    db_group.add_argument(
        "--no-db",
        action="store_true",
        help="Disable database storage",
    )
    db_group.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom database path (default: data/polymarket_analysis.db)",
    )

    # Spread analysis
    spread_group = parser.add_argument_group("Spread Analysis")
    spread_group.add_argument(
        "--spread-analysis",
        action="store_true",
        help="Enable spread/slippage analysis on ranked markets",
    )

    # LLM bias analysis
    bias_llm_group = parser.add_argument_group("LLM Bias Analysis")
    bias_llm_group.add_argument(
        "--llm-bias-analysis",
        action="store_true",
        help="Enable LLM-based bias direction refinement",
    )

    # Multi-model consensus
    consensus_group = parser.add_argument_group("Multi-Model Consensus")
    consensus_group.add_argument(
        "--consensus-models",
        nargs="+",
        default=[],
        help="Run multiple models for consensus (e.g., claude-sonnet-4-5 gpt-5-mini)",
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
        if args.model != DEFAULT_LLM_MODEL:
            config.llm_model = args.model
        if args.categories:
            config.filters.categories = args.categories
        if args.keywords:
            config.filters.keywords = args.keywords
        if args.risk_tolerance != "moderate":
            config.risk_tolerance = RiskTolerance(args.risk_tolerance)
        
        # Always override dry_run and verbose from CLI
        if args.dry_run:
            config.dry_run = True
        if args.verbose:
            config.verbose = True

        # New feature flags from CLI
        if args.no_db:
            config.enable_database = False
        if args.db_path:
            config.db_path = args.db_path
        if args.spread_analysis:
            config.enable_spread_analysis = True
        if args.llm_bias_analysis:
            config.llm_bias_analysis = True
        if args.consensus_models:
            config.consensus_models = args.consensus_models
        if args.max_markets_per_event is not None:
            config.filters.max_markets_per_event = args.max_markets_per_event

        return config
    
    # Build from CLI arguments
    filters = FilterConfig(
        categories=args.categories,
        keywords=args.keywords,
        exclude_keywords=args.exclude_keywords,
        min_volume=args.min_volume,
        max_volume=args.max_volume,
        min_liquidity=args.min_liquidity,
        max_liquidity=args.max_liquidity,
        max_days_to_expiry=args.max_days,
        geographic_regions=args.regions,
        reasoning_heavy_only=args.reasoning_heavy,
        min_reasoning_score=args.min_reasoning_score,
        llm_edge_levels=args.llm_edge,
        min_blind_spot_score=args.min_blind_spot_score,
        mispricing_levels=args.mispricing_levels,
        max_markets_per_event=args.max_markets_per_event,
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
        enable_database=not args.no_db,
        db_path=args.db_path or "data/polymarket_analysis.db",
        enable_spread_analysis=args.spread_analysis,
        llm_bias_analysis=args.llm_bias_analysis,
        consensus_models=args.consensus_models,
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
    
    # Volume range display
    if config.filters.max_volume:
        print(f"Volume Range: ${config.filters.min_volume:,.0f} - ${config.filters.max_volume:,.0f}")
    else:
        print(f"Min Volume: ${config.filters.min_volume:,.0f}")

    if config.filters.geographic_regions:
        print(f"Regions: {', '.join(config.filters.geographic_regions)}")

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
