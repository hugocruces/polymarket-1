"""
Polymarket Bias Scanner CLI

Scans Polymarket for markets where demographic biases may create mispricing.

Usage:
    python -m polymarket_agent.scan [OPTIONS]

Examples:
    python -m polymarket_agent.scan
    python -m polymarket_agent.scan --min-volume 50000 --min-liquidity 10000
    python -m polymarket_agent.scan --model claude-sonnet-4-5 --output reports/scan.md
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.scanner import BiasScanner
from polymarket_agent.bias_reporting import generate_bias_report
from polymarket_agent.config import LLM_MODELS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket Bias Scanner - Find markets with demographic bias potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --min-volume 50000 --min-liquidity 10000
  %(prog)s --model claude-sonnet-4-5 --output reports/scan.md
        """,
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=1000,
        help="Minimum trading volume in USD (default: 1000)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=500,
        help="Minimum liquidity in USD (default: 500)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=90,
        help="Maximum days to resolution (default: 90)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(LLM_MODELS.keys()),
        default="claude-haiku-4-5",
        help="LLM model for classification (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="Maximum markets to fetch (default: 500)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: output/bias_scan_<timestamp>.md)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main_async() -> int:
    """
    Async main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build config
    config = ScannerConfig(
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
        max_days_to_expiry=args.max_days,
        llm_model=args.model,
        max_markets=args.max_markets,
        verbose=args.verbose,
    )

    # Print header
    print("\n" + "=" * 60)
    print("POLYMARKET BIAS SCANNER")
    print("=" * 60)
    print(f"Model: {config.llm_model}")
    print(f"Min Volume: ${config.min_volume:,.0f}")
    print(f"Min Liquidity: ${config.min_liquidity:,.0f}")
    print(f"Max Markets: {config.max_markets}")
    print("=" * 60 + "\n")

    # Run scanner
    scanner = BiasScanner(config)
    grouped = await scanner.run()

    # Count results
    total = sum(len(markets) for markets in grouped.values())

    if total == 0:
        print("No markets with bias potential found.")
        return 0

    # Generate report
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = Path("output") / f"bias_scan_{timestamp}.md"

    report_path = generate_bias_report(
        grouped_markets=grouped,
        output_path=output_path,
        min_volume=config.min_volume,
        min_liquidity=config.min_liquidity,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    for category, markets in grouped.items():
        if markets:
            print(f"  {category.value}: {len(markets)} markets")
    print(f"\nTotal: {total} markets with bias potential")
    print(f"Report: {report_path}")
    print("=" * 60 + "\n")

    return 0


def main():
    """Main entry point."""
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
