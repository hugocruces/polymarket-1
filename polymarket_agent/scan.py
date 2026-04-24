"""
Polymarket Bias Scanner CLI

Scans Polymarket for markets where demographic biases may create mispricing.

Usage:
    python -m polymarket_agent.scan [OPTIONS]

Examples:
    python -m polymarket_agent.scan
    python -m polymarket_agent.scan --min-volume 50000 --min-liquidity 10000
    python -m polymarket_agent.scan --model claude-sonnet-4-6 --output reports/scan.md

Configuration precedence (highest wins):
    1. CLI flags
    2. config.yaml (auto-loaded from ./config.yaml, override with --config PATH)
    3. Defaults on ScannerConfig
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from polymarket_agent.bias_reporting import generate_bias_report
from polymarket_agent.config import LLM_MODELS
from polymarket_agent.scanner import BiasScanner
from polymarket_agent.scanner_config import ScannerConfig, load_yaml_config

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Flags default to argparse.SUPPRESS so the returned namespace only
    contains the flags the user explicitly set. That lets us overlay CLI
    values on top of YAML values without the CLI's defaults clobbering
    the YAML.
    """
    parser = argparse.ArgumentParser(
        description="Polymarket Bias Scanner - Find markets with demographic bias potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --min-volume 50000 --min-liquidity 10000
  %(prog)s --model claude-sonnet-4-6 --output reports/scan.md
  %(prog)s --config my-config.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML config file (default: ./config.yaml). "
             "If the default path is missing, built-in defaults are used. "
             "If an explicit --config path is missing, the scan aborts.",
    )
    parser.add_argument(
        "--min-volume",
        dest="min_volume",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum trading volume in USD (default from YAML/config: 5000)",
    )
    parser.add_argument(
        "--min-liquidity",
        dest="min_liquidity",
        type=float,
        default=argparse.SUPPRESS,
        help="Minimum liquidity in USD (default from YAML/config: 2000)",
    )
    parser.add_argument(
        "--max-days",
        dest="max_days_to_expiry",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum days to resolution (default from YAML/config: 90)",
    )
    parser.add_argument(
        "--model", "-m",
        dest="llm_model",
        choices=list(LLM_MODELS.keys()),
        default=argparse.SUPPRESS,
        help="LLM model for classification (default from YAML/config: claude-sonnet-4-6)",
    )
    parser.add_argument(
        "--max-markets",
        dest="max_markets",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum markets to fetch from API (default from YAML/config: 500)",
    )
    parser.add_argument(
        "--max-reported",
        dest="max_reported_markets",
        type=int,
        default=argparse.SUPPRESS,
        help="Cap on LLM-classified markets in the report (default from YAML/config: 20). "
             "Always-monitored markets are exempt from this cap.",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: <output_dir>/bias_scan_<timestamp>.md)",
    )
    parser.add_argument(
        "--verbose", "-v",
        dest="verbose",
        action="store_const",
        const=True,
        default=argparse.SUPPRESS,
        help="Enable verbose logging",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ScannerConfig:
    """Compose a ScannerConfig from defaults, YAML, and CLI args."""
    config = ScannerConfig()

    yaml_path: Path = args.config
    explicit_yaml = yaml_path != DEFAULT_CONFIG_PATH
    if yaml_path.exists():
        logger.info(f"Loading config from {yaml_path}")
        config = config.with_overrides(load_yaml_config(yaml_path))
    elif explicit_yaml:
        raise FileNotFoundError(
            f"Config file not found: {yaml_path}. "
            f"Pass --config with a valid path, or remove the flag to use defaults."
        )

    cli_overrides = {k: v for k, v in vars(args).items() if k != "config" and k != "output"}
    return config.with_overrides(cli_overrides)


async def main_async() -> int:
    """
    Async main entry point.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    args = parse_args()

    # Verbose logging requested via CLI should take effect immediately —
    # otherwise "Loading config from ..." etc. happen before logging is set up.
    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    config = build_config(args)

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG if config.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
    result = await scanner.run()
    grouped = result.grouped
    failures = result.failures

    # Count results
    total = sum(len(markets) for markets in grouped.values())

    if total == 0 and not failures:
        print("No markets with bias potential found.")
        return 0

    # Generate report
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = Path(config.output_dir) / f"bias_scan_{timestamp}.md"

    report_path = generate_bias_report(
        grouped_markets=grouped,
        output_path=output_path,
        min_volume=config.min_volume,
        min_liquidity=config.min_liquidity,
        failures=failures,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    for category, markets in grouped.items():
        if markets:
            print(f"  {category.value}: {len(markets)} markets")
    print(f"\nTotal: {total} markets with bias potential")
    if failures:
        print(f"Classification failures: {len(failures)}")
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
