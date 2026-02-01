#!/usr/bin/env python3
"""
Backtest Script

Compares predictions stored in the SQLite database with actual market resolutions.
Calculates Brier score, calibration, simulated ROI, and bias category analysis.

Usage:
    python scripts/backtest.py
    python scripts/backtest.py --fetch-resolutions
    python scripts/backtest.py --db-path custom.db --format json --output report.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polymarket_agent.storage.database import AnalysisDatabase


def fetch_resolutions(db: AnalysisDatabase):
    """Fetch resolution outcomes from Polymarket API for unresolved markets."""
    import asyncio
    import httpx

    unresolved = db.get_unresolved_market_ids()
    if not unresolved:
        print("No unresolved markets to fetch.")
        return 0

    print(f"Fetching resolutions for {len(unresolved)} markets...")

    async def _fetch():
        resolved_count = 0
        async with httpx.AsyncClient(timeout=15.0) as client:
            for market_id in unresolved:
                try:
                    resp = await client.get(
                        f"https://gamma-api.polymarket.com/markets/{market_id}"
                    )
                    if resp.status_code != 200:
                        continue

                    data = resp.json()
                    if not data.get("resolved", False):
                        continue

                    # Find winning outcome
                    winning = None
                    for outcome_str in (data.get("outcomePrices") or "").split(","):
                        pass  # outcomePrices is a string of final prices

                    # Check outcomes field
                    outcomes = data.get("outcomes", [])
                    outcome_prices_str = data.get("outcomePrices", "")
                    if outcome_prices_str:
                        try:
                            prices = json.loads(outcome_prices_str)
                            for i, price in enumerate(prices):
                                if float(price) >= 0.99 and i < len(outcomes):
                                    winning = outcomes[i]
                                    break
                        except (json.JSONDecodeError, ValueError):
                            pass

                    if winning:
                        db.store_resolution(
                            market_id=market_id,
                            winning_outcome=winning,
                            resolved_at=data.get("endDate"),
                            resolution_source="api",
                        )
                        resolved_count += 1

                except Exception as e:
                    print(f"  Error fetching {market_id}: {e}")
                    continue

        return resolved_count

    count = asyncio.run(_fetch())
    print(f"Fetched {count} resolutions.")
    return count


def calculate_brier_score(predictions: list[dict]) -> float:
    """
    Calculate Brier score (mean squared error of probability estimates vs outcomes).

    Lower is better. 0 = perfect, 0.25 = random for binary.
    """
    if not predictions:
        return float("nan")

    total = 0.0
    count = 0

    for p in predictions:
        winning = p["winning_outcome"]
        estimates = p["probability_estimates"]

        for outcome, est_range in estimates.items():
            if isinstance(est_range, list) and len(est_range) == 2:
                predicted = (est_range[0] + est_range[1]) / 2
            else:
                continue

            actual = 1.0 if outcome == winning else 0.0
            total += (predicted - actual) ** 2
            count += 1

    return total / count if count > 0 else float("nan")


def calculate_calibration(predictions: list[dict], bins: int = 10) -> list[dict]:
    """
    Group predictions by predicted probability, compare to actual resolution rate.

    Returns list of bin dicts with predicted_avg, actual_rate, count.
    """
    bin_data = [{"predicted_sum": 0.0, "actual_sum": 0.0, "count": 0} for _ in range(bins)]

    for p in predictions:
        winning = p["winning_outcome"]
        estimates = p["probability_estimates"]

        for outcome, est_range in estimates.items():
            if isinstance(est_range, list) and len(est_range) == 2:
                predicted = (est_range[0] + est_range[1]) / 2
            else:
                continue

            actual = 1.0 if outcome == winning else 0.0
            bin_idx = min(int(predicted * bins), bins - 1)
            bin_data[bin_idx]["predicted_sum"] += predicted
            bin_data[bin_idx]["actual_sum"] += actual
            bin_data[bin_idx]["count"] += 1

    result = []
    for i, b in enumerate(bin_data):
        if b["count"] > 0:
            result.append({
                "bin": f"{i * (100 // bins)}-{(i + 1) * (100 // bins)}%",
                "predicted_avg": b["predicted_sum"] / b["count"],
                "actual_rate": b["actual_sum"] / b["count"],
                "count": b["count"],
            })

    return result


def calculate_roi(predictions: list[dict], position_size: float = 100) -> dict:
    """
    Simulated P&L if we'd followed the tool's recommendations.

    For each prediction with mispricing_detected, simulate buying/selling
    at market price and settling at resolution.
    """
    total_invested = 0.0
    total_return = 0.0
    trades = 0
    wins = 0

    for p in predictions:
        if not p["mispricing_detected"]:
            continue

        winning = p["winning_outcome"]
        direction = p["mispricing_direction"]
        market_prices = p.get("outcome_prices", {})

        # Get the "Yes" outcome price
        yes_price = market_prices.get("Yes", 0.5)
        if isinstance(yes_price, str):
            yes_price = float(yes_price)

        if direction == "underpriced":
            # Buy Yes at market price
            cost = yes_price * position_size
            payout = position_size if winning == "Yes" else 0
        elif direction == "overpriced":
            # Buy No (equivalent to selling Yes)
            cost = (1 - yes_price) * position_size
            payout = position_size if winning == "No" else 0
        else:
            continue

        total_invested += cost
        total_return += payout
        trades += 1
        if payout > 0:
            wins += 1

    profit = total_return - total_invested
    roi_pct = (profit / total_invested * 100) if total_invested > 0 else 0

    return {
        "trades": trades,
        "wins": wins,
        "losses": trades - wins,
        "win_rate": wins / trades if trades > 0 else 0,
        "total_invested": total_invested,
        "total_return": total_return,
        "profit": profit,
        "roi_pct": roi_pct,
        "position_size": position_size,
    }


def report_by_bias_category(predictions: list[dict]) -> dict:
    """Analyze which bias categories produced the best alpha."""
    categories: dict[str, list] = {}

    for p in predictions:
        bias = p.get("bias_adjustment")
        if not bias:
            continue

        detected = bias.get("detected_biases", [])
        winning = p["winning_outcome"]
        direction = p.get("mispricing_direction", "fair")
        market_prices = p.get("outcome_prices", {})
        yes_price = market_prices.get("Yes", 0.5)
        if isinstance(yes_price, str):
            yes_price = float(yes_price)

        # Did the prediction get the direction right?
        correct = False
        if direction == "underpriced" and winning == "Yes":
            correct = True
        elif direction == "overpriced" and winning == "No":
            correct = True

        for cat in detected:
            if cat not in categories:
                categories[cat] = []
            categories[cat].append({
                "correct": correct,
                "magnitude": p.get("mispricing_magnitude", 0),
            })

    summary = {}
    for cat, results in categories.items():
        correct_count = sum(1 for r in results if r["correct"])
        summary[cat] = {
            "count": len(results),
            "correct": correct_count,
            "accuracy": correct_count / len(results) if results else 0,
            "avg_magnitude": sum(r["magnitude"] for r in results) / len(results) if results else 0,
        }

    return summary


def print_report(predictions: list[dict]):
    """Print formatted backtest report to console."""
    if not predictions:
        print("\nNo resolved predictions found for backtesting.")
        print("Run the agent first, then wait for markets to resolve.")
        return

    print("\n" + "=" * 60)
    print("BACKTEST REPORT")
    print("=" * 60)

    print(f"\nTotal resolved predictions: {len(predictions)}")

    # Brier score
    brier = calculate_brier_score(predictions)
    print(f"\nBrier Score: {brier:.4f}")
    if brier < 0.15:
        print("  (Good - better than naive baseline)")
    elif brier < 0.25:
        print("  (Moderate - roughly random baseline)")
    else:
        print("  (Poor - worse than random)")

    # Calibration
    print("\nCalibration:")
    print(f"  {'Bin':<12} {'Predicted':>10} {'Actual':>10} {'Count':>6}")
    print(f"  {'-' * 40}")
    for b in calculate_calibration(predictions):
        print(f"  {b['bin']:<12} {b['predicted_avg']:>9.1%} {b['actual_rate']:>9.1%} {b['count']:>6}")

    # ROI
    roi = calculate_roi(predictions)
    print(f"\nSimulated Trading (${roi['position_size']}/trade):")
    print(f"  Trades: {roi['trades']}")
    print(f"  Win rate: {roi['win_rate']:.0%}")
    print(f"  Total invested: ${roi['total_invested']:,.0f}")
    print(f"  Total return: ${roi['total_return']:,.0f}")
    print(f"  Profit: ${roi['profit']:,.0f}")
    print(f"  ROI: {roi['roi_pct']:.1f}%")

    # Bias category analysis
    bias_report = report_by_bias_category(predictions)
    if bias_report:
        print("\nPerformance by Bias Category:")
        for cat, stats in sorted(bias_report.items(), key=lambda x: x[1]["accuracy"], reverse=True):
            print(f"  {cat}: {stats['accuracy']:.0%} accuracy ({stats['correct']}/{stats['count']}), "
                  f"avg magnitude {stats['avg_magnitude']:.1%}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Backtest Polymarket predictions")
    parser.add_argument("--db-path", default="data/polymarket_analysis.db",
                        help="Path to the analysis database")
    parser.add_argument("--fetch-resolutions", action="store_true",
                        help="Fetch resolution outcomes from Polymarket API")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text",
                        help="Output format (default: text)")

    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run the agent first to create the database.")
        sys.exit(1)

    db = AnalysisDatabase(str(db_path))

    if args.fetch_resolutions:
        fetch_resolutions(db)

    predictions = db.get_predictions_for_resolved_markets()

    if args.format == "text":
        print_report(predictions)
        if args.output:
            # Also write to file
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                print_report(predictions)
            Path(args.output).write_text(f.getvalue())
            print(f"\nReport saved to {args.output}")

    elif args.format == "json":
        report = {
            "total_predictions": len(predictions),
            "brier_score": calculate_brier_score(predictions),
            "calibration": calculate_calibration(predictions),
            "roi": calculate_roi(predictions),
            "bias_categories": report_by_bias_category(predictions),
            "predictions": predictions,
        }
        output = json.dumps(report, indent=2, default=str)
        if args.output:
            Path(args.output).write_text(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)

    elif args.format == "csv":
        output_path = args.output or "backtest_results.csv"
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "market_id", "question", "model_used", "confidence",
                "mispricing_detected", "mispricing_direction", "mispricing_magnitude",
                "winning_outcome", "total_score",
            ])
            writer.writeheader()
            for p in predictions:
                writer.writerow({
                    "market_id": p["market_id"],
                    "question": p.get("question", "")[:200],
                    "model_used": p["model_used"],
                    "confidence": p["confidence"],
                    "mispricing_detected": p["mispricing_detected"],
                    "mispricing_direction": p["mispricing_direction"],
                    "mispricing_magnitude": p["mispricing_magnitude"],
                    "winning_outcome": p["winning_outcome"],
                    "total_score": p.get("total_score"),
                })
        print(f"CSV report saved to {output_path}")

    db.close()


if __name__ == "__main__":
    main()
