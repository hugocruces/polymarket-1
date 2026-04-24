"""Bias scanner for Polymarket."""

import logging
import re
from dataclasses import dataclass, field

import httpx

from polymarket_agent.bias_detection.classifier import classify_market
from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
    ClassificationError,
    ClassificationFailure,
    ClassifiedMarket,
)
from polymarket_agent.config import FilterConfig
from polymarket_agent.data_fetching.gamma_api import fetch_active_markets
from polymarket_agent.data_fetching.models import Market
from polymarket_agent.filtering.filters import MarketFilter
from polymarket_agent.scanner_config import ScannerConfig

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Full output of a scan run.

    Attributes:
        grouped: Successfully classified markets grouped by bias category.
        failures: Markets whose LLM classification failed and were excluded
            from `grouped`. Surfaced so silent drops don't masquerade as
            "no bias detected".
    """

    grouped: dict[BiasCategory, list[ClassifiedMarket]]
    failures: list[ClassificationFailure] = field(default_factory=list)


class BiasScanner:
    """
    Scans Polymarket for markets with demographic bias potential.

    Pipeline:
        1. Fetch markets from Polymarket API
        2. Filter by volume/liquidity
        3. Classify each market with LLM
        4. Group by category and rank by bias score

    Example:
        >>> config = ScannerConfig(min_volume=50000)
        >>> scanner = BiasScanner(config)
        >>> result = await scanner.run()
        >>> for category, markets in result.grouped.items():
        ...     print(f"{category.value}: {len(markets)} markets")
    """

    def __init__(self, config: ScannerConfig):
        """
        Initialize the scanner.

        Args:
            config: Scanner configuration. Build one via
                :func:`polymarket_agent.scanner_config.build_scanner_config`
                when running from the CLI.
        """
        self.config = config

    async def fetch_markets(self) -> list[Market]:
        """
        Fetch markets from Polymarket API.

        Returns:
            List of active markets sorted by volume.
        """
        markets = await fetch_active_markets(limit=self.config.max_markets)
        logger.info(f"Fetched {len(markets)} markets")
        return markets

    def filter_markets(self, markets: list[Market]) -> list[Market]:
        """
        Apply volume/liquidity filters to markets.

        Args:
            markets: List of markets to filter.

        Returns:
            Filtered list of markets meeting criteria.
        """
        filter_config = FilterConfig(
            min_volume=self.config.min_volume,
            min_liquidity=self.config.min_liquidity,
            max_days_to_expiry=self.config.max_days_to_expiry,
            always_include_keywords=self.config.always_include_keywords,
        )
        market_filter = MarketFilter(filter_config)
        result = market_filter.apply(markets)
        logger.info(f"Filtered {result.total_before} -> {result.total_after} markets")
        return result.markets

    async def classify_markets(
        self,
        markets: list[Market],
    ) -> tuple[list[ClassifiedMarket], list[ClassificationFailure]]:
        """
        Classify markets with LLM for bias potential.

        Args:
            markets: List of markets to classify.

        Returns:
            Tuple of (classified markets with bias, classification failures).
            Failures are returned separately rather than being silently
            dropped so callers can surface them.
        """
        classified: list[ClassifiedMarket] = []
        failures: list[ClassificationFailure] = []
        total = len(markets)

        for i, market in enumerate(markets):
            logger.debug(f"Classifying market {i+1}/{total}: {market.question[:50]}...")
            try:
                classification = await classify_market(
                    market,
                    model=self.config.llm_model,
                )
            except ClassificationError as e:
                logger.warning(f"Classification error for market {market.id}: {e}")
                failures.append(ClassificationFailure(market=market, error=str(e)))
                continue
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                logger.warning(f"HTTP error classifying market {market.id}: {e}")
                failures.append(
                    ClassificationFailure(market=market, error=f"HTTP error: {e}")
                )
                continue
            except (ImportError, ValueError) as e:
                # ImportError: provider SDK not installed.
                # ValueError: unknown model. Both are configuration errors
                # that will recur for every market — fail fast.
                logger.error(f"Fatal configuration error: {e}")
                raise

            if classification.dominated_by_bias:
                classified.append(
                    ClassifiedMarket(market=market, classification=classification)
                )
                logger.debug(f"  -> Bias detected: {classification.categories}")

        logger.info(
            f"Classified {len(classified)} markets with bias potential, "
            f"{len(failures)} failed, out of {total}"
        )
        return classified, failures

    def group_by_category(
        self,
        classified: list[ClassifiedMarket],
    ) -> dict[BiasCategory, list[ClassifiedMarket]]:
        """
        Group classified markets by bias category and sort by score.

        Args:
            classified: List of classified markets.

        Returns:
            Dictionary mapping categories to sorted lists of markets.
        """
        groups: dict[BiasCategory, list[ClassifiedMarket]] = {
            BiasCategory.ALWAYS_MONITORED: [],
            BiasCategory.POLITICAL: [],
            BiasCategory.PROGRESSIVE_SOCIAL: [],
            BiasCategory.CRYPTO_OPTIMISM: [],
        }

        for cm in classified:
            for category in cm.classification.categories:
                groups[category].append(cm)

        for category in groups:
            groups[category].sort(
                key=lambda x: x.classification.bias_score,
                reverse=True,
            )

        return groups

    async def run(self) -> ScanResult:
        """
        Run the full scan pipeline.

        Returns:
            ScanResult with grouped markets and any classification failures.
        """
        markets = await self.fetch_markets()
        filtered = self.filter_markets(markets)

        # Split into always-monitored pool (no LLM) and normal pool (LLM classified).
        always_pool: list[Market] = []
        normal_pool: list[Market] = []
        if self.config.always_include_keywords:
            pattern = re.compile(
                "|".join(re.escape(k) for k in self.config.always_include_keywords),
                re.IGNORECASE,
            )
            for m in filtered:
                searchable = " ".join(filter(None, [
                    m.question, m.description, m.event_title, " ".join(m.tags),
                ]))
                (always_pool if pattern.search(searchable) else normal_pool).append(m)
        else:
            normal_pool = filtered

        always_classified = [
            ClassifiedMarket(
                market=m,
                classification=BiasClassification(
                    market_id=m.id,
                    dominated_by_bias=True,
                    categories=[BiasCategory.ALWAYS_MONITORED],
                    bias_score=0,
                    european=False,
                    spain=False,
                    reasoning="Always monitored — included without LLM classification.",
                ),
            )
            for m in always_pool
        ]
        logger.info(f"Always-monitored pool: {len(always_classified)} markets (no LLM)")

        normal_classified, failures = await self.classify_markets(normal_pool)
        cap = self.config.max_reported_markets
        if len(normal_classified) > cap:
            logger.info(
                f"Capping normal classified markets: {len(normal_classified)} -> {cap}"
            )
            normal_classified = normal_classified[:cap]

        grouped = self.group_by_category(always_classified + normal_classified)
        return ScanResult(grouped=grouped, failures=failures)
