"""Bias scanner for Polymarket."""

import asyncio
import logging
import re
from typing import Optional

from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.data_fetching.models import Market
from polymarket_agent.data_fetching.gamma_api import fetch_active_markets
from polymarket_agent.filtering.filters import MarketFilter
from polymarket_agent.config import FilterConfig
from polymarket_agent.bias_detection.models import BiasCategory, BiasClassification, ClassifiedMarket
from polymarket_agent.bias_detection.classifier import classify_market

logger = logging.getLogger(__name__)


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
        >>> grouped = await scanner.run()
        >>> for category, markets in grouped.items():
        ...     print(f"{category.value}: {len(markets)} markets")
    """

    def __init__(self, config: Optional[ScannerConfig] = None):
        """
        Initialize the scanner.

        Args:
            config: Scanner configuration. Uses defaults if not provided.
        """
        self.config = config or ScannerConfig()

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
    ) -> list[ClassifiedMarket]:
        """
        Classify markets with LLM for bias potential.

        Args:
            markets: List of markets to classify.

        Returns:
            List of ClassifiedMarket objects that have bias potential.
        """
        classified = []
        total = len(markets)

        for i, market in enumerate(markets):
            try:
                logger.debug(f"Classifying market {i+1}/{total}: {market.question[:50]}...")
                classification = await classify_market(
                    market,
                    model=self.config.llm_model,
                )
                if classification.dominated_by_bias:
                    classified.append(ClassifiedMarket(
                        market=market,
                        classification=classification,
                    ))
                    logger.debug(f"  -> Bias detected: {classification.categories}")
            except Exception as e:
                logger.error(f"Failed to classify market {market.id}: {e}")

        logger.info(f"Classified {len(classified)} markets with bias potential out of {total}")
        return classified

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

        # Sort each group by bias_score descending
        for category in groups:
            groups[category].sort(
                key=lambda x: x.classification.bias_score,
                reverse=True,
            )

        return groups

    async def run(self) -> dict[BiasCategory, list[ClassifiedMarket]]:
        """
        Run the full scan pipeline.

        Returns:
            Dictionary of bias categories to ranked lists of markets.
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

        # Always-monitored: wrap directly — no LLM call.
        always_classified = [
            ClassifiedMarket(
                market=m,
                classification=BiasClassification(
                    market_id=m.id,
                    dominated_by_bias=True,
                    categories=[BiasCategory.ALWAYS_MONITORED],
                    bias_score=0,
                    mispricing_direction="n/a",
                    european=False,
                    spain=False,
                    reasoning="Always monitored — included without LLM classification.",
                ),
            )
            for m in always_pool
        ]
        logger.info(f"Always-monitored pool: {len(always_classified)} markets (no LLM)")

        # Normal pool: LLM classify, then cap at max_reported_markets.
        normal_classified = await self.classify_markets(normal_pool)
        cap = self.config.max_reported_markets
        if len(normal_classified) > cap:
            logger.info(f"Capping normal classified markets: {len(normal_classified)} -> {cap}")
            normal_classified = normal_classified[:cap]

        return self.group_by_category(always_classified + normal_classified)
