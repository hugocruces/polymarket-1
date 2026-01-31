"""
Polymarket Agent

Main agent class that orchestrates the full analysis pipeline:
1. Fetch active markets from Polymarket APIs
2. Apply filters to narrow down candidates
3. Enrich top candidates with web search context
4. Assess markets using LLM
5. Score and rank opportunities
6. Generate reports
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from polymarket_agent.config import AgentConfig, validate_api_keys
from polymarket_agent.data_fetching import fetch_active_events, fetch_active_markets
from polymarket_agent.data_fetching.models import Market, EnrichedMarket, ScoredMarket
from polymarket_agent.filtering import MarketFilter
from polymarket_agent.enrichment import enrich_markets_batch
from polymarket_agent.llm_assessment import MarketAssessor
from polymarket_agent.scoring import MarketScorer
from polymarket_agent.reporting import Reporter
from polymarket_agent.utils import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Results from an agent run."""
    markets_fetched: int
    markets_filtered: int
    markets_enriched: int
    markets_assessed: int
    ranked_markets: list[ScoredMarket]
    output_paths: dict[str, str]
    run_time_seconds: float
    errors: list[str]


class PolymarketAgent:
    """
    Main agent for analyzing Polymarket markets.
    
    Orchestrates the full pipeline from data fetching to report generation.
    
    Example:
        >>> from polymarket_agent import PolymarketAgent, AgentConfig
        >>> 
        >>> config = AgentConfig(
        ...     llm_model="claude-sonnet-4-5",
        ...     risk_tolerance=RiskTolerance.MODERATE,
        ... )
        >>> agent = PolymarketAgent(config)
        >>> results = agent.run()
        >>> 
        >>> for market in results.ranked_markets[:5]:
        ...     print(f"{market.rank}. {market.market.question}")
        ...     print(f"   Score: {market.total_score:.1f}")
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration (uses defaults if not provided)
        """
        self.config = config or AgentConfig()
        
        # Set up logging
        log_level = "DEBUG" if self.config.verbose else "INFO"
        setup_logging(level=log_level)
        
        # Initialize components
        self.filter = MarketFilter(self.config.filters)
        self.assessor = MarketAssessor(self.config)
        self.scorer = MarketScorer(self.config)
        self.reporter = Reporter(self.config)
        
        # Validate setup
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration and API keys."""
        if not self.config.dry_run:
            missing_keys = validate_api_keys(self.config)
            if missing_keys:
                logger.warning(
                    f"Missing API keys: {', '.join(missing_keys)}. "
                    f"Set these in your .env file or environment."
                )
    
    def run(self) -> AgentResult:
        """
        Execute the full analysis pipeline synchronously.
        
        Returns:
            AgentResult with ranked markets and metadata
        """
        return asyncio.run(self.run_async())
    
    async def run_async(self) -> AgentResult:
        """
        Execute the full analysis pipeline asynchronously.
        
        Returns:
            AgentResult with ranked markets and metadata
        """
        start_time = datetime.now()
        errors = []
        
        print("\n🔍 Polymarket AI Agent Starting...\n")
        
        # Step 1: Fetch markets
        print("📥 Fetching active markets from Polymarket...")
        markets = await self._fetch_markets()
        print(f"   Found {len(markets)} active markets\n")
        
        if not markets:
            return AgentResult(
                markets_fetched=0,
                markets_filtered=0,
                markets_enriched=0,
                markets_assessed=0,
                ranked_markets=[],
                output_paths={},
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=["No markets found"],
            )
        
        # Step 2: Filter markets
        print("📊 Applying filters...")
        
        filter_result = self.filter.apply(markets)
        filtered_markets = filter_result.markets
        print(f"   {filter_result.total_before} -> {filter_result.total_after} markets after filtering\n")
        
        if not filtered_markets:
            print("⚠️  No markets passed filters. Try adjusting filter criteria.\n")
            return AgentResult(
                markets_fetched=len(markets),
                markets_filtered=0,
                markets_enriched=0,
                markets_assessed=0,
                ranked_markets=[],
                output_paths={},
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=["No markets passed filters"],
            )
        
        # When bias filter is enabled, prioritize markets with higher blind spot scores
        if self.config.filters.bias_filter_enabled:
            filtered_markets.sort(
                key=lambda m: (m.raw_data or {}).get('_blind_spot_score', 0),
                reverse=True,
            )

        # Limit for enrichment and assessment
        candidates = filtered_markets[:self.config.enrichment_limit]
        
        # Step 3: Enrich with web search (if not dry run)
        enrichments: dict[str, EnrichedMarket] = {}
        
        if not self.config.dry_run:
            print(f"🔎 Enriching top {len(candidates)} candidates with web search...")
            
            def on_enrich_progress(current, total):
                print(f"\r   Enriching: {current}/{total}", end="")
            
            try:
                enriched_list = await enrich_markets_batch(
                    candidates,
                    progress_callback=on_enrich_progress,
                )
                enrichments = {e.market.id: e for e in enriched_list}
                print(f"\n   Enriched {len(enrichments)} markets\n")
            except Exception as e:
                logger.error(f"Enrichment failed: {e}")
                errors.append(f"Enrichment error: {e}")
                print(f"\n   ⚠️ Enrichment failed: {e}\n")
        
        # Step 4: LLM Assessment (if not dry run)
        assessments = []
        assessment_markets = candidates[:self.config.llm_analysis_limit]
        
        if self.config.dry_run:
            print("🔄 Dry run mode - skipping LLM assessment\n")
        else:
            print(f"🤖 Running LLM assessment ({self.config.llm_model})...")
            
            def on_assess_progress(current, total):
                print(f"\r   Assessing: {current}/{total}", end="")
            
            try:
                assessments = await self.assessor.assess_batch(
                    assessment_markets,
                    enrichments,
                    progress_callback=on_assess_progress,
                )
                print(f"\n   Assessed {len(assessments)} markets\n")
            except Exception as e:
                logger.error(f"Assessment failed: {e}")
                errors.append(f"Assessment error: {e}")
                print(f"\n   ⚠️ Assessment failed: {e}\n")
        
        # Step 5: Score and rank
        print("📈 Scoring and ranking results...")
        
        if assessments:
            scored_markets = self.scorer.score_all(
                assessment_markets,
                assessments,
                enrichments,
            )
            ranked_markets = self.scorer.rank(scored_markets)
            print(f"   Ranked {len(ranked_markets)} markets\n")
        else:
            ranked_markets = []
        
        # Step 6: Generate reports
        output_paths = {}
        if ranked_markets:
            print("📁 Generating reports...")
            output_paths = self.reporter.generate(ranked_markets)
            for fmt, path in output_paths.items():
                print(f"   {fmt.upper()}: {path}")
        
        # Calculate run time
        run_time = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Analysis complete in {run_time:.1f} seconds\n")
        
        return AgentResult(
            markets_fetched=len(markets),
            markets_filtered=len(filtered_markets),
            markets_enriched=len(enrichments),
            markets_assessed=len(assessments),
            ranked_markets=ranked_markets,
            output_paths={k: str(v) for k, v in output_paths.items()},
            run_time_seconds=run_time,
            errors=errors,
        )
    
    async def _fetch_markets(self) -> list[Market]:
        """
        Fetch markets from Polymarket APIs.
        
        Tries events endpoint first, falls back to markets endpoint.
        Uses keywords from filters to perform targeted API search if available.
        """
        all_markets = []
        
        # Use first keyword as search query if available
        search_query = None
        if self.config.filters.keywords:
            search_query = self.config.filters.keywords[0]
            logger.info(f"Using search query from config: '{search_query}'")
        
        # Try fetching via events (includes more metadata)
        try:
            events = await fetch_active_events(
                limit=self.config.max_markets_to_fetch // 5,  # Events contain multiple markets
                tag_ids=self.config.filters.tag_ids,
                query=search_query,
            )
            
            for event in events:
                all_markets.extend(event.markets)
            
            logger.info(f"Fetched {len(all_markets)} markets from {len(events)} events")
            
        except Exception as e:
            logger.warning(f"Failed to fetch events: {e}, trying markets endpoint")
        
        # If we don't have enough, supplement with direct market fetch
        if len(all_markets) < self.config.max_markets_to_fetch // 2:
            try:
                direct_markets = await fetch_active_markets(
                    limit=self.config.max_markets_to_fetch - len(all_markets),
                    query=search_query,
                )
                
                # Deduplicate by ID
                existing_ids = {m.id for m in all_markets}
                for market in direct_markets:
                    if market.id not in existing_ids:
                        all_markets.append(market)
                        existing_ids.add(market.id)
                
                logger.info(f"Added {len(direct_markets)} markets from direct fetch")
                
            except Exception as e:
                logger.error(f"Failed to fetch markets: {e}")
        
        return all_markets[:self.config.max_markets_to_fetch]
    
    async def fetch_and_filter(self) -> list[Market]:
        """
        Fetch and filter markets without LLM assessment.
        
        Useful for quick market screening.
        
        Returns:
            List of filtered markets
        """
        markets = await self._fetch_markets()
        filter_result = self.filter.apply(markets)
        return filter_result.markets
    
    async def analyze_market(
        self,
        market: Market,
        enrich: bool = True,
    ) -> ScoredMarket:
        """
        Analyze a single market.
        
        Args:
            market: Market to analyze
            enrich: Whether to enrich with web search
            
        Returns:
            ScoredMarket with assessment and scores
        """
        enrichment = None
        
        if enrich:
            from polymarket_agent.enrichment import enrich_market
            enrichment = await enrich_market(market)
        
        assessment = await self.assessor.assess(market, enrichment)
        enrichments = {market.id: enrichment} if enrichment else {}
        
        scored = self.scorer.score_all([market], [assessment], enrichments)
        ranked = self.scorer.rank(scored)
        
        return ranked[0] if ranked else scored[0]
