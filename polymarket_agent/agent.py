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
from polymarket_agent.storage.database import AnalysisDatabase

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
    run_id: Optional[str] = None


def _evidence_quality_score(enrichment: Optional[EnrichedMarket]) -> float:
    """Score an enrichment by how much useful evidence the web search found.

    Used to prioritise which enriched candidates get sent to the (expensive)
    LLM assessment step.  Markets with richer external context are more likely
    to produce confident, well-grounded assessments.

    Returns a score between 0 and 1.
    """
    if enrichment is None:
        return 0.0

    score = 0.0
    # Source diversity: 0-5 sources → 0-0.35
    score += min(len(enrichment.sources), 5) / 5 * 0.35
    # Fact richness: 0-10 facts → 0-0.30
    score += min(len(enrichment.key_facts), 10) / 10 * 0.30
    # Context depth: 0-2000 chars → 0-0.20
    score += min(len(enrichment.external_context), 2000) / 2000 * 0.20
    # Freshness bonus: any freshness info → 0.10
    if enrichment.context_freshness and enrichment.context_freshness != "unknown":
        score += 0.10
    # Polling data bonus → 0.05
    if (enrichment.market.raw_data or {}).get("_polling_data"):
        score += 0.05

    return score


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

        # Database
        self.db: Optional[AnalysisDatabase] = None
        if self.config.enable_database:
            try:
                self.db = AnalysisDatabase(self.config.db_path)
            except Exception as e:
                logger.warning(f"Failed to initialize database: {e}")

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
        run_id = None

        print("\n🔍 Polymarket AI Agent Starting...\n")

        # Create database run record
        if self.db:
            try:
                run_id = self.db.create_run(self.config)
            except Exception as e:
                logger.warning(f"Failed to create DB run: {e}")

        # Step 1: Fetch markets
        print("📥 Fetching active markets from Polymarket...")
        markets = await self._fetch_markets()
        print(f"   Found {len(markets)} active markets\n")

        if not markets:
            result = AgentResult(
                markets_fetched=0,
                markets_filtered=0,
                markets_enriched=0,
                markets_assessed=0,
                ranked_markets=[],
                output_paths={},
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=["No markets found"],
                run_id=run_id,
            )
            if self.db and run_id:
                self.db.complete_run(run_id, result)
            return result

        # Store fetched markets in DB
        if self.db and run_id:
            try:
                self.db.store_markets(run_id, markets)
            except Exception as e:
                logger.warning(f"Failed to store markets in DB: {e}")

        # Step 2: Filter markets
        print("📊 Applying filters...")

        filter_result = self.filter.apply(markets)
        filtered_markets = filter_result.markets
        print(f"   {filter_result.total_before} -> {filter_result.total_after} markets after filtering\n")

        if not filtered_markets:
            print("⚠️  No markets passed filters. Try adjusting filter criteria.\n")
            result = AgentResult(
                markets_fetched=len(markets),
                markets_filtered=0,
                markets_enriched=0,
                markets_assessed=0,
                ranked_markets=[],
                output_paths={},
                run_time_seconds=(datetime.now() - start_time).total_seconds(),
                errors=["No markets passed filters"],
                run_id=run_id,
            )
            if self.db and run_id:
                self.db.complete_run(run_id, result)
            return result

        # When bias filter is enabled, prioritize markets with higher blind spot scores
        if self.config.filters.bias_filter_enabled:
            filtered_markets.sort(
                key=lambda m: (m.raw_data or {}).get('_blind_spot_score', 0),
                reverse=True,
            )

        # Step 2b: LLM-based bias refinement (Feature E)
        if self.config.llm_bias_analysis and not self.config.dry_run:
            print("🧠 Refining bias analysis with LLM...")
            try:
                from polymarket_agent.analysis.demographic_bias import analyze_bias_with_llm
                from polymarket_agent.llm_assessment.providers import get_llm_client

                bias_client = get_llm_client(self.config.llm_bias_model)
                refined_count = 0
                for market in filtered_markets:
                    if market.raw_data and '_detected_biases' in market.raw_data:
                        try:
                            updated = await analyze_bias_with_llm(
                                market, market.raw_data, bias_client,
                                self.config.llm_bias_model,
                            )
                            if updated:
                                market.raw_data.update(updated)
                                refined_count += 1
                        except Exception as e:
                            logger.debug(f"LLM bias refinement failed for {market.id}: {e}")
                print(f"   Refined bias for {refined_count} markets\n")
            except Exception as e:
                logger.warning(f"LLM bias analysis failed: {e}")
                errors.append(f"LLM bias analysis error: {e}")

        # Limit for enrichment
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

        # Step 3b: Rank candidates by evidence quality, then pick best for LLM
        if enrichments and len(candidates) > self.config.llm_analysis_limit:
            candidates.sort(
                key=lambda m: _evidence_quality_score(enrichments.get(m.id)),
                reverse=True,
            )
            logger.info(
                f"Ranked {len(candidates)} candidates by evidence quality, "
                f"selecting top {self.config.llm_analysis_limit} for LLM"
            )

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

        # Store assessments in DB
        if self.db and run_id and assessments:
            try:
                self.db.store_assessments(run_id, assessments)
            except Exception as e:
                logger.warning(f"Failed to store assessments in DB: {e}")

        # Step 5: Score and rank
        print("📈 Scoring and ranking results...")

        ranked_markets = []
        if assessments:
            # Step 5b: Spread analysis (Feature F)
            spread_analyses = {}
            if self.config.enable_spread_analysis and not self.config.dry_run:
                print("📊 Analyzing orderbook spreads...")
                try:
                    from polymarket_agent.analysis.spread_analysis import analyze_spreads_batch
                    mispricing_magnitudes = {
                        a.market_id: a.mispricing_magnitude for a in assessments
                    }
                    spread_analyses = await analyze_spreads_batch(
                        assessment_markets, mispricing_magnitudes
                    )
                    tradeable = sum(1 for s in spread_analyses.values() if s.is_tradeable)
                    print(f"   Analyzed {len(spread_analyses)} markets, "
                          f"{tradeable} tradeable after spread costs\n")
                except Exception as e:
                    logger.warning(f"Spread analysis failed: {e}")
                    errors.append(f"Spread analysis error: {e}")

            scored_markets = self.scorer.score_all(
                assessment_markets,
                assessments,
                enrichments,
            )

            # Apply spread analysis data to scored markets
            for sm in scored_markets:
                if sm.market.id in spread_analyses:
                    sa = spread_analyses[sm.market.id]
                    sm.spread_analysis = sa.to_dict()
                    # Re-score with spread penalty
                    if sa.net_edge <= 0:
                        sm.total_score *= 0.3
                    elif sa.net_edge < 0.03:
                        sm.total_score *= 0.7

            ranked_markets = self.scorer.rank(scored_markets)
            print(f"   Ranked {len(ranked_markets)} markets\n")

        # Store scores in DB
        if self.db and run_id and ranked_markets:
            try:
                self.db.store_scores(run_id, ranked_markets)
            except Exception as e:
                logger.warning(f"Failed to store scores in DB: {e}")

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

        result = AgentResult(
            markets_fetched=len(markets),
            markets_filtered=len(filtered_markets),
            markets_enriched=len(enrichments),
            markets_assessed=len(assessments),
            ranked_markets=ranked_markets,
            output_paths={k: str(v) for k, v in output_paths.items()},
            run_time_seconds=run_time,
            errors=errors,
            run_id=run_id,
        )

        # Complete DB run record
        if self.db and run_id:
            try:
                self.db.complete_run(run_id, result)
            except Exception as e:
                logger.warning(f"Failed to complete DB run: {e}")

        return result
    
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
