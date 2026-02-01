"""
SQLite Database for Analysis Persistence

Stores predictions, market data, assessments, and scores from each run.
Supports backtesting by comparing predictions with actual resolutions.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class AnalysisDatabase:
    """
    SQLite database for persisting analysis runs and enabling backtesting.

    Stores market data, LLM assessments, scores, and resolution outcomes.
    """

    def __init__(self, db_path: str = "data/polymarket_analysis.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._create_tables()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _create_tables(self):
        conn = self.conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                config_json TEXT,
                markets_fetched INTEGER DEFAULT 0,
                markets_filtered INTEGER DEFAULT 0,
                markets_assessed INTEGER DEFAULT 0,
                llm_model TEXT,
                risk_tolerance TEXT
            );

            CREATE TABLE IF NOT EXISTS markets (
                run_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                slug TEXT,
                question TEXT,
                category TEXT,
                tags_json TEXT,
                outcome_prices_json TEXT,
                volume REAL DEFAULT 0,
                liquidity REAL DEFAULT 0,
                end_date TEXT,
                fetched_at TEXT NOT NULL,
                UNIQUE(run_id, market_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS assessments (
                run_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                model_used TEXT NOT NULL,
                probability_estimates_json TEXT,
                confidence REAL,
                reasoning TEXT,
                mispricing_detected INTEGER DEFAULT 0,
                mispricing_direction TEXT,
                mispricing_magnitude REAL DEFAULT 0,
                bias_adjustment_json TEXT,
                assessed_at TEXT NOT NULL,
                UNIQUE(run_id, market_id, model_used),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS scores (
                run_id TEXT NOT NULL,
                market_id TEXT NOT NULL,
                rank INTEGER,
                mispricing_score REAL DEFAULT 0,
                confidence_score REAL DEFAULT 0,
                evidence_score REAL DEFAULT 0,
                liquidity_score REAL DEFAULT 0,
                risk_score REAL DEFAULT 0,
                total_score REAL DEFAULT 0,
                net_edge REAL,
                UNIQUE(run_id, market_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS resolutions (
                market_id TEXT PRIMARY KEY,
                resolved_at TEXT,
                winning_outcome TEXT,
                resolution_source TEXT DEFAULT 'api',
                fetched_at TEXT NOT NULL
            );
        """)
        conn.commit()

    def create_run(self, config=None) -> str:
        """Create a new analysis run. Returns run_id (UUID)."""
        run_id = str(uuid.uuid4())
        config_json = None
        llm_model = None
        risk_tolerance = None

        if config is not None:
            config_json = json.dumps({
                "llm_model": config.llm_model,
                "risk_tolerance": config.risk_tolerance.value,
                "enrichment_limit": config.enrichment_limit,
                "llm_analysis_limit": config.llm_analysis_limit,
            })
            llm_model = config.llm_model
            risk_tolerance = config.risk_tolerance.value

        self.conn.execute(
            """INSERT INTO runs (run_id, started_at, config_json, llm_model, risk_tolerance)
               VALUES (?, ?, ?, ?, ?)""",
            (run_id, datetime.now().isoformat(), config_json, llm_model, risk_tolerance),
        )
        self.conn.commit()
        return run_id

    def complete_run(self, run_id: str, result=None):
        """Mark a run as completed."""
        markets_fetched = result.markets_fetched if result else 0
        markets_filtered = result.markets_filtered if result else 0
        markets_assessed = result.markets_assessed if result else 0

        self.conn.execute(
            """UPDATE runs SET completed_at=?, markets_fetched=?,
               markets_filtered=?, markets_assessed=?
               WHERE run_id=?""",
            (datetime.now().isoformat(), markets_fetched, markets_filtered,
             markets_assessed, run_id),
        )
        self.conn.commit()

    def store_markets(self, run_id: str, markets: list):
        """Store fetched markets for a run."""
        now = datetime.now().isoformat()
        rows = []
        for m in markets:
            rows.append((
                run_id,
                m.id,
                m.slug,
                m.question,
                m.category,
                json.dumps(m.tags),
                json.dumps(m.outcome_prices),
                m.volume,
                m.liquidity,
                m.end_date.isoformat() if m.end_date else None,
                now,
            ))

        self.conn.executemany(
            """INSERT OR IGNORE INTO markets
               (run_id, market_id, slug, question, category, tags_json,
                outcome_prices_json, volume, liquidity, end_date, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def store_assessments(self, run_id: str, assessments: list):
        """Store LLM assessments for a run."""
        now = datetime.now().isoformat()
        rows = []
        for a in assessments:
            # Serialize probability estimates
            prob_json = json.dumps({
                k: list(v) for k, v in a.probability_estimates.items()
            })
            bias_json = json.dumps(a.bias_adjustment) if hasattr(a, 'bias_adjustment') and a.bias_adjustment else None

            rows.append((
                run_id,
                a.market_id,
                a.model_used,
                prob_json,
                a.confidence,
                a.reasoning,
                1 if a.mispricing_detected else 0,
                a.mispricing_direction,
                a.mispricing_magnitude,
                bias_json,
                now,
            ))

        self.conn.executemany(
            """INSERT OR IGNORE INTO assessments
               (run_id, market_id, model_used, probability_estimates_json,
                confidence, reasoning, mispricing_detected, mispricing_direction,
                mispricing_magnitude, bias_adjustment_json, assessed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def store_scores(self, run_id: str, scored_markets: list):
        """Store scored/ranked markets for a run."""
        rows = []
        for sm in scored_markets:
            net_edge = None
            if hasattr(sm, 'spread_analysis') and sm.spread_analysis:
                net_edge = sm.spread_analysis.get('net_edge')

            rows.append((
                run_id,
                sm.market.id,
                sm.rank,
                sm.mispricing_score,
                sm.confidence_score,
                sm.evidence_score,
                sm.liquidity_score,
                sm.risk_score,
                sm.total_score,
                net_edge,
            ))

        self.conn.executemany(
            """INSERT OR IGNORE INTO scores
               (run_id, market_id, rank, mispricing_score, confidence_score,
                evidence_score, liquidity_score, risk_score, total_score, net_edge)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def store_resolution(self, market_id: str, winning_outcome: str,
                         resolved_at: Optional[str] = None,
                         resolution_source: str = "api"):
        """Store or update a market resolution."""
        self.conn.execute(
            """INSERT OR REPLACE INTO resolutions
               (market_id, resolved_at, winning_outcome, resolution_source, fetched_at)
               VALUES (?, ?, ?, ?, ?)""",
            (market_id, resolved_at, winning_outcome, resolution_source,
             datetime.now().isoformat()),
        )
        self.conn.commit()

    def get_predictions_for_resolved_markets(self) -> list[dict]:
        """JOIN assessments with resolutions for backtesting."""
        cursor = self.conn.execute("""
            SELECT
                a.market_id,
                a.model_used,
                a.probability_estimates_json,
                a.confidence,
                a.mispricing_detected,
                a.mispricing_direction,
                a.mispricing_magnitude,
                a.bias_adjustment_json,
                r.winning_outcome,
                r.resolved_at,
                m.question,
                m.outcome_prices_json,
                s.total_score,
                s.net_edge
            FROM assessments a
            JOIN resolutions r ON a.market_id = r.market_id
            JOIN markets m ON a.market_id = m.market_id AND a.run_id = m.run_id
            LEFT JOIN scores s ON a.market_id = s.market_id AND a.run_id = s.run_id
            ORDER BY a.assessed_at DESC
        """)

        results = []
        for row in cursor.fetchall():
            results.append({
                "market_id": row["market_id"],
                "model_used": row["model_used"],
                "probability_estimates": json.loads(row["probability_estimates_json"]),
                "confidence": row["confidence"],
                "mispricing_detected": bool(row["mispricing_detected"]),
                "mispricing_direction": row["mispricing_direction"],
                "mispricing_magnitude": row["mispricing_magnitude"],
                "bias_adjustment": json.loads(row["bias_adjustment_json"]) if row["bias_adjustment_json"] else None,
                "winning_outcome": row["winning_outcome"],
                "resolved_at": row["resolved_at"],
                "question": row["question"],
                "outcome_prices": json.loads(row["outcome_prices_json"]),
                "total_score": row["total_score"],
                "net_edge": row["net_edge"],
            })

        return results

    def get_unresolved_market_ids(self) -> list[str]:
        """Get market IDs that have predictions but no resolution."""
        cursor = self.conn.execute("""
            SELECT DISTINCT a.market_id
            FROM assessments a
            LEFT JOIN resolutions r ON a.market_id = r.market_id
            WHERE r.market_id IS NULL
        """)
        return [row["market_id"] for row in cursor.fetchall()]

    def get_run_history(self, limit: int = 20) -> list[dict]:
        """Get recent run history."""
        cursor = self.conn.execute(
            """SELECT * FROM runs ORDER BY started_at DESC LIMIT ?""",
            (limit,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "run_id": row["run_id"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "config": json.loads(row["config_json"]) if row["config_json"] else None,
                "markets_fetched": row["markets_fetched"],
                "markets_filtered": row["markets_filtered"],
                "markets_assessed": row["markets_assessed"],
                "llm_model": row["llm_model"],
                "risk_tolerance": row["risk_tolerance"],
            })
        return results
