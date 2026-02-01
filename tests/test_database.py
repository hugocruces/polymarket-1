"""
Tests for SQLite database storage and backtest metric calculations.
"""

import json
import os
import tempfile
import pytest
from datetime import datetime, timedelta

from polymarket_agent.storage.database import AnalysisDatabase
from polymarket_agent.data_fetching.models import (
    Market,
    Outcome,
    LLMAssessment,
    ScoredMarket,
    EnrichedMarket,
)


@pytest.fixture
def db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = AnalysisDatabase(path)
    yield database
    database.close()
    os.unlink(path)


def make_market(market_id="m1", question="Test?") -> Market:
    return Market(
        id=market_id,
        slug=f"slug-{market_id}",
        question=question,
        description="Test market",
        outcomes=[
            Outcome(name="Yes", token_id="y1", price=0.60),
            Outcome(name="No", token_id="n1", price=0.40),
        ],
        category="test",
        tags=["test"],
        volume=10000,
        liquidity=5000,
        end_date=datetime.now() + timedelta(days=30),
    )


def make_assessment(market_id="m1", model="test-model") -> LLMAssessment:
    return LLMAssessment(
        market_id=market_id,
        probability_estimates={"Yes": (0.65, 0.75), "No": (0.25, 0.35)},
        confidence=0.7,
        reasoning="Test reasoning",
        key_factors=["Factor 1"],
        risks=["Risk 1"],
        mispricing_detected=True,
        mispricing_direction="underpriced",
        mispricing_magnitude=0.10,
        model_used=model,
        bias_adjustment={"detected_biases": ["crypto_optimism"], "estimated_skew_direction": "overpriced", "estimated_skew_magnitude": 0.05, "reasoning": "test"},
    )


def make_scored_market(market_id="m1") -> ScoredMarket:
    market = make_market(market_id)
    assessment = make_assessment(market_id)
    return ScoredMarket(
        market=market,
        enrichment=None,
        assessment=assessment,
        mispricing_score=60.0,
        confidence_score=50.0,
        evidence_score=40.0,
        liquidity_score=70.0,
        risk_score=55.0,
        total_score=55.0,
        rank=1,
    )


class TestSchemaCreation:
    """Test that the database schema is created properly."""

    def test_tables_exist(self, db):
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "runs" in tables
        assert "markets" in tables
        assert "assessments" in tables
        assert "scores" in tables
        assert "resolutions" in tables

    def test_creates_dir_if_needed(self):
        """Database should create parent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "test.db")
            database = AnalysisDatabase(path)
            assert os.path.exists(path)
            database.close()


class TestStoreAndRetrieve:
    """Test storing and retrieving data."""

    def test_create_run(self, db):
        run_id = db.create_run()
        assert run_id is not None
        assert len(run_id) == 36  # UUID format

    def test_complete_run(self, db):
        run_id = db.create_run()
        db.complete_run(run_id)
        history = db.get_run_history(limit=1)
        assert len(history) == 1
        assert history[0]["run_id"] == run_id
        assert history[0]["completed_at"] is not None

    def test_store_markets(self, db):
        run_id = db.create_run()
        markets = [make_market("m1"), make_market("m2")]
        db.store_markets(run_id, markets)

        cursor = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM markets WHERE run_id=?", (run_id,)
        )
        assert cursor.fetchone()["cnt"] == 2

    def test_store_assessments(self, db):
        run_id = db.create_run()
        assessments = [make_assessment("m1"), make_assessment("m2")]
        db.store_assessments(run_id, assessments)

        cursor = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM assessments WHERE run_id=?", (run_id,)
        )
        assert cursor.fetchone()["cnt"] == 2

    def test_store_assessments_with_bias(self, db):
        """Assessments with bias_adjustment should be stored."""
        run_id = db.create_run()
        assessment = make_assessment("m1")
        db.store_assessments(run_id, [assessment])

        cursor = db.conn.execute(
            "SELECT bias_adjustment_json FROM assessments WHERE run_id=? AND market_id=?",
            (run_id, "m1"),
        )
        row = cursor.fetchone()
        assert row is not None
        bias = json.loads(row["bias_adjustment_json"])
        assert "detected_biases" in bias

    def test_store_scores(self, db):
        run_id = db.create_run()
        scored = [make_scored_market("m1")]
        db.store_scores(run_id, scored)

        cursor = db.conn.execute(
            "SELECT total_score FROM scores WHERE run_id=? AND market_id=?",
            (run_id, "m1"),
        )
        row = cursor.fetchone()
        assert row is not None
        assert abs(row["total_score"] - 55.0) < 0.1

    def test_store_resolution(self, db):
        db.store_resolution("m1", "Yes", resolved_at="2025-06-01")
        cursor = db.conn.execute(
            "SELECT * FROM resolutions WHERE market_id=?", ("m1",)
        )
        row = cursor.fetchone()
        assert row["winning_outcome"] == "Yes"

    def test_get_unresolved_market_ids(self, db):
        run_id = db.create_run()
        db.store_assessments(run_id, [make_assessment("m1"), make_assessment("m2")])
        db.store_resolution("m1", "Yes")

        unresolved = db.get_unresolved_market_ids()
        assert "m2" in unresolved
        assert "m1" not in unresolved

    def test_get_predictions_for_resolved(self, db):
        run_id = db.create_run()
        market = make_market("m1")
        db.store_markets(run_id, [market])
        db.store_assessments(run_id, [make_assessment("m1")])
        db.store_scores(run_id, [make_scored_market("m1")])
        db.store_resolution("m1", "Yes")

        predictions = db.get_predictions_for_resolved_markets()
        assert len(predictions) >= 1
        p = predictions[0]
        assert p["market_id"] == "m1"
        assert p["winning_outcome"] == "Yes"
        assert p["confidence"] == 0.7

    def test_run_history(self, db):
        db.create_run()
        db.create_run()
        history = db.get_run_history(limit=10)
        assert len(history) == 2

    def test_duplicate_market_ignored(self, db):
        """Duplicate (run_id, market_id) should be ignored."""
        run_id = db.create_run()
        market = make_market("m1")
        db.store_markets(run_id, [market])
        db.store_markets(run_id, [market])  # duplicate

        cursor = db.conn.execute(
            "SELECT COUNT(*) as cnt FROM markets WHERE run_id=?", (run_id,)
        )
        assert cursor.fetchone()["cnt"] == 1


class TestBacktestMetrics:
    """Test backtest metric calculations with synthetic data."""

    def test_brier_score_perfect(self):
        """Perfect predictions should give Brier score of 0."""
        from scripts.backtest import calculate_brier_score

        predictions = [
            {
                "winning_outcome": "Yes",
                "probability_estimates": {"Yes": [1.0, 1.0], "No": [0.0, 0.0]},
            }
        ]
        score = calculate_brier_score(predictions)
        assert abs(score) < 0.001

    def test_brier_score_worst(self):
        """Completely wrong predictions should give high Brier score."""
        from scripts.backtest import calculate_brier_score

        predictions = [
            {
                "winning_outcome": "Yes",
                "probability_estimates": {"Yes": [0.0, 0.0], "No": [1.0, 1.0]},
            }
        ]
        score = calculate_brier_score(predictions)
        assert score >= 0.9

    def test_calibration_bins(self):
        """Calibration should return bin data."""
        from scripts.backtest import calculate_calibration

        predictions = [
            {
                "winning_outcome": "Yes",
                "probability_estimates": {"Yes": [0.7, 0.8]},
            },
            {
                "winning_outcome": "No",
                "probability_estimates": {"Yes": [0.2, 0.3]},
            },
        ]
        cal = calculate_calibration(predictions, bins=10)
        assert len(cal) > 0
        assert all("predicted_avg" in b and "actual_rate" in b for b in cal)

    def test_roi_calculation(self):
        """ROI should reflect win/loss outcomes."""
        from scripts.backtest import calculate_roi

        predictions = [
            {
                "mispricing_detected": True,
                "mispricing_direction": "underpriced",
                "winning_outcome": "Yes",
                "outcome_prices": {"Yes": 0.50},
            },
            {
                "mispricing_detected": True,
                "mispricing_direction": "underpriced",
                "winning_outcome": "No",
                "outcome_prices": {"Yes": 0.50},
            },
        ]
        roi = calculate_roi(predictions, position_size=100)
        assert roi["trades"] == 2
        assert roi["wins"] == 1
        assert roi["losses"] == 1
