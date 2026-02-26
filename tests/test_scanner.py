"""Tests for bias scanner."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from polymarket_agent.scanner import BiasScanner
from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.data_fetching.models import Market, Outcome
from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory


@pytest.fixture
def sample_markets():
    """Create sample markets for testing."""
    return [
        Market(
            id="m1",
            slug="democrat-win",
            question="Will Democrats win the Senate?",
            description="",
            outcomes=[
                Outcome(name="Yes", token_id="t1", price=0.45),
                Outcome(name="No", token_id="t2", price=0.55),
            ],
            volume=100000,
            liquidity=25000,
        ),
        Market(
            id="m2",
            slug="btc-200k",
            question="Will Bitcoin reach $200K?",
            description="",
            outcomes=[
                Outcome(name="Yes", token_id="t3", price=0.30),
                Outcome(name="No", token_id="t4", price=0.70),
            ],
            volume=500000,
            liquidity=100000,
        ),
    ]


class TestBiasScannerInit:
    """Tests for BiasScanner initialization."""

    def test_default_config(self):
        """Test BiasScanner with default config."""
        scanner = BiasScanner()
        assert scanner.config.min_volume == 1000
        assert scanner.config.min_liquidity == 500

    def test_custom_config(self):
        """Test BiasScanner with custom config."""
        config = ScannerConfig(min_volume=50000, llm_model="claude-sonnet-4-5")
        scanner = BiasScanner(config)
        assert scanner.config.min_volume == 50000
        assert scanner.config.llm_model == "claude-sonnet-4-5"


class TestBiasScannerClassifyMarkets:
    """Tests for classify_markets method."""

    @pytest.mark.asyncio
    async def test_classify_markets_returns_biased_only(self, sample_markets):
        """Test that classify_markets only returns markets with bias."""
        config = ScannerConfig()
        scanner = BiasScanner(config)

        # First market has bias, second doesn't
        mock_classifications = [
            BiasClassification(
                market_id="m1",
                dominated_by_bias=True,
                categories=[BiasCategory.POLITICAL],
                bias_score=75,
                mispricing_direction="underpriced",
                european=False,
                spain=False,
                reasoning="Left-favorable",
            ),
            BiasClassification(
                market_id="m2",
                dominated_by_bias=False,
                categories=[],
                bias_score=0,
                mispricing_direction="unclear",
                european=False,
                spain=False,
                reasoning="No bias detected",
            ),
        ]

        with patch('polymarket_agent.scanner.classify_market') as mock_classify:
            # Make classify_market return different results for each call
            mock_classify.side_effect = mock_classifications

            classified = await scanner.classify_markets(sample_markets)

            # Should only return the one with bias
            assert len(classified) == 1
            assert classified[0].market.id == "m1"
            assert classified[0].classification.bias_score == 75


class TestBiasScannerGroupByCategory:
    """Tests for group_by_category method."""

    def test_groups_by_category(self, sample_markets):
        """Test that markets are grouped by their bias categories."""
        scanner = BiasScanner()

        classified = [
            MagicMock(
                market=sample_markets[0],
                classification=MagicMock(
                    categories=[BiasCategory.POLITICAL],
                    bias_score=75,
                ),
            ),
            MagicMock(
                market=sample_markets[1],
                classification=MagicMock(
                    categories=[BiasCategory.CRYPTO_OPTIMISM],
                    bias_score=85,
                ),
            ),
        ]

        grouped = scanner.group_by_category(classified)

        assert len(grouped[BiasCategory.POLITICAL]) == 1
        assert len(grouped[BiasCategory.CRYPTO_OPTIMISM]) == 1
        assert len(grouped[BiasCategory.PROGRESSIVE_SOCIAL]) == 0

    def test_sorts_by_bias_score(self, sample_markets):
        """Test that markets within a category are sorted by bias score."""
        scanner = BiasScanner()

        classified = [
            MagicMock(
                market=sample_markets[0],
                classification=MagicMock(
                    categories=[BiasCategory.POLITICAL],
                    bias_score=60,
                ),
            ),
            MagicMock(
                market=sample_markets[1],
                classification=MagicMock(
                    categories=[BiasCategory.POLITICAL],
                    bias_score=85,
                ),
            ),
        ]

        grouped = scanner.group_by_category(classified)

        # Higher score should come first
        assert grouped[BiasCategory.POLITICAL][0].classification.bias_score == 85
        assert grouped[BiasCategory.POLITICAL][1].classification.bias_score == 60

    def test_market_in_multiple_categories(self, sample_markets):
        """Test that a market can appear in multiple categories."""
        scanner = BiasScanner()

        classified = [
            MagicMock(
                market=sample_markets[0],
                classification=MagicMock(
                    categories=[BiasCategory.POLITICAL, BiasCategory.PROGRESSIVE_SOCIAL],
                    bias_score=70,
                ),
            ),
        ]

        grouped = scanner.group_by_category(classified)

        assert len(grouped[BiasCategory.POLITICAL]) == 1
        assert len(grouped[BiasCategory.PROGRESSIVE_SOCIAL]) == 1
        assert len(grouped[BiasCategory.CRYPTO_OPTIMISM]) == 0
