"""Tests for bias report generation."""

import pytest
from pathlib import Path

from polymarket_agent.bias_reporting import generate_bias_report, format_currency
from polymarket_agent.bias_detection.models import BiasCategory, BiasClassification, ClassifiedMarket
from polymarket_agent.data_fetching.models import Market, Outcome


@pytest.fixture
def sample_grouped_markets():
    """Create sample grouped markets for testing."""
    m1 = Market(
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
    )
    c1 = BiasClassification(
        market_id="m1",
        dominated_by_bias=True,
        categories=[BiasCategory.POLITICAL],
        bias_score=75,
        reasoning="Left-favorable outcome",
    )

    m2 = Market(
        id="m2",
        slug="sanchez-vote",
        question="Will Sanchez survive confidence vote?",
        description="",
        outcomes=[
            Outcome(name="Yes", token_id="t3", price=0.60),
            Outcome(name="No", token_id="t4", price=0.40),
        ],
        volume=50000,
        liquidity=10000,
    )
    c2 = BiasClassification(
        market_id="m2",
        dominated_by_bias=True,
        categories=[BiasCategory.POLITICAL],
        bias_score=72,
        reasoning="Spanish politics, left-leaning",
    )

    return {
        BiasCategory.POLITICAL: [
            ClassifiedMarket(market=m1, classification=c1),
            ClassifiedMarket(market=m2, classification=c2),
        ],
        BiasCategory.PROGRESSIVE_SOCIAL: [],
        BiasCategory.CRYPTO_OPTIMISM: [],
    }


class TestFormatCurrency:
    """Tests for format_currency function."""

    def test_millions(self):
        assert format_currency(1_500_000) == "$1.5M"
        assert format_currency(2_000_000) == "$2.0M"

    def test_thousands(self):
        assert format_currency(50_000) == "$50K"
        assert format_currency(1_000) == "$1K"

    def test_small_amounts(self):
        assert format_currency(500) == "$500"
        assert format_currency(0) == "$0"


class TestGenerateBiasReport:
    """Tests for generate_bias_report function."""

    def test_creates_file(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "report.md"
        result = generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        assert result.exists()
        assert result == output_path

    def test_contains_header(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "# Polymarket Bias Scanner Report" in content
        assert "Generated:" in content

    def test_contains_political_section(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "## Political Bias" in content
        assert "Will Democrats win the Senate?" in content
        assert "75" in content  # bias score

    def test_contains_spain_flag(self, sample_grouped_markets, tmp_path):
        """Sanchez in the market question triggers Spain detection."""
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "🇪🇸" in content

    def test_no_direction_column(self, sample_grouped_markets, tmp_path):
        """Direction column should not appear in the report."""
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "Direction" not in content
        assert "underpriced" not in content
        assert "overpriced" not in content

    def test_full_question_in_report(self, sample_grouped_markets, tmp_path):
        """Market questions should not be truncated."""
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "Will Democrats win the Senate?" in content
        assert "Will Sanchez survive confidence vote?" in content

    def test_contains_footer(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "report.md"
        generate_bias_report(
            grouped_markets=sample_grouped_markets,
            output_path=output_path,
            min_volume=50000,
            min_liquidity=10000,
        )
        content = output_path.read_text()
        assert "Filters applied:" in content
        assert "min_volume=$50K" in content
        assert "Markets classified with bias: 2" in content

    def test_empty_categories_not_shown(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        content = output_path.read_text()
        assert "## Progressive Social" not in content
        assert "## Crypto Optimism" not in content

    def test_creates_parent_directories(self, sample_grouped_markets, tmp_path):
        output_path = tmp_path / "nested" / "dir" / "report.md"
        generate_bias_report(grouped_markets=sample_grouped_markets, output_path=output_path)
        assert output_path.exists()

    def test_no_markets_message(self, tmp_path):
        output_path = tmp_path / "report.md"
        generate_bias_report(
            grouped_markets={
                BiasCategory.POLITICAL: [],
                BiasCategory.PROGRESSIVE_SOCIAL: [],
                BiasCategory.CRYPTO_OPTIMISM: [],
            },
            output_path=output_path,
        )
        content = output_path.read_text()
        assert "No markets with bias potential found" in content
