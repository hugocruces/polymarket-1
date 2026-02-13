"""
Tests for bias detection data models.

These tests verify that bias detection models work correctly.
"""

import pytest

from polymarket_agent.bias_detection.models import (
    BiasCategory,
    BiasClassification,
)


class TestBiasCategory:
    """Tests for BiasCategory enum."""

    def test_political_value(self):
        """Test POLITICAL has correct value."""
        assert BiasCategory.POLITICAL.value == "political"

    def test_progressive_social_value(self):
        """Test PROGRESSIVE_SOCIAL has correct value."""
        assert BiasCategory.PROGRESSIVE_SOCIAL.value == "progressive_social"

    def test_crypto_optimism_value(self):
        """Test CRYPTO_OPTIMISM has correct value."""
        assert BiasCategory.CRYPTO_OPTIMISM.value == "crypto_optimism"

    def test_all_categories_present(self):
        """Test all expected categories are present."""
        expected = {"POLITICAL", "PROGRESSIVE_SOCIAL", "CRYPTO_OPTIMISM"}
        actual = {member.name for member in BiasCategory}
        assert actual == expected


class TestBiasClassification:
    """Tests for BiasClassification dataclass."""

    def test_creation_with_all_fields(self):
        """Test BiasClassification can be created with all fields."""
        classification = BiasClassification(
            market_id="test-market-123",
            dominated_by_bias=True,
            categories=[BiasCategory.POLITICAL, BiasCategory.CRYPTO_OPTIMISM],
            bias_score=75,
            mispricing_direction="overpriced",
            european=False,
            spain=False,
            reasoning="This market shows strong political bias from US-centric demographics.",
        )

        assert classification.market_id == "test-market-123"
        assert classification.dominated_by_bias is True
        assert len(classification.categories) == 2
        assert BiasCategory.POLITICAL in classification.categories
        assert BiasCategory.CRYPTO_OPTIMISM in classification.categories
        assert classification.bias_score == 75
        assert classification.mispricing_direction == "overpriced"
        assert classification.european is False
        assert classification.spain is False
        assert "political bias" in classification.reasoning

    def test_creation_with_empty_categories(self):
        """Test BiasClassification works with empty categories list."""
        classification = BiasClassification(
            market_id="neutral-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            mispricing_direction="unclear",
            european=True,
            spain=False,
            reasoning="No significant bias detected.",
        )

        assert classification.market_id == "neutral-market"
        assert classification.dominated_by_bias is False
        assert classification.categories == []
        assert classification.bias_score == 0
        assert classification.mispricing_direction == "unclear"
        assert classification.european is True
        assert classification.spain is False

    def test_underpriced_direction(self):
        """Test BiasClassification with underpriced direction."""
        classification = BiasClassification(
            market_id="crypto-market",
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=60,
            mispricing_direction="underpriced",
            european=False,
            spain=False,
            reasoning="Crypto optimism may lead to underpricing of negative outcomes.",
        )

        assert classification.mispricing_direction == "underpriced"

    def test_spain_implies_european(self):
        """Test classification with Spain set to True and European True."""
        classification = BiasClassification(
            market_id="spain-market",
            dominated_by_bias=False,
            categories=[],
            bias_score=10,
            mispricing_direction="unclear",
            european=True,
            spain=True,
            reasoning="Market about Spanish politics.",
        )

        assert classification.spain is True
        assert classification.european is True

    def test_single_category(self):
        """Test BiasClassification with a single category."""
        classification = BiasClassification(
            market_id="progressive-market",
            dominated_by_bias=True,
            categories=[BiasCategory.PROGRESSIVE_SOCIAL],
            bias_score=80,
            mispricing_direction="overpriced",
            european=False,
            spain=False,
            reasoning="Progressive social bias detected.",
        )

        assert len(classification.categories) == 1
        assert classification.categories[0] == BiasCategory.PROGRESSIVE_SOCIAL
