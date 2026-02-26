"""Tests for scanner configuration."""

import pytest
from polymarket_agent.scanner_config import ScannerConfig


class TestScannerConfig:
    """Tests for ScannerConfig dataclass."""

    def test_default_values(self):
        """Test ScannerConfig has sensible defaults."""
        config = ScannerConfig()

        assert config.min_volume == 1000
        assert config.min_liquidity == 500
        assert config.max_days_to_expiry == 90
        assert config.llm_model == "claude-haiku-4-5"
        assert config.max_markets == 500
        assert config.output_dir == "output"
        assert config.verbose is False

    def test_custom_values(self):
        """Test ScannerConfig with custom values."""
        config = ScannerConfig(
            min_volume=50000,
            min_liquidity=10000,
            max_days_to_expiry=30,
            llm_model="claude-sonnet-4-5",
            max_markets=100,
            output_dir="reports",
            verbose=True,
        )

        assert config.min_volume == 50000
        assert config.min_liquidity == 10000
        assert config.max_days_to_expiry == 30
        assert config.llm_model == "claude-sonnet-4-5"
        assert config.max_markets == 100
        assert config.output_dir == "reports"
        assert config.verbose is True

    def test_partial_custom_values(self):
        """Test ScannerConfig with only some custom values."""
        config = ScannerConfig(
            min_volume=25000,
            llm_model="gpt-5-mini",
        )

        assert config.min_volume == 25000
        assert config.llm_model == "gpt-5-mini"
        # Defaults should still apply
        assert config.min_liquidity == 500
        assert config.max_markets == 500
