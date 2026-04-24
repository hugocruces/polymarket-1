"""Tests for scanner configuration and YAML loading."""

import textwrap

import pytest

from polymarket_agent.scanner_config import ScannerConfig, load_yaml_config


class TestScannerConfig:
    """Tests for ScannerConfig dataclass."""

    def test_default_values(self):
        """Test ScannerConfig has sensible defaults."""
        config = ScannerConfig()

        assert config.min_volume == 5000
        assert config.min_liquidity == 2000
        assert config.max_days_to_expiry == 90
        assert config.llm_model == "claude-sonnet-4-6"
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
        assert config.min_liquidity == 2000
        assert config.max_markets == 500


class TestWithOverrides:
    """Tests for ScannerConfig.with_overrides."""

    def test_applies_known_keys(self):
        base = ScannerConfig()
        result = base.with_overrides({"min_volume": 99999, "verbose": True})
        assert result.min_volume == 99999
        assert result.verbose is True
        # Unchanged fields keep defaults.
        assert result.min_liquidity == base.min_liquidity

    def test_ignores_unknown_keys(self, caplog):
        base = ScannerConfig()
        with caplog.at_level("WARNING"):
            result = base.with_overrides({"min_volume": 100, "bogus": "x"})
        assert result.min_volume == 100
        assert "bogus" in caplog.text

    def test_none_values_are_skipped(self):
        """None means 'not set' — treat as no-op, don't overwrite."""
        base = ScannerConfig(min_volume=7777)
        result = base.with_overrides({"min_volume": None, "min_liquidity": 3333})
        assert result.min_volume == 7777
        assert result.min_liquidity == 3333

    def test_empty_dict_is_noop(self):
        base = ScannerConfig()
        result = base.with_overrides({})
        assert result == base

    def test_returns_new_instance(self):
        """Override should not mutate the original."""
        base = ScannerConfig(min_volume=5000)
        result = base.with_overrides({"min_volume": 9999})
        assert base.min_volume == 5000
        assert result.min_volume == 9999


class TestLoadYamlConfig:
    """Tests for load_yaml_config."""

    def test_reads_flat_schema(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(textwrap.dedent("""
            min_volume: 12345
            llm_model: gpt-5-mini
            always_include_keywords:
              - Foo
              - Bar
        """))
        data = load_yaml_config(path)
        assert data["min_volume"] == 12345
        assert data["llm_model"] == "gpt-5-mini"
        assert data["always_include_keywords"] == ["Foo", "Bar"]

    def test_empty_file_returns_empty_dict(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        assert load_yaml_config(path) == {}

    def test_non_mapping_top_level_raises(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- foo\n- bar\n")
        with pytest.raises(ValueError):
            load_yaml_config(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_yaml_config(tmp_path / "nope.yaml")


class TestPrecedenceChain:
    """End-to-end test: defaults -> YAML -> CLI."""

    def test_full_chain(self, tmp_path):
        # YAML overrides two fields.
        path = tmp_path / "c.yaml"
        path.write_text(textwrap.dedent("""
            min_volume: 10000
            llm_model: claude-sonnet-4-5
            bogus_key: ignored
        """))
        yaml_data = load_yaml_config(path)
        config = ScannerConfig().with_overrides(yaml_data)

        # YAML-derived values win over defaults.
        assert config.min_volume == 10000
        assert config.llm_model == "claude-sonnet-4-5"
        # Defaults survive where YAML is silent.
        assert config.min_liquidity == 2000

        # CLI overrides YAML.
        cli = {"min_volume": 77777}
        final = config.with_overrides(cli)
        assert final.min_volume == 77777
        assert final.llm_model == "claude-sonnet-4-5"  # from YAML
        assert final.min_liquidity == 2000  # from defaults
