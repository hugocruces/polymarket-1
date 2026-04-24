"""Tests for scanner configuration and YAML loading."""

import textwrap

import pytest

from polymarket_agent.scanner_config import (
    MissingConfigError,
    ScannerConfig,
    build_scanner_config,
    load_yaml_config,
)


# Minimal config that satisfies all required fields, used by tests that
# don't care about the specific values.
_REQUIRED = {
    "min_volume": 5000,
    "min_liquidity": 2000,
    "max_days_to_expiry": 90,
    "llm_model": "claude-sonnet-4-6",
    "max_markets": 500,
    "max_reported_markets": 20,
    "output_dir": "output",
    "verbose": False,
}


class TestScannerConfig:
    """Tests for ScannerConfig dataclass."""

    def test_construction_with_all_fields(self):
        """All fields can be set explicitly."""
        config = ScannerConfig(**_REQUIRED)
        assert config.min_volume == 5000
        assert config.llm_model == "claude-sonnet-4-6"
        assert config.verbose is False

    def test_missing_fields_raise_typeerror(self):
        """Required fields have no defaults — missing them raises TypeError."""
        with pytest.raises(TypeError):
            ScannerConfig()  # type: ignore[call-arg]

    def test_always_include_keywords_defaults(self):
        """always_include_keywords is the only field with a default."""
        config = ScannerConfig(**_REQUIRED)
        assert isinstance(config.always_include_keywords, list)
        assert len(config.always_include_keywords) > 0

    def test_always_include_keywords_is_a_fresh_list(self):
        """Each instance gets its own list — no mutable default sharing."""
        a = ScannerConfig(**_REQUIRED)
        b = ScannerConfig(**_REQUIRED)
        a.always_include_keywords.append("new keyword")
        assert "new keyword" not in b.always_include_keywords


class TestWithOverrides:
    """Tests for ScannerConfig.with_overrides."""

    def test_applies_known_keys(self):
        base = ScannerConfig(**_REQUIRED)
        result = base.with_overrides({"min_volume": 99999, "verbose": True})
        assert result.min_volume == 99999
        assert result.verbose is True
        assert result.min_liquidity == base.min_liquidity

    def test_ignores_unknown_keys(self, caplog):
        base = ScannerConfig(**_REQUIRED)
        with caplog.at_level("WARNING"):
            result = base.with_overrides({"min_volume": 100, "bogus": "x"})
        assert result.min_volume == 100
        assert "bogus" in caplog.text

    def test_none_values_are_skipped(self):
        """None means 'not set' — treat as no-op, don't overwrite."""
        base = ScannerConfig(**{**_REQUIRED, "min_volume": 7777})
        result = base.with_overrides({"min_volume": None, "min_liquidity": 3333})
        assert result.min_volume == 7777
        assert result.min_liquidity == 3333

    def test_returns_new_instance(self):
        base = ScannerConfig(**_REQUIRED)
        result = base.with_overrides({"min_volume": 9999})
        assert base.min_volume == _REQUIRED["min_volume"]
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


class TestBuildScannerConfig:
    """Tests for build_scanner_config (YAML + CLI merge)."""

    def test_yaml_only(self):
        """YAML alone can provide every required field."""
        config = build_scanner_config(yaml_values=_REQUIRED)
        assert config.min_volume == _REQUIRED["min_volume"]
        assert config.llm_model == _REQUIRED["llm_model"]

    def test_cli_only(self):
        """CLI alone can provide every required field (no YAML)."""
        config = build_scanner_config(cli_values=_REQUIRED)
        assert config.min_volume == _REQUIRED["min_volume"]

    def test_cli_overrides_yaml(self):
        yaml = dict(_REQUIRED, min_volume=1000)
        cli = {"min_volume": 99999}
        config = build_scanner_config(yaml_values=yaml, cli_values=cli)
        assert config.min_volume == 99999

    def test_yaml_fills_gaps_cli_doesnt_cover(self):
        cli = {"llm_model": "gpt-5.2"}
        config = build_scanner_config(yaml_values=_REQUIRED, cli_values=cli)
        assert config.llm_model == "gpt-5.2"  # from CLI
        assert config.min_volume == _REQUIRED["min_volume"]  # from YAML

    def test_missing_required_raises(self):
        """Running with neither YAML nor CLI providing required fields fails."""
        with pytest.raises(MissingConfigError) as exc:
            build_scanner_config(yaml_values={}, cli_values={})
        msg = str(exc.value)
        assert "min_volume" in msg
        assert "llm_model" in msg

    def test_missing_one_field_raises(self):
        yaml = {k: v for k, v in _REQUIRED.items() if k != "llm_model"}
        with pytest.raises(MissingConfigError) as exc:
            build_scanner_config(yaml_values=yaml)
        assert "llm_model" in str(exc.value)
        # Other fields shouldn't be in the error since they're provided.
        assert "min_volume" not in str(exc.value)

    def test_none_cli_values_dont_count_as_provided(self):
        """argparse.SUPPRESS keeps unset flags out of the dict; a caller
        passing None explicitly must not mask a missing field."""
        cli = {k: None for k in _REQUIRED}
        with pytest.raises(MissingConfigError):
            build_scanner_config(cli_values=cli)

    def test_always_include_keywords_default_when_omitted(self):
        """always_include_keywords is the one field that can be omitted."""
        config = build_scanner_config(yaml_values=_REQUIRED)
        assert isinstance(config.always_include_keywords, list)
        assert len(config.always_include_keywords) > 0

    def test_unknown_keys_are_logged_and_dropped(self, caplog):
        yaml = dict(_REQUIRED, bogus_key="should warn")
        with caplog.at_level("WARNING"):
            config = build_scanner_config(yaml_values=yaml)
        assert config.min_volume == _REQUIRED["min_volume"]
        assert "bogus_key" in caplog.text
