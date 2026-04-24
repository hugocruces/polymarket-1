"""Configuration for the bias scanner.

All runtime-configurable values live on :class:`ScannerConfig`. Every
field is required except ``always_include_keywords`` — values must come
from ``config.yaml`` or CLI flags, not from Python-level defaults. This
keeps a single source of truth for what a "default" value is:
``./config.yaml`` at the repo root.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


_DEFAULT_ALWAYS_INCLUDE_KEYWORDS = [
    # PSOE — historic & current leadership
    "Pedro Sanchez", "Zapatero", "Felipe Gonzalez",
    "Alfredo Perez Rubalcaba", "Joaquin Almunia", "Patxi Lopez",
    "Carmen Calvo", "Maria Jesus Montero", "Teresa Ribera", "Adriana Lastra",
    "Abalos", "Koldo",
    # Podemos / Sumar / IU
    "Pablo Iglesias", "Yolanda Diaz", "Irene Montero",
    "Ione Belarra", "Alberto Garzon",
    # Regional / municipal left
    "Ada Colau", "Monica Oltra",
    # UFO / UAP / extraterrestrial
    "UAP", "UFO", "extraterrestrial life", "alien life", "alien civilization",
    "alien contact", "alien disclosure", "Area 51", "Roswell", "first contact",
    # Religious prophecy & end-times
    "Second Coming", "return of Jesus", "return of Christ",
    "Rapture", "End Times", "Antichrist", "Armageddon", "biblical prophecy",
    "apocalypse",
]


class MissingConfigError(ValueError):
    """Raised when required configuration values are not provided by any source."""


@dataclass
class ScannerConfig:
    """Configuration for the Polymarket bias scanner.

    Every field is required. Values must come from ``config.yaml`` or
    CLI flags. The only exception is ``always_include_keywords``, which
    defaults to a baked-in list because it's long and rarely overridden.

    Precedence when built through :func:`polymarket_agent.scan.build_config`,
    highest first:
        1. CLI flags
        2. ``config.yaml``

    Attributes:
        min_volume: Minimum trading volume in USD.
        min_liquidity: Minimum liquidity in USD.
        max_days_to_expiry: Maximum days until resolution.
        llm_model: LLM model alias for classification.
        max_markets: Maximum markets to fetch from the API.
        max_reported_markets: Cap on LLM-classified markets in the report.
            Always-monitored markets are exempt.
        output_dir: Directory for auto-named reports.
        verbose: Enable verbose logging.
        always_include_keywords: Markets matching any of these keywords bypass
            all other filters and the LLM classification step.
    """

    min_volume: float
    min_liquidity: float
    max_days_to_expiry: int
    llm_model: str
    max_markets: int
    max_reported_markets: int
    output_dir: str
    verbose: bool
    always_include_keywords: list[str] = field(
        default_factory=lambda: list(_DEFAULT_ALWAYS_INCLUDE_KEYWORDS)
    )

    def with_overrides(self, overrides: dict[str, Any]) -> "ScannerConfig":
        """Return a copy with the given overrides applied.

        Unknown keys are logged and ignored. Keys whose value is ``None``
        are skipped, which lets callers pass a dict of partial overrides
        without clobbering existing values.
        """
        known = {f.name for f in fields(self)}
        unknown = set(overrides) - known
        if unknown:
            logger.warning(f"Ignoring unknown config keys: {sorted(unknown)}")
        applied = {
            k: v for k, v in overrides.items()
            if k in known and v is not None
        }
        return replace(self, **applied) if applied else self


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config file and return its top-level dict.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed mapping. Returns an empty dict if the file is empty.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file's top-level value is not a mapping.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {path} must contain a top-level YAML mapping, "
            f"got {type(data).__name__}"
        )
    return data


def build_scanner_config(
    yaml_values: dict[str, Any] | None = None,
    cli_values: dict[str, Any] | None = None,
) -> ScannerConfig:
    """Merge YAML and CLI values into a ScannerConfig.

    CLI values take precedence over YAML values. Required fields that
    are absent from both sources raise :class:`MissingConfigError`.
    """
    required = {
        f.name for f in fields(ScannerConfig)
        if f.name != "always_include_keywords"
    }

    merged: dict[str, Any] = {}
    if yaml_values:
        merged.update({k: v for k, v in yaml_values.items() if v is not None})
    if cli_values:
        merged.update({k: v for k, v in cli_values.items() if v is not None})

    known = {f.name for f in fields(ScannerConfig)}
    unknown = set(merged) - known
    if unknown:
        logger.warning(f"Ignoring unknown config keys: {sorted(unknown)}")
    merged = {k: v for k, v in merged.items() if k in known}

    missing = sorted(required - set(merged))
    if missing:
        raise MissingConfigError(
            "Missing required configuration values: "
            + ", ".join(missing)
            + ". Provide them via config.yaml or CLI flags."
        )

    return ScannerConfig(**merged)
