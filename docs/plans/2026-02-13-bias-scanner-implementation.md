# Bias Scanner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the Polymarket agent from a full misprice-assessment pipeline to a bias-detection-only scanner that uses LLM classification.

**Architecture:** Fetch markets → filter by volume/liquidity → LLM classifies each market into bias categories with scores → group by category and rank → output markdown report.

**Tech Stack:** Python 3.11+, existing Anthropic/OpenAI/Google LLM clients, pytest

---

## Task 1: Create BiasClassification data model

**Files:**
- Create: `polymarket_agent/bias_detection/__init__.py`
- Create: `polymarket_agent/bias_detection/models.py`
- Test: `tests/test_bias_detection.py`

**Step 1: Write the failing test**

```python
# tests/test_bias_detection.py
"""Tests for bias detection module."""

import pytest
from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory


def test_bias_classification_creation():
    """Test creating a BiasClassification."""
    classification = BiasClassification(
        market_id="test-123",
        dominated_by_bias=True,
        categories=[BiasCategory.POLITICAL],
        bias_score=75,
        mispricing_direction="underpriced",
        european=False,
        spain=False,
        reasoning="Left-favorable outcome likely underpriced",
    )

    assert classification.market_id == "test-123"
    assert classification.dominated_by_bias is True
    assert BiasCategory.POLITICAL in classification.categories
    assert classification.bias_score == 75
    assert classification.mispricing_direction == "underpriced"


def test_bias_classification_no_bias():
    """Test classification with no bias detected."""
    classification = BiasClassification(
        market_id="test-456",
        dominated_by_bias=False,
        categories=[],
        bias_score=0,
        mispricing_direction="unclear",
        european=False,
        spain=False,
        reasoning="No demographic bias detected",
    )

    assert classification.dominated_by_bias is False
    assert classification.categories == []
    assert classification.bias_score == 0


def test_bias_category_enum():
    """Test BiasCategory enum values."""
    assert BiasCategory.POLITICAL.value == "political"
    assert BiasCategory.PROGRESSIVE_SOCIAL.value == "progressive_social"
    assert BiasCategory.CRYPTO_OPTIMISM.value == "crypto_optimism"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bias_detection.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'polymarket_agent.bias_detection'"

**Step 3: Write minimal implementation**

```python
# polymarket_agent/bias_detection/__init__.py
"""Bias detection module for classifying markets by demographic bias potential."""

from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory

__all__ = ["BiasClassification", "BiasCategory"]
```

```python
# polymarket_agent/bias_detection/models.py
"""Data models for bias detection."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BiasCategory(Enum):
    """Categories of demographic bias that may affect market pricing."""
    POLITICAL = "political"
    PROGRESSIVE_SOCIAL = "progressive_social"
    CRYPTO_OPTIMISM = "crypto_optimism"


@dataclass
class BiasClassification:
    """
    LLM's classification of a market's bias potential.

    Attributes:
        market_id: The market being classified
        dominated_by_bias: Whether demographic bias likely affects pricing
        categories: List of bias categories that apply
        bias_score: Strength of bias potential (0-100)
        mispricing_direction: Expected direction ("overpriced", "underpriced", "unclear")
        european: Whether the market relates to Europe
        spain: Whether the market specifically relates to Spain
        reasoning: Brief explanation of the classification
    """
    market_id: str
    dominated_by_bias: bool
    categories: list[BiasCategory]
    bias_score: int
    mispricing_direction: str  # "overpriced", "underpriced", "unclear"
    european: bool
    spain: bool
    reasoning: str
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bias_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/bias_detection/ tests/test_bias_detection.py
git commit -m "feat: add BiasClassification data model"
```

---

## Task 2: Create LLM bias classifier with prompts

**Files:**
- Create: `polymarket_agent/bias_detection/classifier.py`
- Modify: `tests/test_bias_detection.py`

**Step 1: Write the failing test**

Add to `tests/test_bias_detection.py`:

```python
from polymarket_agent.bias_detection.classifier import (
    build_system_prompt,
    build_user_prompt,
    parse_classification_response,
)
from polymarket_agent.data_fetching.models import Market, Outcome


def test_build_system_prompt():
    """Test system prompt contains key demographic info."""
    prompt = build_system_prompt()

    assert "73% male" in prompt
    assert "25-45" in prompt
    assert "right-leaning" in prompt
    assert "crypto" in prompt.lower()


def test_build_user_prompt():
    """Test user prompt includes market data."""
    market = Market(
        id="test-123",
        slug="test-market",
        question="Will Democrats win the Senate?",
        description="Resolution based on election results",
        outcomes=[
            Outcome(name="Yes", token_id="tok1", price=0.45),
            Outcome(name="No", token_id="tok2", price=0.55),
        ],
    )

    prompt = build_user_prompt(market)

    assert "Will Democrats win the Senate?" in prompt
    assert "Yes" in prompt
    assert "0.45" in prompt or "45" in prompt


def test_parse_classification_response_valid():
    """Test parsing a valid JSON response."""
    response = '''
    {
        "dominated_by_bias": true,
        "categories": ["political"],
        "bias_score": 80,
        "mispricing_direction": "underpriced",
        "european": false,
        "spain": false,
        "reasoning": "Democrat victory market - left-favorable outcome"
    }
    '''

    result = parse_classification_response(response, "test-123")

    assert result.dominated_by_bias is True
    assert BiasCategory.POLITICAL in result.categories
    assert result.bias_score == 80
    assert result.mispricing_direction == "underpriced"


def test_parse_classification_response_no_bias():
    """Test parsing response with no bias."""
    response = '''
    {
        "dominated_by_bias": false,
        "categories": [],
        "bias_score": 0,
        "mispricing_direction": "unclear",
        "european": false,
        "spain": false,
        "reasoning": "Sports market with no demographic bias"
    }
    '''

    result = parse_classification_response(response, "test-456")

    assert result.dominated_by_bias is False
    assert result.categories == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bias_detection.py::test_build_system_prompt -v`
Expected: FAIL with "cannot import name 'build_system_prompt'"

**Step 3: Write minimal implementation**

```python
# polymarket_agent/bias_detection/classifier.py
"""LLM-based bias classification for markets."""

import json
import logging
import re
from typing import Optional

from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory
from polymarket_agent.data_fetching.models import Market

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You classify prediction markets for potential bias-driven mispricing.

Polymarket demographics: ~73% male, ages 25-45, ~31% US-based, right-leaning politically, crypto/tech enthusiasts.

These demographics create predictable biases:
- Political: Right-leaning users overestimate right-favorable outcomes, underestimate left-favorable outcomes
- Progressive social: Topics on gender, race, LGBTQ+, immigration—progressive-favorable outcomes underestimated
- Crypto optimism: Positive crypto outcomes overestimated, negative outcomes underestimated

Classify markets and predict mispricing direction. Be concise."""


USER_PROMPT_TEMPLATE = """Market: {question}
Outcomes: {outcomes}
Current prices: {prices}

Respond in JSON:
{{
  "dominated_by_bias": true/false,
  "categories": ["political", "progressive_social", "crypto_optimism"],
  "bias_score": 0-100,
  "mispricing_direction": "overpriced" | "underpriced" | "unclear",
  "european": true/false,
  "spain": true/false,
  "reasoning": "One sentence"
}}

Only include categories that apply. If no bias applies, set dominated_by_bias to false."""


def build_system_prompt() -> str:
    """Build the system prompt for bias classification."""
    return SYSTEM_PROMPT


def build_user_prompt(market: Market) -> str:
    """Build the user prompt for a specific market."""
    outcome_names = [o.name for o in market.outcomes]
    prices = {o.name: f"{o.price:.2f}" for o in market.outcomes}

    return USER_PROMPT_TEMPLATE.format(
        question=market.question,
        outcomes=", ".join(outcome_names),
        prices=json.dumps(prices),
    )


def parse_classification_response(
    response: str,
    market_id: str,
) -> BiasClassification:
    """
    Parse LLM response into BiasClassification.

    Args:
        response: Raw LLM response text
        market_id: Market ID for the classification

    Returns:
        BiasClassification object
    """
    # Extract JSON from response (handle markdown code blocks)
    json_match = re.search(r'\{[\s\S]*\}', response)
    if not json_match:
        logger.warning(f"No JSON found in response for market {market_id}")
        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            mispricing_direction="unclear",
            european=False,
            spain=False,
            reasoning="Failed to parse LLM response",
        )

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for market {market_id}: {e}")
        return BiasClassification(
            market_id=market_id,
            dominated_by_bias=False,
            categories=[],
            bias_score=0,
            mispricing_direction="unclear",
            european=False,
            spain=False,
            reasoning="Failed to parse LLM response",
        )

    # Parse categories
    category_map = {
        "political": BiasCategory.POLITICAL,
        "progressive_social": BiasCategory.PROGRESSIVE_SOCIAL,
        "crypto_optimism": BiasCategory.CRYPTO_OPTIMISM,
    }
    categories = []
    for cat_str in data.get("categories", []):
        if cat_str in category_map:
            categories.append(category_map[cat_str])

    return BiasClassification(
        market_id=market_id,
        dominated_by_bias=data.get("dominated_by_bias", False),
        categories=categories,
        bias_score=int(data.get("bias_score", 0)),
        mispricing_direction=data.get("mispricing_direction", "unclear"),
        european=data.get("european", False),
        spain=data.get("spain", False),
        reasoning=data.get("reasoning", ""),
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bias_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/bias_detection/classifier.py tests/test_bias_detection.py
git commit -m "feat: add LLM bias classifier with prompts"
```

---

## Task 3: Add async classify_market function

**Files:**
- Modify: `polymarket_agent/bias_detection/classifier.py`
- Modify: `polymarket_agent/bias_detection/__init__.py`
- Modify: `tests/test_bias_detection.py`

**Step 1: Write the failing test**

Add to `tests/test_bias_detection.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch

from polymarket_agent.bias_detection.classifier import classify_market


@pytest.mark.asyncio
async def test_classify_market():
    """Test classify_market calls LLM and parses response."""
    market = Market(
        id="test-123",
        slug="test-market",
        question="Will Bitcoin reach $200K?",
        description="Resolution based on price",
        outcomes=[
            Outcome(name="Yes", token_id="tok1", price=0.35),
            Outcome(name="No", token_id="tok2", price=0.65),
        ],
    )

    mock_response = '''
    {
        "dominated_by_bias": true,
        "categories": ["crypto_optimism"],
        "bias_score": 85,
        "mispricing_direction": "overpriced",
        "european": false,
        "spain": false,
        "reasoning": "Crypto-positive outcome likely overpriced"
    }
    '''

    with patch('polymarket_agent.bias_detection.classifier.get_llm_client') as mock_get_client:
        mock_client = AsyncMock()
        mock_client.complete.return_value = AsyncMock(content=mock_response)
        mock_get_client.return_value = mock_client

        result = await classify_market(market, model="claude-haiku-4-5")

        assert result.dominated_by_bias is True
        assert BiasCategory.CRYPTO_OPTIMISM in result.categories
        assert result.bias_score == 85
        assert result.mispricing_direction == "overpriced"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bias_detection.py::test_classify_market -v`
Expected: FAIL with "cannot import name 'classify_market'"

**Step 3: Write minimal implementation**

Add to `polymarket_agent/bias_detection/classifier.py`:

```python
from polymarket_agent.llm_assessment.providers import get_llm_client


async def classify_market(
    market: Market,
    model: str = "claude-haiku-4-5",
) -> BiasClassification:
    """
    Classify a market for demographic bias potential using LLM.

    Args:
        market: Market to classify
        model: LLM model to use

    Returns:
        BiasClassification with category, score, and direction
    """
    client = get_llm_client(model)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(market)

    response = await client.complete(
        prompt=user_prompt,
        system_prompt=system_prompt,
        max_tokens=500,
        temperature=0.1,
    )

    return parse_classification_response(response.content, market.id)
```

Update `polymarket_agent/bias_detection/__init__.py`:

```python
"""Bias detection module for classifying markets by demographic bias potential."""

from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory
from polymarket_agent.bias_detection.classifier import classify_market

__all__ = ["BiasClassification", "BiasCategory", "classify_market"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bias_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/bias_detection/
git commit -m "feat: add async classify_market function"
```

---

## Task 4: Create ClassifiedMarket model for ranked output

**Files:**
- Modify: `polymarket_agent/bias_detection/models.py`
- Modify: `tests/test_bias_detection.py`

**Step 1: Write the failing test**

Add to `tests/test_bias_detection.py`:

```python
from polymarket_agent.bias_detection.models import ClassifiedMarket


def test_classified_market():
    """Test ClassifiedMarket combines market with classification."""
    market = Market(
        id="test-123",
        slug="test-market",
        question="Will Democrats win?",
        description="",
        outcomes=[
            Outcome(name="Yes", token_id="tok1", price=0.45),
            Outcome(name="No", token_id="tok2", price=0.55),
        ],
        volume=100000,
        liquidity=25000,
    )

    classification = BiasClassification(
        market_id="test-123",
        dominated_by_bias=True,
        categories=[BiasCategory.POLITICAL],
        bias_score=75,
        mispricing_direction="underpriced",
        european=False,
        spain=False,
        reasoning="Left-favorable outcome",
    )

    classified = ClassifiedMarket(
        market=market,
        classification=classification,
    )

    assert classified.market.question == "Will Democrats win?"
    assert classified.classification.bias_score == 75
    assert classified.volume == 100000
    assert classified.liquidity == 25000
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bias_detection.py::test_classified_market -v`
Expected: FAIL with "cannot import name 'ClassifiedMarket'"

**Step 3: Write minimal implementation**

Add to `polymarket_agent/bias_detection/models.py`:

```python
from polymarket_agent.data_fetching.models import Market


@dataclass
class ClassifiedMarket:
    """
    A market with its bias classification.

    Attributes:
        market: The base market data
        classification: The LLM's bias classification
    """
    market: Market
    classification: BiasClassification

    @property
    def volume(self) -> float:
        """Market volume for convenience."""
        return self.market.volume

    @property
    def liquidity(self) -> float:
        """Market liquidity for convenience."""
        return self.market.liquidity
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bias_detection.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/bias_detection/models.py tests/test_bias_detection.py
git commit -m "feat: add ClassifiedMarket model"
```

---

## Task 5: Create simplified ScannerConfig

**Files:**
- Create: `polymarket_agent/scanner_config.py`
- Create: `tests/test_scanner_config.py`

**Step 1: Write the failing test**

```python
# tests/test_scanner_config.py
"""Tests for scanner configuration."""

import pytest
from polymarket_agent.scanner_config import ScannerConfig


def test_scanner_config_defaults():
    """Test ScannerConfig has sensible defaults."""
    config = ScannerConfig()

    assert config.min_volume == 1000
    assert config.min_liquidity == 500
    assert config.llm_model == "claude-haiku-4-5"
    assert config.output_dir == "output"
    assert config.max_markets == 500


def test_scanner_config_custom_values():
    """Test ScannerConfig with custom values."""
    config = ScannerConfig(
        min_volume=50000,
        min_liquidity=10000,
        llm_model="claude-sonnet-4-5",
        output_dir="reports",
    )

    assert config.min_volume == 50000
    assert config.min_liquidity == 10000
    assert config.llm_model == "claude-sonnet-4-5"
    assert config.output_dir == "reports"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner_config.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# polymarket_agent/scanner_config.py
"""Configuration for the bias scanner."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScannerConfig:
    """
    Configuration for the Polymarket bias scanner.

    Attributes:
        min_volume: Minimum trading volume in USD
        min_liquidity: Minimum liquidity in USD
        max_days_to_expiry: Maximum days until resolution
        llm_model: LLM model for classification
        max_markets: Maximum markets to fetch
        output_dir: Directory for output reports
        verbose: Enable verbose logging
    """
    min_volume: float = 1000
    min_liquidity: float = 500
    max_days_to_expiry: int = 90
    llm_model: str = "claude-haiku-4-5"
    max_markets: int = 500
    output_dir: str = "output"
    verbose: bool = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/scanner_config.py tests/test_scanner_config.py
git commit -m "feat: add simplified ScannerConfig"
```

---

## Task 6: Create bias scanner orchestrator

**Files:**
- Create: `polymarket_agent/scanner.py`
- Create: `tests/test_scanner.py`

**Step 1: Write the failing test**

```python
# tests/test_scanner.py
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


def test_bias_scanner_init():
    """Test BiasScanner initialization."""
    config = ScannerConfig(min_volume=50000)
    scanner = BiasScanner(config)

    assert scanner.config.min_volume == 50000


@pytest.mark.asyncio
async def test_bias_scanner_classify_markets(sample_markets):
    """Test classifying markets."""
    config = ScannerConfig()
    scanner = BiasScanner(config)

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
            dominated_by_bias=True,
            categories=[BiasCategory.CRYPTO_OPTIMISM],
            bias_score=85,
            mispricing_direction="overpriced",
            european=False,
            spain=False,
            reasoning="Crypto-positive",
        ),
    ]

    with patch('polymarket_agent.scanner.classify_market') as mock_classify:
        mock_classify.side_effect = [
            AsyncMock(return_value=c)() for c in mock_classifications
        ]

        classified = await scanner.classify_markets(sample_markets)

        assert len(classified) == 2
        assert classified[0].classification.bias_score == 75
        assert classified[1].classification.bias_score == 85
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scanner.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# polymarket_agent/scanner.py
"""Bias scanner for Polymarket."""

import asyncio
import logging
from typing import Optional

from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.data_fetching.models import Market
from polymarket_agent.data_fetching.gamma_api import GammaAPIClient
from polymarket_agent.filtering.filters import MarketFilter, FilterConfig
from polymarket_agent.bias_detection.models import BiasClassification, BiasCategory, ClassifiedMarket
from polymarket_agent.bias_detection.classifier import classify_market

logger = logging.getLogger(__name__)


class BiasScanner:
    """
    Scans Polymarket for markets with demographic bias potential.

    Pipeline:
        1. Fetch markets from Polymarket API
        2. Filter by volume/liquidity
        3. Classify each market with LLM
        4. Group by category and rank by bias score
    """

    def __init__(self, config: Optional[ScannerConfig] = None):
        """Initialize the scanner."""
        self.config = config or ScannerConfig()

    async def fetch_markets(self) -> list[Market]:
        """Fetch markets from Polymarket API."""
        client = GammaAPIClient()
        markets = await client.get_markets(limit=self.config.max_markets)
        logger.info(f"Fetched {len(markets)} markets")
        return markets

    def filter_markets(self, markets: list[Market]) -> list[Market]:
        """Apply volume/liquidity filters."""
        filter_config = FilterConfig(
            min_volume=self.config.min_volume,
            min_liquidity=self.config.min_liquidity,
            max_days_to_expiry=self.config.max_days_to_expiry,
        )
        market_filter = MarketFilter(filter_config)
        result = market_filter.apply(markets)
        logger.info(f"Filtered {result.total_before} -> {result.total_after} markets")
        return result.markets

    async def classify_markets(
        self,
        markets: list[Market],
    ) -> list[ClassifiedMarket]:
        """Classify markets with LLM for bias potential."""
        classified = []

        for market in markets:
            try:
                classification = await classify_market(
                    market,
                    model=self.config.llm_model,
                )
                if classification.dominated_by_bias:
                    classified.append(ClassifiedMarket(
                        market=market,
                        classification=classification,
                    ))
            except Exception as e:
                logger.error(f"Failed to classify market {market.id}: {e}")

        logger.info(f"Classified {len(classified)} markets with bias potential")
        return classified

    def group_by_category(
        self,
        classified: list[ClassifiedMarket],
    ) -> dict[BiasCategory, list[ClassifiedMarket]]:
        """Group classified markets by bias category."""
        groups: dict[BiasCategory, list[ClassifiedMarket]] = {
            BiasCategory.POLITICAL: [],
            BiasCategory.PROGRESSIVE_SOCIAL: [],
            BiasCategory.CRYPTO_OPTIMISM: [],
        }

        for cm in classified:
            for category in cm.classification.categories:
                groups[category].append(cm)

        # Sort each group by bias_score descending
        for category in groups:
            groups[category].sort(
                key=lambda x: x.classification.bias_score,
                reverse=True,
            )

        return groups

    async def run(self) -> dict[BiasCategory, list[ClassifiedMarket]]:
        """Run the full scan pipeline."""
        # Fetch
        markets = await self.fetch_markets()

        # Filter
        filtered = self.filter_markets(markets)

        # Classify
        classified = await self.classify_markets(filtered)

        # Group and rank
        grouped = self.group_by_category(classified)

        return grouped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_scanner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/scanner.py tests/test_scanner.py
git commit -m "feat: add BiasScanner orchestrator"
```

---

## Task 7: Create markdown report generator

**Files:**
- Create: `polymarket_agent/bias_reporting.py`
- Create: `tests/test_bias_reporting.py`

**Step 1: Write the failing test**

```python
# tests/test_bias_reporting.py
"""Tests for bias report generation."""

import pytest
from pathlib import Path
from polymarket_agent.bias_reporting import generate_bias_report
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
        mispricing_direction="underpriced",
        european=False,
        spain=False,
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
        mispricing_direction="underpriced",
        european=True,
        spain=True,
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


def test_generate_bias_report(sample_grouped_markets, tmp_path):
    """Test generating markdown report."""
    output_path = tmp_path / "report.md"

    result = generate_bias_report(
        grouped_markets=sample_grouped_markets,
        output_path=output_path,
        min_volume=1000,
        min_liquidity=500,
    )

    assert result.exists()
    content = result.read_text()

    # Check header
    assert "# Polymarket Bias Scanner Report" in content

    # Check political section
    assert "## Political Bias" in content
    assert "Will Democrats win the Senate?" in content
    assert "underpriced" in content
    assert "75" in content  # bias score

    # Check Spain flag
    assert "ES" in content or "🇪🇸" in content

    # Check footer
    assert "min_volume" in content.lower() or "Filters applied" in content
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_bias_reporting.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# polymarket_agent/bias_reporting.py
"""Markdown report generator for bias scanner."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from polymarket_agent.bias_detection.models import BiasCategory, ClassifiedMarket


def format_currency(value: float) -> str:
    """Format a number as currency."""
    if value >= 1_000_000:
        return f"${value/1_000_000:.1f}M"
    elif value >= 1_000:
        return f"${value/1_000:.0f}K"
    else:
        return f"${value:.0f}"


def generate_bias_report(
    grouped_markets: dict[BiasCategory, list[ClassifiedMarket]],
    output_path: str | Path,
    min_volume: float = 1000,
    min_liquidity: float = 500,
) -> Path:
    """
    Generate a markdown report of bias-classified markets.

    Args:
        grouped_markets: Markets grouped by bias category
        output_path: Path to write the report
        min_volume: Minimum volume filter used
        min_liquidity: Minimum liquidity filter used

    Returns:
        Path to the generated report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    # Header
    lines.append("# Polymarket Bias Scanner Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
    lines.append("")

    # Category sections
    category_titles = {
        BiasCategory.POLITICAL: "Political Bias",
        BiasCategory.PROGRESSIVE_SOCIAL: "Progressive Social",
        BiasCategory.CRYPTO_OPTIMISM: "Crypto Optimism",
    }

    total_markets = 0

    for category, title in category_titles.items():
        markets = grouped_markets.get(category, [])
        if not markets:
            continue

        total_markets += len(markets)

        lines.append(f"## {title} ({len(markets)} markets)")
        lines.append("")
        lines.append("| Rank | Market | Direction | Score | Volume | Liquidity | EU |")
        lines.append("|------|--------|-----------|-------|--------|-----------|-----|")

        for rank, cm in enumerate(markets, 1):
            market = cm.market
            classification = cm.classification

            # EU flag
            eu_flag = ""
            if classification.spain:
                eu_flag = "🇪🇸"
            elif classification.european:
                eu_flag = "🇪🇺"

            # Truncate long questions
            question = market.question
            if len(question) > 50:
                question = question[:47] + "..."

            lines.append(
                f"| {rank} | {question} | {classification.mispricing_direction} | "
                f"{classification.bias_score} | {format_currency(market.volume)} | "
                f"{format_currency(market.liquidity)} | {eu_flag} |"
            )

        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"Filters applied: min_volume={format_currency(min_volume)}, min_liquidity={format_currency(min_liquidity)}")
    lines.append(f"Markets classified with bias: {total_markets}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_bias_reporting.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/bias_reporting.py tests/test_bias_reporting.py
git commit -m "feat: add markdown report generator for bias scanner"
```

---

## Task 8: Create simplified CLI main entry point

**Files:**
- Create: `polymarket_agent/scan.py`

**Step 1: Write the implementation**

```python
# polymarket_agent/scan.py
"""
Polymarket Bias Scanner CLI

Scans Polymarket for markets where demographic biases may create mispricing.

Usage:
    python -m polymarket_agent.scan [OPTIONS]

Examples:
    python -m polymarket_agent.scan
    python -m polymarket_agent.scan --min-volume 50000 --min-liquidity 10000
    python -m polymarket_agent.scan --model claude-sonnet-4-5 --output reports/scan.md
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.scanner import BiasScanner
from polymarket_agent.bias_reporting import generate_bias_report
from polymarket_agent.config import LLM_MODELS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Polymarket Bias Scanner - Find markets with demographic bias potential",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--min-volume",
        type=float,
        default=1000,
        help="Minimum trading volume in USD (default: 1000)",
    )
    parser.add_argument(
        "--min-liquidity",
        type=float,
        default=500,
        help="Minimum liquidity in USD (default: 500)",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=90,
        help="Maximum days to resolution (default: 90)",
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(LLM_MODELS.keys()),
        default="claude-haiku-4-5",
        help="LLM model for classification (default: claude-haiku-4-5)",
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=500,
        help="Maximum markets to fetch (default: 500)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: output/bias_scan_<timestamp>.md)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main_async():
    """Async main entry point."""
    args = parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Build config
    config = ScannerConfig(
        min_volume=args.min_volume,
        min_liquidity=args.min_liquidity,
        max_days_to_expiry=args.max_days,
        llm_model=args.model,
        max_markets=args.max_markets,
        verbose=args.verbose,
    )

    # Print header
    print("\n" + "=" * 60)
    print("POLYMARKET BIAS SCANNER")
    print("=" * 60)
    print(f"Model: {config.llm_model}")
    print(f"Min Volume: ${config.min_volume:,.0f}")
    print(f"Min Liquidity: ${config.min_liquidity:,.0f}")
    print("=" * 60 + "\n")

    # Run scanner
    scanner = BiasScanner(config)
    grouped = await scanner.run()

    # Count results
    total = sum(len(markets) for markets in grouped.values())

    if total == 0:
        print("No markets with bias potential found.")
        return

    # Generate report
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = Path("output") / f"bias_scan_{timestamp}.md"

    report_path = generate_bias_report(
        grouped_markets=grouped,
        output_path=output_path,
        min_volume=config.min_volume,
        min_liquidity=config.min_liquidity,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)
    for category, markets in grouped.items():
        if markets:
            print(f"  {category.value}: {len(markets)} markets")
    print(f"\nTotal: {total} markets with bias potential")
    print(f"Report: {report_path}")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 2: Run manual test**

Run: `python -m polymarket_agent.scan --help`
Expected: Help message displayed

**Step 3: Commit**

```bash
git add polymarket_agent/scan.py
git commit -m "feat: add bias scanner CLI entry point"
```

---

## Task 9: Remove obsolete modules

**Files:**
- Delete: `polymarket_agent/llm_assessment/consensus.py`
- Delete: `polymarket_agent/llm_assessment/prompts.py`
- Delete: `polymarket_agent/llm_assessment/assessor.py`
- Delete: `polymarket_agent/enrichment/`
- Delete: `polymarket_agent/scoring/`
- Delete: `polymarket_agent/analysis/spread_analysis.py`
- Delete: `polymarket_agent/analysis/reasoning_classifier.py`
- Delete: `polymarket_agent/analysis/demographic_bias.py`
- Delete: `polymarket_agent/storage/`
- Delete: `polymarket_agent/agent.py`
- Delete: `scripts/backtest.py`
- Delete: Obsolete tests

**Step 1: Delete obsolete modules**

```bash
rm polymarket_agent/llm_assessment/consensus.py
rm polymarket_agent/llm_assessment/prompts.py
rm polymarket_agent/llm_assessment/assessor.py
rm -rf polymarket_agent/enrichment/
rm -rf polymarket_agent/scoring/
rm polymarket_agent/analysis/spread_analysis.py
rm polymarket_agent/analysis/reasoning_classifier.py
rm polymarket_agent/analysis/demographic_bias.py
rm -rf polymarket_agent/storage/
rm polymarket_agent/agent.py
rm -f scripts/backtest.py
```

**Step 2: Delete obsolete tests**

```bash
rm tests/test_scoring.py
rm tests/test_llm_assessment.py
rm tests/test_spread_analysis.py
rm tests/test_database.py
rm tests/test_bias_llm.py
rm tests/test_consensus.py
```

**Step 3: Update __init__ files**

Update `polymarket_agent/llm_assessment/__init__.py`:
```python
"""LLM provider clients."""

from polymarket_agent.llm_assessment.providers import get_llm_client, LLMClient

__all__ = ["get_llm_client", "LLMClient"]
```

Update `polymarket_agent/analysis/__init__.py`:
```python
"""Analysis utilities (minimal after refactor)."""
```

**Step 4: Verify remaining tests pass**

Run: `pytest tests/ -v --ignore=tests/test_scoring.py --ignore=tests/test_llm_assessment.py --ignore=tests/test_spread_analysis.py --ignore=tests/test_database.py --ignore=tests/test_bias_llm.py --ignore=tests/test_consensus.py`
Expected: PASS (remaining tests should pass)

**Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove obsolete LLM assessment, scoring, enrichment modules"
```

---

## Task 10: Update filtering to remove bias-related filters

**Files:**
- Modify: `polymarket_agent/filtering/filters.py`
- Modify: `tests/test_filtering.py`

**Step 1: Remove bias/reasoning filter code from filters.py**

Remove these functions from `filters.py`:
- `filter_by_reasoning`
- `filter_by_demographic_bias`

Remove from `MarketFilter._build_filters_from_config`:
- Reasoning filter block
- Demographic bias filter block

**Step 2: Update FilterConfig references in filters.py**

Remove usage of:
- `config.reasoning_heavy_only`
- `config.min_reasoning_score`
- `config.llm_edge_levels`
- `config.bias_filter_enabled`
- `config.min_blind_spot_score`
- `config.mispricing_levels`

**Step 3: Update test_filtering.py**

Remove tests that reference:
- `filter_by_reasoning`
- `filter_by_demographic_bias`
- `reasoning_heavy_only`
- `bias_filter_enabled`

**Step 4: Run tests**

Run: `pytest tests/test_filtering.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add polymarket_agent/filtering/filters.py tests/test_filtering.py
git commit -m "refactor: remove reasoning and demographic bias filters"
```

---

## Task 11: Clean up config.py

**Files:**
- Modify: `polymarket_agent/config.py`

**Step 1: Remove obsolete config fields**

From `FilterConfig`, remove:
- `reasoning_heavy_only`
- `min_reasoning_score`
- `llm_edge_levels`
- `bias_filter_enabled`
- `min_blind_spot_score`
- `mispricing_levels`

From `AgentConfig`, remove:
- `enrichment_limit`
- `llm_analysis_limit`
- `enable_database`
- `db_path`
- `enable_spread_analysis`
- `llm_bias_analysis`
- `llm_bias_model`
- `consensus_models`
- `risk_tolerance` references
- `RISK_THRESHOLDS`
- `SCORING_WEIGHTS`

**Step 2: Simplify AgentConfig**

Keep only what's needed for basic filtering and LLM model selection.

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 4: Commit**

```bash
git add polymarket_agent/config.py
git commit -m "refactor: simplify config by removing obsolete fields"
```

---

## Task 12: Update package __init__.py

**Files:**
- Modify: `polymarket_agent/__init__.py`

**Step 1: Update exports**

```python
# polymarket_agent/__init__.py
"""Polymarket Bias Scanner - Detect markets with demographic bias potential."""

from polymarket_agent.scanner import BiasScanner
from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.bias_detection import BiasClassification, BiasCategory

__all__ = [
    "BiasScanner",
    "ScannerConfig",
    "BiasClassification",
    "BiasCategory",
]

__version__ = "2.0.0"
```

**Step 2: Commit**

```bash
git add polymarket_agent/__init__.py
git commit -m "refactor: update package exports for bias scanner"
```

---

## Task 13: Final integration test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration tests for bias scanner."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from polymarket_agent.scanner import BiasScanner
from polymarket_agent.scanner_config import ScannerConfig
from polymarket_agent.bias_detection.models import BiasCategory


@pytest.mark.asyncio
async def test_full_scan_pipeline():
    """Test the full scan pipeline with mocked API and LLM."""
    config = ScannerConfig(
        min_volume=1000,
        min_liquidity=500,
        max_markets=10,
    )
    scanner = BiasScanner(config)

    # Mock the fetch
    mock_markets = [
        MagicMock(
            id="m1",
            slug="test-1",
            question="Will Democrats win?",
            description="",
            outcomes=[MagicMock(name="Yes", token_id="t1", price=0.45)],
            volume=100000,
            liquidity=25000,
            days_to_expiry=30,
        ),
    ]

    with patch.object(scanner, 'fetch_markets', return_value=mock_markets):
        with patch.object(scanner, 'filter_markets', return_value=mock_markets):
            with patch('polymarket_agent.scanner.classify_market') as mock_classify:
                mock_classify.return_value = MagicMock(
                    market_id="m1",
                    dominated_by_bias=True,
                    categories=[BiasCategory.POLITICAL],
                    bias_score=75,
                    mispricing_direction="underpriced",
                    european=False,
                    spain=False,
                    reasoning="Test",
                )

                result = await scanner.run()

                assert BiasCategory.POLITICAL in result
                assert len(result[BiasCategory.POLITICAL]) == 1
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for bias scanner pipeline"
```

---

## Task 14: Update README and documentation

**Files:**
- Modify: `README.md` (if exists)

**Step 1: Update README with new usage**

Add section explaining the new bias scanner approach:

```markdown
## Usage

### Bias Scanner

Scan Polymarket for markets where demographic biases may create mispricing opportunities:

```bash
python -m polymarket_agent.scan

# With filters
python -m polymarket_agent.scan --min-volume 50000 --min-liquidity 10000

# Output to specific file
python -m polymarket_agent.scan --output reports/scan.md
```

### Bias Categories

The scanner identifies markets in three bias categories:

1. **Political** - Markets where right-wing userbase may misprice
   - Left-favorable outcomes → likely underpriced
   - Right-favorable outcomes → likely overpriced

2. **Progressive Social** - Gender, race, LGBTQ+, immigration topics
   - Progressive-favorable outcomes → likely underpriced

3. **Crypto Optimism** - Crypto-related markets
   - Crypto-positive outcomes → likely overpriced

European markets (especially Spain) are flagged for informational edge opportunities.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for bias scanner"
```

---

## Summary

This plan refactors the Polymarket agent from a full misprice-assessment pipeline to a focused bias-detection scanner:

1. **Tasks 1-4**: Create bias detection models and LLM classifier
2. **Tasks 5-6**: Create scanner config and orchestrator
3. **Tasks 7-8**: Create report generator and CLI
4. **Tasks 9-11**: Remove obsolete modules and clean up
5. **Tasks 12-14**: Update package exports and documentation

Total: 14 tasks with TDD approach and frequent commits.
