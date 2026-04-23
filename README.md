# Polymarket Bias Scanner

A tool that scans [Polymarket](https://polymarket.com/) for markets where demographic biases may create mispricing opportunities.

## What It Does

Polymarket's user base has distinct demographics: ~73% male, ages 25-45, ~31% US-based, right-leaning politically, and crypto/tech enthusiastic. These demographics create predictable blind spots that can lead to systematic mispricings.

The scanner identifies markets affected by three bias categories:

1. **Political Bias** — Markets on left/right political topics where the right-leaning userbase may underprice left-favorable outcomes (or overprice right-favorable ones).

2. **Progressive Social Bias** — Markets on social issues (climate, healthcare, social justice) where the predominantly male, younger demographic may underprice progressive outcomes.

3. **Crypto Optimism Bias** — Markets on crypto/tech topics where the crypto-enthusiastic userbase may overprice positive crypto outcomes.

For each market, an LLM classifies:
- Which bias categories apply
- A bias strength score (0-100)
- Whether it's a European market (cross-cutting tag, useful for non-US analysis)

The scanner intentionally does not call the direction of any mispricing —
that judgment is left to the human reviewing the report. The LLM only flags
whether bias could be present.

The output is a markdown report organized by bias category, with markets ranked by bias strength.

## Quick Start

### Prerequisites

- Python 3.10+
- An Anthropic, OpenAI, or Google API key

### Setup

```bash
cd polymarket-1

# Create virtual environment and install dependencies
uv sync          # or: pip install -r requirements.txt

# Add your API key
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY (or OPENAI_API_KEY / GOOGLE_API_KEY)
```

### First Run

```bash
# Basic scan with defaults
python -m polymarket_agent.scan

# Scan high-volume markets only
python -m polymarket_agent.scan --min-volume 50000 --min-liquidity 10000

# Use a specific model
python -m polymarket_agent.scan --model claude-sonnet-4-6
```

Output is saved to `output/bias_scan_<timestamp>.md`.

## CLI Options

```
--min-volume      Minimum trading volume in USD (default: 5000)
--min-liquidity   Minimum liquidity in USD (default: 2000)
--max-days        Maximum days to resolution (default: 90)
--model, -m       LLM model for classification (default: claude-sonnet-4-6)
--max-markets     Maximum markets to fetch (default: 500)
--max-reported    Cap on LLM-classified markets in the report (default: 20)
--output, -o      Output file path
--verbose, -v     Enable verbose logging
```

### Supported Providers

The scanner works with any of Anthropic, OpenAI, or Google — pick the
provider you have an API key for. The exact model aliases accepted by
`--model` are defined in `polymarket_agent/config.py:LLM_MODELS`; run
`python -m polymarket_agent.scan --help` to see the current list.

## Output Format

The scanner produces a markdown report with:

- **Header** — Scan timestamp and filter settings
- **Sections by bias category** — Each non-empty category gets a section
- **Market tables** — Question, bias score, volume/liquidity, link
- **European flags** — Markets tagged as European
- **Classification Failures** — Markets whose LLM classification errored,
  surfaced so silent drops don't masquerade as "no bias"

Example output:

```markdown
## Political Bias (1 markets)

| Rank | Market | URL | Score | Volume | Liquidity | EU |
|------|--------|-----|-------|--------|-----------|-----|
| 1 | Will Democrats win the Senate? | [🔗](https://polymarket.com/market/democrat-win) | 75 | $100K | $25K |  |
```

## Project Structure

```
polymarket_agent/
  scan.py                 CLI entry point
  scanner.py              BiasScanner orchestrator, ScanResult
  scanner_config.py       ScannerConfig dataclass
  config.py               LLM models and API endpoints
  bias_detection/
    models.py             BiasCategory, BiasClassification, ClassificationFailure
    classifier.py         LLM prompts, Pydantic validation, parsing
  bias_reporting.py       Markdown report generation
  data_fetching/
    gamma_api.py          Polymarket Gamma API client
    models.py             Market, Outcome, Event models
  filtering/
    filters.py            Market filtering utilities
  llm_assessment/
    providers.py          LLM clients with retry/backoff (Anthropic, OpenAI, Google)

tests/                    unit tests
```

## Limitations

- **LLM classification is imperfect.** The model may misclassify markets or miss relevant biases.
- **Bias doesn't guarantee mispricing.** A market being in a biased category doesn't mean it's actually mispriced.
- **Market efficiency.** Polymarket prices already reflect collective wisdom. Detected biases may be priced in.
- **Information lag.** Markets can move on breaking news before the scanner runs.

## Disclaimer

This tool is for research and educational purposes. It identifies potential biases, not guaranteed trading opportunities. Prediction markets carry risk of loss. Always do your own research.

## Testing

```bash
pytest tests/           # Run all tests
pytest tests/ -v        # Verbose output
pytest tests/ -k bias   # Run only bias-related tests
```
