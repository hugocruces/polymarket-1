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
- Whether the likely mispricing is "overpriced" or "underpriced"
- A bias strength score (0-100)
- Whether it's a European market (cross-cutting tag, useful for non-US analysis)

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
python -m polymarket_agent.scan --model claude-sonnet-4-5
```

Output is saved to `output/bias_scan_<timestamp>.md`.

## CLI Options

```
--min-volume      Minimum trading volume in USD (default: 1000)
--min-liquidity   Minimum liquidity in USD (default: 500)
--max-days        Maximum days to resolution (default: 90)
--model, -m       LLM model for classification (default: claude-haiku-4-5)
--max-markets     Maximum markets to fetch (default: 500)
--output, -o      Output file path
--verbose, -v     Enable verbose logging
```

### Available Models

| Provider  | Model                    | Notes                        |
| --------- | ------------------------ | ---------------------------- |
| Anthropic | `claude-haiku-4-5`       | Fast, cheap (default)        |
| Anthropic | `claude-sonnet-4-5`      | Better quality               |
| Anthropic | `claude-opus-4-5`        | Highest quality              |
| OpenAI    | `gpt-5.2`                | Flagship                     |
| OpenAI    | `gpt-5-mini`             | Fast                         |
| Google    | `gemini-3-pro-preview`   | Strong alternative           |
| Google    | `gemini-3-flash-preview` | Cheapest                     |

## Output Format

The scanner produces a markdown report with:

- **Header** — Scan timestamp and filter settings
- **Sections by bias category** — Each non-empty category gets a section
- **Market tables** — Question, current prices, bias score, mispricing direction, volume/liquidity
- **European flags** — Markets tagged as European

Example output:

```markdown
## Political Bias (3 markets)

| Market | Yes Price | Bias Score | Direction | Volume | Liquidity |
|--------|-----------|------------|-----------|--------|-----------|
| Will Democrats win the Senate? | 45% | 75 | underpriced | $100K | $25K |
```

## Project Structure

```
polymarket_agent/
  scan.py                 CLI entry point
  scanner.py              BiasScanner orchestrator
  scanner_config.py       ScannerConfig dataclass
  config.py               LLM models and API endpoints
  bias_detection/
    models.py             BiasCategory, BiasClassification, ClassifiedMarket
    classifier.py         LLM classification prompts and parsing
  bias_reporting.py       Markdown report generation
  data_fetching/
    gamma_api.py          Polymarket Gamma API client
    clob_api.py           Polymarket CLOB API client
    models.py             Market, Outcome, Event models
  filtering/
    filters.py            Market filtering utilities
  llm_assessment/
    providers.py          Anthropic, OpenAI, Google LLM clients

tests/                    89 unit tests
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
pytest tests/           # Run all 89 tests
pytest tests/ -v        # Verbose output
pytest tests/ -k bias   # Run only bias-related tests
```
