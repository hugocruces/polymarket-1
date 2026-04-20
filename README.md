# Polymarket Bias Scanner

A tool that scans [Polymarket](https://polymarket.com/) for markets where demographic biases may create mispricing opportunities.

## What It Does

Polymarket's user base has distinct demographics: ~73% male, ages 25-45, ~31% US-based, right-leaning politically, and crypto/tech enthusiastic. These demographics create predictable blind spots that can lead to systematic mispricings.

The scanner identifies markets where the crowd would bet **systematically differently from a demographically neutral group** — regardless of the actual underlying probability. It covers three bias categories:

1. **Political Bias** — The right-leaning userbase systematically favors conservative/Republican outcomes. Target markets: US elections, legislation, regulatory appointments, policy decisions with a clear conservative vs. progressive dimension.

2. **Progressive Social Bias** — The young male tech demographic structurally underweights progressive social outcomes. Target markets: gender policy, abortion rights, climate action, DEI initiatives, social safety net, labor rights.

3. **Crypto Optimism** — Crypto enthusiasts are systematically over-optimistic. Target markets: price targets, ETF approvals, adoption milestones, regulatory outcomes for specific coins or platforms.

For each market the LLM returns:
- Which bias categories apply
- A bias strength score (0–100), calibrated: 0–20 no bias / 20–40 mild / 40–70 moderate / 70–100 strong
- A reasoning explanation

Markets scoring ≥ 50 are flagged. EU and Spain markets are detected automatically via keyword matching — no LLM involvement. You decide whether the flagged mispricing is actionable.

The output is a markdown report organised by bias category, ranked by score.

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
python -m polymarket_agent.scan --model claude-opus-4-7
```

Output is saved to `output/bias_scan_<timestamp>.md`.

## CLI Options

```
--min-volume      Minimum trading volume in USD (default: 1000)
--min-liquidity   Minimum liquidity in USD (default: 500)
--max-days        Maximum days to resolution (default: 90)
--model, -m       LLM model for classification (default: claude-sonnet-4-6)
--max-markets     Maximum markets to fetch (default: 500)
--output, -o      Output file path
--verbose, -v     Enable verbose logging
```

### Available Models

| Provider  | Model                      | Notes                     |
| --------- | -------------------------- | ------------------------- |
| Anthropic | `claude-sonnet-4-6`        | Default — best balance    |
| Anthropic | `claude-haiku-4-5`         | Faster, cheaper           |
| Anthropic | `claude-opus-4-7`          | Highest quality           |
| OpenAI    | `gpt-5.4`                  | Flagship                  |
| OpenAI    | `gpt-5.4-mini`             | Fast                      |
| Google    | `gemini-3.1-pro-preview`   | Strong alternative        |
| Google    | `gemini-3-flash-preview`   | Fast                      |

## Output Format

The scanner produces a markdown report with:

- **Header** — Scan timestamp
- **Sections by bias category** — One section per non-empty category
- **Market tables** — Question, link, bias score, volume, liquidity, EU/Spain flag
- **Footer** — Filter settings and total count

Example:

```markdown
## Political Bias (3 markets)

| Rank | Market | URL | Score | Volume | Liquidity | EU |
|------|--------|-----|-------|--------|-----------|-----|
| 1 | Will Democrats win the Senate? | 🔗 | 75 | $100K | $25K |  |
| 2 | Will Sanchez survive confidence vote? | 🔗 | 72 | $50K | $10K | 🇪🇸 |
```

## Project Structure

```
polymarket_agent/
  scan.py                 CLI entry point
  scanner.py              BiasScanner orchestrator (parallel LLM calls)
  scanner_config.py       ScannerConfig dataclass
  config.py               LLM models and API endpoints
  bias_detection/
    models.py             BiasCategory, BiasClassification, ClassifiedMarket
    classifier.py         LLM prompts, parsing, bias score threshold
  bias_reporting.py       Markdown report + EU/Spain keyword detection
  data_fetching/
    gamma_api.py          Polymarket Gamma API client
    models.py             Market, Outcome, Event models
  filtering/
    filters.py            Market filtering + geographic scoring utilities
  llm_assessment/
    providers.py          Anthropic, OpenAI, Google LLM clients

tests/                    Unit tests
```

## How Classification Works

1. **Fetch** up to 500 markets from Polymarket, sorted by volume
2. **Filter** by min volume, liquidity, and days to expiry
3. **Classify** each market concurrently (up to 10 parallel LLM calls) — the LLM returns `categories`, `bias_score`, and `reasoning`
4. **Flag** markets with `bias_score >= 50` as bias-dominated
5. **Detect** EU/Spain relevance from market text using weighted keyword matching
6. **Report** markets grouped by category, ranked by score

## Limitations

- **LLM classification is imperfect.** The model may misclassify markets or miss relevant biases.
- **Bias doesn't guarantee mispricing.** A flagged market isn't necessarily mispriced — collective wisdom may have already corrected for demographic skew.
- **You decide on the edge.** The scanner surfaces candidates; it does not estimate implied edges or recommend trades.
- **Information lag.** Markets can move on breaking news before the scanner runs.

## Disclaimer

This tool is for research and educational purposes. It identifies potential biases, not guaranteed trading opportunities. Prediction markets carry risk of loss. Always do your own research.

## Testing

```bash
pytest tests/
pytest tests/ -v
pytest tests/ -k bias
```
