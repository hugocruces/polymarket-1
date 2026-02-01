# Polymarket AI Agent

An automated tool that scans [Polymarket](https://polymarket.com/) prediction markets, uses LLMs to estimate fair probabilities, and identifies markets where the crowd price may be wrong.

## What It Does

Prediction markets aggregate the opinions of traders into prices that represent probabilities. But traders have blind spots: Polymarket's user base skews young, male, US-centric, and crypto-enthusiastic, which can cause systematic mispricings on topics outside their expertise.

This agent exploits that by running a six-step pipeline:

1. **Fetch** — Pulls active markets from Polymarket's public APIs (Gamma + CLOB).
2. **Filter** — Narrows tens of thousands of markets down to candidates worth analyzing, using configurable rules for topic, volume, liquidity, geography, and more.
3. **Enrich** — Searches the web for recent news, polling data, and expert analysis relevant to each candidate market.
4. **Assess** — Sends the market question plus the gathered evidence to one or more LLMs, asking each to estimate a fair probability range independent of the market price.
5. **Score** — Compares the LLM estimates to the market price. Markets where the gap is large, the LLM is confident, and the evidence is strong score highest.
6. **Report** — Outputs a ranked list with full reasoning, warnings, and score breakdowns.

The result is a report that says, in effect: "Here are the markets where the crowd is most likely wrong, here's why, and here's how confident we are."

## Quick Start

### Prerequisites

- Python 3.10+
- An API key for at least one LLM provider (Anthropic, OpenAI, or Google)

### Setup

```bash
cd polymarket-1

# Create virtual environment and install dependencies
uv sync          # or: pip install -r requirements.txt

# Add your API keys
cp .env.example .env
# Edit .env and fill in your key(s)
```

### First Run

```bash
# Quick scan — fetches and filters markets without calling any LLM
python -m polymarket_agent.main --config config.yaml --dry-run

# Full analysis — filters, enriches, and assesses with LLM
python -m polymarket_agent.main --config config.yaml
```

Output is saved to `output/` as a Markdown report by default.

## Configuration

All settings live in `config.yaml`. The file is divided into sections and commented throughout — open it for the full reference. Here are the key decisions:

### Choosing an LLM

```yaml
llm:
  model: claude-sonnet-4-5   # Best balance of quality and cost
```

| Provider  | Model                    | Notes                        |
| --------- | ------------------------ | ---------------------------- |
| Anthropic | `claude-sonnet-4-5`      | Recommended default          |
| Anthropic | `claude-haiku-4-5`       | Cheapest, good for testing   |
| Anthropic | `claude-opus-4-5`        | Highest quality              |
| OpenAI    | `gpt-5.2`               | Flagship                     |
| OpenAI    | `gpt-5-mini`            | Fast, but may produce malformed JSON |
| Google    | `gemini-3-pro-preview`   | Strong alternative           |
| Google    | `gemini-3-flash-preview` | Cheapest overall             |

You only need an API key for the provider(s) you use.

### Multi-Model Consensus

Instead of trusting a single LLM, you can run multiple models on each market and aggregate their estimates. When models agree, confidence goes up; when they disagree, the market is flagged.

```yaml
consensus_models:
  - claude-sonnet-4-5
  - gemini-3-flash-preview
```

This multiplies LLM cost by the number of models, but produces more reliable assessments.

### Filtering Markets

The agent fetches up to 100K active markets from Polymarket. Filters narrow these to a manageable set before the expensive LLM step.

```yaml
filters:
  keywords: []                    # Only include markets matching these terms
  exclude_keywords: [NFL, NBA]    # Remove sports, entertainment, etc.
  min_volume: 1000                # Minimum trading volume (USD)
  max_volume: 100000              # Focus on under-the-radar markets
  min_liquidity: 50
  max_days_to_expiry: 0           # 0 = no limit
  geographic_regions: []          # e.g., [EU, ASIA, LATAM]
  max_markets_per_event: 1        # Skip "one market per candidate" events
```

**Geographic filtering** uses weighted keyword matching. Set `geographic_regions` to focus on specific parts of the world where Polymarket's US-centric user base may be less informed.

**Event deduplication**: Events like "Who will be the next PM of Japan?" create dozens of per-candidate markets. The `max_markets_per_event` filter collapses these, keeping only events with a manageable number of sub-markets.

**Tag-based filtering**: For precise topic control, use Polymarket's tag IDs:
```yaml
filters:
  tag_ids: [2]  # 2 = Politics
```
Discover tags with `python scripts/list_tags.py --search "keyword"`.

### Controlling Cost

LLM calls are the main cost driver. Two settings control how many markets reach the LLM:

```yaml
enrichment_limit: 30     # How many markets to enrich with web search
llm_analysis_limit: 20   # How many to send to the LLM (most expensive step)
```

After enrichment, the agent ranks candidates by evidence quality (source diversity, number of facts, context freshness) and sends only the best-evidenced ones to the LLM.

### Risk Tolerance

Controls how aggressive the scoring is:

```yaml
risk_tolerance: moderate
```

- **conservative** — Only surfaces markets with >20% mispricing and high confidence.
- **moderate** — Balanced: >10% mispricing.
- **aggressive** — Includes speculative opportunities at >5% mispricing.

### Bias Analysis

Polymarket's demographics create predictable blind spots. The bias analysis pipeline detects these and tells the LLM to correct for them:

```yaml
llm_bias_analysis: true          # Enable the full bias pipeline
llm_bias_model: claude-haiku-4-5 # Use a cheap model for bias refinement
```

When enabled, the agent:
1. Scores each market for demographic bias potential using keyword matching (right-leaning politics, non-US regions, crypto sentiment, etc.).
2. Runs a cheap LLM call to verify the bias direction — fixing false positives like "Will Bitcoin crash?" being tagged as crypto-optimistic.
3. Includes detected biases in the assessment prompt so the main LLM can correct for them.

### Spread Analysis

Checks whether the detected mispricing survives real trading costs:

```yaml
enable_spread_analysis: true
```

Fetches live orderbook data from Polymarket's CLOB API, calculates bid-ask spread and slippage, and computes a "net edge" (mispricing minus trading costs). Markets where the spread eats the edge are penalised in the ranking.

### Database & Backtesting

Every run is persisted in a local SQLite database:

```yaml
enable_database: true
db_path: data/polymarket_analysis.db
```

This stores all predictions, probability estimates, confidence scores, and model reasoning. Once markets resolve, you can compare predictions against outcomes:

```bash
# Fetch resolution outcomes from Polymarket, then compute accuracy metrics
python scripts/backtest.py --fetch-resolutions
```

The backtest report includes:
- **Brier score** — Overall prediction accuracy (lower is better, 0.25 = random).
- **Calibration** — When the agent says 70%, does it happen 70% of the time?
- **Simulated ROI** — What would have happened if you traded every recommendation.
- **Bias category breakdown** — Which types of demographic bias produced the best alpha.

## CLI Reference

The agent can be run entirely from the command line without a config file:

```bash
# Basic run
python -m polymarket_agent.main --model claude-sonnet-4-5

# Focus on politics with high volume
python -m polymarket_agent.main --keywords "election" --min-volume 50000

# Aggressive scan of non-US markets
python -m polymarket_agent.main \
  --regions ASIA LATAM MIDDLE_EAST \
  --risk-tolerance aggressive \
  --llm-bias-analysis

# Multi-model consensus
python -m polymarket_agent.main \
  --consensus-models claude-sonnet-4-5 gemini-3-flash-preview

# Use a config file (CLI flags override config values)
python -m polymarket_agent.main --config config.yaml --verbose
```

Run `python -m polymarket_agent.main --help` for the full list of flags.

## Scoring Methodology

Each market receives a score from 0 to 100:

| Component            | Weight | What It Measures                                              |
| -------------------- | ------ | ------------------------------------------------------------- |
| Mispricing Magnitude | 30%    | Gap between market price and LLM's fair-value estimate        |
| Model Confidence     | 25%    | How confident the LLM is in its probability range             |
| Evidence Strength    | 20%    | Quality and quantity of external sources from web search       |
| Liquidity            | 15%    | Whether you could actually enter and exit a position           |
| Risk Adjustment      | 10%    | Penalises high-uncertainty situations, scaled by risk tolerance |

A score above 50 suggests a meaningful mispricing with supporting evidence. Below 30 is noise.

When spread analysis is enabled, markets where net edge (mispricing minus trading costs) is zero or negative have their score heavily penalised.

## Project Structure

```
polymarket_agent/
  config.py               Configuration and model definitions
  agent.py                Main pipeline orchestration
  main.py                 CLI entry point
  data_fetching/
    gamma_api.py          Polymarket Gamma API (events, markets)
    clob_api.py           Polymarket CLOB API (prices, orderbooks)
    models.py             Data models (Market, ScoredMarket, etc.)
  filtering/
    filters.py            Configurable filter chain
  enrichment/
    web_search.py         Web search + polling data extraction
  analysis/
    demographic_bias.py   Bias detection and LLM refinement
    reasoning_classifier.py  Identifies markets where LLM reasoning adds value
    spread_analysis.py    Orderbook spread and slippage calculation
  llm_assessment/
    providers.py          Anthropic, OpenAI, and Google LLM clients
    assessor.py           Assessment orchestration + consensus mode
    consensus.py          Multi-model aggregation and agreement scoring
    prompts.py            Prompt templates with bias correction
  scoring/
    scorer.py             Weighted scoring and ranking
  reporting/
    reporter.py           Markdown, JSON, and CSV report generation
  storage/
    database.py           SQLite persistence for backtesting
  utils/
    helpers.py            Logging and shared utilities

scripts/
  backtest.py             Compare predictions with actual outcomes
  list_tags.py            Browse Polymarket tag IDs

tests/                    149 unit tests
```

## Limitations

- **LLMs can be wrong.** They produce plausible-sounding reasoning even when their estimates are off. The confidence score helps, but it's self-reported.
- **Information lag.** Web search results may be hours or days behind. Markets can move on breaking news the agent hasn't seen.
- **Market efficiency.** Polymarket prices already reflect the collective wisdom of many traders. Detected "mispricings" may reflect information the model doesn't have.
- **Resolution ambiguity.** Complex resolution criteria (e.g., "arrest includes house arrest") can trip up both the LLM and the scoring.
- **Spread costs.** Low-liquidity markets may have wide spreads that eat any theoretical edge. Enable spread analysis to account for this.

## Disclaimer

This tool is for research and educational purposes. It provides probabilistic analysis, not financial advice. Prediction markets carry risk of loss. Always do your own research and never risk more than you can afford to lose.

## Testing

```bash
pytest tests/           # Run all 149 tests
pytest tests/ -v        # Verbose output
pytest tests/ -k bias   # Run only bias-related tests
```
