# Polymarket AI Agent

An intelligent agent that monitors active markets on Polymarket, retrieves and filters market data, uses LLMs to assess pricing, and outputs a ranked list of potentially mispriced markets with explanations.

## Overview

This agent performs one-off analysis runs to identify potential mispricings in prediction markets. It combines:

- **Data Fetching**: Retrieves active markets from Polymarket's public APIs
- **Filtering**: Configurable filters for category, keywords, volume, liquidity, and more
- **Enrichment**: Web search to gather relevant external context
- **LLM Assessment**: Uses your choice of LLM to estimate fair probabilities
- **Scoring**: Ranks markets by mispricing magnitude, confidence, and risk profile
- **Reporting**: Outputs structured Markdown reports (also JSON/CSV)

## Architecture

```
polymarket_agent/
├── __init__.py
├── config.py              # Configuration and constants
├── data_fetching/         # Polymarket API interactions
│   ├── __init__.py
│   ├── gamma_api.py       # Gamma API for events/markets
│   ├── clob_api.py        # CLOB API for prices/orderbooks
│   └── models.py          # Data models (Market, ScoredMarket, etc.)
├── filtering/             # Market filtering logic
│   ├── __init__.py
│   └── filters.py
├── enrichment/            # External data enrichment
│   ├── __init__.py
│   └── web_search.py
├── llm_assessment/        # LLM integration
│   ├── __init__.py
│   ├── assessor.py        # LLM assessment orchestration
│   ├── providers.py       # Multi-provider LLM support
│   └── prompts.py         # Prompt templates
├── analysis/              # Market analysis utilities
│   ├── __init__.py
│   ├── reasoning_classifier.py  # Reasoning-heavy market detection
│   └── demographic_bias.py      # Demographic bias scoring
├── scoring/               # Scoring and ranking
│   ├── __init__.py
│   └── scorer.py
├── reporting/             # Output generation
│   ├── __init__.py
│   └── reporter.py
├── utils/                 # Shared utilities
│   ├── __init__.py
│   └── helpers.py
└── main.py                # CLI entry point
```

## Installation

### Prerequisites

- Python 3.10+
- API keys for your chosen LLM provider(s)

### Setup

```bash
# Clone or navigate to the project directory
cd polymarket-1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment template and add your API keys
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

### Environment Variables

Create a `.env` file with your API keys:

```env
# LLM Provider API Keys (add the ones you want to use)
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key

# Optional: Web search API (if using paid search)
SERPER_API_KEY=your_serper_key
```

### Filter Configuration

Filters can be set via CLI arguments, config file, or programmatically:

```yaml
# config.yaml example
filters:
  # Keywords matched against market titles (questions)
  keywords:
    - Trump
    - Bitcoin
    - Election
    - Spain

  # Keywords to exclude (e.g., sports markets)
  exclude_keywords:
    - NFL
    - NBA
    - La Liga

  min_volume: 100
  max_volume: 500000
  min_liquidity: 50
  max_days_to_expiry: 360

  # Demographic bias filter
  bias_filter_enabled: true
  min_blind_spot_score: 10
  mispricing_levels:
    - high
    - medium

risk_tolerance: moderate  # conservative, moderate, aggressive

llm:
  model: gemini-3-flash-preview
```

### Geographical Filtering

The agent uses **weighted keyword matching** to infer market geographic relevance. Each keyword has a confidence weight (1-10), and markets are scored based on matches.

**Available Regions:**

| Region        | Description    | Example Keywords                                    |
| ------------- | -------------- | --------------------------------------------------- |
| `US`          | United States  | congress, federal reserve, biden, trump, california |
| `EU`          | European Union | european commission, ecb, macron, scholz, eurozone  |
| `UK`          | United Kingdom | parliament, bank of england, starmer, london        |
| `ASIA`        | Asia-Pacific   | china, japan, korea, india, xi jinping, modi        |
| `LATAM`       | Latin America  | brazil, mexico, argentina, lula, milei              |
| `MIDDLE_EAST` | Middle East    | israel, iran, saudi, netanyahu, opec                |
| `AFRICA`      | Africa         | south africa, nigeria, kenya, african union         |
| `CRYPTO`      | Cryptocurrency | bitcoin, ethereum, defi, binance, blockchain        |
| `GLOBAL`      | International  | united nations, imf, g20, nato, climate             |

**Configuration:**
```yaml
filters:
  geographic_regions:
    - US
    - EU

  # Confidence threshold (0-100)
  # - 30: Any moderate match (default)
  # - 50: Strong indicator required
  # - 70: Definitive match only
  geo_min_score: 50
```

**How Scoring Works:**
- Keywords have weights 1-10 (e.g., "United States Congress" = 10, "Trump" = 7)
- Score combines highest weight match + bonus for multiple matches
- Markets must meet `geo_min_score` threshold to be included

**Tag-Based Filtering** (Alternative)

For topic-based filtering, use Polymarket's tag system. Note: the Polymarket API only supports filtering by **one** `tag_id` at a time.
```yaml
filters:
  tag_ids:
    - 2        # Politics
```

To discover available tags:
```bash
python scripts/list_tags.py --search "election"
```

### Niche Market Discovery

If you are looking for niche topics (e.g., regional elections in Spain) that don't have massive volume relative to US presidential elections, they may be buried deep in the list of active markets.

To ensure the agent "sees" these markets:
1.  **Increase `max_markets_to_fetch`**: Change this in `config.yaml` to `30000` or `40000`. This allows the agent to scan a larger portion of the many active markets on Polymarket.
2.  **Use Targeted Tag IDs**: If you know the tag (e.g., `101768` for "European"), specify it in `tag_ids` to force the API to return those events first.
3.  **Use Specific Keywords**: The agent will now attempt to use your first keyword as a search query during data fetching to improve discovery.

## Usage

### Command Line Interface

```bash
# Basic run with defaults
python -m polymarket_agent.main

# Specify LLM model
python -m polymarket_agent.main --model claude-sonnet-4-5

# With filters
python -m polymarket_agent.main \
  --categories politics crypto \
  --min-volume 10000 \
  --risk-tolerance aggressive \
  --output output

# Markdown output format
python -m polymarket_agent.main --format markdown

# Using config file
python -m polymarket_agent.main --config config.yaml

# Dry run (fetch and filter only, no LLM calls)
python -m polymarket_agent.main --dry-run
```

### Programmatic Usage

```python
from polymarket_agent import PolymarketAgent
from polymarket_agent.config import AgentConfig, FilterConfig, RiskTolerance

# Configure the agent
config = AgentConfig(
    filters=FilterConfig(
        categories=["politics", "crypto"],
        min_volume=10000,
        keywords=["election"]
    ),
    risk_tolerance=RiskTolerance.MODERATE,
    llm_model="claude-sonnet-4-5"
)

# Run analysis
agent = PolymarketAgent(config)
results = agent.run()

# Access results
for market in results.ranked_markets:
    print(f"{market.market.question}: {market.total_score:.1f}")
    print(f"  Market: Yes @ {market.market.outcome_prices.get('Yes', 0):.1%}")
    est = market.assessment.probability_estimates.get("Yes", (0, 0))
    print(f"  LLM Est: {est[0]:.1%}-{est[1]:.1%}")
    print(f"  Rationale: {market.assessment.reasoning[:200]}")
```

## Model Selection

### Supported Models

| Provider  | Model ID                 | Notes                                    |
| --------- | ------------------------ | ---------------------------------------- |
| Anthropic | `claude-sonnet-4-5`      | Recommended for balance of speed/quality |
| Anthropic | `claude-haiku-4-5`       | Fastest, most cost-effective             |
| Anthropic | `claude-opus-4-5`        | Highest quality, slower                  |
| OpenAI    | `gpt-5.2`                | Flagship model                           |
| OpenAI    | `gpt-5-mini`             | Fast/efficient                           |
| Google    | `gemini-3-flash-preview` | Fast Google model                        |
| Google    | `gemini-3-pro-preview`   | High quality Google model                |

## Interpreting Results

### Ranking Scores

The overall attractiveness score (0-100) combines:

| Component            | Weight | Description                                         |
| -------------------- | ------ | --------------------------------------------------- |
| Mispricing Magnitude | 30%    | Difference between market and estimated probability |
| Model Confidence     | 25%    | LLM's self-reported confidence in estimate          |
| Evidence Strength    | 20%    | Quality/freshness of supporting information         |
| Liquidity Score      | 15%    | Ability to enter/exit position                      |
| Risk Adjustment      | 10%    | Adjusted based on risk tolerance                    |

### Risk Tolerance Effects

- **Conservative**: Only surfaces markets with >20% mispricing AND high confidence
- **Moderate**: Surfaces markets with >10% mispricing, balanced confidence
- **Aggressive**: Includes speculative opportunities with >5% mispricing

### Output Fields

The Markdown report includes a rankings table and detailed per-market analysis. The underlying `ScoredMarket.to_dict()` output looks like:

```json
{
  "rank": 1,
  "market_slug": "will-x-happen",
  "title": "Will X happen by date Y?",
  "category": "politics",
  "market_prices": {"Yes": 0.65, "No": 0.35},
  "llm_estimate": {"Yes": [0.45, 0.55], "No": [0.45, 0.55]},
  "mispricing_detected": true,
  "mispricing_direction": "overpriced",
  "mispricing_magnitude": 0.15,
  "confidence": 0.72,
  "explanation": "Based on recent polling data...",
  "key_factors": ["Recent polling shift", "Historical precedent"],
  "risks": ["Limited sample size"],
  "evidence_sources": ["source1.com", "source2.com"],
  "volume": 245000,
  "liquidity": 180000,
  "days_to_expiry": 30,
  "scores": {
    "mispricing": 78.5,
    "confidence": 65.0,
    "evidence": 55.0,
    "liquidity": 70.0,
    "risk_adjusted": 60.0,
    "total": 68.2
  },
  "warnings": ["Limited historical data available"]
}
```

## Limitations & Assumptions

### Known Limitations

1. **LLM Hallucination Risk**: LLMs may generate plausible-sounding but incorrect probability estimates. Always verify key claims.

2. **Information Lag**: Web search results may not reflect the most recent developments.

3. **Market Efficiency**: Prediction markets often incorporate information quickly. Detected "mispricings" may reflect information the model doesn't have.

4. **Liquidity Constraints**: Low-liquidity markets may have wide spreads that affect actual execution prices.

5. **Resolution Criteria**: Complex resolution criteria may be misinterpreted by the LLM.

### Assumptions

1. **API Availability**: Assumes Polymarket's public APIs remain accessible and maintain current structure.

2. **Price Accuracy**: Assumes `outcomePrices` from the API reflect current market consensus.

3. **Web Search**: Uses free search methods which may have rate limits.

### Ethical Considerations

- This tool provides **probabilistic analysis**, not financial advice
- No guarantees of profit—prediction markets carry inherent risk
- Users should conduct their own research before trading
- Past mispricing patterns don't guarantee future opportunities

## Example Run

```bash
$ python -m polymarket_agent.main --categories politics --min-volume 50000 --limit 5

Fetching active markets from Polymarket...
   Found 247 active markets

Applying filters...
   After filtering: 23 markets

Enriching top candidates with web search...
   Enriched 10 markets with external context

Running LLM assessment (claude-sonnet-4-5)...
   Analyzed 10 markets

Scoring and ranking results...

═══════════════════════════════════════════════════════════════
TOP 5 POTENTIALLY MISPRICED MARKETS
═══════════════════════════════════════════════════════════════

#1 [Score: 78.5] Will candidate X win primary?
   Market: Yes @ 72% | LLM Estimate: 52-58%
   Direction: OVERPRICED by ~16%
   Confidence: High | Liquidity: $245K
   Rationale: Recent polling shows significant momentum shift...

#2 [Score: 65.2] Will crypto bill pass by Q2?
   Market: Yes @ 45% | LLM Estimate: 58-65%
   Direction: UNDERPRICED by ~16%
   Confidence: Medium | Liquidity: $180K
   Rationale: Committee markup scheduled for next week...

...

Full report saved to: output/report_2026-01-24_143052.md
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=polymarket_agent

# Run specific test module
pytest tests/test_filtering.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It does not constitute financial advice. Trading on prediction markets involves risk of loss. Always do your own research and never risk more than you can afford to lose.
