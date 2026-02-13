# Polymarket Bias Scanner Redesign

## Summary

Pivot from a full misprice-assessment pipeline (fetch → filter → enrich → LLM assess probability → score → report) to a focused bias-detection tool that identifies markets where Polymarket's demographic biases may create mispricing opportunities.

## Problem

The original approach used LLMs to estimate probabilities and detect mispricing magnitude. This is the wrong abstraction—the value is in identifying *where* biases exist, not in having an LLM guess probabilities.

## Solution

A scanner that:
1. Fetches markets from Polymarket
2. Filters by volume/liquidity
3. Uses an LLM to classify markets by bias category and predict mispricing direction
4. Outputs ranked lists by category in markdown

## Bias Categories

### Political
Markets where the right-leaning userbase creates predictable mispricing:
- Left-favorable outcomes → likely **underpriced**
- Left-unfavorable outcomes → likely **overpriced**
- Right-favorable outcomes → likely **overpriced**
- Right-unfavorable outcomes → likely **underpriced**

### Progressive Social
Topics on gender, race, LGBTQ+, immigration where progressive perspectives are underrepresented:
- Progressive-favorable outcomes → likely **underpriced**
- Progressive-unfavorable outcomes → likely **overpriced**

### Crypto Optimism
Markets where crypto enthusiasm skews pricing:
- Crypto-positive outcomes → likely **overpriced**
- Crypto-negative outcomes → likely **underpriced**

### European (Cross-cutting Tag)
Not a category itself. Markets in the above categories that relate to Europe (especially Spain) are flagged, representing informational edge opportunities.

## Polymarket Demographics (for LLM context)

- ~73% male
- Ages 25-45 concentration
- ~31% US-based
- Right-leaning politically
- Crypto/tech enthusiasts

## LLM Classification

### System Prompt
```
You classify prediction markets for potential bias-driven mispricing.

Polymarket demographics: ~73% male, ages 25-45, ~31% US-based, right-leaning politically, crypto/tech enthusiasts.

These demographics create predictable biases:
- Political: Right-leaning users overestimate right-favorable outcomes, underestimate left-favorable outcomes
- Progressive social: Topics on gender, race, LGBTQ+, immigration—progressive-favorable outcomes underestimated
- Crypto optimism: Positive crypto outcomes overestimated, negative outcomes underestimated

Classify markets and predict mispricing direction. Be concise.
```

### User Prompt (per market)
```
Market: {question}
Outcomes: {outcome_names}
Current prices: {prices}

Respond in JSON:
{
  "dominated_by_bias": true/false,
  "categories": ["political", "progressive_social", "crypto_optimism"],
  "bias_score": 0-100,
  "mispricing_direction": "overpriced" | "underpriced" | "unclear",
  "european": true/false,
  "spain": true/false,
  "reasoning": "One sentence"
}

Only include categories that apply. If no bias applies, set dominated_by_bias to false.
```

## Module Structure

### Keep (with modifications)
- `data_fetching/gamma_api.py` - Unchanged
- `data_fetching/clob_api.py` - Unchanged
- `data_fetching/models.py` - Unchanged
- `filtering/filters.py` - Remove bias-related filters, keep volume/liquidity/expiry

### Remove
- `llm_assessment/consensus.py`
- `llm_assessment/prompts.py`
- `enrichment/`
- `scoring/`
- `analysis/spread_analysis.py`
- `analysis/reasoning_classifier.py`
- `analysis/demographic_bias.py`
- `storage/database.py`
- `scripts/backtest.py`

### New/Rewritten
- `bias_detection/classifier.py` - LLM classification logic
- `bias_detection/models.py` - BiasClassification dataclass
- `reporting/reporter.py` - Markdown output by category
- `scanner.py` - Orchestrator: fetch → filter → classify → report
- `main.py` - Simplified CLI
- `config.py` - Stripped down config

### Final Structure
```
polymarket_agent/
├── data_fetching/
│   ├── gamma_api.py
│   ├── clob_api.py
│   └── models.py
├── filtering/
│   └── filters.py
├── bias_detection/
│   ├── classifier.py
│   └── models.py
├── reporting/
│   └── reporter.py
├── config.py
├── scanner.py
└── main.py
```

## Output Format

Markdown report grouped by category:

```markdown
# Polymarket Bias Scanner Report
Generated: 2026-02-13 14:30 UTC

## Political Bias (12 markets)

| Rank | Market | Direction | Score | Volume | Liquidity | EU |
|------|--------|-----------|-------|--------|-----------|-----|
| 1 | Will Democrats win the Senate? | underpriced | 85 | $2.3M | $450K | |
| 2 | Trump approval above 50%? | overpriced | 78 | $1.8M | $320K | |
| 3 | Sánchez survives confidence vote? | underpriced | 72 | $89K | $12K | ES |

## Progressive Social (5 markets)
...

## Crypto Optimism (8 markets)
...

---
Filters applied: min_volume=$50K, min_liquidity=$10K
Markets scanned: 1,247 | Classified with bias: 25
```

EU column: EU for European markets, ES for Spain.

## CLI Usage

```bash
# Basic run
python -m polymarket_agent.main

# With filters
python -m polymarket_agent.main --min-volume 50000 --min-liquidity 10000

# Output to specific file
python -m polymarket_agent.main --output reports/scan.md
```
