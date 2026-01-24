# Geographical Filtering Guide

This guide explains how to filter Polymarket markets by geographical region.

## Overview

The Polymarket Agent supports two methods for geographical/topical filtering:

1. **Tag IDs** (Recommended) - Uses Polymarket's official tag system for precise filtering
2. **Keyword Matching** - Searches market titles/descriptions for geographical terms (less precise)

## Method 1: Tag ID Filtering (Recommended)

### How It Works

Polymarket categorizes events using tags. You can filter events by tag ID using the `tag_ids` field in your configuration.

**Important**: The Polymarket API only supports filtering by **ONE** tag at a time. If you specify multiple tag IDs, only the first one will be used.

### Finding Tag IDs

**Option 1: Use the helper script**
```bash
# List all tags
python scripts/list_tags.py

# Search for specific topics
python scripts/list_tags.py --search "politics"
python scripts/list_tags.py --search "crypto"
python scripts/list_tags.py --search "ukraine"
```

**Option 2: Check the API directly**
```bash
curl "https://gamma-api.polymarket.com/tags?limit=500"
```

### Common Active Tags (January 2026)

| Tag ID | Label | Description |
|--------|-------|-------------|
| 2 | Politics | Most active political category |
| 101191 | Trump Presidency | Trump administration events |
| 126 | Trump | Trump-related markets |
| 96 | Ukraine | Ukraine conflict |
| 21 | Crypto | Cryptocurrency markets |
| 450 | NFL | American football |
| 101970 | World | World events |
| 100265 | Geopolitics | International politics |
| 188 | U.S. Politics | US domestic politics |

### Configuration Example

```yaml
filters:
  # Filter by tag ID (only first tag used)
  tag_ids:
    - 101191  # Trump Presidency
  
  # Further refine with keywords
  keywords:
    - cabinet
    - nomination
    - policy
  
  min_volume: 50000
  min_liquidity: 10000
```

### Example Configs

**US Politics:**
```bash
python -m polymarket_agent.main --config config.us-election.yaml --dry-run
```

**Crypto Markets:**
```bash
python -m polymarket_agent.main --config config.crypto.yaml --dry-run
```

## Method 2: Keyword Matching

### How It Works

The `geographic_regions` field uses keyword matching to find markets related to specific regions. This searches through market titles, descriptions, and tags.

### Predefined Regions

The agent includes keyword mappings for common regions:

| Region Code | Keywords Matched |
|-------------|-----------------|
| US | United States, American, USA, U.S. |
| EU | Europe, European, EU |
| UK | United Kingdom, British, UK, England |
| ASIA | Asia, Asian, China, Japan, Korea |
| CRYPTO | Bitcoin, Ethereum, cryptocurrency, DeFi |
| GLOBAL | World, Global, International |

### Configuration Example

```yaml
filters:
  geographic_regions:
    - US
    - EU
  
  keywords:
    - election
    - president
```

**Note**: This method is less precise than tag filtering and may include false positives.

## Combining Filters

You can combine multiple filtering methods for precise results:

```yaml
filters:
  # Start with a tag for the main category
  tag_ids:
    - 2  # Politics
  
  # Add keywords for subcategory
  keywords:
    - immigration
    - border
  
  # Exclude unwanted topics
  exclude_keywords:
    - sports
    - entertainment
  
  # Volume/liquidity filters
  min_volume: 100000
  min_liquidity: 20000
  
  # Time horizon
  max_days_to_expiry: 90
```

## Workflow

### 1. Discover Available Tags

```bash
# Find tags related to your topic
python scripts/list_tags.py --search "election"
python scripts/list_tags.py --search "crypto"
python scripts/list_tags.py --search "sports"
```

### 2. Test Your Filters

```bash
# Run in dry-run mode to see how many markets match
python -m polymarket_agent.main --config your-config.yaml --dry-run
```

### 3. Adjust and Run

```bash
# Once satisfied, run the full analysis
python -m polymarket_agent.main --config your-config.yaml
```

## API Limitations

1. **Single Tag Filter**: The Polymarket API only supports filtering by one `tag_id` at a time
2. **Tag Availability**: Not all topics have dedicated tags
3. **Dynamic Tags**: Tags and their IDs may change over time

## Tips

1. **Start Broad**: Begin with tag filtering, then refine with keywords
2. **Test First**: Always run `--dry-run` to verify your filters
3. **Check Volume**: High-volume tags (like Politics) may return many markets
4. **Update Tags**: Periodically run `list_tags.py` to find new relevant tags
5. **Combine Methods**: Use tag_ids for category + keywords for specifics

## Examples

### US Political Markets
```yaml
filters:
  tag_ids: [101191]  # Trump Presidency
  keywords: [policy, nomination]
  min_volume: 50000
```

### European Markets
```yaml
filters:
  geographic_regions: [EU]
  keywords: [election, parliament]
  min_volume: 25000
```

### Cryptocurrency Markets
```yaml
filters:
  tag_ids: [21]  # Crypto
  keywords: [Bitcoin, Ethereum, price]
  min_volume: 100000
```

### Sports (NFL)
```yaml
filters:
  tag_ids: [450]  # NFL
  keywords: [Super Bowl, playoffs]
  min_volume: 50000
```
