"""
LLM Prompt Templates

Contains all prompt templates used for market assessment.
These prompts are designed to elicit structured, probabilistic reasoning
from LLMs about prediction market pricing.

IMPORTANT: These prompts emphasize uncertainty and avoid absolutist language.
The LLM is instructed to provide probability ranges, not point estimates,
and to clearly articulate confidence levels and potential errors.

Prompt Design Principles:
1. Structured output format for reliable parsing
2. Emphasis on probabilistic reasoning
3. Clear instruction to consider base rates
4. Warning about information limitations
5. Request for explicit confidence levels
"""

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """You are a quantitative analyst specializing in prediction markets and probabilistic forecasting. Your role is to assess whether prediction market prices accurately reflect true probabilities.

Key principles for your analysis:
1. PROBABILISTIC THINKING: Always think in probability ranges, not point estimates. The world is uncertain.
2. BASE RATES: Consider historical base rates for similar events before adjusting based on specific information.
3. INFORMATION QUALITY: Evaluate the reliability and recency of available information.
4. MARKET EFFICIENCY: Prediction markets often incorporate information quickly. Be humble about detecting mispricings.
5. UNCERTAINTY: Clearly state when you're uncertain. It's better to admit ignorance than to hallucinate facts.

You will analyze markets and provide structured assessments including:
- Probability estimates as ranges (e.g., 35-45%)
- Confidence levels in your assessment
- Key factors supporting your estimate
- Potential risks and blind spots
- Comparison to market prices

CRITICAL WARNINGS:
- Do NOT fabricate statistics or data. If you don't know, say so.
- Do NOT provide financial advice. You are analyzing probabilities, not recommending trades.
- Do NOT claim certainty. All predictions are probabilistic.
- Do NOT ignore the possibility that the market is right and you are wrong.

Output your assessments in the structured format requested."""


# ============================================================================
# Assessment Prompt Template
# ============================================================================

ASSESSMENT_PROMPT_TEMPLATE = """Analyze the following prediction market and assess whether its current pricing is accurate.

## MARKET INFORMATION

**Question:** {question}

**Description/Resolution Criteria:**
{description}

**Current Market Prices:**
{outcome_prices}

**Market Metadata:**
- Category: {category}
- Trading Volume: ${volume:,.0f}
- Liquidity: ${liquidity:,.0f}
- Days to Resolution: {days_to_expiry}

## EXTERNAL CONTEXT

The following information was gathered from web search to help inform your analysis:

{external_context}

**Key Facts Extracted:**
{key_facts}
{demographic_bias_context}
---

## YOUR TASK

Analyze this market and provide your assessment in the following JSON format:

```json
{{
    "probability_estimates": {{
        "<outcome_name>": [<lower_bound>, <upper_bound>],
        ...
    }},
    "confidence": <0.0 to 1.0>,
    "reasoning": "<detailed reasoning>",
    "key_factors": [
        "<factor 1>",
        "<factor 2>",
        ...
    ],
    "risks": [
        "<risk or uncertainty 1>",
        "<risk or uncertainty 2>",
        ...
    ],
    "mispricing_detected": <true/false>,
    "mispricing_direction": "<overpriced|underpriced|fair>",
    "mispricing_magnitude": <0.0 to 1.0>,
    "warnings": [
        "<any warnings about this assessment>",
        ...
    ],
    "bias_adjustment": {{
        "detected_biases": ["<bias category 1>", ...],
        "estimated_skew_direction": "<overpriced|underpriced|neutral>",
        "estimated_skew_magnitude": 0.0,
        "reasoning": "<explanation of bias correction applied>"
    }}
}}
```

## GUIDELINES FOR YOUR ANALYSIS

1. **Probability Estimates**: Provide a range [lower, upper] for each outcome. The ranges should reflect your uncertainty. Wider ranges = more uncertainty.

2. **Confidence**: Rate your overall confidence from 0.0 (no confidence) to 1.0 (very confident). Consider:
   - Quality and recency of available information
   - Complexity of the prediction
   - Whether this is in your knowledge domain

3. **Reasoning**: Explain your logic step by step. Reference specific evidence.

4. **Key Factors**: List the 3-5 most important factors influencing your estimate.

5. **Risks**: What could make your estimate wrong? What are you uncertain about?

6. **Mispricing Detection**:
   - Compare the MIDPOINT of your probability range to the market price
   - `mispricing_detected`: true if difference > 5 percentage points
   - `mispricing_direction`: 
     - "overpriced" if market price > your estimate (market is too bullish)
     - "underpriced" if market price < your estimate (market is too bearish)
     - "fair" if within 5 percentage points
   - `mispricing_magnitude`: absolute difference between market and your midpoint estimate

7. **Warnings**: Include any caveats, e.g.:
   - "Limited information available"
   - "Rapidly evolving situation"
   - "Resolution criteria ambiguous"
   - "Outside my knowledge cutoff"

Remember: It's better to express high uncertainty than to appear confident when you're not. Markets are often more efficient than they appear.

Provide your assessment now:"""


# ============================================================================
# Prompt Building Functions
# ============================================================================

def format_outcome_prices(outcomes: list) -> str:
    """
    Format outcome prices for the prompt.
    
    Args:
        outcomes: List of Outcome objects
        
    Returns:
        Formatted string of outcome prices
    """
    lines = []
    for outcome in outcomes:
        percentage = outcome.price * 100
        lines.append(f"- {outcome.name}: {percentage:.1f}%")
    return "\n".join(lines)


def format_key_facts(facts: list[str]) -> str:
    """
    Format key facts for the prompt.
    
    Args:
        facts: List of key fact strings
        
    Returns:
        Formatted bullet list
    """
    if not facts:
        return "- No key facts extracted"
    
    return "\n".join(f"- {fact}" for fact in facts)


def build_assessment_prompt(
    market,  # Market object
    enrichment = None,  # EnrichedMarket object
    bias_analysis = None,  # Optional BiasAnalysis object
) -> str:
    """
    Build the full assessment prompt for a market.
    
    This function populates the prompt template with market data
    and any enrichment context available.
    
    Args:
        market: Market object to assess
        enrichment: Optional EnrichedMarket with external context
        
    Returns:
        Complete prompt string ready to send to LLM
        
    Example:
        >>> prompt = build_assessment_prompt(market, enriched_market)
        >>> response = await llm_client.complete(prompt)
    """
    # Get external context
    if enrichment:
        external_context = enrichment.external_context or "No external context available."
        key_facts = format_key_facts(enrichment.key_facts)
    else:
        external_context = "No external context available. Analysis based on market information only."
        key_facts = "- No key facts available"
    
    # Handle missing days to expiry
    days_to_expiry = market.days_to_expiry
    if days_to_expiry is None:
        days_to_expiry_str = "Unknown"
    else:
        days_to_expiry_str = str(days_to_expiry)
    
    # Build demographic bias context from raw_data (populated by bias filter)
    demographic_bias_context = ""

    # Always include demographics awareness section
    bias_correction_section = (
        "\n## BIAS CORRECTION TASK\n"
        "\n"
        "Polymarket's user base is approximately 73% male, concentrated ages 25-45, ~31% US-based,\n"
        "with strong crypto/tech affinity and a documented right-leaning political skew.\n"
        "\n"
    )

    has_specific_biases = market.raw_data and '_detected_biases' in market.raw_data

    if has_specific_biases:
        biases = market.raw_data['_detected_biases']
        direction = market.raw_data.get('_bias_direction', 'uncertain')
        blind_spot = market.raw_data.get('_blind_spot_score', 0)
        bias_list = "\n".join(f"- {b}" for b in biases)

        # Map bias categories to correction guidance
        guidance_lines = []
        for bias in biases:
            if bias == "political_right_bias":
                guidance_lines.append(
                    "- This market involves right-leaning political content. "
                    "The user base tends to overestimate right-leaning outcomes. "
                    "Consider whether the current price is inflated."
                )
            elif bias == "political_left_bias":
                guidance_lines.append(
                    "- This market involves left-leaning political content. "
                    "The user base may systematically underestimate left-leaning outcomes. "
                    "Consider whether the current price is deflated."
                )
            elif bias == "crypto_optimism":
                guidance_lines.append(
                    "- This market involves cryptocurrency. "
                    "The user base has strong crypto enthusiasm and may overestimate "
                    "positive crypto outcomes. Apply skepticism to bullish pricing."
                )
            elif bias in ("geographic_blind_spots", "non_us_politics"):
                guidance_lines.append(
                    "- This market involves a region/topic unfamiliar to most users. "
                    "Low user familiarity increases mispricing potential. "
                    "Your external knowledge may provide significant edge here."
                )
            elif bias == "gender_topics":
                guidance_lines.append(
                    "- This market touches on gender-related topics. "
                    "The male-dominated user base may have blind spots. "
                    "Consider perspectives that may be underrepresented."
                )
            elif bias == "age_blind_spots":
                guidance_lines.append(
                    "- This market involves age demographics outside the 25-45 core. "
                    "Users may have limited familiarity with these demographics."
                )

        guidance_text = "\n".join(guidance_lines) if guidance_lines else ""

        bias_correction_section += (
            "This market was flagged for the following potential demographic biases:\n"
            f"{bias_list}\n"
            "\n"
            f"Likely bias direction: {direction}\n"
            f"Blind spot score: {blind_spot}/100\n"
            "\n"
            "**Specific correction guidance:**\n"
            f"{guidance_text}\n"
            "\n"
        )
    else:
        bias_correction_section += (
            "No specific demographic biases were detected for this market, "
            "but keep the user demographics in mind as a general check.\n"
            "\n"
        )

    bias_correction_section += (
        "Your probability_estimates should ALREADY INCORPORATE your bias correction.\n"
        "\n"
        'In addition to the standard fields, include a "bias_adjustment" field in your JSON:\n'
        '```\n'
        '"bias_adjustment": {\n'
        '    "detected_biases": ["list of relevant bias categories"],\n'
        '    "estimated_skew_direction": "overpriced|underpriced|neutral",\n'
        '    "estimated_skew_magnitude": 0.0,\n'
        '    "reasoning": "explanation of bias correction applied"\n'
        '}\n'
        '```\n'
    )

    demographic_bias_context = bias_correction_section

    # Add polling data section if available
    if market.raw_data and '_polling_data' in market.raw_data:
        polling_data = market.raw_data['_polling_data']
        if polling_data:
            polling_lines = []
            for poll in polling_data:
                polling_lines.append(
                    f"- {poll.get('entity', '?')}: {poll.get('percentage', '?')}% "
                    f"(source: {poll.get('source', 'unknown')})"
                )
            polling_section = (
                "\n## POLLING DATA\n"
                "\n"
                "Recent polling data found for this market:\n"
                + "\n".join(polling_lines) + "\n"
                "\n"
                "**Caveat**: Polls have known biases including non-response bias, "
                "likely voter screens, and herding effects. Use as one input among many.\n"
            )
            demographic_bias_context += polling_section

    # Build the prompt
    prompt = ASSESSMENT_PROMPT_TEMPLATE.format(
        question=market.question,
        description=market.description[:2000] if market.description else "No detailed description provided.",
        outcome_prices=format_outcome_prices(market.outcomes),
        category=market.category or "Unknown",
        volume=market.volume,
        liquidity=market.liquidity,
        days_to_expiry=days_to_expiry_str,
        external_context=external_context,
        key_facts=key_facts,
        demographic_bias_context=demographic_bias_context,
    )
    
    return prompt


# ============================================================================
# Batch Assessment Prompt
# ============================================================================

BATCH_ASSESSMENT_INTRO = """You will analyze multiple prediction markets in sequence. For each market, provide a complete assessment in the JSON format specified.

Analyze each market independently, but feel free to note if markets are related.

---

"""


def build_batch_prompt(
    markets: list,  # List of Market objects
    enrichments: dict = None,  # Dict of market_id -> EnrichedMarket
) -> str:
    """
    Build a prompt for batch assessment of multiple markets.
    
    For efficiency, multiple markets can be assessed in a single LLM call.
    
    Args:
        markets: List of Market objects
        enrichments: Optional dict mapping market IDs to EnrichedMarket
        
    Returns:
        Combined prompt for all markets
    """
    if enrichments is None:
        enrichments = {}
    
    parts = [BATCH_ASSESSMENT_INTRO]
    
    for i, market in enumerate(markets, 1):
        enrichment = enrichments.get(market.id)
        market_prompt = build_assessment_prompt(market, enrichment)
        
        parts.append(f"## MARKET {i} of {len(markets)}\n\n")
        parts.append(market_prompt)
        parts.append("\n\n---\n\n")
    
    return "".join(parts)


# ============================================================================
# Response Parsing Helpers
# ============================================================================

def extract_json_from_response(response: str) -> str:
    """
    Extract JSON from an LLM response that may contain markdown code blocks.
    
    Args:
        response: Raw LLM response text
        
    Returns:
        Extracted JSON string
    """
    import re
    
    # Try to find JSON in code blocks first
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, response)
    
    if matches:
        return matches[0].strip()
    
    # Try to find raw JSON (starts with { and ends with })
    json_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_pattern, response)
    
    if matches:
        # Return the longest match (likely the full JSON)
        return max(matches, key=len)
    
    # Return as-is if no JSON found
    return response.strip()
