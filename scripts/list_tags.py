#!/usr/bin/env python3
"""
List available Polymarket tags for filtering.

This script fetches all available tags from the Polymarket API and displays them.
Use tag IDs in your config.yaml to filter events by specific categories.

Usage:
    python scripts/list_tags.py [--search KEYWORD]
"""

import asyncio
import argparse
import httpx
from typing import Optional


async def fetch_tags(search: Optional[str] = None) -> None:
    """Fetch and display Polymarket tags."""
    url = "https://gamma-api.polymarket.com/tags"
    params = {"limit": 500}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        tags = response.json()
    
    print(f"\n{'='*80}")
    print(f"POLYMARKET TAGS ({len(tags)} total)")
    print(f"{'='*80}\n")
    
    # Filter by search term if provided
    if search:
        search_lower = search.lower()
        tags = [t for t in tags if search_lower in t.get("label", "").lower()]
        print(f"Filtered by '{search}': {len(tags)} tags found\n")
    
    # Group tags by category
    print(f"{'ID':<10} {'Label':<40} {'Slug':<30}")
    print("-" * 80)
    
    for tag in sorted(tags, key=lambda t: t.get("label", "")):
        tag_id = tag.get("id", "")
        label = tag.get("label", "")
        slug = tag.get("slug", "")
        
        print(f"{tag_id:<10} {label:<40} {slug:<30}")
    
    print(f"\n{'='*80}")
    print("\n💡 Usage:")
    print("   Add tag IDs to config.yaml under filters.tag_ids:")
    print("   ")
    print("   filters:")
    print("     tag_ids: [589, 101768]  # US presidential election, European")
    print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="List available Polymarket tags for filtering"
    )
    parser.add_argument(
        "--search", "-s",
        help="Filter tags by keyword (case-insensitive)",
    )
    
    args = parser.parse_args()
    asyncio.run(fetch_tags(search=args.search))


if __name__ == "__main__":
    main()
