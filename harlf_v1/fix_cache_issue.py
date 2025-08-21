#!/usr/bin/env python3
"""
Cache Management Utility for Sentiment Analysis

This script provides functions to fix cache issues and improve cache management
for the sentiment analysis pipeline.
"""

import json
import time
from pathlib import Path

def get_cache_path(source, ticker, year):
    """Get cache file path for specific data"""
    cache_dir = Path("news_cache")
    return cache_dir / f"{source}_{ticker}_{year}.json"

def load_cached_data(cache_path, force_refresh=False):
    """Load data from cache if available and recent (24 hours)"""
    if force_refresh:
        return None
        
    if cache_path.exists():
        # Check if cache is less than 24 hours old
        if time.time() - cache_path.stat().st_mtime < 86400:  # 24 hours
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    # Don't return empty arrays from cache - force fresh fetch
                    if isinstance(data, list) and len(data) == 0:
                        print(f"Empty cache detected for {cache_path.name}, forcing fresh fetch")
                        return None
                    return data
            except Exception as e:
                print(f"Cache read error for {cache_path.name}: {e}")
                return None
    return None

def save_to_cache(cache_path, data):
    """Save data to cache"""
    try:
        # Don't cache empty results to avoid the empty cache problem
        if isinstance(data, list) and len(data) == 0:
            print(f"Skipping cache save for empty results: {cache_path.name}")
            return
            
        with open(cache_path, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Cache save error: {e}")

def clear_cache_for_ticker(source, ticker, year):
    """Clear cache for specific ticker to force fresh fetch"""
    cache_path = get_cache_path(source, ticker, year)
    if cache_path.exists():
        cache_path.unlink()
        print(f"Cleared cache: {cache_path.name}")

def clear_empty_caches():
    """Clear all empty cache files that might cause issues"""
    cache_dir = Path("news_cache")
    if not cache_dir.exists():
        print("No cache directory found")
        return
        
    cleared_count = 0
    for cache_file in cache_dir.glob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) == 0:
                    cache_file.unlink()
                    print(f"Cleared empty cache: {cache_file.name}")
                    cleared_count += 1
        except Exception as e:
            print(f"Error reading {cache_file.name}: {e}")
            # If we can't read it, it might be corrupted, so delete it
            cache_file.unlink()
            print(f"Cleared corrupted cache: {cache_file.name}")
            cleared_count += 1
    
    print(f"Cleared {cleared_count} problematic cache files")

def list_cache_status():
    """List all cache files and their status"""
    cache_dir = Path("news_cache")
    if not cache_dir.exists():
        print("No cache directory found")
        return
        
    print("Cache Status:")
    print("-" * 50)
    
    for cache_file in sorted(cache_dir.glob("*.json")):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    status = f"{len(data)} items"
                    if len(data) == 0:
                        status = "EMPTY (problematic)"
                else:
                    status = "Invalid format"
                    
                age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                print(f"{cache_file.name}: {status} (age: {age_hours:.1f}h)")
        except Exception as e:
            print(f"{cache_file.name}: CORRUPTED ({e})")

if __name__ == "__main__":
    print("Cache Management Utility")
    print("=" * 30)
    
    # List current cache status
    list_cache_status()
    
    print("\nClearing problematic cache files...")
    clear_empty_caches()
    
    print("\nUpdated cache status:")
    list_cache_status() 