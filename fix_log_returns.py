#!/usr/bin/env python3
"""
Fix log returns data to include all tickers from price data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_log_returns_from_prices():
    """Create log returns data from price data."""
    
    # Load price data
    price_file = Path("outputs/price_data_for_training.csv")
    if not price_file.exists():
        print("❌ Price data file not found!")
        return
    
    print("Loading price data...")
    prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
    print(f"✓ Price data loaded: {prices.shape}")
    print(f"Tickers: {list(prices.columns)}")
    
    # Calculate log returns
    print("Calculating log returns...")
    log_returns = np.log(prices / prices.shift(1)).fillna(0)
    
    # Save log returns
    output_file = Path("outputs/log_returns_data.csv")
    log_returns.to_csv(output_file)
    
    print(f"✓ Log returns saved: {log_returns.shape}")
    print(f"✓ File: {output_file}")
    print(f"Tickers in log returns: {list(log_returns.columns)}")
    
    # Verify ARBE and RGTI are included
    if 'ARBE' in log_returns.columns and 'RGTI' in log_returns.columns:
        print("✓ ARBE and RGTI are included in log returns data")
    else:
        print("❌ ARBE and RGTI are missing from log returns data")
        missing = []
        if 'ARBE' not in log_returns.columns:
            missing.append('ARBE')
        if 'RGTI' not in log_returns.columns:
            missing.append('RGTI')
        print(f"Missing tickers: {missing}")

if __name__ == "__main__":
    create_log_returns_from_prices() 