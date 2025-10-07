import pandas as pd


def add_regime_indicator(prices, window=6):
    """
    Add market regime indicator (bull/bear) for each asset.
    
    Args:
        prices: DataFrame of asset prices
        window: Rolling window for SMA (default 6 months)
    
    Returns:
        DataFrame with regime indicators (1=bull, 0=bear) per asset
    """
    sma = prices.rolling(window=window).mean()
    regime = (prices > sma).astype(int)
    regime.columns = [f'Regime_{col}' for col in regime.columns]
    return regime


def split_data_chronologically(price_data, technical_features, sentiment_features,
                               train_ratio=0.60, val_ratio=0.20, test_ratio=0.20):
    
    print("\n" + "="*70)
    print("DATA SPLITTING")
    print("="*70)
    
    # Verify all data has same length
    assert len(price_data) == len(technical_features) == len(sentiment_features), \
        "All dataframes must have same length!"
    
    n_total = len(price_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    print(f"\nTotal data: {n_total} months")
    print(f"Date range: {price_data.index[0]} to {price_data.index[-1]}")
    
    # Check minimum data requirements
    if n_total < 36:
        print(f"\n⚠️  WARNING: Only {n_total} months of data!")
        print(f"   Recommended: 60+ months")
        print(f"   Consider using more data or weekly frequency")
    
    # Split chronologically (keeping time order!)
    train_prices = price_data.iloc[:n_train]
    train_technical = technical_features.iloc[:n_train]
    train_sentiment = sentiment_features.iloc[:n_train]
    
    val_prices = price_data.iloc[n_train:n_train + n_val]
    val_technical = technical_features.iloc[n_train:n_train + n_val]
    val_sentiment = sentiment_features.iloc[n_train:n_train + n_val]
    
    test_prices = price_data.iloc[n_train + n_val:]
    test_technical = technical_features.iloc[n_train + n_val:]
    test_sentiment = sentiment_features.iloc[n_train + n_val:]
    
    # Print split info
    print(f"\n{'='*70}")
    print("DATA SPLIT SUMMARY:")
    print(f"{'='*70}")
    print(f"✓ Train: {len(train_prices):3d} months ({train_ratio:.0%}) | "
          f"{train_prices.index[0].strftime('%Y-%m')} to {train_prices.index[-1].strftime('%Y-%m')}")
    print(f"✓ Val:   {len(val_prices):3d} months ({val_ratio:.0%}) | "
          f"{val_prices.index[0].strftime('%Y-%m')} to {val_prices.index[-1].strftime('%Y-%m')}")
    print(f"✓ Test:  {len(test_prices):3d} months ({test_ratio:.0%}) | "
          f"{test_prices.index[0].strftime('%Y-%m')} to {test_prices.index[-1].strftime('%Y-%m')}")
    print(f"{'='*70}")
    
    # Verify no overlap
    assert train_prices.index[-1] < val_prices.index[0], "Train/Val overlap!"
    assert val_prices.index[-1] < test_prices.index[0], "Val/Test overlap!"
    
    print("\n✓ No data leakage - splits are properly separated")
    
    return {
        'train': (train_prices, train_technical, train_sentiment),
        'val': (val_prices, val_technical, val_sentiment),
        'test': (test_prices, test_technical, test_sentiment)
    }