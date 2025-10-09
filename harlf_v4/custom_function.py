import pandas as pd
import numpy as np


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
                               train_ratio=0.60, val_ratio=0.20):
    """
    Split data chronologically for machine learning training.
    
    Args:
        price_data: DataFrame of asset prices (rows=dates, cols=assets)
        technical_features: DataFrame of technical indicators
        sentiment_features: DataFrame of sentiment features
        train_ratio: Proportion for training set (default 0.60)
        val_ratio: Proportion for validation set (default 0.20)
        
    Returns:
        dict: Contains 'train', 'val', 'test' tuples of (prices, technical, sentiment)
        
    Note: test_ratio is implicit (1 - train_ratio - val_ratio)
    """
    print("\n" + "="*70)
    print("DATA SPLITTING")
    print("="*70)
    
    # Input validation
    assert 0 < train_ratio < 1, "train_ratio must be between 0 and 1"
    assert 0 < val_ratio < 1, "val_ratio must be between 0 and 1"
    assert train_ratio + val_ratio < 1, "train_ratio + val_ratio must be < 1"
    
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
        print(f"\nWARNING: Only {n_total} months of data!")
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
    test_ratio = 1 - train_ratio - val_ratio
    print(f"\n{'='*70}")
    print("DATA SPLIT SUMMARY:")
    print(f"{'='*70}")
    print(f"Train: {len(train_prices):3d} months ({train_ratio:.0%}) | "
          f"{train_prices.index[0].strftime('%Y-%m')} to {train_prices.index[-1].strftime('%Y-%m')}")
    print(f"Val:   {len(val_prices):3d} months ({val_ratio:.0%}) | "
          f"{val_prices.index[0].strftime('%Y-%m')} to {val_prices.index[-1].strftime('%Y-%m')}")
    print(f"Test:  {len(test_prices):3d} months ({test_ratio:.0%}) | "
          f"{test_prices.index[0].strftime('%Y-%m')} to {test_prices.index[-1].strftime('%Y-%m')}")
    print(f"{'='*70}")
    
    # Verify no overlap
    assert train_prices.index[-1] < val_prices.index[0], "Train/Val overlap!"
    assert val_prices.index[-1] < test_prices.index[0], "Val/Test overlap!"
    
    print("\nNo data leakage - splits are properly separated")
    
    return {
        'train': (train_prices, train_technical, train_sentiment),
        'val': (val_prices, val_technical, val_sentiment),
        'test': (test_prices, test_technical, test_sentiment)
    }


def create_performance_comparison_table(super_metrics, meta_metrics, test_periods):
    """
    Create comprehensive performance comparison table for hierarchical agents.
    
    Args:
        super_metrics: Dictionary of metrics from super agent test evaluation
        meta_metrics: Dictionary of metrics from meta agent test evaluation
        test_periods: Number of test periods (months)
        
    Returns:
        DataFrame with formatted comparison including improvement percentages
    """
    comparison_data = {
        'Metric': [
            'Sharpe Ratio', 
            'Total Return (%)', 
            'Annualized Return (%)', 
            'Max Drawdown (%)', 
            'Volatility (%)', 
            'Win Rate (%)', 
            'Final Value ($)'
        ],
        'Super Agent': [
            f"{super_metrics['sharpe_ratio']:.3f}",
            f"{super_metrics['total_return']*100:.2f}",
            f"{(np.power(1 + super_metrics['total_return'], 12/test_periods) - 1)*100:.2f}",
            f"{super_metrics['max_drawdown']*100:.2f}",
            f"{super_metrics['volatility']*100:.2f}",
            f"{super_metrics['win_rate']*100:.2f}",
            f"{super_metrics['final_value']:,.2f}"
        ],
        'Meta Agent': [
            f"{meta_metrics['sharpe_ratio']:.3f}",
            f"{meta_metrics['total_return']*100:.2f}",
            f"{(np.power(1 + meta_metrics['total_return'], 12/test_periods) - 1)*100:.2f}",
            f"{meta_metrics['max_drawdown']*100:.2f}",
            f"{meta_metrics['volatility']*100:.2f}",
            f"{meta_metrics['win_rate']*100:.2f}",
            f"{meta_metrics['final_value']:,.2f}"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate improvement percentages
    improvement = []
    for i in range(len(comparison_df)):
        metric_name = comparison_df.iloc[i, 0]
        
        # Extract numeric values
        super_val = float(comparison_df.iloc[i, 1].replace(',', '').replace('%', '').replace('$', ''))
        meta_val = float(comparison_df.iloc[i, 2].replace(',', '').replace('%', '').replace('$', ''))
        
        # Calculate improvement based on metric type
        if 'Drawdown' in metric_name:
            # Lower is better for drawdown
            imp = ((super_val - meta_val) / super_val * 100) if super_val != 0 else 0
        else:
            # Higher is better for all other metrics
            imp = ((meta_val - super_val) / super_val * 100) if super_val != 0 else 0
        
        improvement.append(f"{imp:+.1f}%")
    
    comparison_df['Improvement'] = improvement
    
    return comparison_df


def print_final_statistics(super_metrics, meta_metrics, test_periods):
    """
    Print comprehensive final statistics comparing super and meta agents.
    
    Args:
        super_metrics: Dictionary of metrics from super agent test evaluation
        meta_metrics: Dictionary of metrics from meta agent test evaluation
        test_periods: Number of test periods (months)
    """
    comparison_df = create_performance_comparison_table(super_metrics, meta_metrics, test_periods)
    
    print("\n" + "="*80)
    print("HIERARCHICAL AGENT TEST SET PERFORMANCE COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)
    print("\nNote: Positive improvement % is better for all metrics")
    print("      (including Max Drawdown, where reduction = improvement)")
    
    # Print key highlights
    sharpe_improvement = ((meta_metrics['sharpe_ratio'] - super_metrics['sharpe_ratio']) / 
                          super_metrics['sharpe_ratio'] * 100)
    return_improvement = ((meta_metrics['total_return'] - super_metrics['total_return']) / 
                          super_metrics['total_return'] * 100)
    
    print("\n" + "="*80)
    print("KEY HIGHLIGHTS")
    print("="*80)
    print(f"Meta Agent Sharpe Ratio Improvement: {sharpe_improvement:+.1f}%")
    print(f"Meta Agent Total Return Improvement: {return_improvement:+.1f}%")
    print(f"Meta Agent Final Portfolio Value: ${meta_metrics['final_value']:,.2f}")
    print("="*80)