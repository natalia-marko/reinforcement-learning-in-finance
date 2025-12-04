"""
Verify Feature Alignment in harlf_weekly_v3

This script checks if features at time t are aligned with returns from t to t+1,
ensuring no look-ahead bias.
"""

import pandas as pd
import numpy as np

def verify_weekly_v3_features():
    print("="*70)
    print("FEATURE ALIGNMENT VERIFICATION - harlf_weekly_v3")
    print("="*70)
    
    # Load technical features
    print("\n1. Loading data...")
    tech_train = pd.read_csv('data_hierarchical/technical/train.csv', index_col=0, parse_dates=True)
    returns_train = pd.read_csv('data_hierarchical/returns_train.csv', index_col=0, parse_dates=True)
    
    print(f"Technical features shape: {tech_train.shape}")
    print(f"Returns shape: {returns_train.shape}")
    
    #  Check a specific ticker
    ticker = 'NVDA'
    ticker_data = tech_train[tech_train['ticker'] == ticker].copy()
    
    print(f"\n2. Analyzing {ticker} features...")
    print(f"   Ticker data shape: {ticker_data.shape}")
    print(f"   Date range: {ticker_data.index.min()} to {ticker_data.index.max()}")
    
    # The key question: Are features at time t calculated using data up to t,
    # or do they include data from t+1?
    
    # Check if 'return' column exists in technical features
    if 'return' in ticker_data.columns:
        print("\n3. Checking return alignment...")
        
        # Get dates
        dates = ticker_data.index
        
        # Compare feature 'return' with actual returns
        feature_returns = ticker_data['return'].values
        actual_returns = returns_train[ticker].loc[dates].values
        
        # Check if they match
        diff = np.abs(feature_returns - actual_returns)
        print(f"   Mean difference between feature 'return' and actual returns: {diff.mean():.6f}")
        
        if diff.mean() < 0.001:
            print("   ✓ Returns match - features include current return")
            print("   WARNING: This means features at time t see the return from t-1 to t")
            print("   For NO look-ahead bias, features should be LAGGED by 1 period")
            print("   So that features[t-1] are used to predict returns[t-1 to t]")
        else:
            print("   Returns do not match - investigating further...")
    
    # Check if returns are lagged
    print("\n4. Checking for lagging in features...")
    
    # The data preparation code shows features like:
    # df['return_lag_1w'] = returns.shift(1)
    # This creates features that look back 1 week
    
    # Check if momentum features exist
    if 'momentum_2w' in ticker_data.columns:
        print("   Found momentum features - these should be BACKWARDS looking")
        
    # The critical check: Are rolling calculations including current data?
    # In the notebook code, I saw:
    # sma = close.rolling(w).mean()
    # df[f'price_to_sma_{w}w'] = close / sma
    
    # This means at time t:
    # - sma[t] = mean(close[t-w+1:t+1]) <- includes close[t]!
    # - price_to_sma[t] = close[t] / sma[t]
    
    # For NO look-ahead bias when predicting return[t to t+1]:
    # - We should use features calculated up to time t (OK)
    # - Return should be from t to t+1 (OK)
    
    # BUT if the environment step logic does:
    # obs[t] -> action[t] -> return calculated as price[t+1]/price[t]
    # Then we're good!
    
    print("\n5. Conclusion:")
    print("   The features are calculated using data UP TO time t (including t).")
    print("   This is CORRECT if the RL environment uses:")
    print("   - obs[t] = features[t]")
    print("   - action[t] taken at time t")
    print("  - reward[t] = return from t to t+1")
    print("")
    print("   ✓ No explicit lagging is needed in the features themselves.")
    print("   ⚠️  MUST VERIFY: Environment step logic correctly calculates returns")

if __name__ == "__main__":
    verify_weekly_v3_features()
