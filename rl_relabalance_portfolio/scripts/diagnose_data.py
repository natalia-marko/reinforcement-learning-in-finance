"""
Quick diagnostic script to check validation data
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from rl_system import PortfolioEnv, create_walk_forward_folds

# Load data
DATA_DIR = Path('data/processed')
train_df = pd.read_parquet(DATA_DIR / 'train.parquet')

with open(DATA_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['feature_cols']
tickers = metadata['tickers']

print("=" * 80)
print("DATA DIAGNOSTIC")
print("=" * 80)

print(f"\nTrain data shape: {train_df.shape}")
print(f"Columns: {train_df.columns.tolist()[:10]}...")
print(f"Tickers: {tickers}")
print(f"Features: {len(feature_cols)} features")

# Create folds
folds = create_walk_forward_folds(train_df, n_folds=5)

# Check first validation fold
fold_0 = folds[0]
val_data = fold_0['val']

print(f"\n{'=' * 80}")
print(f"FOLD 0 VALIDATION DATA")
print(f"{'=' * 80}")
print(f"Val data shape: {val_data.shape}")
print(f"Val date range: {val_data['date'].min()} to {val_data['date'].max()}")
print(f"Val unique dates: {len(val_data['date'].unique())}")
print(f"Val unique tickers: {val_data['ticker'].unique().tolist()}")

# Check for price data
print(f"\n{'=' * 80}")
print(f"PRICE DATA CHECK")
print(f"{'=' * 80}")

for ticker in tickers:
    ticker_data = val_data[val_data['ticker'] == ticker]
    if 'close' in ticker_data.columns:
        prices = ticker_data['close'].values
        print(f"\n{ticker}:")
        print(f"  Rows: {len(ticker_data)}")
        print(f"  Close prices: min={prices.min():.2f}, max={prices.max():.2f}, mean={prices.mean():.2f}")
        print(f"  Price changes: {np.diff(prices)[:5]}")
        print(f"  Has NaN: {ticker_data['close'].isna().any()}")

        # Calculate log returns
        log_returns = np.log(prices[1:] / prices[:-1])
        print(f"  Log returns: min={log_returns.min():.4f}, max={log_returns.max():.4f}, mean={log_returns.mean():.4f}")
        print(f"  Log returns std: {log_returns.std():.4f}")
    else:
        print(f"\n{ticker}: NO CLOSE COLUMN!")

# Test environment creation
print(f"\n{'=' * 80}")
print(f"ENVIRONMENT TEST")
print(f"{'=' * 80}")

try:
    env = PortfolioEnv(
        data=val_data,
        feature_cols=feature_cols,
        tickers=tickers,
        rebalance_frequency=4,
        transaction_cost=0.00025,
        max_weight_per_asset=0.4,
        reward_lookback=4,
        seed=42
    )

    print(f"Environment created successfully!")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action space: {env.action_space}")
    print(f"  N periods: {env.n_periods}")
    print(f"  Initial capital: {env.initial_capital}")
    print(f"  Dates: {len(env.dates)}")

    # Test reset
    obs, info = env.reset()
    print(f"\nAfter reset:")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Portfolio value: {env.portfolio_value}")
    print(f"  Weights: {env.weights}")
    print(f"  Info: {info}")

    # Test one step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nAfter 1 step:")
    print(f"  Reward: {reward}")
    print(f"  Portfolio value: {env.portfolio_value}")
    print(f"  Period return: {info.get('period_return', 'N/A')}")
    print(f"  Terminated: {terminated}")
    print(f"  Info: {info}")

    # Run full episode
    env.reset()
    total_steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        if terminated:
            break

    print(f"\nAfter full episode:")
    print(f"  Total steps: {total_steps}")
    print(f"  Portfolio returns length: {len(env.portfolio_returns)}")
    print(f"  Portfolio returns sample: {env.portfolio_returns[:5]}")

    stats = env.get_portfolio_stats()
    print(f"\nFinal stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

except Exception as e:
    print(f"ERROR creating/running environment: {e}")
    import traceback
    traceback.print_exc()
