"""
Quick test of training environment
"""
import pandas as pd
import json
from pathlib import Path
from rl_system import PortfolioEnv, create_walk_forward_folds
import numpy as np

# Load data
DATA_DIR = Path('data/processed')
train_df = pd.read_parquet(DATA_DIR / 'train.parquet')

with open(DATA_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

feature_cols = metadata['feature_cols']
tickers = metadata['tickers']

# Create folds
folds = create_walk_forward_folds(train_df, n_folds=5)
fold_0 = folds[0]
train_data = fold_0['train']

print("=" * 80)
print("TESTING TRAINING ENVIRONMENT")
print("=" * 80)

print(f"\nTrain data:")
print(f"  Unique dates: {len(train_data['date'].unique())}")
print(f"  Date range: {train_data['date'].min().date()} to {train_data['date'].max().date()}")

# Create environment
env = PortfolioEnv(
    data=train_data.copy(),
    feature_cols=feature_cols,
    tickers=tickers,
    rebalance_frequency=4,
    transaction_cost=0.00025,
    max_weight_per_asset=0.4,
    reward_lookback=4,
    seed=42
)

print(f"\nEnvironment created:")
print(f"  N periods: {env.n_periods}")
print(f"  State dim: {env.state_dim}")

# Test one episode with RANDOM actions
obs, info = env.reset()
print(f"\nRunning episode with random actions...")

total_reward = 0
steps = 0
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if steps <= 5:
        print(f"  Step {steps}: reward={reward:.4f}, portfolio_value={info['portfolio_value']:.2f}")

    if terminated:
        break

print(f"\nEpisode complete:")
print(f"  Total steps: {steps}")
print(f"  Total reward: {total_reward:.4f}")
print(f"  Portfolio returns length: {len(env.portfolio_returns)}")

if len(env.portfolio_returns) > 0:
    print(f"  Portfolio returns sample: {env.portfolio_returns[:5]}")
    print(f"  Portfolio returns mean: {np.mean(env.portfolio_returns):.6f}")
    print(f"  Portfolio returns std: {np.std(env.portfolio_returns):.6f}")

stats = env.get_portfolio_stats()
print(f"\nFinal portfolio stats:")
for key, value in stats.items():
    if key == 'portfolio_value':
        print(f"  {key}: ${value:,.2f}")
    elif 'return' in key or 'drawdown' in key or 'volatility' in key:
        print(f"  {key}: {value*100:.2f}%")
    else:
        print(f"  {key}: {value:.4f}")

# Check if environment is broken
if stats['total_return'] == 0.0 and stats['sharpe_ratio'] == 0.0:
    print("\n" + "!" * 80)
    print("WARNING: Environment produced zero returns!")
    print("!" * 80)

    # Debug
    print("\nDebug info:")
    print(f"  portfolio_returns: {env.portfolio_returns}")
    print(f"  portfolio_value: {env.portfolio_value}")
    print(f"  current_step: {env.current_step}")
    print(f"  n_periods: {env.n_periods}")
