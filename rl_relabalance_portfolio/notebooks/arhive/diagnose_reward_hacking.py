"""
Diagnostic script to understand why training Sharpe is 2-3.5 but validation is 0.5-1.0
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load data
DATA_DIR = Path('data/processed')
train_df = pd.read_parquet(DATA_DIR / 'train.parquet')

with open(DATA_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

print("="*80)
print("DIAGNOSTIC: Why High Training Sharpe vs Low Validation Sharpe?")
print("="*80)

# Simulate fold 2 (the one tested)
dates = sorted(train_df['date'].unique())
n_dates = len(dates)

# Fold 2: Train on first 180 periods, validate on next 64
fold_2_train_dates = dates[:180]
fold_2_val_dates = dates[180:180+64]

train_data_fold2 = train_df[train_df['date'].isin(fold_2_train_dates)]
val_data_fold2 = train_df[train_df['date'].isin(fold_2_val_dates)]

print(f"\nFold 2 Structure:")
print(f"  Train: {len(fold_2_train_dates)} periods ({fold_2_train_dates[0].date()} to {fold_2_train_dates[-1].date()})")
print(f"  Val:   {len(fold_2_val_dates)} periods ({fold_2_val_dates[0].date()} to {fold_2_val_dates[-1].date()})")

# Analyze returns during training vs validation periods
tickers = metadata['tickers']

print(f"\n{'='*80}")
print("ASSET RETURNS COMPARISON")
print(f"{'='*80}")

train_returns = []
val_returns = []

for ticker in tickers:
    # Training period returns
    train_ticker = train_data_fold2[train_data_fold2['ticker'] == ticker].sort_values('date')
    if len(train_ticker) > 1:
        train_total_return = (train_ticker['close'].iloc[-1] / train_ticker['close'].iloc[0]) - 1
        train_returns.append(train_total_return)

    # Validation period returns
    val_ticker = val_data_fold2[val_data_fold2['ticker'] == ticker].sort_values('date')
    if len(val_ticker) > 1:
        val_total_return = (val_ticker['close'].iloc[-1] / val_ticker['close'].iloc[0]) - 1
        val_returns.append(val_total_return)

    print(f"{ticker:6s}: Train={train_total_return:7.2%}  Val={val_total_return:7.2%}")

print(f"\nAverage: Train={np.mean(train_returns):7.2%}  Val={np.mean(val_returns):7.2%}")
print(f"Std Dev: Train={np.std(train_returns):7.2%}  Val={np.std(val_returns):7.2%}")

# Check volatility
print(f"\n{'='*80}")
print("VOLATILITY COMPARISON")
print(f"{'='*80}")

for ticker in tickers:
    train_ticker = train_data_fold2[train_data_fold2['ticker'] == ticker].sort_values('date')
    val_ticker = val_data_fold2[val_data_fold2['ticker'] == ticker].sort_values('date')

    if len(train_ticker) > 1:
        train_log_returns = np.log(train_ticker['close'] / train_ticker['close'].shift(1)).dropna()
        train_vol = train_log_returns.std() * np.sqrt(52)
    else:
        train_vol = 0

    if len(val_ticker) > 1:
        val_log_returns = np.log(val_ticker['close'] / val_ticker['close'].shift(1)).dropna()
        val_vol = val_log_returns.std() * np.sqrt(52)
    else:
        val_vol = 0

    print(f"{ticker:6s}: Train Vol={train_vol:6.2%}  Val Vol={val_vol:6.2%}")

# Check regime features
print(f"\n{'='*80}")
print("REGIME INDICATORS COMPARISON")
print(f"{'='*80}")

regime_features = ['bull_market', 'bear_market', 'market_concentration']

for feat in regime_features:
    train_mean = train_data_fold2[feat].mean()
    val_mean = val_data_fold2[feat].mean()
    train_std = train_data_fold2[feat].std()
    val_std = val_data_fold2[feat].std()

    print(f"{feat:25s}: Train μ={train_mean:6.3f} σ={train_std:6.3f}  |  Val μ={val_mean:6.3f} σ={val_std:6.3f}")

# Simulate what a simple equal-weight portfolio would achieve
print(f"\n{'='*80}")
print("EQUAL WEIGHT PORTFOLIO SIMULATION")
print(f"{'='*80}")

def simulate_equal_weight(data, tickers, rebalance_freq=4):
    """Simulate equal weight portfolio"""
    dates_list = sorted(data['date'].unique())
    n_assets = len(tickers)
    weights = np.ones(n_assets) / n_assets
    portfolio_value = 100000.0
    returns = []

    for i in range(len(dates_list) - 1):
        current_date = dates_list[i]
        next_date = dates_list[i + 1]

        # Get price changes
        price_changes = []
        for ticker in tickers:
            curr_price = data[(data['date'] == current_date) & (data['ticker'] == ticker)]['close'].iloc[0]
            next_price = data[(data['date'] == next_date) & (data['ticker'] == ticker)]['close'].iloc[0]
            price_changes.append(np.log(next_price / curr_price))

        # Portfolio return
        portfolio_return = np.sum(weights * np.array(price_changes))
        returns.append(portfolio_return)
        portfolio_value *= np.exp(portfolio_return)

    # Calculate stats
    returns = np.array(returns)
    total_return = (portfolio_value / 100000.0) - 1.0
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    sharpe = (mean_return / std_return) * np.sqrt(52) if std_return > 0 else 0
    vol = std_return * np.sqrt(52)

    return {
        'total_return': total_return,
        'sharpe': sharpe,
        'volatility': vol,
        'final_value': portfolio_value
    }

train_eq = simulate_equal_weight(train_data_fold2, tickers)
val_eq = simulate_equal_weight(val_data_fold2, tickers)

print("\nEqual Weight (Naive Benchmark):")
print(f"  Train: Return={train_eq['total_return']:7.2%}  Sharpe={train_eq['sharpe']:5.2f}  Vol={train_eq['volatility']:6.2%}")
print(f"  Val:   Return={val_eq['total_return']:7.2%}  Sharpe={val_eq['sharpe']:5.2f}  Vol={val_eq['volatility']:6.2%}")

# Key insight
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")

print("""
1. REWARD STRUCTURE ISSUE:
   - Current reward: simple_return (just log return, no risk penalty)
   - Agent learns: "Maximize returns at ANY cost"
   - Result: Takes huge risks that happen to work in training but fail in validation

2. WHY TRAINING SHARPE IS SO HIGH:
   - Training episodes run multiple times through the same data
   - Agent "memorizes" good periods and learns to exploit them
   - With enough training steps, it finds strategies that work specifically on training data
   - These are NOT generalizable strategies, just data-specific hacks

3. WHY VALIDATION SHARPE IS LOW:
   - Validation data is different time period with different market conditions
   - The risky strategies learned on training data don't work here
   - Agent hasn't learned to ADAPT to different regimes

4. WHY REGIME FEATURES DON'T HELP:
   - The reward function doesn't incentivize using them!
   - "simple_return" reward = "maximize returns, ignore risk"
   - Regime features would be useful for "maximize risk-adjusted returns"
   - But current reward doesn't care about risk adjustment

5. THE SMOKING GUN:
   - Training Sharpe 2-3.5 is UNREALISTIC (world's best hedge funds: 1.5-2.5)
   - This is a clear sign of overfitting/reward hacking
   - The agent isn't learning robust strategies, it's exploiting training data quirks
""")

print(f"\n{'='*80}")
print("RECOMMENDED FIXES")
print(f"{'='*80}")

print("""
IMMEDIATE ACTION (Choose ONE to start):

Option A: Switch to multi_component reward (EASIEST)
  - Already implemented in your code
  - Change: reward_type='simple_return' → 'multi_component'
  - This adds 50% Sharpe, 25% return, 25% drawdown to reward
  - Expected: Training Sharpe drops to 1.5-2.0, Validation improves to 1.2-1.5

Option B: Add explicit volatility penalty to observation (COMPLEMENT)
  - Add portfolio volatility as explicit signal in state
  - Agent can learn "when vol is high, reduce risk"
  - This makes regime features more useful

Option C: Reduce training timesteps / early stopping (QUICK FIX)
  - Your plots show peak at 16-62% of training
  - Stop training earlier (100k steps instead of 200k)
  - Or reduce patience from 20 to 10
  - Expected: Less overfitting to training data

CRITICAL UNDERSTANDING:
Your model CAN learn from regime indicators, but only if:
1. The reward function incentivizes risk-aware behavior
2. The observation includes explicit risk signals
3. Training doesn't overfit to specific data patterns

The current "simple_return" reward is like telling a race car driver:
"Go as fast as possible" without mentioning "and don't crash"
Result: They crash. A lot.
""")
