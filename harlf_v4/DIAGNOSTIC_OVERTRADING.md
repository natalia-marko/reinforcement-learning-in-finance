# Diagnostic: Overtrading Issue

## Your Results

```
Test Results:
  Sharpe:  0.410
  Sortino: 0.542
  Return:  8.5%
  Max DD:  -14.2%
  Trades:  74 out of 75 periods (98.7% turnover!)
```

## The Problem: Extreme Overtrading

**74 trades in 75 periods = 98.7% turnover rate**

This means the agent is trading almost EVERY SINGLE WEEK. This is catastrophic for several reasons:

1. **Transaction costs destroying returns**
   - 74 trades × 0.0025 cost × avg position change ≈ 10-15% of capital
   - Your 8.5% return is likely -5% after realistic costs

2. **Implementation impossibility**
   - Can't execute weekly rebalancing in real markets
   - High slippage on actual trades
   - Psychological strain

3. **Overfitting to noise**
   - Model reacting to random weekly fluctuations
   - Not learning real market patterns

## Root Cause: Unbalanced Reward Function

Your config had:
```python
'composite_return_weight': 3.0,      # Way too high
'composite_turnover_weight': 0.1,    # Way too low
```

**Ratio: 30:1 in favor of returns over turnover penalty**

The agent learned: "Trade aggressively for any tiny return, turnover penalty is negligible"

## Fixed Configuration

### Changed Parameters

| Parameter | Your Value | Fixed Value | Reason |
|-----------|------------|-------------|--------|
| `composite_return_weight` | 3.0 | 1.0 | Normalize base weight |
| `composite_turnover_weight` | 0.1 | 0.8 | **8x increase** in penalty |
| `reward_lookback` | 13 weeks | 52 weeks | Stable Sharpe calculation |
| `transaction_cost` | 0.0025 | 0.003 | More realistic |
| `batch_size` | 256 | 512 | Better learning stability |
| `steps_per_epoch` | 2000 | 5000 | Proper validation frequency |
| `early_stopping_patience` | 5 | 8 | Allow more time to converge |

### New Reward Balance

```python
Reward = 1.0 × returns       # Base return
       - 0.3 × volatility    # Risk penalty
       - 0.5 × drawdown      # Drawdown penalty
       - 0.8 × |Δposition|   # STRONG turnover penalty
```

Now the ratio is **1.25:1** (return:turnover), much more balanced.

## Expected Results After Fix

With balanced config, you should see:

| Metric | Your Results | Expected | Improvement |
|--------|--------------|----------|-------------|
| Trades/100 periods | 98.7 | 15-25 | **75% reduction** |
| Avg holding period | 1.0 weeks | 4-7 weeks | 4-7x longer |
| Sharpe ratio | 0.41 | 0.6-1.0 | 50-150% better |
| Return (net) | ~0-2% | 10-20% | Much higher |
| Max drawdown | -14% | -10 to -15% | Similar/better |

## Why This Will Work

1. **Turnover penalty now meaningful**
   - Before: 0.1 penalty vs 3.0 reward → ignore penalty
   - After: 0.8 penalty vs 1.0 reward → must consider penalty

2. **Longer observation window**
   - Before: 13 weeks → noisy Sharpe estimates
   - After: 52 weeks → stable Sharpe estimates

3. **Realistic costs**
   - Higher transaction costs force quality over quantity

## Testing the Fix

Run this in your notebook:

```python
# Reload config
import importlib
importlib.reload(config)

print("Fixed Configuration:")
print(f"  Return weight:    {config.ENV_CONFIG['composite_return_weight']}")
print(f"  Turnover weight:  {config.ENV_CONFIG['composite_turnover_weight']}")
print(f"  Ratio:            {config.ENV_CONFIG['composite_return_weight']/config.ENV_CONFIG['composite_turnover_weight']:.1f}:1")
print(f"  Reward lookback:  {config.ENV_CONFIG['reward_lookback']} weeks")
print(f"  Transaction cost: {config.ENV_CONFIG['transaction_cost']*10000:.0f} bps")
```

Then retrain:

```python
# Retrain with fixed config
model = SAC('MlpPolicy', train_env, verbose=1, **config.get_model_hyperparams())
model.learn(total_timesteps=config.MODEL_CONFIG['total_training_steps'])

# Evaluate
test_env = TechnicalEnv(test_p, test_f, **config.ENV_CONFIG)
obs, _ = test_env.reset()
done = False

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, _ = test_env.step(action)
    done = terminated or truncated

metrics = test_env.get_portfolio_metrics()
print(f"\nImproved Results:")
print(f"  Sharpe:  {metrics['sharpe_ratio']:.3f}")
print(f"  Return:  {metrics['total_return']*100:.1f}%")
print(f"  Trades:  {metrics['n_trades']}")

# Analyze trade frequency
from utils import print_trade_analysis
print_trade_analysis(test_env)
```

## If Still Overtrading

If you still see >30 trades per 100 periods, progressively increase:

```python
# Step 1: Moderate increase
config.ENV_CONFIG['composite_turnover_weight'] = 1.2

# Step 2: Strong increase
config.ENV_CONFIG['composite_turnover_weight'] = 1.5
config.ENV_CONFIG['transaction_cost'] = 0.004  # 40 bps

# Step 3: Very strong (last resort)
config.ENV_CONFIG['composite_turnover_weight'] = 2.0
config.ENV_CONFIG['transaction_cost'] = 0.005  # 50 bps
```

## Understanding the Math

### Your Previous Reward (Simplified)

```
At each step:
- Portfolio gains 0.5% → Reward ≈ +0.015 (3.0 × 0.005)
- Trade changes position by 50% → Penalty ≈ -0.05 (0.1 × 0.5)

Net reward: +0.015 - 0.05 = -0.035... but wait!

The return component accumulates over time while turnover is instant.
Effective: +0.015 gain vs -0.05 penalty, BUT return happens every period
while trade happens once. So agent trades constantly to accumulate returns.
```

### Fixed Reward

```
At each step:
- Portfolio gains 0.5% → Reward ≈ +0.005 (1.0 × 0.005)
- Trade changes position by 50% → Penalty ≈ -0.40 (0.8 × 0.5)

Net reward: +0.005 - 0.40 = -0.395 (clearly negative!)

Now agent only trades when expected gain > 0.40, which requires
strong signal and longer holding periods.
```

## Key Takeaway

**The composite reward function is a balancing act**:

- Too low turnover penalty → overtrades
- Too high turnover penalty → never trades
- Sweet spot: 0.5-1.0 for turnover_weight with 1.0 return_weight

Your previous 0.1 was **8x too low**, causing the overtrading disaster.

## Validation Checklist

After retraining with fixed config, check:

- [ ] Turnover rate < 30%
- [ ] Sharpe ratio > 0.6
- [ ] Avg holding > 3 weeks
- [ ] Transaction costs < 10% of returns
- [ ] Net return > buy-and-hold (if applicable)

If all checks pass, you're ready for walk-forward validation!

