# Managing Trade Frequency in RL Trading Systems

## Problem: Excessive Trading

High-frequency trading in RL agents causes:
1. **High transaction costs** - Eats into profits
2. **Implementation challenges** - Hard to execute in real markets
3. **Overfitting** - Model reacting to noise, not signal
4. **Slippage** - Real execution worse than simulated

**Rule of thumb for weekly data**: 
- Good: 5-15 trades per 100 periods (~5-15% turnover)
- Acceptable: 15-30 trades per 100 periods
- Too high: >30 trades per 100 periods

## Solutions (Ranked by Effectiveness)

### 1. Use Composite Reward Function (BEST)

**Current Change Applied:**
```python
'reward_function': 'composite',           # Changed from 'differential_sharpe'
'composite_turnover_weight': 0.5,        # Increased from 0.1
```

**How it works:**
- Directly penalizes position changes in the reward
- Agent learns that trading has a cost beyond transaction fees
- Most effective solution

**Tuning turnover weight:**
- Start: 0.5 (moderate penalty)
- If still too many trades: 1.0-2.0 (strong penalty)
- If too few trades: 0.2-0.3 (light penalty)

### 2. Increase Transaction Costs

**Current Change Applied:**
```python
'transaction_cost': 0.003,  # Increased from 0.001 (10 bps → 30 bps)
```

**Guidelines:**
- Real-world costs (including slippage): 20-50 bps
- Conservative modeling: 30-40 bps
- Aggressive (assumes good execution): 10-20 bps

**Effect:** Agent learns trades must be worthwhile to overcome costs.

### 3. Position Discretization

Add discrete action space to prevent micro-adjustments:

```python
# Add to tech_env_module.py step() method
def _discretize_position(self, target_position: float, num_levels: int = 5) -> float:
    """
    Discretize position to fixed levels to reduce micro-trading.
    
    Example with 5 levels and max_position=0.7:
    -0.7, -0.35, 0.0, 0.35, 0.7
    """
    levels = np.linspace(-self.max_position, self.max_position, num_levels)
    closest_level = levels[np.argmin(np.abs(levels - target_position))]
    return closest_level
```

**Usage:** Call this before applying position change in `step()`.

### 4. Minimum Trade Threshold

Only execute trades above a minimum size:

```python
# Add to tech_env_module.py step() method
MIN_TRADE_SIZE = 0.10  # Only trade if position change > 10%

position_change = target_position - self.position

if abs(position_change) < MIN_TRADE_SIZE:
    # Skip trade, keep current position
    target_position = self.position
    position_change = 0.0
```

### 5. Trade Cooldown Period

Prevent trading on consecutive periods:

```python
# Add to TechnicalEnv __init__:
self.last_trade_step = -999
self.cooldown_periods = 2  # Must wait 2 weeks between trades

# In step() method:
if (self.current_step - self.last_trade_step) < self.cooldown_periods:
    # Force no trade during cooldown
    target_position = self.position
    position_change = 0.0
else:
    # Allow trade
    if abs(position_change) > 1e-6:
        self.last_trade_step = self.current_step
```

## Recommended Configuration Profiles

### Conservative (Fewer Trades, More Stable)
```python
ENV_CONFIG = {
    'transaction_cost': 0.004,              # 40 bps
    'reward_function': 'composite',
    'composite_return_weight': 1.0,
    'composite_risk_weight': 0.3,
    'composite_drawdown_weight': 0.5,
    'composite_turnover_weight': 1.5,       # Strong penalty
}
```
**Expected:** 8-15 trades per 100 periods

### Balanced (Current Configuration)
```python
ENV_CONFIG = {
    'transaction_cost': 0.003,              # 30 bps
    'reward_function': 'composite',
    'composite_return_weight': 1.0,
    'composite_risk_weight': 0.3,
    'composite_drawdown_weight': 0.5,
    'composite_turnover_weight': 0.5,       # Moderate penalty
}
```
**Expected:** 15-25 trades per 100 periods

### Active (More Trades, Higher Returns Potential)
```python
ENV_CONFIG = {
    'transaction_cost': 0.002,              # 20 bps
    'reward_function': 'composite',
    'composite_return_weight': 1.0,
    'composite_risk_weight': 0.3,
    'composite_drawdown_weight': 0.5,
    'composite_turnover_weight': 0.2,       # Light penalty
}
```
**Expected:** 25-40 trades per 100 periods

## How to Analyze Trade Frequency

### In Your Notebook

```python
# After training and evaluation
metrics = test_env.get_portfolio_metrics()
n_trades = metrics['n_trades']
n_periods = len(test_env.portfolio_history) - 1

# Calculate turnover rate
turnover_rate = (n_trades / n_periods) * 100

print(f"\nTrade Frequency Analysis:")
print(f"  Total trades:    {n_trades}")
print(f"  Total periods:   {n_periods}")
print(f"  Turnover rate:   {turnover_rate:.1f}%")
print(f"  Avg holding:     {n_periods/max(n_trades, 1):.1f} periods")

if turnover_rate > 30:
    print("  ⚠️  WARNING: High turnover rate")
elif turnover_rate < 5:
    print("  ⚠️  WARNING: Very low activity (might be undertrading)")
else:
    print("  ✓ Acceptable turnover rate")
```

### Visualize Trade Frequency

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Portfolio value with trade markers
portfolio_values = test_env.portfolio_history
trade_steps = [t['step'] for t in test_env.trade_history]

ax1.plot(portfolio_values, label='Portfolio Value')
ax1.scatter(trade_steps, [portfolio_values[s] for s in trade_steps], 
           color='red', marker='v', s=50, alpha=0.7, label='Trades')
ax1.set_title('Portfolio Value with Trade Markers')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Position changes
position_history = test_env.position_history
position_changes = np.diff(position_history)

ax2.bar(range(len(position_changes)), position_changes, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_title('Position Changes Over Time')
ax2.set_ylabel('Position Change')
ax2.set_xlabel('Time Step')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Troubleshooting

### Issue: Still Too Many Trades After Changes

**Try in order:**

1. **Increase turnover weight**
   ```python
   'composite_turnover_weight': 2.0  # Very strong penalty
   ```

2. **Add minimum trade threshold**
   - See "Solution 4" above
   - Start with 0.10 (10% minimum change)

3. **Use position discretization**
   - See "Solution 3" above
   - Use 3-5 discrete levels

4. **Add trade cooldown**
   - See "Solution 5" above
   - Start with 2-3 period cooldown

### Issue: Too Few Trades (Underfitting)

**Signs:**
- <5 trades per 100 periods
- Mostly staying at zero position
- Low Sharpe despite low volatility

**Solutions:**

1. **Reduce turnover penalty**
   ```python
   'composite_turnover_weight': 0.1
   ```

2. **Reduce transaction costs**
   ```python
   'transaction_cost': 0.001  # Back to 10 bps
   ```

3. **Switch reward function**
   ```python
   'reward_function': 'sortino'  # More aggressive
   ```

### Issue: Trades Not Improving Performance

This often indicates the model is overfitting to noise. Solutions:

1. **Increase training data** (longer train window)
2. **Add regularization** (L2 penalty on network)
3. **Reduce network size** (already at [256,256])
4. **Use simpler features** (remove correlated features)

## Post-Processing: Trade Filtering

Apply after getting model predictions:

```python
def filter_small_trades(positions, threshold=0.05):
    """
    Remove trades smaller than threshold to reduce turnover.
    """
    filtered = [positions[0]]
    
    for pos in positions[1:]:
        if abs(pos - filtered[-1]) < threshold:
            filtered.append(filtered[-1])  # Keep previous position
        else:
            filtered.append(pos)  # Accept new position
    
    return filtered

# Apply during evaluation
original_positions = []
obs, _ = test_env.reset()

# Collect all positions first
for step in range(len(test_env.price_data) - 1):
    action, _ = model.predict(obs, deterministic=True)
    original_positions.append(action[0] * test_env.max_position)
    obs, _, _, _, _ = test_env.step(action)

# Filter and re-evaluate
filtered_positions = filter_small_trades(original_positions, threshold=0.10)
```

## Best Practice Workflow

1. **Start with conservative settings** (current config is good)
2. **Train and evaluate**
3. **Analyze trade frequency** (use visualization above)
4. **If too many trades:**
   - First increase `composite_turnover_weight`
   - Then increase `transaction_cost`
   - Finally add mechanical filters
5. **Re-train and compare** Sharpe ratios
6. **Choose best tradeoff** between frequency and performance

## Target Metrics for Weekly Trading

| Metric | Good | Acceptable | Poor |
|--------|------|------------|------|
| Trades per 100 periods | 8-15 | 15-25 | >30 |
| Avg holding period | >7 weeks | 4-7 weeks | <4 weeks |
| Transaction cost impact | <5% of returns | 5-10% | >10% |

Remember: **In real markets, fewer quality trades beat many mediocre trades.**

