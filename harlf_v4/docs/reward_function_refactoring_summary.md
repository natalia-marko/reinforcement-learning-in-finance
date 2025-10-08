# Reward Function Refactoring Summary

## Overview
Refactored reward functions across all agent environments to better align with risk-adjusted performance metrics and encourage exploration.

## Changes by Agent Type

### 1. Base Agents (Sentiment & Technical)

**Reward Function**: Risk-Adjusted Returns (Sharpe-like ratio)

**Implementation**:
```python
# Calculate rolling Sharpe-like ratio
recent_returns = self.returns_history[-self.vol_window:]
if len(recent_returns) > 1:
    mean_return = np.mean(recent_returns)
    volatility = np.std(recent_returns)
    if volatility > 1e-6:
        reward = mean_return / volatility
    else:
        reward = mean_return
else:
    reward = portfolio_return
```

**Key Features**:
- Uses rolling window (default 10 periods) for Sharpe-like calculation
- Rewards consistent returns while penalizing volatility
- Falls back to raw return when insufficient history
- More stable training signal than raw returns

**New Parameters**:
- `vol_window`: Rolling window size for risk calculation (default: 10)

---

### 2. Super Agent

**Reward Function**: Complex Multi-Component Reward

**Formula**:
```
reward = alpha1 * log_returns - alpha2 * mdd - alpha3 * volatility + exploration_bias
```

**Components**:
- **Log Returns** (alpha1 = 1.0): Primary return signal, scaled by alpha1
- **Maximum Drawdown** (alpha2 = 0.5): Penalizes peak-to-current losses
- **Volatility** (alpha3 = 0.5): Penalizes return instability
- **Exploration Bias** (0.001): Small positive constant to encourage exploration

**Implementation Details**:
```python
# Maximum drawdown calculation
portfolio_series = pd.Series(self.portfolio_history)
peak = portfolio_series.expanding().max()
drawdown = (portfolio_series - peak) / peak
current_mdd = abs(drawdown.iloc[-1])

# Volatility calculation
volatility = np.std(self.returns_history)

# Composite reward
reward = (self.alpha1 * portfolio_log_return - 
         self.alpha2 * current_mdd - 
         self.alpha3 * volatility + 
         self.exploration_bias)
```

**New Parameters**:
- `alpha1`: Weight for log returns (default: 1.0)
- `alpha2`: Weight for MDD penalty (default: 0.5)
- `alpha3`: Weight for volatility penalty (default: 0.5)
- `exploration_bias`: Small positive constant (default: 0.001)

---

### 3. Meta Agent

**Reward Function**: Identical to Super Agent

Uses the same complex multi-component reward structure as the Super Agent.

**Formula**:
```
reward = alpha1 * log_returns - alpha2 * mdd - alpha3 * volatility + exploration_bias
```

**Same parameters as Super Agent**

---

## Rationale

### Base Agents
- **Sharpe-like ratio** provides a better learning signal than raw returns
- Encourages consistent performance over volatile gains
- Rolling window approach adapts to recent performance
- More robust to market regime changes

### Super & Meta Agents
- **Multi-component reward** balances multiple objectives:
  1. Maximize returns (log returns)
  2. Minimize risk exposure (volatility)
  3. Avoid catastrophic losses (maximum drawdown)
  4. Encourage exploration (small positive bias)

- **Exploration bias** prevents premature convergence to suboptimal policies
- **Tunable alphas** allow customization based on risk preferences
- **MDD penalty** specifically addresses tail risk concerns

---

## Additional Changes

### All Environments
1. Added `returns_history` tracking for risk calculations
2. Added `log_returns_history` tracking where needed
3. Properly initialized new tracking lists in `reset()` methods
4. Maintained backward compatibility with existing code

---

## Parameter Tuning Guidance

### Base Agents
- **vol_window** (10): Increase for smoother signals, decrease for faster adaptation
- Typical range: 5-20 periods

### Super & Meta Agents
- **alpha1** (1.0): Baseline return weight, typically kept at 1.0
- **alpha2** (0.5): Increase to be more conservative, decrease for more risk
- **alpha3** (0.5): Increase to prefer stable returns, decrease to tolerate volatility
- **exploration_bias** (0.001): Small value, typically 0.0001-0.01

**Conservative Profile**: alpha2=1.0, alpha3=1.0
**Aggressive Profile**: alpha2=0.2, alpha3=0.2
**Balanced Profile**: alpha2=0.5, alpha3=0.5 (default)

---

## Testing Recommendations

1. Compare training curves with old vs new reward functions
2. Monitor:
   - Convergence speed
   - Final policy performance
   - Out-of-sample Sharpe ratio
   - Maximum drawdown
3. Experiment with different alpha combinations
4. Validate that exploration bias doesn't dominate the signal

---

## Files Modified

1. `sentiment_enviroment.py`
2. `technical_enviroment.py`
3. `super_agent_envoriment.py`
4. `meta_agent_enviroment.py`

All files passed linter checks with no errors.

