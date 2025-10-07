# Market Regime Indicators Guide

## Overview

Market regime indicators help hierarchical agents (Super and Meta) adapt their strategies based on market conditions (bull vs bear markets).

**Key Concept**: Base agents (Sentiment & Technical) already have specialized features. Hierarchical agents need broader market context to coordinate strategies effectively.

## What Are Regime Indicators?

Simple binary indicators per asset:
- **1 (Bull)**: Price > 6-month moving average
- **0 (Bear)**: Price < 6-month moving average

## Why Only for Super & Meta Agents?

### Base Agents (Sentiment & Technical)
- Already have rich, specialized features
- Technical agent: 110+ technical indicators per asset
- Sentiment agent: 50+ sentiment features
- Don't need additional regime signals

### Hierarchical Agents (Super & Meta)
- Coordinate multiple base agents
- Need to know when to trust which agent more
- Example: Trust technical signals more in bull markets, sentiment signals more in bear markets
- Regime context helps with strategy blending

## Implementation

### 1. Function Definition

```python
# In custom_function.py
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
```

### 2. Usage in Notebook

```python
# Cell 3 in notebook
from custom_function import add_regime_indicator

# Create regime indicators
regime_indicators = add_regime_indicator(price_data, window=6)

# Shows shape and current regimes
print(f"Regime indicators shape: {regime_indicators.shape}")
print(regime_indicators.tail(3))
```

### 3. Pass to Super Agent

```python
super_env = SuperAgentEnv(
    price_data=price_data,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    regime_indicators=regime_indicators,  # Add this
    **super_agent_config
)
```

### 4. Pass to Meta Agent

```python
meta_env = MetaAgentEnv(
    price_data=price_data,
    features=technical_features,
    sentiment_features=sentiment_features,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    super_agent=super_agent,
    regime_indicators=regime_indicators,  # Add this
    **meta_agent_config
)
```

## How It Works

### Observation Space Changes

**Super Agent**:
```python
# Without regime indicators
obs = [sentiment_weights, technical_weights]  # Shape: (2 * n_assets,)

# With regime indicators (10 assets)
obs = [sentiment_weights, technical_weights, regime_indicators]  # Shape: (30,)
#      [10 values]         [10 values]         [10 values]
```

**Meta Agent**:
```python
# Without regime indicators
obs = [tech_features, sent_features, agent_weights]  # Shape: (163,)

# With regime indicators (10 assets)
obs = [tech_features, sent_features, agent_weights, regime_indicators]  # Shape: (173,)
#      [110 values]   [53 values]    [30 values]      [10 values]
```

### Agent Learning

The RL agent learns patterns like:
- "When NVDA is in bull regime (1), trust technical agent more"
- "When multiple assets are in bear regime (0), reduce overall exposure"
- "In mixed regimes, blend strategies more carefully"

## Tuning the Regime Window

### Short Window (3-4 months)
```python
regime_indicators = add_regime_indicator(price_data, window=3)
```
- **Pros**: Faster response to market changes
- **Cons**: More noise, frequent regime switches
- **Use when**: Trading volatile assets, shorter holding periods

### Medium Window (6 months) - Default
```python
regime_indicators = add_regime_indicator(price_data, window=6)
```
- **Pros**: Balanced signal, good for monthly rebalancing
- **Cons**: Moderate lag
- **Use when**: General portfolio management (recommended)

### Long Window (12+ months)
```python
regime_indicators = add_regime_indicator(price_data, window=12)
```
- **Pros**: Very smooth, captures major trends
- **Cons**: Slow to detect regime changes
- **Use when**: Long-term strategic allocation

## Example: Full Workflow

```python
# 1. Load data
price_data = pd.read_csv("data/monthly_prices.csv", index_col=0, parse_dates=True)
technical_features = pd.read_csv("data/technical_indicators.csv", index_col=0, parse_dates=True)
sentiment_features = pd.read_csv("data/nlp_features.csv", index_col=0, parse_dates=True)

# 2. Create regime indicators
from custom_function import add_regime_indicator
regime_indicators = add_regime_indicator(price_data, window=6)

# 3. Train base agents (no regime indicators)
sentiment_env = SentimentEnv(price_data, sentiment_features)
technical_env = TechnicalEnv(price_data, technical_features)
# ... train PPO/SAC ...

# 4. Create super agent (WITH regime indicators)
super_env = SuperAgentEnv(
    price_data=price_data,
    sentiment_agent=sent_ppo,
    technical_agent=tech_ppo,
    regime_indicators=regime_indicators,  # Add this!
    alpha1=1.0,
    alpha2=0.5,
    alpha3=0.5
)

# 5. Train super agent
super_model = PPO("MlpPolicy", super_env, ...)
super_model.learn(total_timesteps=100000)

# 6. Create meta agent (WITH regime indicators)
meta_env = MetaAgentEnv(
    price_data=price_data,
    features=technical_features,
    sentiment_features=sentiment_features,
    sentiment_agent=sent_wrapper,
    technical_agent=tech_wrapper,
    super_agent=super_wrapper,
    regime_indicators=regime_indicators,  # Add this!
    alpha1=1.0,
    alpha2=0.5,
    alpha3=0.5
)

# 7. Train meta agent
meta_model = PPO("MlpPolicy", meta_env, ...)
meta_model.learn(total_timesteps=100000)
```

## Analyzing Regime Impact

After training, analyze how regimes affect performance:

```python
# Check regime distribution
print("Bull market periods:", (regime_indicators == 1).sum().sum() / regime_indicators.size)
print("Bear market periods:", (regime_indicators == 0).sum().sum() / regime_indicators.size)

# Visualize regimes over time
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot prices
price_data['NVDA'].plot(ax=axes[0], label='NVDA Price')
price_data['NVDA'].rolling(6).mean().plot(ax=axes[0], label='6-month SMA', linestyle='--')
axes[0].set_title('NVDA Price and Regime')
axes[0].legend()

# Plot regime
regime_indicators['Regime_NVDA'].plot(ax=axes[1], label='Regime (1=Bull, 0=Bear)')
axes[1].fill_between(regime_indicators.index, 0, regime_indicators['Regime_NVDA'], alpha=0.3)
axes[1].set_title('Market Regime')
axes[1].set_ylim(-0.1, 1.1)
axes[1].legend()

plt.tight_layout()
plt.savefig('regime_analysis.png')
```

## Common Patterns Agents Learn

### Super Agent Strategies
1. **Bull Market**: Weight technical signals more (momentum works)
2. **Bear Market**: Weight sentiment more (news drives fear)
3. **Mixed Regimes**: Balanced blending
4. **Regime Transitions**: Reduce exposure temporarily

### Meta Agent Strategies
1. **All Bull**: Aggressive, trust all agents
2. **All Bear**: Conservative, lower overall weights
3. **Divergent Regimes**: Sector rotation opportunities
4. **Stable Regimes**: Higher confidence in predictions

## Configuration Options

Add to your notebook configuration cell:

```python
# Configuration Cell
regime_config = {
    'enabled': True,       # Enable/disable regime indicators
    'window': 6,           # SMA window (months)
}

# Create regime indicators conditionally
if regime_config['enabled']:
    regime_indicators = add_regime_indicator(
        price_data, 
        window=regime_config['window']
    )
else:
    regime_indicators = None  # Agents will work without them
```

## Performance Impact

**Expected Benefits**:
- Better strategy coordination
- Improved drawdown control in bear markets
- Higher Sharpe ratios through regime-aware allocation

**Trade-offs**:
- Slightly larger observation space
- More complex decision-making
- Needs more training data for good generalization

## Testing With/Without Regimes

Compare performance:

```python
# Experiment 1: Without regime indicators
super_env_v1 = SuperAgentEnv(
    price_data, sent_agent, tech_agent,
    regime_indicators=None  # No regimes
)

# Experiment 2: With regime indicators
super_env_v2 = SuperAgentEnv(
    price_data, sent_agent, tech_agent,
    regime_indicators=regime_indicators  # With regimes
)

# Train both and compare metrics
```

## Summary

**When to use regime indicators**:
- Training super or meta agents
- Want market-adaptive strategies
- Have sufficient data (60+ months)

**When NOT to use**:
- Training base agents (already have features)
- Very short time series (<36 months)
- Assets with erratic price movements

**Best practices**:
1. Start with default window (6 months)
2. Visualize regime transitions
3. Compare performance with/without
4. Fine-tune window based on your assets
5. Document which configuration works best

## Files Modified

- `custom_function.py`: Added `add_regime_indicator()` function
- `super_agent_envoriment.py`: Added regime support
- `meta_agent_enviroment.py`: Added regime support
- `harlf_system.ipynb`: Added regime creation cell

## Next Steps

1. Run notebook cells 1-3 to create regime indicators
2. Train base agents (cells 4-5)
3. Train hierarchical agents with regimes
4. Compare performance vs baseline (no regimes)
5. Fine-tune regime window if needed

