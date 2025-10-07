# Regime Indicators - Quick Summary

## What Was Added

Market regime indicators (bull/bear signals) for Super and Meta agents to adapt strategies based on market conditions.

## Quick Usage

### 1. In Notebook (Cell 3)

```python
from custom_function import add_regime_indicator

# Create regime indicators
regime_indicators = add_regime_indicator(price_data, window=6)
```

### 2. When Creating Super/Meta Agents

```python
# Super Agent
super_env = SuperAgentEnv(
    price_data=price_data,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    regime_indicators=regime_indicators,  # Add this line
    alpha1=1.0, alpha2=0.5, alpha3=0.5
)

# Meta Agent
meta_env = MetaAgentEnv(
    price_data=price_data,
    features=technical_features,
    sentiment_features=sentiment_features,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    super_agent=super_agent,
    regime_indicators=regime_indicators,  # Add this line
    alpha1=1.0, alpha2=0.5, alpha3=0.5
)
```

## What It Does

- **1 = Bull Market**: Price > 6-month moving average
- **0 = Bear Market**: Price < 6-month moving average
- One indicator per asset in your portfolio

## Why Only Super/Meta?

- **Base agents** (Sentiment/Technical): Already have 100+ specialized features, don't need regime signals
- **Hierarchical agents** (Super/Meta): Need market context to coordinate strategies effectively

Example: Super agent learns "trust technical signals more in bull markets, sentiment signals more in bear markets"

## Files Modified

1. **`custom_function.py`**: Added `add_regime_indicator()` function
2. **`super_agent_envoriment.py`**: Added `regime_indicators` parameter to `__init__` and observations
3. **`meta_agent_enviroment.py`**: Added `regime_indicators` parameter to `__init__` and observations
4. **`harlf_system.ipynb`**: Added Cell 3 to create regime indicators
5. **`experiment_configs.py`**: Added `regime_config` options

## Configuration Options

```python
# In experiment_configs.py or your notebook
regime_config = {
    'enabled': True,   # Turn on/off
    'window': 6,       # SMA window (months)
}

# To disable
regime_indicators = None  # Agents will work without them
```

### Tuning the Window

- **3-4 months**: Fast response, more noise
- **6 months**: Balanced (default, recommended)
- **12+ months**: Smooth, captures major trends only

## Observation Space Changes

### Super Agent
- **Without**: `(2 * n_assets,)` - just agent weights
- **With**: `(2 * n_assets + n_assets,)` - agent weights + regime

### Meta Agent
- **Without**: `(tech + sent features + 3*n_assets,)`
- **With**: `(tech + sent features + 3*n_assets + n_assets,)`

Example with 10 assets:
- Super: 20 dimensions → 30 dimensions
- Meta: 163 dimensions → 173 dimensions

## Expected Benefits

1. Better strategy coordination
2. Improved drawdown control in bear markets
3. Higher Sharpe ratios through regime-aware allocation
4. Adaptive blending of base agent signals

## Testing

Compare with/without:

```python
# Without regime
super_env_v1 = SuperAgentEnv(
    price_data, sent_agent, tech_agent,
    regime_indicators=None
)

# With regime
super_env_v2 = SuperAgentEnv(
    price_data, sent_agent, tech_agent,
    regime_indicators=regime_indicators
)

# Train both and compare metrics
```

## Next Steps

1. **Run notebook**: Create regime indicators in Cell 3
2. **Train hierarchical agents**: Pass `regime_indicators` parameter
3. **Compare**: Test performance vs baseline (without regimes)
4. **Tune**: Adjust window if needed (3, 6, or 12 months)

## Documentation

See **`REGIME_INDICATORS_GUIDE.md`** for:
- Detailed explanation
- Full examples
- Performance analysis
- Visualization code
- Best practices

## Backward Compatibility

- Regime indicators are **optional** (`regime_indicators=None` by default)
- Base agents **unchanged** (don't use regimes)
- Existing code **still works** without modifications
- New functionality is **opt-in**

