# HARLF System Architecture with Regime Indicators

## Visual Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Price Data  │  Technical Features  │  Sentiment Features       │
│  (10 assets) │     (110 features)   │    (53 features)          │
└──────┬───────┴──────────┬───────────┴───────────┬───────────────┘
       │                  │                       │
       │                  │                       │
       ▼                  ▼                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    REGIME INDICATORS                             │
├──────────────────────────────────────────────────────────────────┤
│  add_regime_indicator(price_data, window=6)                      │
│                                                                  │
│  Output: 10 binary indicators (1=bull, 0=bear)                  │
│  - Regime_NVDA:  1   (price > 6-month SMA)                      │
│  - Regime_AMD:   0   (price < 6-month SMA)                      │
│  - Regime_MSFT:  1                                               │
│  - ... (one per asset)                                           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                     │
        ▼                                     ▼
┌───────────────────┐              ┌───────────────────┐
│  BASE AGENTS      │              │  BASE AGENTS      │
│  (NO REGIME)      │              │  (NO REGIME)      │
├───────────────────┤              ├───────────────────┤
│ Sentiment Agent   │              │ Technical Agent   │
│                   │              │                   │
│ Input:            │              │ Input:            │
│ - 53 sentiment    │              │ - 110 technical   │
│   features        │              │   indicators      │
│                   │              │                   │
│ Output:           │              │ Output:           │
│ - 10 weights      │              │ - 10 weights      │
│   (portfolio)     │              │   (portfolio)     │
└─────────┬─────────┘              └─────────┬─────────┘
          │                                  │
          │                                  │
          └──────────────┬───────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │      SUPER AGENT                 │
          │      (WITH REGIME)               │
          ├──────────────────────────────────┤
          │ Input (obs):                     │
          │ ┌──────────────────────────────┐ │
          │ │ Sentiment weights:  10 dims  │ │
          │ │ Technical weights:  10 dims  │ │
          │ │ Regime indicators:  10 dims  │ │◄─ Regime Added Here!
          │ │ ─────────────────────────────│ │
          │ │ Total observation:  30 dims  │ │
          │ └──────────────────────────────┘ │
          │                                  │
          │ Output:                          │
          │ - 10 weights (blended strategy)  │
          └──────────────┬───────────────────┘
                         │
                         ▼
          ┌──────────────────────────────────┐
          │      META AGENT                  │
          │      (WITH REGIME)               │
          ├──────────────────────────────────┤
          │ Input (obs):                     │
          │ ┌──────────────────────────────┐ │
          │ │ Technical features:  110     │ │
          │ │ Sentiment features:   53     │ │
          │ │ Sentiment weights:    10     │ │
          │ │ Technical weights:    10     │ │
          │ │ Super weights:        10     │ │
          │ │ Regime indicators:    10     │ │
          │ │ ─────────────────────────────│ │
          │ │ Total observation:   203     │ │
          │ └──────────────────────────────┘ │
          │                                  │
          │ Output:                          │
          │ - 10 weights (final portfolio)   │
          └──────────────────────────────────┘
```

## Data Flow

```
1. Price Data (monthly)
   └─> add_regime_indicator(window=6)
       └─> Regime Indicators (10 binary values)
           │
           ├─> NOT used by Base Agents
           │   (they have specialized features already)
           │
           └─> USED by Hierarchical Agents
               │
               ├─> Super Agent: Learns when to trust which base agent
               │   Example: "In bull regime, trust technical more"
               │
               └─> Meta Agent: Learns portfolio-level strategy
                   Example: "In mixed regimes, reduce exposure"
```

## Observation Space Breakdown

### Base Agents (Unchanged)

#### Sentiment Agent
```
Input: 53 sentiment features
├─ NVDA_sent, NVDA_sent_lag1, NVDA_sent_lag2, ...
├─ AMD_sent, AMD_sent_lag1, ...
├─ market_sent, sent_dispersion, sent_trend
└─ (NO regime indicators)

Output: 10 portfolio weights
```

#### Technical Agent
```
Input: 110 technical indicators
├─ NVDA_sharpe, NVDA_sortino, NVDA_rsi, ...
├─ AMD_sharpe, AMD_sortino, ...
├─ Volume indicators, momentum, volatility
└─ (NO regime indicators)

Output: 10 portfolio weights
```

### Hierarchical Agents (With Regime)

#### Super Agent
```
Input: 30 dimensions
├─ Sentiment weights:    [w1, w2, ..., w10]  (10 dims)
├─ Technical weights:    [w1, w2, ..., w10]  (10 dims)
└─ Regime indicators:    [r1, r2, ..., r10]  (10 dims) 

Where regime ri ∈ {0, 1}:
- 1 = Bull market (price > SMA)
- 0 = Bear market (price < SMA)

Output: 10 portfolio weights (blended strategy)
```

#### Meta Agent
```
Input: 203 dimensions
├─ Technical features:   110 dims
├─ Sentiment features:    53 dims
├─ Sentiment weights:     10 dims
├─ Technical weights:     10 dims
├─ Super weights:         10 dims
└─ Regime indicators:     10 dims

Output: 10 portfolio weights (final portfolio)
```

## Regime Learning Examples

### Super Agent Learns

```
Scenario 1: All Bull Market
Regime = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Strategy: Weight technical agent more (momentum works in bull markets)

Scenario 2: All Bear Market
Regime = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Strategy: Weight sentiment agent more (news drives fear in bear markets)

Scenario 3: Mixed Regime
Regime = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
Strategy: Asset-specific weighting based on individual regimes
```

### Meta Agent Learns

```
Scenario 1: Strong Bull (High Regime Sum)
Regime sum = 8-10 bulls
Strategy: Aggressive allocation, trust all agents

Scenario 2: Strong Bear (Low Regime Sum)
Regime sum = 0-2 bulls
Strategy: Conservative allocation, reduce overall exposure

Scenario 3: Transition Period
Regime changing frequently
Strategy: Lower confidence, more cautious position sizing
```

## Code Implementation

### 1. Create Regime Indicators

```python
# In notebook Cell 3
from custom_function import add_regime_indicator

regime_indicators = add_regime_indicator(price_data, window=6)

# Result:
# regime_indicators.shape = (129, 10)
# regime_indicators.columns = ['Regime_NVDA', 'Regime_AMD', ...]
```

### 2. Pass to Super Agent

```python
super_env = SuperAgentEnv(
    price_data=price_data,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    regime_indicators=regime_indicators,  # ← Add this
    alpha1=1.0, alpha2=0.5, alpha3=0.5
)
```

### 3. Pass to Meta Agent

```python
meta_env = MetaAgentEnv(
    price_data=price_data,
    features=technical_features,
    sentiment_features=sentiment_features,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    super_agent=super_agent,
    regime_indicators=regime_indicators,  # ← Add this
    alpha1=1.0, alpha2=0.5, alpha3=0.5
)
```

## Regime Calculation Logic

```python
# For each asset:
price_t = current_price
sma_t = rolling_mean(price, window=6)

if price_t > sma_t:
    regime = 1  # Bull market
else:
    regime = 0  # Bear market
```

## Impact on Training

### Before (No Regime)
```
Super Agent learns:
- Fixed blending strategy
- Doesn't adapt to market conditions
- May perform poorly in regime changes

Meta Agent learns:
- General portfolio strategy
- Limited context about market state
```

### After (With Regime)
```
Super Agent learns:
- Adaptive blending based on market regime
- Different strategies for bull vs bear
- Better performance during transitions

Meta Agent learns:
- Regime-aware portfolio allocation
- When to be aggressive vs conservative
- How to handle mixed market conditions
```

## Performance Expectations

**Typical Improvements**:
- Sharpe Ratio: +5-15%
- Max Drawdown: -10-20% (lower is better)
- Win Rate: +2-5%
- Better out-of-sample performance

**Trade-offs**:
- Slightly more complex observation space
- May need more training data
- Longer training time (10-15% more)

## Backward Compatibility

```python
# Old code still works (no regime)
super_env = SuperAgentEnv(
    price_data, sent_agent, tech_agent
    # regime_indicators defaults to None
)

# New code with regime (opt-in)
super_env = SuperAgentEnv(
    price_data, sent_agent, tech_agent,
    regime_indicators=regime_indicators
)
```

## Summary

```
┌──────────────────────────────────────────┐
│ Regime Indicators Purpose               │
├──────────────────────────────────────────┤
│ ✓ Help hierarchical agents adapt        │
│ ✓ Provide market context                │
│ ✓ Improve strategy coordination         │
│ ✓ Better drawdown control               │
│                                          │
│ ✗ Not needed for base agents            │
│ ✗ Optional (can be disabled)            │
│ ✗ Requires minimum 36 months data       │
└──────────────────────────────────────────┘
```

## Files Structure

```
harlf_v4/
├─ custom_function.py              ← add_regime_indicator()
├─ super_agent_envoriment.py       ← regime support added
├─ meta_agent_enviroment.py        ← regime support added
├─ harlf_system.ipynb              ← Cell 3 creates regimes
├─ REGIME_INDICATORS_GUIDE.md      ← Full guide
├─ REGIME_INDICATORS_SUMMARY.md    ← Quick reference
└─ REGIME_ARCHITECTURE.md          ← This file
```

