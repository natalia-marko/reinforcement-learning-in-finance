# Sentiment Agent Analysis & Issues

**Generated**: 2025-11-05
**Project**: Multi-Hierarchical RL Portfolio System

---

## Executive Summary

The "Sentiment Agent" has significant conceptual and implementation issues. Despite its name, it **does not use actual sentiment data**. Instead, it uses technical indicators that largely duplicate the Technical Agent's features. This creates redundancy and may explain poor performance.

**Grade**: ⚠️ **C-** (Needs Major Revision)

---

## 1. Critical Issues

### Issue #1: Misleading Nomenclature ⚠️ CRITICAL

**Problem**: The agent is called "Sentiment" but doesn't analyze sentiment.

**What It Actually Is**: A secondary technical analysis agent using alternative momentum/volatility indicators.

**True Sentiment Features Should Include**:
- News sentiment scores (e.g., from Bloomberg, Reuters APIs)
- Social media sentiment (Twitter/X, Reddit, StockTwits)
- Analyst ratings and recommendation changes
- Options market sentiment (put/call ratios, implied volatility skew)
- Earnings call transcripts sentiment (NLP analysis)
- Insider trading activity
- Short interest data
- Fund flows and institutional positioning

**Current "Sentiment" Features** (All Technical):
```python
SENTIMENT = [
    # Momentum Signals (3) - THESE ARE TECHNICAL
    'momentum_2w', 'momentum_4w', 'momentum_6w',

    # Trend Quality (2) - THESE ARE TECHNICAL
    'positive_weeks_pct', 'price_dispersion',

    # Volume Sentiment (2) - THESE ARE TECHNICAL
    'volume_sentiment', 'volume_surge',

    # Market Regime (2) - THESE ARE TECHNICAL
    'vol_regime', 'drawdown',

    # Fear/Greed (3) - THESE ARE PROXIES, NOT REAL SENTIMENT
    'market_fear', 'credit_sentiment', 'news_sentiment'
]
```

---

### Issue #2: High Redundancy with Technical Agent ⚠️ CRITICAL

**Overlap Analysis**:

| Sentiment Feature | Technical Equivalent | Redundancy Level |
|-------------------|---------------------|------------------|
| `momentum_2w/4w/6w` | `return_lag_1w/2w/3w`, `roc_4w` | **HIGH** - Same concept, different timeframes |
| `positive_weeks_pct` | Can be derived from `return_lag_*` | **MEDIUM** - Aggregation of returns |
| `volume_sentiment` | `volume_ratio_20w` | **HIGH** - Both measure abnormal volume |
| `volume_surge` | `volume_ratio_20w` | **HIGH** - Z-score of volume ratio |
| `vol_regime` | `volatility_12w` | **HIGH** - Percentile rank vs raw volatility |
| `drawdown` | Can be derived from returns | **MEDIUM** - Cumulative calculation |
| `market_fear` | `volatility_12w` spike | **HIGH** - Volatility spike detection |

**Result**: ~70% feature redundancy between Technical and Sentiment agents.

**Impact on Hierarchical System**:
- Super Agent receives **highly correlated** inputs from both base agents
- Reduces diversity of information
- May cause the Super Agent to overweight technical signals
- Limits the benefit of ensemble learning

---

### Issue #3: Fake "News Sentiment" Feature ⚠️ CRITICAL

**Line 290 in run_data_prep.py**:
```python
features['news_sentiment'] = features['momentum_4w'] * np.tanh(vol_norm - 1)
```

**Problem**: This is **not news sentiment**. It's a mathematical transformation of momentum and volume.

**What It Actually Measures**:
- High momentum + high volume = positive "news sentiment"
- Low momentum + low volume = negative "news sentiment"

**Why This is Wrong**:
- No connection to actual news events
- No NLP or sentiment analysis
- Purely price/volume derived
- The name is misleading

**Correct Implementation Would Require**:
```python
# Example using real news sentiment
from transformers import pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")

news_data = fetch_news_for_ticker(ticker, date_range)  # From news API
sentiments = [sentiment_analyzer(article['text'])[0] for article in news_data]
features['news_sentiment'] = np.mean([s['score'] * (1 if s['label'] == 'positive' else -1)
                                       for s in sentiments])
```

---

### Issue #4: Only One Unique Feature ⚠️ MEDIUM

**The Only Truly Different Feature**:
```python
# Line 299 in run_data_prep.py
price_dispersion = all_returns.std(axis=1)  # Cross-sectional std of returns
```

**Why This Matters**:
- `price_dispersion` is the ONLY cross-sectional feature in the entire system
- It measures market dispersion (sector rotation, correlation regime)
- This should be elevated to a more prominent role

**What It Actually Captures**:
- High dispersion = Stock picking environment (low correlation)
- Low dispersion = Macro-driven market (high correlation)
- Useful for portfolio construction and risk management

**Recommendation**: Move this to **Macro Agent** or create a separate "Market Regime" feature set.

---

## 2. Environment Construction Issues

### Issue #5: No Special Handling for Sentiment Features

**Current Implementation** (`helpers/environments.py`):
```python
def create_env(agent_type: str, split: str = 'train', ...):
    # Load data
    features, returns = prepare_env_data(agent_type, split)

    # Create environment (SAME for all agents)
    env = PortfolioEnv(
        features=features,
        returns=returns,
        reward_type=reward_type,
        **env_kwargs
    )
    return env
```

**Observation Construction** (line 229-242):
```python
# Get current features (n_assets, n_features)
current_features = self.features[self.current_step]

# Flatten features
obs = current_features.flatten()  # Shape: (n_assets * n_features,)

# Add current positions
if self.include_positions:
    obs = np.concatenate([obs, self.positions])  # +n_assets
```

**For Sentiment Agent**:
- **Observation Shape**: `(7 assets × 12 features) + 7 positions = 91 dimensions`
- **Technical Agent**: `(7 assets × 20 features) + 7 positions = 147 dimensions`

**Problem**: Sentiment agent has **40% fewer input dimensions** but uses the **same network architecture** as Technical agent.

---

## 3. Performance Analysis

### Observed Performance Issues

From the conversation context:
```
Sentiment Agent Test Performance:
  - Total Return: -33.32%
  - Sharpe Ratio: -1.05
  - Early Termination: Hit max drawdown at week 31/62
```

**Why It Failed**:

1. **Feature Redundancy**:
   - Agent learns similar patterns to Technical agent
   - No unique signal to exploit
   - Correlated losses

2. **Smaller Input Space**:
   - 12 features vs 20 features
   - Less information to learn from
   - May need smaller network architecture

3. **No True Alpha Source**:
   - Real sentiment data provides alpha
   - Fake sentiment data is just noise
   - Pure price/volume features are crowded

4. **Overfitting to Spurious Patterns**:
   - Mathematical transformations like `momentum * tanh(volume - 1)`
   - No economic meaning
   - Don't generalize out-of-sample

---

## 4. Comparison: Technical vs "Sentiment" Features

### Technical Agent Features (20 features)

**✅ Well-Designed**:
- **Trend Following** (5): SMA/EMA ratios at multiple timeframes
- **Momentum** (7): MACD, RSI, Stochastic, ROC, lagged returns
- **Volatility** (3): Rolling vol, ATR, Bollinger Bands
- **Volume** (3): Volume ratio, MFI, OBV
- **Benchmark** (2): Beta, correlation with QQQ

**Characteristics**:
- Diverse indicators across multiple categories
- Multiple timeframes (4w, 8w, 12w, 14w, 20w)
- Both trend and mean-reversion signals
- Validated technical analysis tools

### Sentiment Agent Features (12 features)

**⚠️ Poorly Designed**:
- **Momentum** (3): Duplicates Technical momentum
- **Trend Quality** (2): Derived from returns + one unique cross-sectional feature
- **Volume** (2): Duplicates Technical volume indicators
- **Volatility Regime** (2): Duplicates Technical volatility
- **Fake Sentiment** (3): Mathematical proxies with no real sentiment

**Characteristics**:
- Heavy overlap with Technical features
- Only 1 truly unique feature (`price_dispersion`)
- Misleading feature names
- No actual sentiment analysis

---

## 5. Recommendations

### Option A: Fix the Sentiment Agent (Recommended) ⭐

**Rename** to "Alternative Technical Agent" or "Momentum Agent"

**Keep Only Unique Features**:
```python
ALTERNATIVE_TECHNICAL = [
    # Cross-Sectional Features (NEW)
    'price_dispersion',           # Already implemented
    'relative_strength',          # Stock vs sector/market
    'correlation_to_market',      # Rolling correlation
    'sector_momentum',            # If we add sector data

    # Advanced Momentum (Different from Technical)
    'momentum_consistency',       # Consistency of momentum direction
    'momentum_acceleration',      # 2nd derivative of price
    'reversal_signal',            # Mean-reversion indicator

    # Unique Volume Patterns
    'volume_trend',               # Volume moving average slope
    'price_volume_correlation',   # Correlation between price and volume
    'volume_persistence',         # Autocorrelation of volume
]
```

### Option B: Build a Real Sentiment Agent 🎯

**Integrate Real Sentiment Data Sources**:

1. **News Sentiment**:
   ```python
   # Use FinBERT or similar
   from transformers import pipeline
   sentiment = pipeline("sentiment-analysis", model="ProsusAI/finbert")
   ```

2. **Social Media Sentiment**:
   ```python
   # Twitter/X sentiment via API
   # Reddit WallStreetBets activity
   # StockTwits sentiment scores
   ```

3. **Options Market Sentiment**:
   ```python
   # Put/Call Ratio
   # Implied Volatility Skew
   # Options volume relative to stock volume
   ```

4. **Analyst Sentiment**:
   ```python
   # Recommendation changes
   # Price target changes
   # Number of analysts covering stock
   ```

5. **Insider Trading**:
   ```python
   # Insider buying/selling activity
   # Form 4 filings
   ```

**New Feature Set**:
```python
SENTIMENT = [
    # News Sentiment (4)
    'news_sentiment_score',
    'news_volume_24h',
    'news_sentiment_change',
    'negative_news_count',

    # Social Sentiment (3)
    'twitter_sentiment',
    'reddit_mentions',
    'stocktwits_bullish_pct',

    # Options Sentiment (3)
    'put_call_ratio',
    'iv_skew',
    'unusual_options_activity',

    # Analyst Sentiment (2)
    'analyst_recommendation_avg',
    'price_target_vs_current'
]
```

### Option C: Remove Sentiment Agent Entirely 🗑️

**Arguments For Removal**:
- Adds no unique information
- Increases computational cost
- Confuses hierarchical system with redundant signals
- Current implementation provides no value

**Alternative Architecture**:
- **Technical Agent**: Price/volume/technical indicators
- **Macro Agent**: Economic regime, rates, VIX, calendar
- **Cross-Sectional Agent**: Relative strength, correlation, dispersion
- **Super Agent**: Blends Technical + Macro + Cross-Sectional
- **Meta Agent**: Adjusts Super Agent based on market regime

---

## 6. Data Quality Issues

### Issue #6: Normalization Applied Uniformly

**Current Implementation** (`helpers/utils.py:464`):
```python
scaler = StandardScaler()
train_features_normalized = scaler.fit_transform(train_df[feature_cols])
```

**Problem**: StandardScaler assumes features are normally distributed.

**Sentiment Features That Violate This**:
- `momentum_*`: Can have extreme outliers during crashes/rallies
- `drawdown`: Bounded at [-1, 0], not normal
- `vol_regime`: Percentile rank, bounded at [0, 1]
- `positive_weeks_pct`: Bounded at [0, 1]

**Better Approach**:
```python
from sklearn.preprocessing import RobustScaler, QuantileTransformer

# For momentum features (handle outliers)
robust_scaler = RobustScaler()

# For bounded features (map to normal distribution)
quantile_transformer = QuantileTransformer(output_distribution='normal')
```

---

## 7. Testing Recommendations

### Test #1: Remove Sentiment Agent, Compare Performance

**Hypothesis**: Super Agent performs **better** without Sentiment Agent due to reduced noise.

**Experiment**:
1. Train Super Agent with only Technical agent (no Sentiment)
2. Compare test set Sharpe ratio to current system
3. If Sharpe improves, Sentiment agent adds negative value

### Test #2: Add Real Sentiment Data

**Hypothesis**: Real sentiment data provides alpha.

**Experiment**:
1. Integrate FinBERT news sentiment for past 2 weeks
2. Add put/call ratio from options market
3. Retrain Sentiment agent
4. Compare performance

**Expected Result**: Sharpe ratio should improve if sentiment data has predictive power.

### Test #3: Feature Importance Analysis

**Method**: Use SHAP values or permutation importance to measure feature impact.

**Questions to Answer**:
- Which "sentiment" features actually matter?
- Is `price_dispersion` the only useful feature?
- Are fake sentiment features (news_sentiment, credit_sentiment) contributing?

---

## 8. Code-Level Issues

### Issue #7: Inconsistent Data Shapes

**In `calculate_sentiment_features()` (line 268)**:
```python
features['price_dispersion'] = np.nan  # Placeholder
```

Then later filled cross-sectionally:
```python
# Line 305-306
sent_feat['price_dispersion'] = price_dispersion
```

**Problem**: Creates temporary NaN values, then fills them. Inefficient and error-prone.

**Better Approach**:
```python
def calculate_sentiment_features(df, all_stocks_df, price_dispersion_series):
    """Pass pre-calculated price_dispersion as argument."""
    features['price_dispersion'] = price_dispersion_series
```

---

### Issue #8: No Feature Validation

**Missing Checks**:
- Are all features finite? (no inf, no nan)
- Are feature ranges reasonable?
- Are there any constant features (zero variance)?

**Add Validation**:
```python
def validate_features(features_df, feature_cols):
    """Validate feature quality."""
    issues = []

    for col in feature_cols:
        # Check for inf
        if np.isinf(features_df[col]).any():
            issues.append(f"{col}: Contains infinite values")

        # Check for constant
        if features_df[col].std() < 1e-10:
            issues.append(f"{col}: Near-zero variance (constant feature)")

        # Check for high skewness
        if abs(features_df[col].skew()) > 5:
            issues.append(f"{col}: Highly skewed (skew={features_df[col].skew():.2f})")

    return issues
```

---

## 9. Summary Table

| Aspect | Technical Agent | "Sentiment" Agent | Actual Sentiment Agent |
|--------|----------------|-------------------|------------------------|
| **Feature Type** | Technical indicators | Technical indicators (mislabeled) | News, social, analyst data |
| **Data Sources** | Price, volume, benchmark | Price, volume | NLP APIs, social media, options |
| **Feature Count** | 20 | 12 | 12-15 |
| **Redundancy** | N/A | ~70% overlap with Technical | ~0% overlap |
| **Alpha Potential** | Medium (crowded) | Low (duplicate) | High (alternative data) |
| **Implementation** | ✅ Correct | ⚠️ Incorrect naming | ❌ Not implemented |
| **Test Sharpe** | 1.65 | -1.05 | ??? |
| **Value Add** | Baseline | Negative | Potentially high |

---

## 10. Action Items

### Immediate (This Week)

1. ✅ **Rename Agent**: Change "Sentiment" → "Alternative Technical" in all code and docs
2. ✅ **Remove Misleading Features**: Delete `news_sentiment`, `credit_sentiment` (fake sentiment)
3. ✅ **Feature Audit**: Run validation to check for inf/nan/constant features
4. ✅ **Correlation Analysis**: Measure correlation between Technical and "Sentiment" features

### Short-term (This Month)

5. ⚠️ **Experiment**: Train Super Agent without Sentiment agent, compare performance
6. ⚠️ **Document**: Add warning in docs that current "Sentiment" agent doesn't use sentiment data
7. ⚠️ **Feature Engineering**: Extract truly unique features (cross-sectional, relative strength)

### Long-term (Next Quarter)

8. 🎯 **Integration**: Add real sentiment data sources (FinBERT, options data)
9. 🎯 **Architecture**: Redesign as 4-agent system (Technical, Macro, Cross-Sectional, Super)
10. 🎯 **Validation**: Run A/B test: current vs improved Sentiment agent

---

## 11. Conclusion

The current "Sentiment Agent" is **misnamed and redundant**. It should either be:

**Option A**: Renamed to "Alternative Technical Agent" and refactored to use truly different technical indicators

**Option B**: Replaced with a Real Sentiment Agent using NLP and alternative data

**Option C**: Removed entirely to simplify the system

**Recommendation**: Start with **Option C** (remove), then implement **Option B** (real sentiment) if resources allow.

The current implementation provides negative value due to feature redundancy and lack of true sentiment signals. The hierarchical system would likely perform better with 2 agents (Technical + Macro) than with 3 agents where 2 provide redundant information.

---

**End of Analysis**
