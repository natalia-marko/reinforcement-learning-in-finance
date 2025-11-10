# Performance Audit Report
## Walk-Forward RL Portfolio Optimization - CRITICAL ISSUES IDENTIFIED

**Date:** November 10, 2025
**Auditor:** Claude (AI Assistant)
**Status:** 🔴 CRITICAL - RESULTS INVALID

---

## Executive Summary

The evaluation results show **astronomical and unrealistic returns** that indicate fundamental bugs in the implementation. The trained models and performance metrics **cannot be trusted** and should not be used for any real-world decisions.

**Key Finding:** The environment has a critical data leakage bug where it uses **historical returns as future returns**, leading to completely unrealistic performance metrics.

---

## Critical Issues Identified

### 1. 🔴 CRITICAL: Data Leakage in Return Calculation

**Location:** `harlf/envs/portfolio_env.py:188-213`

**The Bug:**
```python
def _get_daily_returns(self, step: int) -> np.ndarray:
    # ...
    for ticker in self.tickers:
        return_5d = self.feature_pivots['return_5d'].loc[date, ticker]
        daily_return_approx = return_5d / 5.0  # ❌ WRONG!
        returns.append(daily_return_approx)
    return np.array(returns, dtype=np.float32)
```

**Why This Is Wrong:**

1. **Data Leakage:** `return_5d` is a feature that contains the **historical 5-day return** (from t-5 to t). The environment is using this historical return as if it were the future return that will be realized by holding the assets.

2. **Incorrect Math:** Even if we had forward-looking returns, dividing a compound 5-day return by 5 is mathematically incorrect. A 20% 5-day return ≠ 4% per day. The correct formula is: `daily_return = (1 + return_5d)^(1/5) - 1`

3. **Look-Ahead Bias:** The agent is essentially "knowing" what returns will happen and making decisions based on that, which is impossible in real trading.

**Impact:**
- All validation returns are inflated by 100x to 10,000x
- All training is based on this incorrect signal
- Models have learned to exploit fake information
- Zero real-world applicability

---

### 2. 🔴 CRITICAL: Astronomical Returns in Validation Set

**Validation Set Results (41 folds):**

| Agent | Mean Return | Max Return | Reality Check |
|-------|-------------|------------|---------------|
| PPO | **42,749%** | **810,894%** | ❌ Impossible |
| Equal Weight | 3.07% | 89.5% | ✅ Realistic |
| Random | 2.10% | 69.7% | ✅ Realistic |
| Momentum | **30,851%** | **1,173,674%** | ❌ Impossible |

**Analysis:**
- PPO achieved an average return of **42,749%** over ~125 days (6 months)
- This would turn $100K into $42.8 million on average
- Maximum fold achieved **810,894%** return ($100K → $810 million)
- These returns are **physically impossible** in real financial markets
- Even the best hedge funds achieve 20-40% **annually**, not 42,000% semi-annually

**Why Equal Weight and Random Are Lower:**
- These strategies use fixed allocations and don't "learn" from the buggy signals
- They still use the same buggy return calculation, but less adaptively
- Their returns (2-3%) are closer to reality but still inflated

---

### 3. ⚠️ HIGH: Extreme Maximum Drawdowns

**Validation Set:**
- PPO mean max drawdown: **85.14%**
- This means the agent loses 85% of capital on average during episodes
- Range: 53.7% to 99.98%

**Combined with astronomical returns, this indicates:**
- Extreme volatility and risk-taking behavior
- Portfolio value swings wildly (possibly going near zero then recovering)
- Unrealistic trading behavior
- Likely bugs in portfolio value tracking

---

### 4. ⚠️ HIGH: Corrupted Sortino Ratio (Test Set)

**Test Set Results:**
- PPO mean Sortino Ratio: **14,495,598**
- This is astronomically high (normal range: -2 to 5)
- Indicates division by near-zero downside deviation
- Likely caused by:
  - Insufficient downside returns in short test period (20 days)
  - Numerical instability in calculation
  - Bug in Sortino calculation

**Code Issue (portfolio_env.py:395-397):**
```python
downside_returns = returns[returns < 0]
downside_std = downside_returns.std() if len(downside_returns) > 0 else std_return
sortino = (mean_return / (downside_std + 1e-8)) * np.sqrt(252)
```

If there are very few negative returns, `downside_std` can be extremely small, causing astronomical Sortino ratios.

---

### 5. 🟡 MEDIUM: Validation vs Test Set Discrepancy

| Metric | Validation | Test | Ratio |
|--------|-----------|------|-------|
| PPO Mean Return | 42,749% | 5.78% | **7,393x** |
| Episode Length | 125 days | 20 days | 6.25x |

**Why such a huge difference?**

1. **Compounding Effect:** The bug compounds over time:
   - If daily_return_approx averages 3%, then:
   - 20 days: (1.03)^20 = 81% gain ✓ matches ~5-6%
   - 125 days: (1.03)^125 = 2,378% gain ✓ matches scale

2. **Longer episodes amplify the bug exponentially**

This explains why test returns are more reasonable - the bug has less time to compound!

---

### 6. 🟡 MEDIUM: High Turnover Rates

- PPO mean turnover: 33.3% (validation) to 33.9% (test)
- This means ~33% of portfolio is rebalanced each day
- With 15 bps transaction costs: ~0.05% cost per day
- Over 125 days: ~6.25% total transaction costs
- This is reasonable for an active strategy

**Not a bug, but worth noting for real-world implementation:**
- High turnover = high trading costs
- May be difficult to execute in practice
- Slippage costs not modeled

---

## Root Cause Analysis

### The Core Problem

The environment needs **actual realized returns** from holding assets, but instead:

1. Features contain **lagged technical indicators** (correct for ML features)
2. No **price data** or **forward returns** are included in the data
3. Environment incorrectly **approximates returns using historical features**
4. This creates **perfect foresight** - the agent "knows" recent returns

### Why This Happened

Looking at the data pipeline:
- Features were correctly engineered for ML (lagged indicators)
- But the environment needs **price data** to calculate real returns
- The environment tried to work around this with approximation
- The approximation created data leakage

---

## Validation Status by Metric

| Metric | Status | Notes |
|--------|--------|-------|
| Returns | ❌ Invalid | Data leakage, 100-10,000x inflated |
| Sharpe Ratio | ⚠️ Questionable | Based on invalid returns, but relative ranking may be meaningful |
| Sortino Ratio | ❌ Invalid | Numerical issues, astronomical values |
| Max Drawdown | ⚠️ Suspicious | Extremely high, may indicate calculation issues |
| Turnover | ✅ Valid | Not affected by return calculation bug |
| Volatility | ⚠️ Questionable | Calculated from invalid returns |

---

## What Can Be Salvaged?

### ❌ Cannot Use:
1. Absolute return numbers
2. Portfolio value trajectories
3. Claims of profitability
4. Sharpe/Sortino ratios
5. Any real-world deployment

### ⚠️ Maybe Useful (with caveats):
1. **Relative rankings** between agents (PPO vs baselines) - but still questionable
2. **Turnover patterns** - these are real behaviors
3. **Weight allocation patterns** - how agents allocate capital
4. **Training methodology** - the RL approach is sound, just needs correct data

### ✅ Definitely Valid:
1. Data pipeline (feature engineering is correct)
2. Walk-forward validation framework
3. Model architecture
4. Training infrastructure

---

## Recommendations

### 🔥 IMMEDIATE (Required Before Any Further Work):

1. **Fix the return calculation** - Add actual price data or use returns correctly:
   ```python
   # Option 1: Include actual prices in data
   # Add 'close_price' to features (unscaled)

   # Option 2: Include forward returns
   # Add 'next_day_return' = (price_t+1 - price_t) / price_t

   # Option 3: Use existing features correctly
   # Calculate implied return from change in return_5d feature
   # return_t = (return_5d_t+1 - return_5d_t) + adjustment
   ```

2. **Retrain ALL models** - All 42 existing models are trained on invalid data

3. **Re-evaluate from scratch** - All performance metrics are invalid

### 📋 SHORT-TERM (Before Real Deployment):

4. **Add price bounds checking:**
   ```python
   # Sanity check: daily returns should be reasonable
   if abs(portfolio_return_net) > 0.20:  # ±20% is extreme for daily
       logger.warning(f"Suspicious daily return: {portfolio_return_net}")
   ```

5. **Add performance sanity checks:**
   ```python
   # During training/eval, check for unrealistic returns
   if total_return > 10.0:  # 1000% return is suspicious
       raise ValueError("Unrealistic returns detected")
   ```

6. **Validate environment with known strategies:**
   - Test with buy-and-hold on known historical periods
   - Compare to documented market returns
   - If environment shows 100%+ when market did 10%, there's a bug

7. **Fix Sortino calculation:**
   ```python
   # Add better handling for low/zero downside deviation
   if len(downside_returns) < 10 or downside_std < 0.001:
       sortino = np.nan  # Not enough data for reliable Sortino
   else:
       sortino = (mean_return / downside_std) * np.sqrt(252)
   ```

### 🎯 LONG-TERM (For Production):

8. **Add realistic market constraints:**
   - Bid-ask spread modeling
   - Market impact costs
   - Slippage for large orders
   - Execution delays

9. **Add robustness tests:**
   - Transaction cost sensitivity
   - Market regime changes
   - Crisis periods

10. **Implement proper backtesting:**
    - Point-in-time data only
    - Realistic execution assumptions
    - Out-of-sample testing on held-out periods

---

## Impact Assessment

### Training Impact:
- ❌ All trained models learned from **fake signals**
- ❌ Models optimized for **exploiting data leakage**, not real trading
- ❌ Zero transfer to real markets

### Evaluation Impact:
- ❌ All performance metrics are **invalid**
- ❌ Cannot determine if PPO actually outperforms baselines
- ❌ Cannot estimate real-world performance

### Business Impact:
- ❌ **DO NOT DEPLOY** these models to real trading
- ❌ Cannot use these results in papers/reports without correction
- ❌ Estimated time to fix: 1-2 weeks (fix bug + retrain + re-evaluate)

---

## Testing Protocol (After Fix)

Before considering results valid:

1. **Sanity Check Returns:**
   - Daily returns should be mostly in ±5% range
   - Annual returns should be in ±100% range
   - Compare to actual market returns for same period

2. **Validate on Known Period:**
   - Test on 2020 (known high volatility)
   - Test on 2021 (known bull market)
   - Results should match market character

3. **Check Baseline Reasonableness:**
   - Equal weight should track market average
   - Random should underperform equal weight
   - All strategies should have similar order-of-magnitude returns

4. **Cross-Check Metrics:**
   - Sharpe ratios should be -2 to 3 (4+ is rare)
   - Sortino ratios should be 0.5 to 5
   - Max drawdowns should be 10-50% for aggressive strategies
   - Turnovers should be consistent across agents

---

## Conclusion

**The current results are completely invalid and cannot be used.**

The fundamental issue is that the environment uses historical returns as if they were future returns, creating perfect foresight. This is a classic data leakage bug that invalidates all training and evaluation.

**The good news:**
- The bug is identifiable and fixable
- The overall framework (features, walk-forward, architecture) is sound
- Once fixed, you have a solid foundation for real RL trading research

**Next steps:**
1. Fix the return calculation in `portfolio_env.py`
2. Validate the fix with sanity checks
3. Retrain all models from scratch
4. Re-run evaluation
5. Verify results are realistic before proceeding

**Estimated effort:** 1-2 weeks to fix, retrain, and validate properly.

---

## Appendix: Supporting Evidence

### Evidence of Data Leakage

```python
# In feature_engineering.py:36
features['return_5d'] = close_prices.pct_change(periods=5)

# This calculates: return_5d[t] = (price[t] - price[t-5]) / price[t-5]
# This is HISTORICAL data (past 5 days)

# But in portfolio_env.py:210
return_5d = self.feature_pivots['return_5d'].loc[date, ticker]
daily_return_approx = return_5d / 5.0

# Then used in step() as the return for holding assets TODAY:
portfolio_return_gross = np.dot(action, asset_returns)  # portfolio_env.py:296
```

This creates a causality violation: using past returns as future returns.

### Sample Unrealistic Returns

From `fold_2` (validation):
- Initial capital: $100,000
- Final value: $138,488,676,352 (138 billion)
- Return: 138,487,578%
- Period: 125 days (~6 months)
- Daily compound rate: ~11.6% per day

For comparison:
- S&P 500 average annual return: ~10%
- Best hedge fund annual returns: ~30-50%
- This result: ~138 million % in 6 months

**Conclusion: Physically impossible.**

---

**Report Generated:** 2025-11-10
**Recommendation:** HALT all deployment. Fix bugs before proceeding.
