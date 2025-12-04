# CRITICAL ANALYSIS: Training Results & Data Leakage Check

**Date:** 2025-11-20
**Status:** ⚠️ MAJOR CONCERNS IDENTIFIED

---

## Executive Summary

✓ **NO DATA LEAKAGE** detected in fold splits or feature normalization
⚠️ **EARLY STOPPING FAILED** - all folds showed 19-44% validation performance drops
⚠️ **HIGH VARIANCE** across folds (Sharpe: 0.461 → 1.378)
🚨 **VALIDATION/TEST MISMATCH** - worst validation fold performed best on test!

---

## 1. Data Leakage Check: ✓ PASSED

### Fold Split Analysis
```
Fold 0:
  Train: 2015-06-25 to 2017-02-02 (595 samples)
  Val:   2017-08-10 to 2019-04-11 (616 samples)
  Embargo: 26 weeks
  ✓ No overlap, chronological

Fold 1:
  Train: 2015-06-25 to 2019-04-11 (1,393 samples)
  Val:   2019-10-17 to 2021-06-17 (616 samples)
  Embargo: 26 weeks
  ✓ No overlap, chronological

Fold 2:
  Train: 2015-06-25 to 2021-06-17 (2,191 samples)
  Val:   2021-12-23 to 2023-08-24 (616 samples)
  Embargo: 26 weeks
  ✓ No overlap, chronological
```

**Verdict:** Fold splits are correct. 26-week embargo prevents lookback leakage.

### Feature Normalization Check
- Method: Per-ticker z-score with 52-week rolling window
- Test: Manually calculated vs actual normalized values
- Difference: 0.009 (< 0.01 threshold)
- **Verdict:** ✓ Normalization uses only past data, no leakage

---

## 2. Overfitting Detection: 🚨 FAILED

### Validation Performance During Training

| Fold | Best Val Reward | Final Val Reward | Drop | Status |
|------|----------------|------------------|------|--------|
| 0 | 0.9658 @ 5k | 0.7827 @ 80k | **19.0%** | ⚠️ Overfitting |
| 1 | 1.1561 @ 30k | 0.8731 @ 80k | **24.5%** | ⚠️ Overfitting |
| 2 | 0.2365 @ 55k | 0.1325 @ 85k | **44.0%** | 🚨 Severe overfitting |

**All folds continued training far past peak performance!**

### Why Early Stopping Failed

Config setting: `EARLY_STOP_PATIENCE = 5`

Expected: Stop if validation doesn't improve for 5 evaluations
Actual: Trained for full 80k-85k steps despite degrading performance

**Possible causes:**
1. Callback not properly connected
2. `min_evals=10` requirement delayed triggering
3. Evaluation frequency (5k steps) too coarse

**Evidence-based recommendation:** Should have stopped at:
- Fold 0: ~5k steps (saved 75k wasted steps)
- Fold 1: ~30k steps (saved 50k wasted steps)
- Fold 2: ~55k steps (saved 30k wasted steps)

---

## 3. High Variance Across Folds: ⚠️ CONCERNING

### Test Performance

| Fold | Test Sharpe | Test Return | Final Value | Val Reward (final) |
|------|------------|-------------|-------------|-------------------|
| 0 | 0.461 | 54.3% | $154,302 | 0.7827 |
| 1 | 0.976 | 145.5% | $245,450 | 0.8731 |
| 2 | **1.378** | **305.0%** | **$404,967** | 0.1325 |

**Statistics:**
- Mean Sharpe: 0.938
- Std Sharpe: 0.375
- CV: **0.40** (high variance)

### Baseline Comparison

| Strategy | Sharpe | Return |
|----------|--------|--------|
| Equal Weight | 1.174 | 121.3% |
| Buy & Hold | 1.058 | 115.9% |
| Risk Parity | 1.133 | 93.2% |
| Momentum | 1.064 | 120.7% |
| Min Variance | 0.940 | 79.4% |

**Fold 2 outperforms Equal Weight by 2.5x in returns!**

---

## 4. Validation/Test Mismatch: 🚨 CRITICAL

### The Paradox

**Fold 2:**
- Validation reward: 0.1325 (WORST)
- Test Sharpe: 1.378 (BEST)

**Fold 0:**
- Validation reward: 0.7827 (BEST)
- Test Sharpe: 0.461 (WORST)

**This inverse relationship is highly suspicious!**

### Possible Explanations

1. **Reward Function Mismatch**
   - Log return reward during training
   - Sharpe ratio during testing
   - These may not correlate well

2. **Market Regime Change**
   - Fold 2 Val: 2021-12 to 2023-08 (Bear market, recovery)
   - Test: 2023-08 to 2025-10 (AI boom, massive rally)
   - Fold 2 learned "bearish" strategies that worked in "bullish" test period?

3. **Lucky Timing**
   - Fold 2 model happened to align with AI boom
   - Not reliable for future performance

4. **Overfitting to Validation**
   - Models fit validation-specific patterns
   - Don't generalize to test

---

## 5. Market Regime Analysis

### Validation Period Characteristics

| Fold | Val Period | Avg Momentum | Regime |
|------|-----------|--------------|--------|
| 0 | 2017-08 to 2019-04 | +0.0136 | Bullish |
| 1 | 2019-10 to 2021-06 | +0.0836 | Very Bullish |
| 2 | 2021-12 to 2023-08 | **-0.0078** | **Bearish** |

### Test Period: 2023-08 to 2025-10

- NVDA: +500% (AI chip boom)
- AMD: +200% (AI chips)
- Overall: Extremely bullish tech market

**Observation:** Fold 2 trained in bearish period but tested in bullish period. This regime mismatch might explain the paradox.

---

## 6. Root Cause Analysis

### Why Results Are Poor (High Variance)

**Primary Issues:**
1. ✗ Early stopping didn't work → Models overtrained
2. ✗ Log return reward ≠ Sharpe ratio → Training objective mismatch
3. ✗ High variance across folds → Models unstable
4. ✗ Market regime mismatch → Fold 2 got lucky

**Secondary Issues:**
5. Only 3 folds → Not enough to assess true performance
6. Validation periods too short (616 samples each)
7. LSTM may be too complex for weekly data

### Why No Leakage Detected

✓ Fold splits are correct with 26-week embargo
✓ Feature normalization uses only past data
✓ Walk-forward validation properly implemented

**The problem is NOT data leakage. The problem is overfitting and training instability.**

---

## 7. Recommendations

### Immediate Actions (Fix Early Stopping)

1. **Verify Early Stopping Callback**
   ```python
   # Current config
   EARLY_STOP_PATIENCE = 5

   # Check: Is callback actually connected?
   # Check: Is min_evals=10 preventing early triggering?
   ```

2. **Use Best Model, Not Final Model**
   - Models saved at peak validation performance
   - Don't use final models (overtrained)

3. **Lower Min Evals**
   ```python
   stop_callback = StopTrainingOnNoModelImprovement(
       max_no_improvement_evals=5,
       min_evals=5,  # Was 10, reduce to 5
       verbose=1
   )
   ```

### Medium-term Fixes

4. **Align Training and Testing Objectives**
   - Option A: Use Sharpe-based reward during training
   - Option B: Evaluate models using log return on test (not Sharpe)

5. **Reduce Model Capacity**
   ```python
   LSTM_HIDDEN_SIZE = 50  # Was 100, reduce to 50 (Jiang 2017 minimum)
   ```

6. **Increase Validation Frequency**
   ```python
   EVAL_FREQ = 2500  # Was 5000, check more often
   ```

7. **Ensemble Models**
   - Don't pick single best fold
   - Average predictions from all 3 folds
   - More stable, less prone to overfitting

### Long-term Solutions

8. **More Folds** (5-10 instead of 3)
   - Better estimate of true performance
   - Reduce impact of any single lucky/unlucky fold

9. **Longer Validation Periods**
   - Current: 616 samples (~12 weeks per ticker)
   - Target: 1000+ samples (~20 weeks per ticker)

10. **Alternative Architecture**
    - Try simpler MLP (no LSTM)
    - Try smaller LSTM (32-50 hidden units)
    - Literature shows simpler often better (Jiang 2017)

---

## 8. What to Do Now

### Conservative Approach (Recommended)

**Use Fold 1 model** instead of Fold 2:
- Fold 1: Sharpe 0.976, Return 145%
- More stable validation performance (24.5% drop vs 44% for Fold 2)
- Trained on bull market (2019-2021) closer to test period
- Still beats most baselines

**Why not Fold 2?**
- Validation/test mismatch is too suspicious
- 44% validation performance drop indicates severe overfitting
- 305% return might be luck, not skill

**Why not Fold 0?**
- Poor test performance (Sharpe 0.461)
- Below all baselines
- Trained on too little data (only 595 samples)

### Aggressive Approach (If You Believe in Fold 2)

**Use Fold 2 model** but with caveats:
- Acknowledge high uncertainty
- Monitor closely in production
- Have stop-loss ready
- Don't expect 305% returns to continue

### Ensemble Approach (Best Practice)

**Average predictions from all 3 folds:**
```python
# At each timestep
action_fold0 = model_fold0.predict(obs)
action_fold1 = model_fold1.predict(obs)
action_fold2 = model_fold2.predict(obs)

# Average the actions
action_ensemble = (action_fold0 + action_fold1 + action_fold2) / 3
```

Expected result: More stable than any single fold

---

## 9. Comparison: Minimal Features vs Original

**You asked if minimal features (22) are better than original (100).**

**Can't answer yet because:**
1. Early stopping failed (models overtrained)
2. High variance across folds (unstable)
3. Need to retrain with fixes first

**To properly compare:**
1. Fix early stopping
2. Retrain minimal features (22)
3. Retrain original features (100)
4. Compare with same experimental setup

**Hypothesis:** Minimal features SHOULD be better because:
- Lower overfitting risk (sample/feature ratio: 32 → 136)
- Research-backed feature selection
- Less redundancy

**But current results don't prove this due to training issues.**

---

## 10. Final Verdict

### Data Leakage: ✓ NO LEAKAGE

- Fold splits: ✓ Correct
- Feature normalization: ✓ Correct
- Walk-forward: ✓ Correct

### Training Quality: ✗ MAJOR ISSUES

- Early stopping: ✗ Failed
- Overfitting: ✗ All folds
- Stability: ✗ High variance
- Val/Test correlation: ✗ Inverse!

### Performance: 🤷 UNCLEAR

- Fold 2 results (305%) are suspicious
- Could be overfitting or lucky timing
- Need to fix training process and retest

---

## Next Steps

1. **Immediate:** Fix early stopping (min_evals=5)
2. **Short-term:** Retrain with fixes, use best validation checkpoint
3. **Medium-term:** Implement ensemble, reduce LSTM size
4. **Long-term:** More folds, align training/test objectives

**Bottom line:** The system works mechanically (no leakage), but the training process needs refinement (early stopping, model capacity, stability).
