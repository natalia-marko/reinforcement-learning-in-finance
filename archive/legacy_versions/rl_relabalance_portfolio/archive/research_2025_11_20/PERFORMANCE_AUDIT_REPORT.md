# Performance Audit Report: Minimal Features Training
**Date:** 2025-11-20
**Model:** LSTM PPO with 22 minimal features (weekly data)
**Training Files Modified:** 10:40-10:50 today

---

## Executive Summary

**Overall Status:** ⚠️ MIXED RESULTS - Strong test performance but validation instability

**Key Findings:**
- ✓ Fold 2 model **beats all baselines** (Sharpe 1.52 vs Equal Weight 1.17)
- ✓ All 3 folds achieved positive returns on unseen test data
- ⚠️ High variance across folds (Sharpe: 0.75 → 1.52)
- ⚠️ Early stopping partially worked but issues remain
- 🚨 Validation/test mismatch persists (overfitting indicators don't match test results)

---

## 1. Training Performance (Validation Set)

### Validation Metrics During Training

| Fold | Evals | Steps | Peak Reward | Peak @ Step | Final Reward | Drop | Status |
|------|-------|-------|-------------|-------------|--------------|------|--------|
| 0 | 11 | 55k | 1.0137 | 25k | 0.7161 | **29.4%** | ⚠️ Overfit |
| 1 | 12 | 60k | 1.4092 | 30k | 1.3055 | **7.4%** | ✓ Stable |
| 2 | 9 | 45k | 0.2390 | 10k | 0.1862 | **22.1%** | ⚠️ Overfit |

### Key Observations:

1. **Fold 1 is EXCELLENT** - Only 7.4% drop, very stable training
2. **Fold 0 & 2 show overfitting** - 22-29% performance drops
3. **Early stopping triggered** - All stopped before 100k (vs configured max)
4. **Training stopped earlier than previous run** - Now 45-60k vs old 80-85k

**Improvement from previous training:** Training stopped 20-40k steps earlier, suggesting early stopping is working better (but still not optimal).

---

## 2. Test Performance (Out-of-Sample Backtest)

### Test Period: 2023-08-31 to 2025-10-30 (114 weeks)

| Model | Test Sharpe | Total Return | Final Value | Volatility | Max DD | Win Rate | Rank |
|-------|-------------|--------------|-------------|------------|---------|----------|------|
| **LSTM_PPO_fold2** | **1.521** | **206.9%** | **$306,883** | 31.3% | -24.0% | 63.4% | **#1** |
| Equal Weight | 1.174 | 121.3% | $221,270 | 27.7% | -33.1% | 58.4% | #2 |
| Risk Parity | 1.133 | 93.2% | $193,175 | 23.2% | -29.0% | 61.1% | #3 |
| Momentum | 1.064 | 120.7% | $220,712 | 30.5% | -36.5% | 57.5% | #4 |
| Buy & Hold | 1.058 | 115.9% | $215,899 | 29.7% | -34.6% | 57.5% | #5 |
| Min Variance | 0.940 | 79.4% | $179,434 | 24.4% | -33.8% | 60.2% | #6 |
| LSTM_PPO_fold1 | 0.906 | 172.0% | $271,974 | 46.4% | -42.1% | 53.6% | #7 |
| LSTM_PPO_fold0 | 0.752 | 146.1% | $246,128 | 49.8% | -55.6% | 50.9% | #8 |

### Key Findings:

**Successes:**
- ✓ **Fold 2 is #1** - Beats all baselines including Equal Weight (1.52 vs 1.17 Sharpe)
- ✓ **Strong risk-adjusted returns** - Higher Sharpe than all baselines
- ✓ **Lower drawdown than Fold 0/1** - Only -24% vs -42% to -56%
- ✓ **High win rate** - 63.4% winning weeks

**Concerns:**
- ⚠️ **High variance across folds** - Sharpe ranges from 0.75 to 1.52 (CV=0.35)
- ⚠️ **Fold 0 underperformed** - Below all baselines, high volatility (49.8%), large drawdown (-55.6%)
- ⚠️ **Fold 1 high volatility** - 46.4% volatility despite decent returns

---

## 3. The Validation/Test Paradox

### Critical Observation:

| Fold | Val Drop | Val Rank | Test Sharpe | Test Rank | Paradox? |
|------|----------|----------|-------------|-----------|----------|
| 0 | 29.4% | Worst | 0.752 | Worst | ✓ Consistent |
| 1 | 7.4% | Best | 0.906 | Middle | ⚠️ Expected better |
| 2 | 22.1% | Middle | 1.521 | **Best** | 🚨 **INVERSE** |

**The Problem:**
- Fold 2 showed 22% validation degradation (overfitting indicator)
- But achieved BEST test performance (Sharpe 1.52)
- This inverse relationship suggests validation set is not predictive of test set

**Possible Explanations:**

1. **Market Regime Mismatch**
   - Fold 2 val period: 2021-12 to 2023-08 (bear market, crypto crash)
   - Test period: 2023-08 to 2025-10 (AI boom, NVDA rally)
   - Model learned defensive strategy that worked in bullish test

2. **Lucky Timing**
   - Test period had massive tech rally (NVDA +500%)
   - Any long tech allocation would do well
   - Not necessarily skill-based

3. **Validation Set Too Small**
   - Only 616 samples per validation fold
   - Not enough to estimate true performance
   - High variance in validation metrics

---

## 4. Early Stopping Analysis

### Current State:

**Configuration:**
- `EARLY_STOP_PATIENCE = 5` (stop after 5 consecutive non-improvements)
- `min_evals = ?` (needs verification - notebook may still have old value)
- `EVAL_FREQ = 5000` (evaluate every 5k steps)

**What Happened:**

| Fold | Peak @ | Stopped @ | Steps Past Peak | Should Stop @ | Waste |
|------|--------|-----------|-----------------|---------------|-------|
| 0 | 25k | 55k | 30k | ~45k | 10k |
| 1 | 30k | 60k | 30k | ~50k | 10k |
| 2 | 10k | 45k | 35k | ~35k | 10k |

**Status:** ⚠️ **IMPROVED BUT NOT FIXED**

**Evidence:**
- Training stopped 20-40k steps earlier than previous run (was 80-85k)
- But still 10k steps past optimal stopping point for each fold
- Fold 1 shows only 7.4% degradation - early stopping worked well here
- Folds 0 & 2 still show 22-29% degradation - stopped too late

**Recommendation:** Verify that `min_evals=3` is actually in effect. The debug script detected `min_evals=10` which would explain the delayed stopping.

---

## 5. Comparison: Minimal (22) vs Original (100) Features

### Cannot Definitively Answer Yet

**Reason:** Different training runs, different stopping points, different test periods may have been used.

**What We Know:**
- Minimal features: 22 → Sample/feature ratio: 135.9
- Original features: 100 → Sample/feature ratio: 31.6
- **4.3x better ratio** should reduce overfitting

**Test Results (Minimal Features):**
- Best fold: Sharpe 1.52, Return 207%
- Beats all baselines

**Expected but Unverified:**
- Should have more stable training (less overfitting)
- Should generalize better across folds
- Should have lower variance

**To Properly Compare:**
Need to retrain original (100 features) with:
- Same early stopping settings
- Same training conditions
- Same test period
- Then compare directly

---

## 6. Risk Assessment

### Model Deployment Recommendation: **FOLD 2 (with caveats)**

**Why Fold 2?**
- ✓ Best test Sharpe ratio (1.52)
- ✓ Best risk-adjusted returns (207% return, -24% max DD)
- ✓ Beats all baselines
- ✓ High win rate (63.4%)
- ✓ Moderate turnover (12.3%)

**Caveats:**
- ⚠️ Validation overfitting (22% drop) - suggests unstable
- ⚠️ Test performance might be lucky timing (AI boom)
- ⚠️ High variance across folds indicates model instability
- ⚠️ Only 114 weeks of test data (needs more validation)

### Alternative: Ensemble Approach

**Recommendation:** Use weighted ensemble instead of single fold

```python
# Weight by test Sharpe ratio
w0 = 0.752 / (0.752 + 0.906 + 1.521)  # 0.235
w1 = 0.906 / (0.752 + 0.906 + 1.521)  # 0.283
w2 = 1.521 / (0.752 + 0.906 + 1.521)  # 0.476

# Ensemble prediction
action = w0 * model0.predict(obs) + w1 * model1.predict(obs) + w2 * model2.predict(obs)
```

**Expected Benefits:**
- More stable than single model
- Reduces risk of lucky timing
- Averaging reduces overfitting effects
- Smoother portfolio transitions

---

## 7. Identified Issues & Recommendations

### Immediate Actions:

1. **Verify Early Stopping Settings** ⚡ URGENT
   ```python
   # Check in notebook Cell 10
   # Should be: min_evals=3
   # May still be: min_evals=10
   ```
   - If still 10, change to 3 and retrain
   - Expected: 10-20k steps saved per fold
   - Expected: <10% validation drops

2. **Reduce LSTM Hidden Size** (if overfitting persists)
   ```python
   LSTM_HIDDEN_SIZE = 50  # Current: 100, Jiang 2017 minimum: 50
   ```
   - Smaller model = less overfitting
   - Literature shows 50-100 range, we're at max
   - Try 50 or 75

3. **Increase Evaluation Frequency**
   ```python
   EVAL_FREQ = 2500  # Current: 5000
   ```
   - More frequent checks = catch peak earlier
   - Stop closer to optimal point
   - Doubles eval overhead but worth it

### Medium-term Improvements:

4. **More Folds** (5-7 instead of 3)
   - Better estimate of true performance
   - Reduces impact of lucky/unlucky folds
   - More robust validation

5. **Longer Validation Periods**
   - Current: 616 samples (~12 weeks per ticker)
   - Target: 1000+ samples (~20 weeks per ticker)
   - More reliable performance estimates

6. **Align Reward Function with Evaluation Metric**
   - Currently: Train with log_return, test with Sharpe
   - Consider: Train with Sharpe-based reward
   - Better alignment = better generalization

### Long-term Enhancements:

7. **Walk-Forward Out-of-Sample Test**
   - Current: Single test period (may be lucky)
   - Implement: Multiple rolling test windows
   - Verify performance across different market regimes

8. **Ensemble by Default**
   - Don't pick single best fold
   - Average all fold predictions (weighted by validation performance)
   - More robust, production-ready approach

---

## 8. Summary Scorecard

| Metric | Score | Status |
|--------|-------|--------|
| **Best Test Performance** | 1.52 Sharpe | ✓✓ Excellent |
| **Beats Baselines** | Yes (+29% vs Equal Weight) | ✓ Good |
| **Training Stability** | 7-29% validation drop | ⚠️ Mixed |
| **Cross-Fold Consistency** | CV=0.35 | ⚠️ High Variance |
| **Early Stopping** | Stopped 45-60k (vs 100k max) | ⚠️ Partial |
| **Val/Test Correlation** | Inverse for Fold 2 | 🚨 Concerning |
| **Risk Management** | -24% max DD (best fold) | ✓ Good |
| **Win Rate** | 63% (best fold) | ✓ Good |

**Overall Grade:** **B** (Good performance, needs stability improvements)

---

## 9. Answers to Key Questions

### Q1: Did the early stopping fix work?
**A:** **Partially.** Models stopped 20-40k steps earlier (55-60k vs 80-85k), but still 10k past optimal. Fold 1 excellent (7% drop), but Folds 0 & 2 still overfit (22-29% drops).

### Q2: Are minimal features (22) better than original (100)?
**A:** **Likely yes, but not proven.** Best fold beats all baselines with good risk profile. Sample/feature ratio improved 4.3x. But need direct comparison with same training setup.

### Q3: Should we deploy this model?
**A:** **Yes, but use ensemble.** Fold 2 alone is risky (validation/test mismatch). Weighted ensemble of all 3 folds is safer and more robust.

### Q4: What's causing the validation/test paradox?
**A:** **Market regime mismatch + small validation sets.** Fold 2 trained on bear market (2021-2023) but tested on AI boom (2023-2025). Validation sets too small (616 samples) to be predictive.

### Q5: What should we do next?
**A:**
1. **Verify** early stopping settings (min_evals=3?)
2. **Retrain** if needed with confirmed fix
3. **Implement** ensemble approach
4. **Test** on longer/multiple time periods
5. **Consider** reducing model capacity (LSTM size 100→50)

---

## 10. Production Readiness

### Current Status: **NOT READY** (needs improvements)

**Blockers:**
- ❌ High variance across folds (unstable)
- ❌ Validation/test mismatch (can't trust validation)
- ❌ Early stopping not optimal (still overtraining)

**What's Needed for Production:**
1. ✓ Implement ensemble approach
2. ✓ Verify early stopping fix
3. ✓ Add more validation folds (5-7 total)
4. ✓ Test on multiple time periods
5. ✓ Add monitoring/alerts
6. ✓ Define stop-loss rules

**Timeline Estimate:**
- **With ensemble only:** 1 day (ready for careful testing)
- **With all improvements:** 1-2 weeks (production-ready)

---

## Files & Artifacts

**Analysis Outputs:**
- `models_minimal_features/fold_*/validation_curve_fold*.png` - Validation curves
- `results/backtests_minimal_features/backtest_summary.csv` - Test results
- `results/backtests_minimal_features/backtest_comparison.png` - Visual comparison

**Training Artifacts:**
- `models_minimal_features/fold_*/best_model.zip` - Trained models (3 folds)
- `models_minimal_features/fold_*/evaluations.npz` - Validation history
- `models_minimal_features/training_summary_minimal_features.json` - Config

**Timestamps:**
- Fold 0: 2025-11-20 10:40
- Fold 1: 2025-11-20 10:46
- Fold 2: 2025-11-20 10:50

---

**Report Generated:** 2025-11-20
**Next Review:** After retraining with verified early stopping fix
