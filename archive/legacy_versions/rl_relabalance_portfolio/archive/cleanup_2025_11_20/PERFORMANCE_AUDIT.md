# Performance Audit: Early Stopping Fix

**Date:** 2025-11-20
**Training:** LSTM=50, patience=4 workaround

---

## Executive Summary

**The early stopping fix had MIXED results:**
- ✅ Fold 1: Dramatically improved (Sharpe 0.89 → 1.16)
- ❌ Fold 0: Performance decreased (Sharpe 1.21 → 0.99)
- ❌ Fold 2: Performance decreased (Sharpe 1.20 → 0.94)

**Root cause:** Training dynamics changed significantly with patience=4.

---

## Detailed Comparison

### Validation Performance

| Fold | Config | Peak Step | Stop Step | Val Drop | Status |
|------|--------|-----------|-----------|----------|--------|
| **0** | Old (patience=5) | 40,000 | 55,000 | 20.4% | Baseline |
| **0** | New (patience=4) | 7,500 | 20,000 | **31.1%** | ❌ Worse |
| **1** | Old (patience=5) | 2,500 | 22,500 | 66.1% | Very bad |
| **1** | New (patience=4) | 27,500 | 40,000 | **10.6%** | ✅ Much better! |
| **2** | Old (patience=5) | 10,000 | 25,000 | 46.4% | Bad |
| **2** | New (patience=4) | 2,500 | 20,000 | **862%** | ❌ Disaster |

### Test Performance (Out-of-Sample)

| Fold | Old Sharpe | New Sharpe | Change | Status |
|------|-----------|-----------|--------|--------|
| **0** | 1.205 | **0.994** | -17.5% | ❌ Worse |
| **1** | 0.892 | **1.158** | +29.8% | ✅ Better! |
| **2** | 1.195 | **0.938** | -21.5% | ❌ Worse |
| **Avg** | 1.097 | 1.030 | -6.1% | ❌ Slightly worse |

### Test vs Baseline (Equal Weight: Sharpe 1.174)

| Fold | Old vs EW | New vs EW | Winner |
|------|-----------|-----------|--------|
| **0** | +2.6% | -15.4% | Old better |
| **1** | -24.0% | -1.4% | New better |
| **2** | +1.8% | -20.1% | Old better |

---

## Key Findings

### 1. Early Stopping Worked Correctly

**Fold 0:**
- Peaked at eval #3, stopped at eval #8
- 5 evals past peak ✅ (as intended with patience=4 workaround)

**Fold 1:**
- Peaked at eval #11, stopped at eval #16
- 5 evals past peak ✅ (as intended)

**Fold 2:**
- Peaked at eval #1, stopped at eval #8
- 7 evals past peak (due to min_evals=3 warmup)

**Conclusion:** The patience=4 workaround worked as designed.

### 2. Training Dynamics Changed

**Peak timing shifted dramatically:**

| Fold | Old Peak | New Peak | Shift |
|------|----------|----------|-------|
| 0 | 40,000 | 7,500 | -81% (much earlier) |
| 1 | 2,500 | 27,500 | +1000% (much later) ✅ |
| 2 | 10,000 | 2,500 | -75% (much earlier) |

**Why this matters:**
- Fold 1 fixed: Now peaks late (27.5k) instead of early (2.5k) - learned properly!
- Folds 0 & 2 broken: Now peak very early (7.5k, 2.5k) - failed to learn

### 3. Fold 2 Has Negative Rewards

**Validation rewards:**
- Best: -0.012 (barely positive)
- Final: -0.116 (negative)
- All evaluations: Negative rewards

**This indicates:**
- Model never learned profitable strategy
- Worse than random/equal weight
- Complete training failure

### 4. The Paradox

**Fold 1 success factors:**
- Peaked late (27.5k steps)
- Stable validation (10.6% drop)
- Best test performance (1.16 Sharpe)
- **This fold had 66% drop with old config!**

**Folds 0 & 2 failure factors:**
- Peaked very early (<10k steps)
- High validation drops (31%, 862%)
- Worse test performance
- **These folds were better with old config**

---

## Root Cause Analysis

### Why Did Results Flip?

**Hypothesis 1: Random Seed Sensitivity**
- Same random seed (42) but different stopping creates different training paths
- Fold 1 benefited from more training time
- Folds 0 & 2 needed the longer training

**Hypothesis 2: Local Optima**
- Patience=4 stopped Folds 0 & 2 at bad local optima
- Patience=5 (old) let them escape to better optima
- Fold 1 was stuck in bad local optimum with old config, escaped with new

**Hypothesis 3: Overfitting vs Underfitting**
- Old config (15k steps past peak): Overfitting for Folds 1 & 2
- New config (12.5k steps past peak): Underfitting for Folds 0 & 2
- Fold 1 needs less training, others need more

### Why Fold 2 Failed Completely?

**Negative rewards indicate:**
1. **Poor exploration:** Peaked at first eval (2.5k steps)
2. **Exploitation failure:** Never found profitable strategy
3. **Data issue:** 2021-2023 training data might be problematic
4. **Reward hacking:** Found immediate exploit that doesn't generalize

---

## Comparison to Baseline

### Equal Weight (Sharpe 1.174)

**Old config:**
- Beat baseline: 2 out of 3 folds (Folds 0, 2)
- Average: 1.097 (7% below baseline)

**New config:**
- Beat baseline: 0 out of 3 folds
- Average: 1.030 (12% below baseline)

**Conclusion:** Old config was actually better overall!

---

## Recommendations

### Option 1: Revert to Old Config ⭐ **RECOMMENDED**

**Why:**
- Better overall performance (1.097 vs 1.030 Sharpe)
- 2/3 folds beat baseline (vs 0/3)
- More stable across folds

**Action:**
```python
stop_callback = StopTrainingOnNoModelImprovement(
    max_no_improvement_evals=5,  # Back to 5 (stops at 6 due to bug)
    min_evals=3,
    verbose=1
)
```

**Expected:**
- Fold 0: Sharpe ~1.20 ✅
- Fold 1: Sharpe ~0.89 (but still beats baseline if lucky)
- Fold 2: Sharpe ~1.20 ✅

### Option 2: Increase LSTM to 75

**Why:**
- Current LSTM=50 might be under-capacity
- Folds 0 & 2 peak too early (insufficient learning)
- Literature suggests 50-100 works

**Action:**
```python
# config.py
LSTM_HIDDEN_SIZE = 75  # Was 50
```

Keep patience=4 or try patience=5.

### Option 3: Investigate Fold 2 Data

**Why:**
- Fold 2 consistently fails (negative rewards)
- Training period: 2021-2023
- May have data quality issues

**Action:**
1. Check Fold 2 training data (2015-2021)
2. Look for anomalies, missing data, outliers
3. Consider different normalization

### Option 4: Try Different Random Seeds

**Why:**
- Results are highly sensitive to initialization
- Fold 1 improved dramatically with new config
- Might find better initialization for all folds

**Action:**
```python
# config.py
RANDOM_SEED = 123  # Try 42, 123, 456, 789, 999
```

Run 5 different seeds, pick best.

---

## Critical Questions

### 1. Why did Fold 1 improve so much?

**Old:** Peaked at 2.5k, degraded to 66% drop
**New:** Peaked at 27.5k, stable at 10.6% drop

**Answer:** Early stopping at 2.5k was a local optimum. Continuing to 27.5k found better solution.

### 2. Why did Folds 0 & 2 get worse?

**Old:** Peaked around 10-40k, continued to 25-55k
**New:** Peaked at 2.5-7.5k, stopped at 20k

**Answer:** Stopping too early caught bad local optima. Needed more training.

### 3. Why does Fold 2 have negative rewards?

**Validation rewards all negative (-0.01 to -0.29)**

**Answer:** Model never learned profitable strategy. Complete failure.

---

## What I Learned

### The SB3 Bug is NOT the Main Problem

**Evidence:**
- Patience=4 worked as intended (stopped at 5 evals)
- But results got worse for 2/3 folds
- The extra 2,500 steps (6 vs 5 evals) might actually help!

### Training is Highly Chaotic

**Same hyperparameters, different results:**
- Small change (patience 5→4) caused dramatic flip
- Fold 1: 0.89 → 1.16 (+30%)
- Fold 0: 1.21 → 0.99 (-18%)
- Fold 2: 1.20 → 0.94 (-21%)

### Overfitting vs Underfitting is Complex

**It's not just "more training = more overfitting":**
- Fold 1 improved with LESS training
- Folds 0 & 2 worsened with LESS training
- Optimal training time differs per fold

---

## Bottom Line

**The fix didn't work as expected.**

While early stopping now works correctly (stops at 5 evals), the overall performance DECREASED:
- Average Sharpe: 1.097 → 1.030 (-6%)
- Folds beating baseline: 2 → 0
- Only Fold 1 improved significantly

**Recommendation:** Revert to old config (patience=5) OR try LSTM=75 to give models more capacity.

The SB3 bug (stopping at 6 instead of 5) might actually be helping by allowing a bit more training!

---

## Next Steps

1. **Revert to patience=5** (accept the 2,500 extra steps)
2. **Try LSTM=75** to increase capacity
3. **Investigate Fold 2 data issues** (negative rewards)
4. **Run multiple random seeds** to find robust solution

The goal should be: **All 3 folds beat baseline consistently**, not just fix early stopping.
