# Performance Audit: Quick Summary

**Training:** LSTM=50, patience=4 workaround
**Result:** ❌ **Overall performance DECREASED**

---

## Test Performance Comparison

| Fold | Old Sharpe | New Sharpe | Change | Winner |
|------|-----------|-----------|--------|--------|
| 0 | 1.205 | **0.994** | **-17%** | ❌ Old better |
| 1 | 0.892 | **1.158** | **+30%** | ✅ New better |
| 2 | 1.195 | **0.938** | **-22%** | ❌ Old better |
| **Avg** | **1.097** | **1.030** | **-6%** | ❌ Old better |

**Baseline (Equal Weight):** Sharpe 1.174

---

## What Happened?

### ✅ The Good: Fold 1 Fixed!
- **Old:** Peaked at 2.5k steps → 66% val drop → Sharpe 0.89
- **New:** Peaked at 27.5k steps → 10% val drop → Sharpe 1.16
- **Why:** Early stopping let it escape bad local optimum

### ❌ The Bad: Folds 0 & 2 Broke
- **Old:** Peaked around 10-40k → continued training → Good test performance
- **New:** Peaked at 2.5-7.5k → stopped too early → Bad test performance
- **Why:** Stopped at bad local optima, needed more training

### ❌ The Ugly: Fold 2 Total Failure
- **Validation rewards:** ALL NEGATIVE (-0.01 to -0.29)
- **Peaked at:** First evaluation (2.5k steps)
- **Status:** Model never learned profitable strategy

---

## The Paradox

**Reducing training time:**
- ✅ Fixed the overfit fold (Fold 1)
- ❌ Broke the good folds (Folds 0 & 2)

**This shows:** Different folds need different amounts of training!

---

## Key Metrics

### Early Stopping Behavior

| Fold | Peak Step | Stop Step | Evals Past Peak | Target |
|------|-----------|-----------|-----------------|--------|
| 0 | 7,500 | 20,000 | 5 | ✅ Correct |
| 1 | 27,500 | 40,000 | 5 | ✅ Correct |
| 2 | 2,500 | 20,000 | 7* | ⚠️ Warmup |

*Fold 2 has 7 because it peaked during warmup period

**Conclusion:** Early stopping worked as intended, but results got worse.

---

## Why Did This Happen?

### Training Sensitivity
Small change (patience 5→4) caused dramatic shifts:
- Training paths diverged completely
- Peak times shifted by -81% to +1000%
- Random effects amplified

### Local Optima Problem
- Fold 1 escaped bad local optimum (good!)
- Folds 0 & 2 got stuck in bad local optima (bad!)
- No way to predict which folds benefit from less training

### The SB3 "Bug" Might Actually Help
- Extra 2,500 steps (6 vs 5 evals) lets models escape local optima
- Helps 2/3 folds, hurts 1/3
- Overall: Old config better

---

## Recommendations (Ranked)

### 1. Revert to Old Config ⭐ **RECOMMENDED**
```python
max_no_improvement_evals=5  # Accept 6 evals due to SB3 bug
```
**Why:** Better overall (Sharpe 1.097 vs 1.030), beats baseline 2/3 times

### 2. Increase LSTM to 75
```python
LSTM_HIDDEN_SIZE = 75  # Was 50
```
**Why:** More capacity might prevent early peaks

### 3. Try Multiple Random Seeds
```python
RANDOM_SEED = 123  # Try 42, 123, 456, 789
```
**Why:** Results are highly sensitive to initialization

### 4. Investigate Fold 2 Data
**Why:** Consistently fails with negative rewards

---

## Bottom Line

**The early stopping "fix" made things worse.**

While it technically works (stops at 5 evals), the overall performance decreased by 6%. The old config with the "bug" (stops at 6 evals) actually performs better.

**Recommendation:** Revert to `patience=5` and accept the extra 2,500 training steps. Those steps help 2 out of 3 folds.

---

**Full analysis in:** `PERFORMANCE_AUDIT.md`
