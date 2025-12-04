# Performance Comparison: LSTM=100 vs LSTM=50

**Date:** 2025-11-20
**Experiment:** Reduce LSTM size (100→50) + increase eval frequency (5000→2500)

---

## Configuration Changes

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| LSTM_HIDDEN_SIZE | 100 | 50 | Literature shows 50-75 units sufficient |
| EVAL_FREQ | 5000 | 2500 | More frequent monitoring to catch degradation |

---

## Results Summary

### Validation Performance (Overfitting Detection)

| Fold | LSTM Size | Best Step | Final Step | Val Drop | Status |
|------|-----------|-----------|------------|----------|--------|
| **0** | 100 | ? | ? | 29.4% | ⚠️ |
| **0** | 50 | 40k | 55k | **20.4%** | ✓ **IMPROVED** |
| **1** | 100 | ? | ? | 7.4% | ✓ Best before |
| **1** | 50 | 2.5k | 22.5k | **66.1%** | ❌ **CATASTROPHIC** |
| **2** | 100 | ? | ? | 22.1% | ⚠️ |
| **2** | 50 | 10k | 25k | **46.4%** | ❌ **WORSE** |

### Test Performance (Out-of-Sample)

| Fold | LSTM Size | Test Sharpe | Test Return | Status |
|------|-----------|-------------|-------------|--------|
| **0** | 100 | 0.752 | ? | - |
| **0** | 50 | **1.205** | **226.1%** | ✓ **+60% Sharpe!** |
| **1** | 100 | 0.906 | ? | - |
| **1** | 50 | 0.892 | 89.9% | ≈ Similar |
| **2** | 100 | 1.521 | ? | - |
| **2** | 50 | **1.195** | 151.5% | ❌ **-21% Sharpe** |

### Baseline Comparison (LSTM=50, Test Set)

| Strategy | Sharpe | Return | Status |
|----------|--------|--------|--------|
| **LSTM_PPO_fold0** | 1.205 | 226.1% | 🥇 BEST |
| **LSTM_PPO_fold2** | 1.195 | 151.5% | 🥈 2nd |
| Equal Weight | 1.174 | 121.3% | Baseline |
| LSTM_PPO_fold1 | 0.892 | 89.9% | 🔴 Worst |

---

## Critical Findings

### 1. **Early Stopping Still Not Working**
- Fold 1: Peaked at step 2,500, continued to 22,500 (20k steps past peak!)
- Fold 2: Peaked at step 10,000, continued to 25,000 (15k steps past peak)
- **Root cause:** Early stopping should trigger at 5 evals without improvement, but it's allowing 8+ evals

### 2. **Fold 1 Collapse**
- Validation reward dropped 66% from peak (1.08 → 0.36)
- Peaked EXTREMELY early (2.5k steps = 1st evaluation!)
- Model never recovered, yet training continued for 20k more steps
- **This is reward hacking** - found local optimum immediately, then diverged

### 3. **Mixed Test Results**
- Fold 0: Dramatic improvement (Sharpe 0.75 → 1.21) ✓
- Fold 1: Stable despite val collapse (Sharpe 0.91 → 0.89) ≈
- Fold 2: Performance decreased (Sharpe 1.52 → 1.20) ✗

### 4. **Validation-Test Disconnect**
- Fold 1: Worst validation (66% drop) → Similar test performance
- Fold 2: Moderate validation (46% drop) → Good test performance
- **Implication:** Models are overfitting to validation set patterns

---

## What Went Wrong?

### Expected Outcome (Literature):
- Smaller model (50 units) should reduce overfitting
- More frequent eval (2500) should catch degradation faster
- **Expected result:** 10-15% validation drops, better test performance

### Actual Outcome:
- ❌ Validation drops INCREASED (20-66% vs 7-29%)
- ❌ Early stopping STILL not working (training 15-20k steps past peak)
- ⚠️ Test results mixed (1 better, 1 same, 1 worse)

### Root Causes:

1. **Model too small?**
   - 50 units might be under-capacity for 22 features
   - Literature (Jiang 2017) used 100 features → 50 units
   - We have 22 features → maybe 50 is actually appropriate?

2. **Early stopping broken**
   - Patience=5 should stop after 5 evals without improvement
   - Actually running 8+ evals past peak
   - This explains ALL the overfitting

3. **Fold 1 reward hacking**
   - Peaked at FIRST evaluation (2.5k steps)
   - This suggests immediate exploitation of some pattern
   - Then degraded for 20k steps while training continued

---

## Recommendations

### Option 1: Fix Early Stopping FIRST ⭐ RECOMMENDED
**Why:** Early stopping is clearly broken - all folds trained 15-20k steps past peak

**Action:**
1. Debug early stopping callback implementation
2. Verify `patience=5` actually stops after 5 evaluations
3. Check if `min_evals=3` is interfering with stopping logic
4. Rerun with current LSTM=50 once early stopping works

**Expected:** Dramatic improvement even with LSTM=50

### Option 2: Revert to LSTM=100 + Keep EVAL_FREQ=2500
**Why:** LSTM=100 might be the right size, just needed more frequent monitoring

**Action:**
1. Change `LSTM_HIDDEN_SIZE = 100` in config.py
2. Keep `EVAL_FREQ = 2500` (more frequent monitoring is good)
3. Fix early stopping first
4. Retrain

**Expected:** Better stability with proper early stopping

### Option 3: Try Intermediate Size (LSTM=75)
**Why:** Compromise between capacity (100) and regularization (50)

**Action:**
1. Change `LSTM_HIDDEN_SIZE = 75` in config.py
2. Keep `EVAL_FREQ = 2500`
3. Fix early stopping first
4. Retrain

**Expected:** Balanced performance

### Option 4: Investigate Fold 1 Collapse
**Why:** Peaking at first eval (2.5k) suggests reward hacking

**Action:**
1. Examine Fold 1 training data (2017-2019 dates)
2. Check if specific market regime allows easy exploitation
3. Plot training rewards vs validation rewards for Fold 1
4. May need different reward scaling or clipping for this period

**Expected:** Understand why model found immediate local optimum

---

## Next Steps

**PRIORITY 1:** Fix early stopping
- Current early stopping allows 8+ evals past peak (should be 5)
- This is THE ROOT CAUSE of all overfitting

**PRIORITY 2:** Investigate Fold 1 reward hacking
- Peaking at 2.5k steps (first eval) is suspicious
- Model found immediate exploitation, never learned proper policy

**PRIORITY 3:** Once early stopping works, decide on LSTM size
- Rerun with LSTM=50 and working early stopping
- Compare to LSTM=75 and LSTM=100
- Choose based on validation stability

---

## Bottom Line

**The experiment failed, but we learned why:**

1. ❌ Smaller model didn't help because early stopping is STILL broken
2. ❌ More frequent eval helped catch degradation, but couldn't stop it
3. ⚠️ Fold 1 shows severe reward hacking (peaked at first eval)

**The fix is NOT changing LSTM size - it's fixing early stopping!**

Once early stopping works properly, LSTM=50 might actually perform as literature predicts.
