# Final Trial: Old Patience + Larger LSTM

**Date:** 2025-11-20
**Configuration:** patience=5 (old) + LSTM=75 (new)
**Status:** ✅ Ready to train

---

## Configuration

### Changes Applied
```python
# config.py
LSTM_HIDDEN_SIZE = 75  # Was: 50 → Increased by 50%

# Notebook (cell #10)
max_no_improvement_evals = 5  # Was: 4 → Reverted to old config
```

### Full Training Config
```
LSTM_HIDDEN_SIZE = 75           # Compromise between 50 and 100
EARLY_STOP_PATIENCE = 5         # Stops at 6 due to SB3 bug (old behavior)
EVAL_FREQ = 2500                # Frequent monitoring
MIN_EVALS = 3                   # Warmup period
TOTAL_TIMESTEPS = 100,000       # Max training duration
FEATURES = 22                   # Minimal feature set
```

---

## Rationale: Best of Both Worlds

### Why Old Patience (5)?
**Previous trials showed:**
- patience=5 (stops at 6): Overall Sharpe 1.097, beats baseline 2/3 times ✅
- patience=4 (stops at 5): Overall Sharpe 1.030, beats baseline 0/3 times ❌

**The extra 2,500 steps help:**
- Folds 0 & 2 escape local optima
- Only Fold 1 suffered from overtraining
- Net benefit: +6% performance

### Why LSTM=75?
**Previous trials showed:**
- LSTM=50: Peaked too early (2.5k-7.5k steps) for Folds 0 & 2
- LSTM=100: Original overfitting issues (20-66% val drops)
- LSTM=75: Compromise - more capacity, less overfitting

**Expected benefits:**
- More parameters → Better learning capacity
- Prevents early peaks (Folds 0 & 2 issue with LSTM=50)
- Still regularized (less than 100)

---

## Expected Results

### Training Behavior

| Fold | Expected Peak | Expected Stop | Evals Past Peak |
|------|--------------|---------------|-----------------|
| 0 | 30-50k steps | 45-65k | 6 |
| 1 | 20-40k steps | 35-55k | 6 |
| 2 | 20-40k steps | 35-55k | 6 |

**Key improvement:** Later peaks due to increased capacity (LSTM=75)

### Performance Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| Average Sharpe | > 1.10 | 1.097 (old config) |
| Folds beating baseline | ≥ 2/3 | Equal Weight: 1.174 |
| Validation drops | < 25% | Previous: 20-66% |

---

## Comparison to Previous Trials

### Trial 1: LSTM=100, patience=5
- **Result:** Sharpe 1.097, but high overfitting (20-66% val drops)
- **Issue:** Too much capacity → overfit

### Trial 2: LSTM=50, patience=5
- **Result:** Sharpe 1.097 (same as Trial 1)
- **Issue:** Folds 0 & 2 peaked early, needed more capacity

### Trial 3: LSTM=50, patience=4
- **Result:** Sharpe 1.030 (worse!)
- **Issue:** Stopped too early, caught bad local optima

### Trial 4: LSTM=75, patience=5 ⭐ **THIS TRIAL**
- **Expected:** Sharpe > 1.10
- **Why:** More capacity (vs 50) + better training duration (patience=5)

---

## What to Watch For

### Success Indicators
✅ All folds peak after 20k steps (not 2.5k-7.5k like LSTM=50)
✅ Validation drops < 25% for all folds
✅ Test Sharpe > 1.0 for all folds
✅ At least 2/3 folds beat Equal Weight baseline (1.174)

### Warning Signs
⚠️ Any fold peaks before 10k steps → capacity still too low
⚠️ Validation drops > 40% → still overfitting
⚠️ Fold 2 has negative rewards → data/initialization issue

---

## Training Plan

### Step 1: Start Training
```bash
jupyter notebook 07_train_minimal_features.ipynb
```
Then: **Kernel → Restart & Run All**

### Step 2: Monitor Progress (4-6 hours)
Watch for:
- Peak times (should be later than LSTM=50)
- Early stopping messages (should stop at ~6 evals past peak)
- Validation rewards (should be positive for all folds)

### Step 3: Analyze Results
```bash
python analyze_overfitting.py
```

Compare to previous trials:
- LSTM=50, patience=5: Sharpe 1.097
- LSTM=50, patience=4: Sharpe 1.030
- Target: Sharpe > 1.10

---

## Decision Tree After Training

### If Results are Good (Sharpe > 1.10, 2/3 beat baseline)
✅ **SUCCESS!** This is the final configuration.
- Document results
- Use for production/paper

### If Results are Similar (Sharpe ~1.10)
⚠️ LSTM=75 didn't help much
- Try LSTM=100 with patience=5
- Or try different random seeds

### If Results are Worse (Sharpe < 1.05)
❌ LSTM=75 overcomplicated
- Revert to LSTM=50, patience=5 (Trial 2)
- Or investigate Fold 2 data issues

---

## Files Ready

```
✓ config.py                         - LSTM=75
✓ 07_train_minimal_features.ipynb  - patience=5
✓ models_minimal_features/          - Deleted (fresh start)
✓ analyze_overfitting.py           - Analysis tool
```

---

## Quick Checklist

- [x] LSTM_HIDDEN_SIZE = 75
- [x] max_no_improvement_evals = 5
- [x] Models directory deleted
- [x] Imports verified
- [x] Training data ready (22 features)
- [x] Config loaded successfully

---

## Expected Timeline

```
Training:        4-6 hours
Analysis:        5 minutes
Decision:        Based on results
Next iteration:  If needed
```

---

## Bottom Line

**This trial combines:**
- ✅ Better training duration (patience=5, worked in Trials 1 & 2)
- ✅ More model capacity (LSTM=75, compromise between 50 and 100)
- ✅ Evidence-based features (22 minimal features)

**Expected outcome:**
- Folds 0 & 2: Peak later (not 2.5k-7.5k) → Better learning
- Fold 1: Similar or better (already good with patience=5)
- Overall: Sharpe > 1.10, beat baseline 2-3/3 times

**This should be the best configuration yet!** 🎯

---

**Ready to train. Just open the notebook and run all cells!**
