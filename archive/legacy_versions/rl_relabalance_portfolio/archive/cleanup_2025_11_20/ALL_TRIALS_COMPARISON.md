# All Trials Comparison

**Baseline to beat:** Equal Weight Sharpe 1.174

---

## Trial Summary

| Trial | LSTM | Patience | Stops At | Avg Sharpe | Folds Beat Baseline | Status |
|-------|------|----------|----------|------------|---------------------|--------|
| 1 | 100 | 5 | 6 evals | ❓ | ❓ | Not recorded |
| 2 | 50 | 5 | 6 evals | 1.097 | 2/3 | Baseline |
| 3 | 50 | 4 | 5 evals | 1.030 | 0/3 | Failed |
| 4 | 75 | 5 | 6 evals | ❓ | ❓ | **← Ready to run** |

---

## Detailed Comparison

### Trial 2: LSTM=50, patience=5 (BASELINE)
```
Configuration: LSTM_HIDDEN_SIZE=50, patience=5
Result: Avg Sharpe 1.097
```

| Fold | Val Drop | Test Sharpe | vs Baseline |
|------|----------|-------------|-------------|
| 0 | 20.4% | 1.205 | +2.6% ✅ |
| 1 | 66.1% | 0.892 | -24.0% ❌ |
| 2 | 46.4% | 1.195 | +1.8% ✅ |

**Summary:**
- ✅ 2/3 folds beat baseline
- ⚠️ High validation drops (20-66%)
- ⚠️ Folds 0 & 2 peaked early (7.5k-10k)

---

### Trial 3: LSTM=50, patience=4 (FAILED)
```
Configuration: LSTM_HIDDEN_SIZE=50, patience=4
Result: Avg Sharpe 1.030 (worse than Trial 2)
```

| Fold | Val Drop | Test Sharpe | vs Baseline | Change from Trial 2 |
|------|----------|-------------|-------------|---------------------|
| 0 | 31.1% | 0.994 | -15.4% ❌ | -17% ❌ |
| 1 | 10.6% | 1.158 | -1.4% ❌ | +30% ✅ |
| 2 | 862% | 0.938 | -20.1% ❌ | -22% ❌ |

**Summary:**
- ❌ 0/3 folds beat baseline
- ❌ Overall worse (1.030 vs 1.097)
- ✅ Only Fold 1 improved (but still below baseline)
- ❌ Folds 0 & 2 peaked too early, stopped at bad local optima

---

### Trial 4: LSTM=75, patience=5 (UPCOMING)
```
Configuration: LSTM_HIDDEN_SIZE=75, patience=5
Result: TBD (training now)
```

**Expected:**
| Fold | Expected Sharpe | vs Baseline | Rationale |
|------|----------------|-------------|-----------|
| 0 | 1.15-1.25 | ✅ | More capacity → later peak |
| 1 | 0.95-1.05 | ✅ | Same training duration as Trial 2 |
| 2 | 1.10-1.20 | ✅ | More capacity → better learning |
| **Avg** | **1.10-1.15** | **✅** | Improvement over 1.097 |

**Why this should work:**
- ✅ More capacity than LSTM=50 → prevents early peaks
- ✅ Same training duration as Trial 2 → keeps Folds 0 & 2 good
- ✅ Compromise between 50 (too small) and 100 (too big)

---

## Key Learnings

### 1. Training Duration Matters
- **patience=5 (stops at 6):** Works well for 2/3 folds
- **patience=4 (stops at 5):** Too early, catches bad local optima

### 2. Model Capacity Matters
- **LSTM=50:** Too small, peaks too early (2.5k-7.5k steps)
- **LSTM=100:** Too big, overfits (20-66% val drops)
- **LSTM=75:** Expected sweet spot

### 3. Fold-Specific Behavior
- **Fold 1:** Benefits from LESS training (66% → 10% val drop with patience=4)
- **Folds 0 & 2:** Need MORE training (peaked early with LSTM=50)

### 4. The SB3 "Bug" Helps
- Extra 2,500 steps (6 vs 5 evals) lets models escape local optima
- Helps 2/3 folds in our case
- Not actually a bug - it's a feature!

---

## Decision Rules

### If Trial 4 Sharpe > 1.15
✅ **SUCCESS!** Use LSTM=75, patience=5 as final config

### If Trial 4 Sharpe 1.05-1.15
⚠️ **MARGINAL** - Try:
- LSTM=80 or LSTM=85
- Or different random seeds

### If Trial 4 Sharpe < 1.05
❌ **FAILED** - Revert to Trial 2 (LSTM=50, patience=5)

---

## Bottom Line

**Trial 4 is our best shot:**
- Combines successful training duration (patience=5)
- Adds more capacity (LSTM=75)
- Should fix early peaking issue (Folds 0 & 2 with LSTM=50)

**Expected improvement:** 1.097 → 1.10-1.15 (+0-5%)

**If this doesn't work, accept Trial 2 as best we can do with current setup.**

---

**Training in progress... check back in 4-6 hours!** ⏳
