# Quick Start: Minimal Feature Implementation

**Goal:** Reduce features from 100 → 17 and retrain models
**Time:** 30 minutes
**Expected Impact:** Reduced overfitting, better generalization

---

## Step-by-Step Instructions

### **Step 1: Test the Minimal Feature Pipeline (5 min)**

Run the diagnostic and test script:

```bash
# First, see current state
python diagnose_features.py

# Then, create minimal feature dataset
python test_minimal_features.py
```

**Expected Output:**
```
Feature reduction: 100 → 17
Sample/feature ratio improvement:
  Before: 4.5 samples/feature
  After:  26.5 samples/feature

✓ Saved processed data to: data/processed/train_minimal_features.parquet
✓ Saved metadata to: data/processed/metadata_minimal.json
```

**Verify files created:**
```bash
ls -lh data/processed/train_minimal_features.parquet
ls -lh data/processed/metadata_minimal.json
```

---

### **Step 2: Update Training Configuration (2 min)**

Open `config.py` and add these lines:

```python
# At the top, around line 17-19
DATA_PATH_MINIMAL = PROJECT_ROOT / 'data' / 'processed' / 'train_minimal_features.parquet'
METADATA_PATH_MINIMAL = PROJECT_ROOT / 'data' / 'processed' / 'metadata_minimal.json'

# Update feature loading (around line 31-37)
try:
    with open(METADATA_PATH_MINIMAL, 'r') as f:
        FEATURE_COLS = json.load(f)['feature_cols']
except FileNotFoundError:
    # Fallback to original if minimal doesn't exist
    try:
        with open(METADATA_PATH, 'r') as f:
            FEATURE_COLS = json.load(f)['feature_cols']
    except FileNotFoundError:
        FEATURE_COLS = []
        print(f"Warning: No metadata found. FEATURE_COLS will be empty.")
```

---

### **Step 3: Create Training Notebook with Minimal Features (10 min)**

Copy existing training notebook and modify:

```bash
# Copy your best training notebook
cp 06_train_lstm_ppo.ipynb 07_train_lstm_minimal_features.ipynb
```

In the new notebook, change the data loading cell:

```python
# OLD (cell ~3-5)
train_data = pd.read_parquet('data/processed/train.parquet')

# NEW
train_data = pd.read_parquet('data/processed/train_minimal_features.parquet')

# Update metadata loading
with open('data/processed/metadata_minimal.json', 'r') as f:
    metadata = json.load(f)
feature_cols = metadata['feature_cols']

print(f"✓ Loaded {len(feature_cols)} minimal features (was 100)")
print(f"Sample/feature ratio: {len(train_data) / len(feature_cols):.1f}")
```

**Important:** Update the output directory so results don't overwrite old models:

```python
# In training cell
OUTPUT_DIR = Path('models_minimal_features')
OUTPUT_DIR.mkdir(exist_ok=True)
```

---

### **Step 4: Run Training (10 min)**

Run the notebook cells:

1. Data loading → Should show "17 features" instead of 100
2. Environment creation → Should work identically
3. Model training → May be slightly faster (fewer features to process)
4. Evaluation → Compare to baseline

**Monitor these metrics:**
- Training reward curve (should be smoother)
- Validation Sharpe ratio (should stabilize better)
- Training time per episode (may be slightly faster)

---

### **Step 5: Compare Results (5 min)**

After training completes, compare to baseline:

```python
# In notebook, add comparison cell
import json

# Load old results (if you saved them)
try:
    with open('models/training_summary.txt', 'r') as f:
        baseline_results = f.read()
    print("=== BASELINE (100 features) ===")
    print(baseline_results)
except:
    print("Baseline results not found")

# Load new results
with open('models_minimal_features/training_summary.txt', 'r') as f:
    minimal_results = f.read()
print("\n=== MINIMAL FEATURES (17 features) ===")
print(minimal_results)

# Key metrics to compare:
# - Final validation Sharpe ratio
# - Overfitting (train vs validation gap)
# - Training stability (fewer spikes in reward curve)
```

---

## What to Expect

### **Positive Signs (Success):**
✓ Validation performance more stable (fewer spikes/drops)
✓ Smaller gap between training and validation Sharpe
✓ Test set performance improves
✓ Reward curves smoother

### **Neutral Signs (Expected):**
- Training may be marginally faster (~5-10%)
- Peak training Sharpe may be similar or slightly lower
- Model converges to similar final policy

### **Warning Signs (Investigate):**
⚠️ Validation Sharpe drops significantly (>20%)
⚠️ Training becomes unstable (more spikes)
⚠️ Model fails to converge

If you see warning signs:
1. Check feature normalization applied correctly
2. Verify all 17 features exist in data
3. Ensure no NaN/inf values introduced
4. Compare feature distributions (old vs new)

---

## Troubleshooting

### **Error: "Feature 'X' not found in dataframe"**

**Cause:** Minimal feature set requires these features to exist in original data.

**Solution:** Check which features are missing:
```python
import pandas as pd
import json

df = pd.read_parquet('data/processed/train.parquet')
with open('data/processed/metadata.json', 'r') as f:
    current_features = json.load(f)['feature_cols']

from feature_fixes import get_all_minimal_features
minimal_features = get_all_minimal_features()

missing = [f for f in minimal_features if f not in current_features]
print(f"Missing features: {missing}")
```

If many features missing, your feature engineering pipeline may be different. Use only features that exist:
```python
available = [f for f in minimal_features if f in current_features]
print(f"Using {len(available)} available features")
```

---

### **Error: "NaN values in normalized features"**

**Cause:** Normalization window too large for early data.

**Solution:** Reduce window or increase min_periods:
```python
# In test_minimal_features.py or feature_fixes.py
normalize_features_per_ticker(
    df,
    features,
    window=52,         # Was: 52 (1 year)
    min_periods=26     # Was: 13 (reduce to 6 months)
)
```

---

### **Warning: "Sample/feature ratio still < 10"**

**Cause:** Not enough data or too many features kept.

**Check:**
```python
n_samples = len(train_data)
n_features = len(feature_cols)
ratio = n_samples / n_features

if ratio < 10:
    print(f"⚠️ Ratio too low: {ratio:.1f}")
    print(f"Need to either:")
    print(f"  - Add more data (current: {n_samples} samples)")
    print(f"  - Reduce features (current: {n_features} features)")
```

---

### **Performance didn't improve**

**Possible causes:**

1. **Overfitting wasn't the main issue**
   - Check if baseline had large train/val gap
   - If gap was small, overfitting wasn't the problem

2. **Features removed were informative**
   - Run feature importance analysis:
   ```python
   from sklearn.ensemble import RandomForestRegressor
   # ... analyze feature importance ...
   ```

3. **Other issues dominate**
   - Reward function design
   - Hyperparameter tuning
   - Environment formulation
   - Training instability

**Next step:** Move to graph-enhanced features (Option 2) or investigate other issues.

---

## Success Criteria

After completing this quick start, you should have:

- [x] Reduced features from 100 → 17
- [x] Applied per-ticker z-score normalization
- [x] Retrained model with minimal features
- [x] Compared performance to baseline
- [x] Documented results

**If performance improved or stayed similar:** ✓ Success! You've reduced overfitting risk with no performance loss. Consider adding graph features next (Option 2).

**If performance degraded significantly:** Need to investigate. Check feature distributions, normalization, and ensure features are informative.

---

## Next Steps After Success

### **Short-term (Week 2)**
1. Run same pipeline on test set
2. Compare multiple folds (walk-forward validation)
3. Document which features were most important
4. Consider adding graph-enhanced features (Option 2)

### **Medium-term (Week 3-4)**
5. Implement graph-enhanced features:
   ```bash
   python feature_engineering_advanced.py
   # Use feature_set='graph_enhanced'
   ```
6. Scrape sector/industry data
7. Add correlation features
8. Retrain and compare

### **Long-term (Month 2+)**
9. If graph features help, consider full GNN or Transformer
10. Implement one of:
    - R-GCN (GPM 2022)
    - Portfolio Transformer (2023)
    - Wavelet Coherence Graph (WCG-RL 2024)

---

## Quick Reference

### **Files Created:**
- `diagnose_features.py` - Diagnostic tool
- `test_minimal_features.py` - Create minimal dataset
- `feature_fixes.py` - Foundational feature engineering
- `feature_engineering_advanced.py` - Advanced features (graph, attention)

### **Data Files Generated:**
- `data/processed/train_minimal_features.parquet` - 17 features, normalized
- `data/processed/metadata_minimal.json` - Feature metadata

### **Key Metrics to Track:**
- Sample/feature ratio: 4.5 → 26.5 ✓
- Training Sharpe: (compare to baseline)
- Validation Sharpe: (compare to baseline)
- Overfitting gap: (should decrease)

### **Time Estimates:**
- Diagnostic: 2 min
- Create minimal dataset: 3 min
- Update config: 2 min
- Modify notebook: 10 min
- Training: 10-30 min (depends on TOTAL_TIMESTEPS)
- Evaluation: 5 min

**Total: ~30-60 minutes**

---

**Ready to start? Run:**
```bash
python diagnose_features.py
python test_minimal_features.py
```

Good luck! 🚀
