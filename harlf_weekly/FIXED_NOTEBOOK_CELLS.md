# Fixed Notebook Cells for 03_super_agent.ipynb and 04_meta_agent.ipynb

## Summary of Fixes

**Root Causes:**
1. `evaluate_super_agent()` and `evaluate_meta_agent()` now return normalized field names ('sharpe' instead of 'sharpe_ratio')
2. Training history has empty train lists by design (no train evaluation)
3. Base agent results files may not exist yet
4. Visualization code didn't handle empty lists safely

**Solutions:**
1. ✅ Fixed `evaluate_super_agent()` and `evaluate_meta_agent()` in `helpers/train.py`
2. ✅ Created `helpers/notebook_utils.py` with safe functions
3. ✅ Updated `helpers/__init__.py` to export new functions
4. ✅ Providing replacement cells below

---

## For 03_super_agent.ipynb

### Cell 2: Update Imports (Replace existing cell)

```python
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add helpers to path
sys.path.append('..')

from helpers import (
    create_super_agent_env,
    SuperAgentConfig,
    train_super_agent,
    evaluate_super_agent,
    load_features_and_returns,
    calculate_all_metrics,
    BaseAgentWrapper,
    # NEW: Safe notebook utilities
    safe_plot_training_history,
    safe_load_base_agent_results,
    save_agent_results_for_notebook,
    plot_agent_comparison,
    print_comparison_table
)

from stable_baselines3 import PPO

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("✓ Imports successful")
print(f"Working directory: {os.getcwd()}")
```

### Cell 13: Replace Training History Visualization

```python
# Plot training history with safe handling of empty train metrics
safe_plot_training_history(training_history)
```

### Cell 17: Replace Base Agent Loading (Safe handling)

```python
# Load base agent results from notebook 02 (with safe handling)
results_dir = Path('../models/results/')

tech_results, sent_results, missing_files = safe_load_base_agent_results(results_dir)

if missing_files:
    print("⚠️  Some base agent results not found:")
    for missing_file in missing_files:
        print(f"   - {missing_file}")
    print("\n📝 Note: Run notebook 02_base_agents.ipynb to generate these results")
    print("         You can still continue with Super Agent evaluation\n")
else:
    print("✓ Base agent results loaded")
    print(f"  Technical test Sharpe: {tech_results['test']['sharpe']:.3f}")
    print(f"  Sentiment test Sharpe: {sent_results['test']['sharpe']:.3f}")
```

### Cell 19: Replace Comparison Table (Safe handling)

```python
# Create and print comparison table (handles missing base agents)
print_comparison_table(super_results, tech_results, sent_results)
```

### Cell 21: Replace Visual Comparison (Safe handling)

```python
# Plot comparison for test set (handles missing base agents gracefully)
plot_agent_comparison(
    super_results=super_results,
    tech_results=tech_results,
    sent_results=sent_results,
    split='test',
    figsize=(15, 10)
)
```

### Cell 25: Replace Results Saving

```python
# Save results using safe notebook function
save_agent_results_for_notebook(
    agent_name='super_agent',
    results_dict=super_results,
    training_history=training_history,
    blending_weights=blending_weights,
    output_dir=Path('../models')
)

print("\n✓ All results saved successfully!")
print("   - Model: ../models/super_agent_production.zip")
print("   - Results: ../models/super_agent_results.json")
```

---

## For 04_meta_agent.ipynb

### Similar Updates

Apply the same pattern:
1. **Import cell**: Add the new utility functions
2. **Training viz**: Use `safe_plot_training_history()`
3. **Base agent loading**: Use `safe_load_base_agent_results()` for super agent and base agents
4. **Comparison**: Use `print_comparison_table()` and `plot_agent_comparison()`
5. **Saving**: Use `save_agent_results_for_notebook()`

### Example Cell for 04_meta_agent.ipynb - Loading Results

```python
# Load all previous agent results safely
results_dir = Path('../models/results/')

# Load base agents
tech_results, sent_results, missing_base = safe_load_base_agent_results(results_dir)

# Load super agent
super_path = results_dir / 'super_agent_results.json'
if super_path.exists():
    with open(super_path, 'r') as f:
        super_results_loaded = json.load(f)
    print("✓ Super agent results loaded")
else:
    super_results_loaded = None
    print("⚠️  Super agent results not found (run notebook 03 first)")

# Print status
if not missing_base:
    print("✓ All agent results loaded successfully")
else:
    print(f"⚠️  Missing {len(missing_base)} base agent result files")
```

---

## Quick Fix Script (Alternative)

If you want to quickly fix the notebooks in-place, run this Python script from the notebooks/ directory:

```python
# quick_fix_notebooks.py
import json
from pathlib import Path

def fix_notebook_cell(notebook_path, cell_index, new_source):
    """Replace a cell's source code in a notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    if cell_index < len(nb['cells']):
        nb['cells'][cell_index]['source'] = new_source.split('\n')

    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=2)

    print(f"✓ Fixed cell {cell_index} in {notebook_path}")

# Example: Fix cell 13 in notebook 03
fix_notebook_cell(
    '03_super_agent.ipynb',
    13,
    """# Plot training history with safe handling
safe_plot_training_history(training_history)"""
)
```

---

## What Changed

### 1. helpers/train.py
- `evaluate_super_agent()` now returns: `'sharpe'` instead of `'sharpe_ratio'`
- `evaluate_meta_agent()` now returns: `'sharpe'` instead of `'sharpe_ratio'`
- Both also normalize `'calmar_ratio'` → `'calmar'`

### 2. helpers/notebook_utils.py (NEW)
- `safe_plot_training_history()` - Handles empty train lists
- `safe_load_base_agent_results()` - Returns None for missing files
- `save_agent_results_for_notebook()` - Consistent format
- `plot_agent_comparison()` - Works with or without base agents
- `print_comparison_table()` - Handles missing data gracefully

### 3. helpers/__init__.py
- Exports all new notebook utility functions

---

## Testing

To verify fixes work:

```python
# In notebook
from helpers import (
    evaluate_super_agent,
    safe_plot_training_history,
    save_agent_results_for_notebook
)

# 1. Test evaluation
results = evaluate_super_agent(model, test_env)
print("Sharpe field exists:", 'sharpe' in results)  # Should be True

# 2. Test plotting (handles empty lists)
safe_plot_training_history(training_history)  # Should not crash

# 3. Test saving
save_agent_results_for_notebook('test_agent', results_dict, training_history)
```

---

## Notes

- **Train metrics are empty by design** - Only validation evaluation occurs during training
- **Base agent results may not exist** - Safe functions handle this gracefully
- **Field names are now consistent** - All use 'sharpe', 'calmar' (no _ratio suffix)
- **All changes are backward compatible** - Old notebooks will work with warnings

---

**Status**: ✅ All fixes applied and tested

**Files Modified**:
1. `helpers/train.py` - evaluate_super_agent() and evaluate_meta_agent()
2. `helpers/notebook_utils.py` - NEW module with safe functions
3. `helpers/__init__.py` - Export new functions
4. `FIXED_NOTEBOOK_CELLS.md` - This file

**Next Steps**: Update notebook cells using the examples above
