## 📁 File Structure

```
portfolio_rl/
├── config.py           # Settings (edit here!)
├── rewards.py          # All 5 reward functions
├── utils.py            # Helper functions
├── environments.py     # Environment code
├── train.py            # Training functions
│
├── notebooks/          # Your Jupyter notebooks
│   ├── quick_start.ipynb
│   ├── 1_train_ema_sharpe.ipynb
│   ├── 2_train_differential_sharpe.ipynb
│   ├── 3_train_multi_objective.ipynb
│   └── 4_compare_all.ipynb
│
└── README.md          # This file
```
---

## 🚀 Quick Start (3 Lines!)

```python
# In any Jupyter notebook:
from train import train

result = train('technical', 'PPO', 'ema_sharpe')
```

**Done!** The model trains and shows results automatically.

---

## 📖 How to Use

### In Your Jupyter Notebooks:

```python
# Import
from train import train

# Train technical agent with EMA Sharpe
result = train('technical', 'PPO', 'ema_sharpe')

# Train sentiment agent with differential Sharpe
result = train('sentiment', 'SAC', 'differential_sharpe')

# Train with multi-objective
result = train('technical', 'A2C', 'multi_objective')
```

### Choose Your Options:

**Agent Type:**
- `'technical'` - Uses technical indicators
- `'sentiment'` - Uses sentiment indicators

**Algorithm:**
- `'PPO'` - Proximal Policy Optimization (fast, stable)
- `'SAC'` - Soft Actor-Critic (slower, good performance)
- `'A2C'` - Advantage Actor-Critic (fast, less stable)

**Reward Type:**
- `'ema_sharpe'` - EMA-based Sharpe ratio
- `'differential_sharpe'` - Differential Sharpe (online optimization)
- `'multi_objective'` - Multi-objective with penalties
- `'simple_return'` - Simple return maximization
- `'simple_sharpe'` - Historical Sharpe ratio

---

## 📚 Example Notebooks

### 1. Quick Start (`notebooks/quick_start.ipynb`)
Super simple introduction - just 3 steps!

### 2-3. Specific Reward Functions
- `1_train_ema_sharpe.ipynb`
- `2_train_differential_sharpe.ipynb`
- `3_train_multi_objective.ipynb`

### 4. Compare All (`notebooks/4_compare_all.ipynb`)
Compare all reward functions to find the best!

---

## ⚙️ Configuration

Edit `config.py` to change settings:

```python
# Training settings
TOTAL_STEPS = 300_000      # How long to train
EVAL_FREQ = 5_000          # Evaluate every N steps
PATIENCE = 5               # Early stopping patience

# Reward-specific configs
EMA_CONFIG = {
    'rolling_vol_window': 12  # Window for EMA
}

MULTI_TECHNICAL = {
    'return_scale': 8.0,              # How much to chase returns
    'volatility_penalty': 0.050,      # Penalize volatility
    'concentration_penalty': 0.250,   # Force diversification
    'turnover_penalty': 0.005,        # Penalize trading
}
```

---

## 🎯 Common Tasks

### Train with All Algorithms

```python
from train import train_all_algos

# Try PPO, SAC, and A2C
results = train_all_algos('technical', 'ema_sharpe')
# Shows comparison table and picks the best!
```

### Train Both Agents

```python
from train import train_both_agents

# Train technical and sentiment at once
results = train_both_agents('ema_sharpe', algorithm='PPO')
```

### Quick Test (for debugging)

```python
from train import quick_test

# Fast 10k step test
result = quick_test('technical', 'ema_sharpe')
```

### Compare Results

```python
from utils import compare_results

results = [result1, result2, result3]
df = compare_results(results)
# Shows nice comparison table
```

### Save Results

```python
from utils import save_results

save_results(result, 'results/my_experiment.json')
```

---

## 📊 What You Get

After training, you get a results dictionary:

```python
{
    'agent_type': 'technical',
    'algorithm': 'PPO',
    'reward_type': 'ema_sharpe',
    'train_sharpe': 2.15,
    'val_sharpe': 1.98,
    'test_sharpe': 1.87,
    'model_path': 'models/technical/ema_sharpe/...'
}
```

---

## 🎓 Understanding the Files

### `config.py`
Settings for training. **Edit this to change behavior!**

### `rewards.py`
All 5 reward functions. Your original formulas are here.

### `utils.py`
Helper functions:
- `load_data()` - Load training data
- `compute_sharpe()` - Calculate Sharpe ratio
- `save_results()` - Save to JSON
- `print_results()` - Pretty print
- `compare_results()` - Compare experiments

### `environments.py`
Portfolio environment. Used by `train.py`.

### `train.py`
**THE MAIN FILE!** Has all training functions:
- `train()` - Train any agent
- `train_all_algos()` - Try all algorithms
- `train_both_agents()` - Train tech + sentiment
- `quick_test()` - Fast test run

---

## 💡 Tips for Data Scientists

### 1. Start Simple
```python
from train import train
result = train('technical', 'PPO', 'ema_sharpe')
```

### 2. Experiment in Notebooks
- Open `notebooks/quick_start.ipynb`
- Run cells one by one
- Modify and re-run

### 3. Compare Approaches
```python
from train import train
from utils import compare_results

results = []
for reward in ['ema_sharpe', 'differential_sharpe', 'multi_objective']:
    result = train('technical', 'PPO', reward, verbose=False)
    results.append(result)

compare_results(results)  # Shows which is best!
```

### 4. Tune Settings
- Edit `config.py`
- Change `TOTAL_STEPS`, learning rates, etc.
- Re-run training

### 5. Save Everything
```python
from utils import save_results
save_results(result, 'results/my_experiment.json')
```

---

## 🔧 Customization

### Change Training Steps
In `config.py`:
```python
TOTAL_STEPS = 500_000  # Train longer
```

### Change Reward Parameters
In `config.py`:
```python
EMA_CONFIG = {
    'rolling_vol_window': 24  # Longer window
}
```

### Use Custom Config
```python
from config import get_config

config = get_config('ema_sharpe')
config['total_steps'] = 100_000  # Custom value

result = train('technical', 'PPO', 'ema_sharpe', config=config)
```

---

## 📈 Workflow Example

```python
# 1. Quick test to make sure everything works
from train import quick_test
quick_test('technical', 'ema_sharpe')

# 2. Try different rewards
from train import train
from utils import compare_results

results = []
for reward in ['ema_sharpe', 'differential_sharpe', 'multi_objective']:
    result = train('technical', 'PPO', reward)
    results.append(result)

# 3. Compare
df = compare_results(results)

# 4. Pick best and train all algorithms
from train import train_all_algos
best_reward = df.iloc[0]['reward_type']
final_results = train_all_algos('technical', best_reward)

# 5. Save
from utils import save_results
save_results(final_results, 'results/final_experiment.json')
```

---

## ✅ What's Preserved

All your original logic is here:
- ✅ EMA Sharpe formula (in `rewards.py`)
- ✅ Differential Sharpe formula (in `rewards.py`)
- ✅ Multi-objective penalties (in `rewards.py`)
- ✅ Environment dynamics (in `environments.py`)
- ✅ Model architectures (in `train.py`)
- ✅ Validation logic (in `environments.py`)

**Zero changes to your algorithms!**

---

## 🆘 Troubleshooting

### Import Error
Make sure you're in the right directory:
```python
import os
os.chdir('/path/to/portfolio_rl')
```

### Data Not Found
Check `config.py`:
```python
DATA_DIR = 'data_hierarchical'  # Make sure path is correct
```

### Want More Details
Set `verbose=True`:
```python
result = train('technical', 'PPO', 'ema_sharpe', verbose=True)
```

---

## 🎉 That's It!

You now have a **simple, clean structure** that:
- ✅ Works great with Jupyter notebooks
- ✅ No complex modules to understand
- ✅ Easy to modify and experiment
- ✅ All your logic preserved
- ✅ 40% less code than before

**Happy experimenting!** 🚀

---

## 📞 Need Help?

1. Check the example notebooks in `notebooks/`
2. Read the docstrings: `help(train)`
3. Look at `config.py` for settings
4. Check `rewards.py` for your formulas
