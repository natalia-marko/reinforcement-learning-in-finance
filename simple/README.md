# RL Portfolio Rebalancing System

A reinforcement learning system for dynamic portfolio allocation using PPO (Proximal Policy Optimization) with both LSTM and MLP architectures.

## 🎯 Project Overview

This project implements a **weekly portfolio rebalancing agent** that learns optimal allocation strategies across 7 tech stocks (NVDA, MU, AAPL, AMD, ASML, MSFT, GOOG) using reinforcement learning.

**Key Features:**
- **Walk-Forward Validation**: Prevents data leakage with proper train/val/test splits
- **Sophisticated Reward Function**: Combines Sharpe, Sortino, diversification, and transaction costs
- **Feature Engineering**: 83 features including momentum, volatility, volume signals, and macro indicators
- **Two Model Architectures**: LSTM (for temporal patterns) and MLP (lightweight, faster)
- **Feature Importance Analysis**: Permutation-based pruning to identify signal vs. noise

## 📊 Performance Summary

| Model | Features | Total Return | Sharpe | Alpha vs QQQ | Training Time |
|-------|----------|--------------|--------|--------------|---------------|
| **Lean MLP** ⭐ | 37 | 37.08% | 0.85 | +7.26% | ~60s |
| Full MLP | 83 | 37.97% | 0.86 | +7.86% | ~60s |
| LSTM (deprecated) | 83 | - | - | - | ~5 min |

**Recommendation**: Use **Lean MLP** for production (simpler, more robust).

---

## Project Structure

```
simple/
│   ├── plot_mlp_curves.py       # Learning curve visualization
│   ├── models/                  # Full MLP checkpoints
│   ├── models_lean/             # ⭐ Lean MLP checkpoints (PRODUCTION)
│   └── logs/                    # Training logs
│
├── data/                        # Data storage
│   ├── raw_data_train.csv       # Train+Val prices & volumes
│   ├── qqq_benchmark.csv        # QQQ benchmark
│   └── test_set_NEVER_TOUCH.csv # Held-out test set
│
├── outputs/                     # Generated plots and CSVs
│
├── app.py                       # FastAPI visualization server
└── static/                      # Web UI assets
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python data_eng.py
```
This creates:
- `data/raw_data_train.csv` (Train+Val set)
- `data/test_set_NEVER_TOUCH.csv` (True test set)

### 3. Train the Model (Lean MLP - Recommended)
```bash
python mlp/train_mlp.py --lean
```
**Output**: `mlp/models_lean/best_overall_model.zip`

### 4. Backtest
```bash
python mlp/backtest_mlp.py --lean
```
**Output**: `outputs/mlp_lean_backtest.png`

### 5. Visualize Learning Curves
```bash
python mlp/plot_mlp_curves.py
```

---

## 📈 System Architecture

### Environment (`rl_system.py`)

**State Space (101 dimensions):**
- Market features: 83 (momentum, volatility, volume, macro)
- Correlation features: 14 (cross-asset correlations + betas)
- Portfolio state: 4 (recent return, vol, time in market, total return)

**Action Space:**
- 7 continuous values → Softmax with max 40% concentration per asset

**Reward Function:**
```python
reward = (
    sharpe_reward * 0.4 +
    sortino_reward * 0.3 +  # Downside protection
    diversification_reward * 0.2 +
    cost_penalty * 0.1
)
```

### Data Engineering (`data_eng.py`)

**Features (83 total):**
- **Per-Asset (7 × 11 = 77)**:
  - Returns (1), Momentum (2), Volatility (2), Technical (2), Risk (1), Volume (3)
- **Macro (4)**: VIX, DXY, Yield Curve (TNX-IRX), 10Y Treasury
- **Calendar (2)**: Month sine/cosine encoding

**Critical**: Uses `expanding_zscore()` to prevent data leakage. Normalization only uses past data.

---

## 🔬 Advanced Usage

### Feature Importance Analysis
```bash
python mlp/analyze_features.py
```
This runs **permutation importance** on the test set to identify which features drive performance.

**Output**: `outputs/feature_importance.csv` + visualization

### Compare Full vs Lean MLP
```bash
# Train both
python mlp/train_mlp.py          # Full (83 features)
python mlp/train_mlp.py --lean   # Lean (37 features)

# Compare results
python mlp/backtest_mlp.py
python mlp/backtest_mlp.py --lean
```

---

## 🧪 Hyperparameters

Defined in `config.py`:

```python
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 256,        # Optimized for dataset size
    "batch_size": 64,      # Ensures frequent updates
    "ent_coef": 0.01,      # Prevents premature convergence
    "gamma": 0.99,
    "clip_range": 0.2,
}
```

**Training**:
- Walk-Forward Validation: 3 folds
- Timesteps per fold: 100,000
- Gap between folds: 56 weeks (prevents leakage)

---

## 📊 Visualization & Monitoring

### Real-Time Dashboard (Optional)
```bash
uvicorn app:app --reload
```
Then open: `http://localhost:8000`

**Features**:
- Live portfolio allocation chart (Chart.js)
- Historical performance tracking
- Risk metrics dashboard

---

## 🐛 Troubleshooting

### Issue: "KeyError: None of [*_simple_ret] are in columns"
**Cause**: Feature pruning removed return columns needed by `PortfolioEnv`.
**Fix**: Return columns are now force-included in `train_mlp.py`.

### Issue: LSTM overfitting (Fold 3 gap > 30)
**Cause**: LSTM has 500K+ parameters, dataset only has 566 samples.
**Fix**: Use MLP instead (15K parameters). See `mlp/` directory.

### Issue: Training is slow
**Cause**: Using LSTM instead of MLP.
**Fix**: Switch to `mlp/train_mlp.py --lean` (7.5x faster).

---

## 🔑 Key Design Decisions

1. **Why Weekly Rebalancing?**
   - Balances transaction costs vs. responsiveness
   - Aligns with typical institutional rebalancing frequencies

2. **Why MLP over LSTM?**
   - Features already encode temporal info (momentum, MA)
   - LSTM requires 100K+ samples for stable training
   - MLP is 10x faster and generalizes better on small data

3. **Why Lean Features (37 vs 83)?**
   - Removes noisy signals (raw returns of volatile stocks)
   - Reduces overfitting risk
   - Easier to interpret and debug

4. **Why Softmax + Constraints?**
   - Ensures valid portfolio (sums to 1.0)
   - Prevents concentration risk (max 40% per asset)
   - Avoids shorting (all weights ≥ 0)

---

## 📚 File Descriptions

### Core Files
- **`core/config.py`**: All hyperparameters, paths, asset lists
- **`core/data_eng.py`**: Data pipeline (download, features, train/test split)
- **`core/rl_system.py`**: Gymnasium environment + reward function

### Training Scripts
- **`mlp/train_mlp.py`**: Training script (use `--lean` for optimized features)
- **`lstm/train_lstm.py`**: LSTM training script (use `--lean` for optimized features)

### Evaluation
- **`mlp/backtest_mlp.py`**: Out-of-sample evaluation (use `--lean` for lean model)
- **`mlp/plot_mlp_curves.py`**: Learning curve analysis
- **`mlp/analyze_features.py`**: Feature importance via permutation

### Outputs
- **`mlp/models_lean/best_overall_model.zip`**: Trained agent
- **`outputs/mlp_lean_backtest.png`**: Performance visualization
- **`outputs/feature_importance.csv`**: Feature rankings

---

## 🎓 References & Methodology

**Reinforcement Learning**:
- Algorithm: PPO (Proximal Policy Optimization)
- Framework: Stable-Baselines3
- Environment: Gymnasium (formerly OpenAI Gym)

**Backtesting Integrity**:
- Walk-Forward Validation with 56-week gap (prevents leakage)
- True held-out test set (never seen during training)
- Expanding window normalization (only uses past data)

**Feature Engineering**:
- Inspired by: "Advances in Financial Machine Learning" (Marcos López de Prado)
- Volume signals: OBV, MFI, Volume Ratio
- Risk-adjusted metrics: Sharpe, Sortino, Max Drawdown

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **Stable-Baselines3**: RL training framework
- **yfinance**: Historical market data
- **TA-Lib**: Technical indicators
- **Chart.js**: Portfolio visualization

---

## 🚨 Disclaimer

This is an educational project. **Not financial advice**. Do not deploy real capital without:
1. Extensive backtesting across multiple market regimes
2. Consulting a qualified financial advisor
3. Understanding the risks of algorithmic trading

Past performance does not guarantee future results.
