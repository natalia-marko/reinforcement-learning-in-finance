# Hierarchical Reinforcement Learning for Finance - Project Structure

## 📁 Directory Organization

```
harlf_weekly_v2/
│
├── 01_data_preparation/          # Data acquisition and preprocessing
├── 02_part1_base_agents/         # Base agents + reward function experiments
├── 03_part2_super_agent/         # Hierarchical super agent
├── data_hierarchical/            # Processed data (train/val/test splits)
├── sentiment_data/               # Raw sentiment data
├── models_part1/                 # Trained base agent models
├── models_diff_sharpe/           # Differential Sharpe models
├── models_multi_objective/       # Multi-objective models
├── models_super_agent/           # Super agent models
├── plots/                        # All visualizations and figures
├── docs/                         # Documentation
├── configs/                      # Configuration files
└── README.md                     # Main project documentation
```

---

## 📂 Detailed Directory Contents

### `01_data_preparation/`
**Purpose:** Data acquisition, feature engineering, and preprocessing

**Files:**
- `data_preparation_part1_v2.py` - Main data preparation script
  - Loads raw stock data
  - Computes technical indicators (SMA, EMA, RSI, MACD, etc.)
  - Computes sentiment indicators
  - Applies StandardScaler normalization
  - Creates train/val/test splits
  - Saves to `data_hierarchical/`

- `get_sentiment_data.ipynb` - Fetch sentiment data from sources
- `sentiment_data_retrieve.ipynb` - Process and clean sentiment data

**Output:** `data_hierarchical/` with normalized train/val/test CSVs

**Run Order:**
1. Get sentiment data (notebooks)
2. Run `python data_preparation_part1_v2.py`

---

### `02_part1_base_agents/`
**Purpose:** Train individual base agents using technical and sentiment features. Compare different reward functions to find the best approach for your data.

**Core Training Files:**
- `environments_part1.py` - Gymnasium environments
  - `TechnicalAgentEnv` - Uses technical indicators
  - `SentimentAgentEnv` - Uses sentiment indicators
  - Portfolio allocation via softmax
  - EMA-based Sharpe ratio reward (default)

- `train_part1_agents.ipynb` - Original training (EMA Sharpe)
- `retrain_base_agents.ipynb` - Retraining with visualizations

**Reward Function Approaches:**
- `ema_sharpe_approach.py` - EMA-based Sharpe (baseline, proven)
- `differential_sharpe_approach.py` - Differential Sharpe (theoretical)
- `multi_objective.py` - Multi-objective with penalties (optimized)

**Comparison & Tuning:**
- `compare_reward_functions.py` - Compare all approaches
- `compare_reward_functions.ipynb` - Interactive comparison
- `tune_multi_objective.py` - Hyperparameter tuning

**Models Saved:** 
- `models_part1/` - EMA Sharpe models (baseline)
- `models_diff_sharpe/` - Differential Sharpe models
- `models_multi_objective/` - Multi-objective models

**Key Results:**
- EMA Sharpe: Technical 1.78, Sentiment 1.92
- Multi-Objective (optimized): Expected 2.0-2.2 (Technical), 1.4-1.6 (Sentiment)

---

### `03_part2_super_agent/`
**Purpose:** Hierarchical agent that combines base agents

**Files:**
- `super_agent_enviroment.py` - Super agent environment
  - Takes recommendations from base agents
  - Meta-policy for agent selection/blending
  - Higher-level portfolio management
  
- `train_super_agent.ipynb` - Super agent training
  - Loads best base agents
  - Trains meta-controller
  - Final ensemble model

**Models Saved:** `models_super_agent/`

**Concept:** Hierarchical RL where super agent learns to:
1. Select which base agent to follow
2. Blend recommendations
3. Adjust based on market conditions

---

### `data_hierarchical/`
**Structure:**
```
data_hierarchical/
├── technical/
│   ├── train.csv       # Technical features (normalized)
│   ├── val.csv
│   └── test.csv
├── sentiment/
│   ├── train.csv       # Sentiment features (normalized)
│   ├── val.csv
│   └── test.csv
├── returns_train.csv   # Portfolio returns
├── returns_val.csv
├── returns_test.csv
└── metadata.json       # Feature lists, tickers, normalization params
```

**Tickers:** NVDA, MU, AAPL, AMD, ASML, MSFT, GOOG  
**Date Range:** 2020-01-01 to 2025-10-23  
**Splits:** Train 60%, Val 20%, Test 20%

---

### `plots/`
**Purpose:** All visualizations and figures

**Generated Plots:**
- `reward_function_comparison.png` - Performance across reward functions
- `reward_comparison_heatmap.png` - Test Sharpe heatmap
- `generalization_gap.png` - Train-test gap analysis
- `overall_winner.png` - Best approach highlight
- `tech_performance.png` - Technical agent metrics
- `sent_performance.png` - Sentiment agent metrics
- `training_summary_report.png` - Comprehensive training report

---

### `docs/`
**Documentation Files:**
- `parameters_used.md` - Feature specifications
- `PENALTY_OPTIMIZATION.md` - Multi-objective tuning guide
- `TUNING_GUIDE.md` - How to tune penalties
- `REWARD_COMPARISON_README.md` - Reward function comparison guide

---

### `configs/`
**Purpose:** Configuration files (future use)

Suggested contents:
- `data_config.yaml` - Data parameters
- `technical_agent_config.yaml` - Technical agent settings
- `sentiment_agent_config.yaml` - Sentiment agent settings
- `super_agent_config.yaml` - Super agent settings

---

## 🚀 Quick Start Guide

### 1. Data Preparation
```bash
cd 01_data_preparation
python data_preparation_part1_v2.py
```

### 2. Train Base Agents
```bash
cd 02_part1_base_agents

# Option A: Quick start with proven EMA Sharpe
jupyter notebook retrain_base_agents.ipynb

# Option B: Compare all reward functions
jupyter notebook compare_reward_functions.ipynb

# Option C: Tune multi-objective
python tune_multi_objective.py --quick
```

### 3. Train Super Agent
```bash
cd 03_part2_super_agent
jupyter notebook train_super_agent.ipynb
```

---

## 📊 Current Best Results

| Agent | Algorithm | Reward Type | Test Sharpe |
|-------|-----------|-------------|-------------|
| Technical | PPO | EMA Sharpe | 1.78 |
| Sentiment | SAC | EMA Sharpe | 1.92 |
| Technical | SAC | Multi-Obj (old) | 1.91 |
| Sentiment | PPO | Multi-Obj (old) | 1.14 |

**Note:** Multi-objective with optimized penalties expected to improve significantly.

---

## 🔧 Dependencies

```bash
pip install numpy pandas gymnasium stable-baselines3 matplotlib seaborn scikit-learn torch
```

---

## 📝 Workflow Summary

1. **Data Prep** → Fetch & process data → Normalize → Split
2. **Part 1** → Train base agents → Compare reward functions → Select best
3. **Part 2** → Train super agent using best base agents
4. **Evaluation** → Backtest & analyze performance

---

## 🎯 Project Goals

1. ✅ Develop robust base agents (technical & sentiment)
2. ✅ Compare reward function approaches (integrated in Part 1)
3. 🔄 Optimize multi-objective penalties (in progress)
4. ⏳ Build hierarchical super agent
5. ⏳ Production deployment

---

## 👥 Contributors

- Project Lead: [Your Name]
- Last Updated: 2025-01-24
- Version: 2.0 (Organized Structure)

