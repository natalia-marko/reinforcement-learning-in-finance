# Project Structure

```
reinforcement_learning_in_finance/simple/
├── core/                       # Core modules
│   ├── config.py              # Global configuration
│   ├── rl_system.py           # Portfolio RL environment
│   ├── data_eng_simple.py     # Simple feature engineering (83 features)
│   └── data_eng_expanded.py   # Expanded feature engineering (175 features)
│
├── src/                        # Main pipeline
│   ├── analyze_features.py    # Feature importance analysis
│   ├── train.py               # Model training
│   ├── backtest.py            # Backtesting
│   ├── plot.py                # Learning curve visualization
│   ├── pipeline.ipynb         # End-to-end pipeline notebook
│   ├── models/                # Trained models
│   ├── models_lean/           # Lean models (optimal features)
│   ├── models_expanded/       # Expanded feature models
│   ├── models_expanded_lean/  # Expanded + lean models
│   ├── logs/                  # Training logs
│   ├── logs_lean/
│   ├── logs_expanded/
│   └── logs_expanded_lean/
│
├── data/                       # Data storage
│   ├── simple/                # Simple feature set data
│   └── expanded/              # Expanded feature set data
│
├── outputs/                    # Generated plots and results
├── archive/                    # Archived code
│   └── lstm/                  # LSTM implementation (archived)
│
├── static/                     # Web app assets
├── app.py                      # Streamlit dashboard
├── requirements.txt
├── README.md
├── DECISION_LOG.md            # Decision to use MLP over LSTM
└── STRUCTURE.md               # This file
```

## Usage

### 1. Prepare Data
```bash
# For simple feature set (83 features)
python -m core.data_eng_simple

# For expanded feature set (175 features)
python -m core.data_eng_expanded
```

### 2. Run Complete Pipeline
```bash
cd src
jupyter notebook pipeline.ipynb
```

Or run individual scripts:
```bash
cd src
python analyze_features.py --expanded
python train.py --expanded --lean
python plot.py --expanded --lean
python backtest.py --expanded --lean
```

### 3. Launch Dashboard
```bash
streamlit run app.py
```

## Modes

- **`--expanded`**: Use 175-feature expanded set (richer technical indicators)
- **`--lean`**: Use optimal feature subset based on importance analysis
- Both flags can be combined for best results
