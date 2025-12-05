# Training Pipeline Guide

## 📓 Training Notebooks (Now in daily_adviser/)

The training pipeline consists of two notebooks:

### 1. Data Preparation: `01_ai_rebalancer.ipynb`
- Fetches historical market data
- Engineers features (volatility, RSI, correlations)
- Creates the training environment
- **Run this first** to prepare data

### 2. Model Training: `02_board_of_directors.ipynb`
- Trains Bull, Bear, and Sniper agents
- Implements Board of Directors ensemble
- Saves models as `agent_bull.zip`, `agent_bear.zip`, `agent_sniper.zip`
- **Run this second** to create models

## 🚀 Quick Start - Retraining Models

### Step 1: Open Jupyter
```bash
cd daily_adviser
jupyter notebook
```

### Step 2: Run Data Prep Notebook
1. Open `01_ai_rebalancer.ipynb`
2. Run all cells
3. Verify features are generated correctly

### Step 3: Run Training Notebook
1. Open `02_board_of_directors.ipynb`
2. Run all cells (this takes time - PPO training)
3. New model files will be created: `agent_bull.zip`, `agent_bear.zip`, `agent_sniper.zip`

### Step 4: Test New Models
```bash
python daily_advisor.py
```

## 📝 Important Notes

### Before Retraining
- Current models in daily_adviser are from the archive
- They're working correctly (as you just tested)
- Only retrain if:
  - You want more recent market data
  - Market regime has shifted
  - You changed asset composition

### During Training
- Training takes significant time (1-2 hours depending on system)
- Make sure you have enough disk space (~500MB for data + logs)
- Don't interrupt the training process

### After Training
- Old models are overwritten automatically
- Test immediately with `python daily_advisor.py`
- Compare performance before committing to use

### Supporting Files (Already Copied)
- `rl_system.py` - Core classes (FeatureEngineer, Environment, Board)
- `config.py` - Transaction costs and risk parameters

## 🔧 Dependencies

Make sure you have all required packages:
```bash
pip install -r requirements.txt
```

Additional for training (not in requirements.txt):
```bash
pip install jupyter
```

## ⚙️ Configuration

Edit in notebooks if needed:
- **Assets**: `TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']`
- **Data Period**: Adjust date ranges in 01_ai_rebalancer.ipynb
- **Training Steps**: Modify in 02_board_of_directors.ipynb

## 🎯 Expected Output

After successful training:
- ✅ `agent_bull.zip` (~2.1 MB)
- ✅ `agent_bear.zip` (~2.1 MB)
- ✅ `agent_sniper.zip` (~2.1 MB)
- ✅ Backtest performance metrics printed
- ✅ Training logs in console

## 🐛 Troubleshooting

**"No module named 'config'"**
- Make sure you're running from the daily_adviser directory

**Training takes forever**
- This is normal - PPO training is computationally intensive
- Bull/Bear/Sniper each need separate training runs

**Models perform poorly**
- Check data quality (no NaNs, correct date ranges)
- Verify feature engineering matches production
- Try adjusting training hyperparameters

**Memory errors**
- Reduce batch size in training configuration
- Close other applications
- Use smaller date range for initial testing

---

**Ready to retrain?** Start Jupyter and run the notebooks in order!
