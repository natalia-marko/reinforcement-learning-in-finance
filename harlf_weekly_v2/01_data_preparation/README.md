# Data Preparation

## Overview
This directory contains all scripts and notebooks for data acquisition, feature engineering, and preprocessing.

## Files

### `data_preparation_part1_v2.py` ⭐ Main Script
**Purpose:** Complete data preparation pipeline

**Features:**
- Loads raw stock price data (OHLCV)
- Computes 20+ technical indicators:
  - Trend: SMA, EMA, MACD
  - Momentum: RSI, Stochastic, ROC
  - Volatility: Historical Vol, ATR, Bollinger Bands
  - Volume: Volume Ratio, MFI, OBV
  - Benchmark: Correlation, Beta
- Computes sentiment indicators
- Applies StandardScaler normalization (fitted on train only)
- Creates train/val/test splits (60/20/20)
- Saves processed data to `../data_hierarchical/`

**Usage:**
```bash
python data_preparation_part1_v2.py
```

**Output:**
```
../data_hierarchical/
├── technical/
│   ├── train.csv       # Normalized technical features
│   ├── val.csv
│   └── test.csv
├── sentiment/
│   ├── train.csv       # Normalized sentiment features
│   ├── val.csv
│   └── test.csv
├── returns_train.csv   # Weekly returns
├── returns_val.csv
├── returns_test.csv
└── metadata.json       # Tickers, features, normalization params
```

---

### `get_sentiment_data.ipynb`
Fetch sentiment data from various sources

### `sentiment_data_retrieve.ipynb`
Process and clean raw sentiment data

---

## Technical Indicators Details

### Trend Indicators
- **SMA (4w, 8w, 12w)**: Price-to-SMA ratios
- **EMA (8w, 12w)**: Price-to-EMA ratios
- **MACD Histogram**: 12w fast, 26w slow, 9w signal

### Momentum Indicators
- **Return Lags**: 1w, 2w, 3w lagged returns
- **RSI**: 14-week Relative Strength Index
- **Stochastic**: 14-week %K
- **ROC**: 4-week Rate of Change

### Volatility Indicators
- **Historical Volatility**: 12w rolling std (annualized)
- **ATR**: 14w Average True Range (% of price)
- **Bollinger Band Position**: Price position in 20w bands

### Volume Indicators
- **Volume Ratio**: Current vs 20w average
- **MFI**: 14w Money Flow Index
- **OBV ROC**: 4w On-Balance Volume change

### Benchmark Metrics
- **Correlation**: 12w rolling correlation with QQQ
- **Beta**: 12w rolling beta vs QQQ

---

## Normalization

**Method:** StandardScaler (z-score normalization)
- Fitted on training data only
- Applied to train/val/test
- Binary features (if any) are skipped
- Parameters saved in `metadata.json`

**Why:** Neural networks perform better with normalized inputs

---

## Data Splits

| Split | Percentage | Purpose |
|-------|------------|---------|
| Train | 60% | Agent learning |
| Val | 20% | Hyperparameter tuning, early stopping |
| Test | 20% | Final performance evaluation |

**Note:** Splits are chronological (no data leakage)

---

## Key Improvements (v2)

1. ✅ Fixed technical indicator windows
2. ✅ Added missing indicators (EMA, MACD, etc.)
3. ✅ Implemented StandardScaler normalization
4. ✅ Separated base columns from indicator features
5. ✅ Explicit feature lists in metadata

---

## Next Steps

After running data preparation:
1. Verify outputs in `../data_hierarchical/`
2. Check `metadata.json` for feature lists
3. Proceed to `../02_part1_base_agents/` for training

