# Data Preparation Guide


## Introduction & Overview

### Project Objectives

This data preparation pipeline supports a reinforcement learning system for **weekly portfolio rebalancing** with the goal of achieving optimal risk-adjusted returns. The pipeline transforms raw market data into a comprehensive feature set suitable for RL agent training.

### Workflow Overview

```
Raw Data → Feature Engineering → Preprocessing → Train/Test Splits → Normalized Features
   ↓              ↓                    ↓                ↓                    ↓
Yahoo Finance  Technical        Outlier Removal  80/20 Split        Z-score Scaling
FRED API       Indicators       Correlation       (Chronological)    (Train-fitted)
               Macro Data        Filtering
```

### Key Features

- **Comprehensive Feature Set**: 200+ features across 15+ categories
- **Time-Series Aware**: Proper handling of chronological data with no data leakage
- **Production Ready**: Modular code structure with reusable utilities
- **Reproducible**: Complete documentation and version-controlled preprocessing steps

---

## Prerequisites

### Environment Setup

**Python Version**: 3.9+ recommended

**Required Libraries**: Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `pandas>=1.5.0` - Data manipulation
- `numpy>=1.23.0` - Numerical computations
- `yfinance>=0.2.0` - Yahoo Finance data fetching
- `scikit-learn>=1.2.0` - Preprocessing and normalization
- `ta>=0.10.0` - Technical analysis indicators
- `fredapi>=0.5.0` - FRED macroeconomic data (optional)
- `matplotlib>=3.6.0`, `seaborn>=0.12.0` - Visualization
- `pyarrow>=10.0.0` - Parquet file support

### Project Structure

```
rl_relabalance_portfolio/
├── utile.py                      # All utility functions: data fetching, feature engineering, preprocessing
├── notebooks/
│   └── 01_data_preparation.ipynb  # Main execution notebook
├── data/
│   ├── raw/                      # Raw price data and features
│   └── processed/                # Preprocessed train/test splits
├── results/                      # Visualizations and analysis
├── api_keys.json                 # API keys configuration
└── requirements.txt              # Python dependencies
```

### API Key Configuration

**FRED API Key** (for macroeconomic data):

1. Get a free API key from [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Add to `api_keys.json`:
```json
{
    "fred": "your_api_key_here"
}
```

Alternatively, set environment variable:
```bash
export FRED_API_KEY="your_api_key_here"
```

The system checks both locations automatically (see `utile.py::_load_api_key_from_file()`).

---

## Quick Start Guide

### Minimal Example

Execute the complete pipeline using the main notebook:

```python
# In notebooks/01_data_preparation.ipynb

# 1. Import utilities
from utile import (
    fetch_price_data, prepare_price_dataframe, fetch_macro_data,
    engineer_all_features,
    preprocess_pipeline
)

# 2. Fetch price data
tickers = ['AAPL', 'MSFT', 'GOOGL']
price_data = fetch_price_data(
    tickers=tickers,
    start_date='2020-01-01',
    end_date='2025-01-01',
    interval='1wk',
)
price_df = prepare_price_dataframe(price_data)

# 3. Engineer features
features_df = engineer_all_features(price_df)

# 4. Preprocess
feature_cols = [col for col in features_df.columns 
                if col not in ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]

preprocessed = preprocess_pipeline(
    df=features_df.dropna(subset=feature_cols),
    feature_cols=feature_cols,
    train_ratio=0.8
)

# 5. Access results
train_df = preprocessed['train']
test_df = preprocessed['test']
```

### Execution Order

1. **Run `notebooks/01_data_preparation.ipynb`** - Complete pipeline execution
2. **Verify outputs** in `data/processed/` directory
3. **Check metadata** in `data/processed/metadata.json`
4. **Review visualizations** in `results/` directory

### Verification Checkpoints

- [ ] Raw price data saved to `data/raw/weekly_prices.parquet`
- [ ] Features calculated and saved to `data/raw/features_raw.parquet`
- [ ] Train/test splits created (80/20 ratio)
- [ ] Features normalized (mean ≈ 0, std ≈ 1 for train set)
- [ ] Metadata file contains feature lists and date ranges

---

## Data Acquisition

### Price Data Fetching

**Function**: `utile.py::fetch_price_data()`

Fetches weekly OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.

**Important**: Uses `auto_adjust=False` to access both raw and adjusted prices. The pipeline automatically uses **Adjusted Close** prices to account for splits and dividends.

```python
from utile import fetch_price_data

price_data = fetch_price_data(
    tickers=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX'],
    start_date='2020-01-01',
    end_date='2025-01-01',
    interval='1wk'  # Weekly data
)
```

**Returns**: Multi-index DataFrame with structure:
```
                    (AAPL, Open)  (AAPL, High)  ...  (NFLX, Volume)
Date
2020-01-03          75.0          76.0         ...  12345678
2020-01-10          76.5          77.0         ...  23456789
...
```

### Converting to Long Format

**Function**: `utile.py::prepare_price_dataframe()`

Converts multi-index DataFrame to long format with ticker column. Automatically uses Adjusted Close when available.

```python
from utile import prepare_price_dataframe

price_df = prepare_price_dataframe(price_data)
# Returns DataFrame with columns: date, ticker, open, high, low, close, volume
# 'close' column contains Adjusted Close prices
```

**Output Format**:
```
   date       ticker   open    high    low     close   volume
2020-01-03    AAPL    75.0    76.0    74.0    75.5    12345678
2020-01-03    MSFT    150.0   152.0   149.0   151.0   23456789
...
```

### Macroeconomic Data

**Function**: `utile.py::fetch_macro_data()`

Fetches macroeconomic indicators from FRED (Federal Reserve Economic Data).

**Default Indicators**:
- `treasury_10y` (DGS10) - 10-Year Treasury Yield
- `fed_funds_rate` (FEDFUNDS) - Federal Funds Rate
- `vix` (VIXCLS) - VIX Volatility Index
- `unemployment_rate` (UNRATE) - Unemployment Rate
- `gdp_growth` (A191RL1Q225SBEA) - GDP Growth

```python
from utile import fetch_macro_data

macro_data = fetch_macro_data(
    start_date='2020-01-01',
    end_date='2025-01-01'
)
```

**Note**: Returns empty DataFrame if `fredapi` is not installed or API key is missing. The pipeline continues without macro data.

#### Macro Data Merging Strategy

**Function**: `utile.py::merge_macro_data()`

Macroeconomic indicators are merged with price data using a forward-fill and backward-fill strategy to handle sparse updates.

**Approach**:
1. **Date Alignment**: Macro data is filtered to price data date range with a 7-day buffer for weekly alignment
2. **Reindexing**: Macro data is reindexed to match price data dates
3. **Forward Fill**: Each macro column is forward-filled to carry forward the last known value
4. **Backward Fill**: Initial NaN values are backward-filled to handle cases where macro data starts later than price data
5. **Per-Ticker Fill**: After merging, macro columns are forward-filled within each ticker group to handle ticker-specific date gaps

**Rationale**: Macro indicators update infrequently (weekly/monthly), so interpolation is needed for daily/weekly price data. Forward-fill preserves the most recent known macro value, which is appropriate for economic indicators that persist until updated.

**Implementation Details**:
- Uses `pd.reindex()` with `ffill()` for each column separately to handle sparse data
- Applies `bfill()` for any remaining NaN at the start of the series
- Per-ticker forward-fill ensures each asset has complete macro data even if price dates differ slightly

### Handling Multi-Index DataFrames

The `prepare_price_dataframe()` function handles both single and multiple tickers:

- **Single Ticker**: Automatically extracts ticker name from column structure
- **Multiple Tickers**: Iterates through `price_data.columns.get_level_values(0).unique()` to extract each ticker's data

**Error Handling**: Warns and skips tickers with missing Adjusted Close columns.

---

## Feature Engineering

### Overview

**Function**: `utile.py::engineer_all_features()`

Calculates comprehensive features across multiple categories. Features are calculated per-ticker to maintain asset-specific characteristics.

### Feature Categories

#### 1. Momentum & Trend Features

**Function**: `calculate_momentum_features()`

| Feature | Formula/Description | Timeframes |
|---------|-------------------|------------|
| `return_1w`, `return_4w`, `return_13w`, `return_26w` | `ln(P_t / P_{t-n})` (log returns) | 1w, 4w, 13w, 26w |
| `rsi_14d` | RSI = 100 - (100 / (1 + RS)) where RS = Avg Gain / Avg Loss | 14 periods |
| `roc_4w` | Rate of Change: `(P_t - P_{t-4}) / P_{t-4} * 100` | 4 weeks |
| `macd_histogram` | MACD - Signal line (12/26/9 EMA) | - |
| `stochastic_14w` | `%K = 100 * (C - L14) / (H14 - L14)` | 14 weeks |
| `price_to_sma_{period}w` | `P_t / SMA_{period}` | 4w, 8w, 12w, 20w |
| `price_to_ema_{period}w` | `P_t / EMA_{period}` | 4w, 8w, 12w, 20w |

**Implementation**: Uses `ta` library indicators (RSIIndicator, MACD, StochasticOscillator, ROCIndicator)

**Note**: All return features use **log returns** (`ln(P_t / P_{t-n})`) instead of simple percentage returns. See "Implementation Approaches" section for rationale.

#### 2. Volatility & Risk Features

**Function**: `calculate_volatility_features()`

| Feature | Formula/Description | Timeframes |
|---------|-------------------|------------|
| `volatility_{period}w` | `std(log_returns) * sqrt(52)` (annualized) | 4w, 13w, 26w, 52w |
| `atr_14d` | Average True Range / Close price | 14 periods |
| `bb_width` | `(Upper Band - Lower Band) / Close` | 20 periods |
| `bb_position` | `(Close - Lower Band) / (Upper Band - Lower Band)` | 20 periods |
| `downside_volatility_{period}w` | `std(log_returns[returns < 0]) * sqrt(52)` | 4w, 13w, 26w |
| `upside_volatility_{period}w` | `std(log_returns[returns > 0]) * sqrt(52)` | 4w, 13w, 26w |

**Implementation**: Uses BollingerBands and AverageTrueRange from `ta` library. Volatility calculations use log returns.

#### 3. Volume & Liquidity Features

**Function**: `calculate_volume_features()`

| Feature | Formula/Description | Timeframes |
|---------|-------------------|------------|
| `obv` | On-Balance Volume: cumulative sum of `sign(price_change) * volume` | - |
| `obv_roc_4w` | OBV rate of change over 4 weeks | 4 weeks |
| `relative_volume_{period}w` | `Volume / SMA(Volume, period)` | 4w, 13w, 20w |
| `mfi_14d` | Money Flow Index (14-period) | 14 periods |
| `vwap_distance` | `(Close - VWAP) / VWAP` | 20 periods |
| `chaikin_money_flow` | `(MF_positive - MF_negative) / MF_total` | 20 periods |
| `volume_oscillator` | `(Volume_SMA_short - Volume_SMA_long) / Volume_SMA_long` | 5/20 periods |

**Implementation**: Uses MFIIndicator from `ta` library, custom calculations for OBV and VWAP

#### 4. Risk-Adjusted Performance Features

**Function**: `calculate_risk_adjusted_features()`

| Feature | Formula | Timeframes |
|---------|---------|------------|
| `sharpe_{period}w` | `(Mean Log Return * 52) / (Volatility * sqrt(52))` | 4w, 13w, 26w, 52w |
| `sortino_{period}w` | `(Mean Log Return * 52) / Downside Volatility` | 4w, 13w, 26w, 52w |
| `calmar_{period}w` | `(Mean Log Return * 52) / abs(Rolling Max Drawdown)` | 4w, 13w, 26w, 52w |
| `correlation_to_qqq` | Rolling correlation with QQQ log returns | 60 periods |
| `beta_to_market` | `Cov(log_returns, benchmark) / Var(benchmark)` | 60 periods |
| `relative_return` | `log_returns - benchmark_log_returns` | - |

**Mathematical Details**:
- Sharpe Ratio: Annualized excess return per unit of volatility (uses log returns)
- Sortino Ratio: Similar to Sharpe but uses downside deviation (only negative returns)
- Calmar Ratio: Annualized return divided by maximum drawdown within rolling window (prevents data leakage)

**Data Leakage Prevention**: Calmar ratio uses `max_dd.rolling(period).min()` instead of global minimum to ensure only past data is used in calculations.

#### 5. Drawdown & Path Risk Features

**Function**: `calculate_drawdown_features()`

| Feature | Formula/Description | Timeframes |
|---------|-------------------|------------|
| `current_drawdown` | `(Price - Running Max) / Running Max` | Expanding window |
| `max_drawdown_{period}w` | Maximum drawdown over rolling window | 4w, 13w, 26w, 52w |
| `recovery_factor_{period}w` | `(Max - Current) / (Current - Min)` | 4w, 13w, 26w, 52w |
| `ulcer_index` | `sqrt(mean(squared_drawdown))` | 13 weeks |
| `pain_index` | Mean of absolute negative drawdowns | 13 weeks |

#### 6. Statistical/Distribution Features

**Function**: `calculate_statistical_features()`

| Feature | Description | Timeframes |
|---------|-------------|------------|
| `skew_13w`, `skew_26w` | Skewness of log returns distribution | 13w, 26w |
| `kurt_13w`, `kurt_26w` | Kurtosis of log returns distribution | 13w, 26w |
| `autocorr_1w`, `autocorr_4w` | Autocorrelation of log returns at lag 1 and 4 | 13w, 26w windows |

#### 7. Calendar Features

**Function**: `calculate_calendar_features()`

| Feature | Description | Purpose |
|---------|-------------|---------|
| `sin_month`, `cos_month` | Cyclical encoding of month (1-12) | Capture monthly seasonality (January effect, year-end rallies) |
| `sin_dow`, `cos_dow` | Cyclical encoding of day of week (0-6) | Capture weekday effects (Monday effect, options expiry) |
| `sin_dom`, `cos_dom` | Cyclical encoding of day of month (1-31) | Capture turn-of-month patterns, window dressing |

**Encoding**: `sin(2π * value / max_value)` and `cos(2π * value / max_value)` for cyclical representation

### Feature Calculation Order

Features are calculated in this order to respect dependencies:

1. **Macro data merge** (if provided) - Merges macroeconomic indicators with price data
2. **Momentum features** (require: close, high, low) - Calculates log returns and momentum indicators
3. **Volatility features** (require: log returns from momentum) - Calculates volatility metrics using log returns
4. **Volume features** (require: close, high, low, volume) - Calculates volume and liquidity indicators
5. **Risk-adjusted features** (require: log returns, benchmark) - Calculates Sharpe, Sortino, Calmar ratios
6. **Drawdown features** (require: close prices) - Calculates drawdown and path risk metrics
7. **Statistical features** (require: log returns) - Calculates distribution statistics
8. **Calendar features** (require: date column) - Adds cyclical calendar encodings

**Note**: All features are calculated per-ticker to maintain asset-specific characteristics.

### Feature Dependencies

- **Returns-based features** depend on price data
- **Risk-adjusted features** require benchmark returns (QQQ)
- **Rolling window features** require sufficient historical data (may have NaN for initial periods)
- **Calendar features** are independent and can be broadcast across all tickers

---

## Implementation Approaches

### Log Returns Implementation

**Function**: `utile.py::calculate_log_returns()` (lines 26-51)

All price return calculations use **log returns** (`ln(P_t / P_{t-n})`) instead of simple percentage returns (`(P_t - P_{t-n}) / P_{t-n}`).

**Formula**: `r_t = ln(P_t / P_{t-n})`

**Rationale**:
1. **Time-Additive Property**: Log returns are additive over time: `r_total = r1 + r2 + ... + rn`. This makes multi-period calculations straightforward.
2. **Better for Multi-Period Calculations**: When compounding returns, log returns simplify the math.
3. **Standard Practice**: Log returns are standard in quantitative finance and econometrics.
4. **Symmetric**: Log returns treat gains and losses symmetrically (e.g., +50% and -33% are symmetric in log space).

**Scope of Application**:
- All return features: `return_1w`, `return_4w`, `return_13w`, `return_26w`
- Returns used in volatility calculations
- Returns used in risk-adjusted features (Sharpe, Sortino, Calmar)
- Returns used in statistical features (skewness, kurtosis, autocorrelation)
- VWAP momentum and trend features (VWAP is price-like)
- Benchmark returns (QQQ)

**Exception**: Volume features use `pct_change()` because volume is not price-like and percentage changes are more interpretable for volume metrics.

**Implementation**:
```python
def calculate_log_returns(series: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns: ln(P_t / P_{t-n})"""
    if periods == 1:
        return np.log(series / series.shift(1))
    else:
        return np.log(series / series.shift(periods))
```

**Note**: For small returns (< 10%), log returns ≈ simple returns, but log returns are preferred for consistency and mathematical properties.

---

## Data Preprocessing Pipeline

### Overview

**Function**: `utile.py::preprocess_pipeline()`

Complete preprocessing workflow that transforms raw features into normalized, train/test splits ready for RL agent training.

### Pipeline Steps

#### Step 0: Constant Feature Removal

**Function**: `remove_constant_features()`

Removes features with zero or near-zero variance before other preprocessing steps.

```python
from utile import remove_constant_features

df_cleaned, constant_features = remove_constant_features(
    df=features_df,
    feature_cols=feature_cols,
    min_variance=1e-8
)
```

**Criteria**:
- Features with `nunique() <= 1` (single unique value)
- Features with `std() < 1e-8` (near-zero variance)

**Rationale**: 
- Constant features provide no information for model training
- Can cause numerical issues in scaling and correlation calculations
- Removed early to prevent downstream errors

**Note**: This step is automatically called within `preprocess_pipeline()` before outlier removal.

#### Step 1: Missing Data Handling

**Location**: `notebooks/01_data_preparation.ipynb` (Cell 11), `utile.py::preprocess_pipeline()` (lines 943-946)

**Approach**:
1. **Row Filtering**: Drop rows with >50% missing features (notebook-level)
2. **Feature Filling**: Fill remaining NaN values with 0 before preprocessing (pipeline-level)

```python
# Notebook: Drop rows with >50% missing features
row_missing_pct = features_df[feature_cols].isna().sum(axis=1) / len(feature_cols)
features_df_clean = features_df[row_missing_pct <= 0.5].copy()

# Pipeline: Fill remaining NaN with 0
result[available_feature_cols] = result[available_feature_cols].fillna(0)
```

**Rationale**: 
- Balance between data retention and quality
- Filling with 0 is appropriate after removing rows with excessive missingness
- Prevents downstream errors in correlation and scaling calculations

#### Step 2: Outlier Removal

**Function**: `remove_outliers()`

Clips extreme values at specified percentiles to prevent outliers from affecting model training.

```python
from utile import remove_outliers

df_cleaned = remove_outliers(
    df=features_df,
    feature_cols=feature_cols,
    clip_percentiles=(0.01, 0.99)  # Clip at 1st and 99th percentiles
)
```

**Rationale**: 
- Prevents extreme values from dominating feature scaling
- Maintains 98% of data while removing outliers
- Applied before correlation filtering to ensure stable correlation calculations

#### Step 3: Correlation Filtering

**Function**: `remove_highly_correlated_features()`

Removes features with correlation above threshold to reduce redundancy and multicollinearity.

```python
from utile import remove_highly_correlated_features

df_filtered, removed_features = remove_highly_correlated_features(
    df=df_cleaned,
    feature_cols=feature_cols,
    threshold=0.9
)
```

**Algorithm**:
1. Calculate absolute correlation matrix
2. Identify upper triangle pairs with correlation > threshold
3. Remove one feature from each highly correlated pair
4. Return list of removed features for documentation

**Rationale**: Highly correlated features provide redundant information and can cause numerical instability in neural networks.

**Note**: In practice, the threshold is set to **0.9** (not 0.95) for more aggressive filtering, reducing features from ~160 to ~110 for RL model training.

#### Step 4: Optional PCA Dimensionality Reduction

**Function**: `apply_pca_filtering()`

Optional step to reduce dimensionality while preserving variance.

```python
from utile import apply_pca_filtering

df_pca, pca_obj = apply_pca_filtering(
    df=df_filtered,
    feature_cols=feature_cols,
    n_components=None,  # Auto-determine from variance_threshold
    variance_threshold=0.95  # Keep components explaining 95% variance
)
```

**Note**: PCA is disabled by default (`use_pca=False`). Enable only if feature count is very high (>500) or memory constraints exist.

#### Step 5: Train/Test Split

**Function**: `split_data()`

Splits data chronologically (80/20) to prevent data leakage.

```python
from utile import split_data

splits = split_data(
    df=df_filtered,
    train_ratio=0.8,
    date_col='date'
)
train_df = splits['train']
test_df = splits['test']
```

**Critical**: 
- **Chronological split** - No random shuffling
- **20% test set** - Reserved for final evaluation, never used during training
- Split based on unique dates to maintain temporal integrity

#### Step 6: Normalization

**Function**: `normalize_features()`

Applies z-score (standard) scaling fitted on training data only.

```python
from utile import normalize_features

train_norm, test_norm, scaler = normalize_features(
    train_df=train_df,
    test_df=test_df,
    feature_cols=final_feature_cols,
    scaler=None  # Will be fitted on train_df
)
```

**Formula**: `z = (x - μ) / σ` where μ and σ are calculated from training data only

**Critical Rules**:
1. **Fit scaler on training data only** - Prevents data leakage
2. **Transform test data** using training statistics
3. **Never refit** scaler on test/validation data

**Rationale**: Neural networks require normalized inputs for stable training. Using training statistics ensures test set represents true out-of-sample performance.

### Complete Pipeline Usage

```python
from utile import preprocess_pipeline

preprocessed = preprocess_pipeline(
    df=features_df_clean,
    feature_cols=feature_cols,
    train_ratio=0.8,
    clip_percentiles=(0.01, 0.99),
    corr_threshold=0.9,  # More aggressive filtering for RL models
    use_pca=False,  # Set to True for dimensionality reduction
    pca_variance_threshold=0.95,
    date_col='date'
)

# Access results
train_df = preprocessed['train']      # Normalized training data
test_df = preprocessed['test']        # Normalized test data
scaler = preprocessed['scaler']        # Fitted StandardScaler
removed_features = preprocessed['removed_features']  # List of removed features
pca_obj = preprocessed['pca']         # PCA object if used, else None
```

---

## Data Validation & Quality Checks

### Missing Data Handling

**Implementation**: Two-stage approach

1. **Row Filtering** (Notebook): Drop rows with >50% missing features
```python
row_missing_pct = features_df[feature_cols].isna().sum(axis=1) / len(feature_cols)
features_df_clean = features_df[row_missing_pct <= 0.5].copy()
```

2. **Feature Filling** (Pipeline): Fill remaining NaN with 0
```python
# In preprocess_pipeline()
result[available_feature_cols] = result[available_feature_cols].fillna(0)
```

**Check**: Verify missing data percentage per feature

```python
missing_pct = (features_df[feature_cols].isna().sum() / len(features_df) * 100)
print(missing_pct.sort_values())
```

**Action**: Features with >50% missing data should be investigated or removed before preprocessing.

**Rationale**: This two-stage approach balances data retention with quality, ensuring sufficient data for training while handling sparse features appropriately.

### Data Type Validation

**Check**: Ensure correct data types

```python
assert price_df['date'].dtype == 'datetime64[ns]'
assert price_df['close'].dtype in ['float64', 'float32']
assert price_df['ticker'].dtype == 'object'
```

### Date Range Verification

**Check**: Verify date coverage and gaps

```python
dates = sorted(price_df['date'].unique())
print(f"Date range: {dates[0]} to {dates[-1]}")
print(f"Total weeks: {len(dates)}")
print(f"Expected weeks: {(pd.to_datetime('2025-01-01') - pd.to_datetime('2020-01-01')).days // 7}")
```

**Action**: Investigate large gaps (>2 weeks) which may indicate missing data.

### Feature Distribution Checks

**Before Preprocessing**:
```python
features_df[feature_cols].describe()  # Check for extreme values, zeros, etc.
```

**After Normalization**:
```python
# Training set should have mean ≈ 0, std ≈ 1
train_df[feature_cols].mean().abs().max()  # Should be < 0.1
train_df[feature_cols].std().abs().max()    # Should be ≈ 1.0
```

### Correlation Matrix Analysis

**Check**: Verify correlation filtering worked

```python
corr_matrix = train_df[feature_cols].corr().abs()
max_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
print(f"Maximum correlation: {max_corr:.3f}")  # Should be < 0.9 (or 0.95 if using default threshold)
```

**Visualization**: Correlation matrix with hierarchical clustering

The notebook includes a correlation matrix visualization that uses hierarchical clustering to sort features by correlation similarity:

```python
# Calculate correlation matrix
corr_matrix = train_df[final_feature_cols].corr()

# Convert correlation to distance (1 - abs(correlation))
dist_matrix = 1 - np.abs(corr_matrix.values)

# Perform hierarchical clustering (Ward linkage)
linkage_matrix = linkage(squareform(dist_matrix), method='ward')
ordered_indices = leaves_list(linkage_matrix)
ordered_features = [final_feature_cols[i] for i in ordered_indices]

# Reorder and visualize
corr_matrix_sorted = corr_matrix.loc[ordered_features, ordered_features]
```

**Rationale**: Hierarchical clustering groups highly correlated features together, making correlation patterns easier to identify in the heatmap visualization.

### Outlier Detection

**Before Clipping**:
```python
for col in feature_cols[:10]:  # Sample features
    q1 = features_df[col].quantile(0.25)
    q3 = features_df[col].quantile(0.75)
    iqr = q3 - q1
    outliers = ((features_df[col] < q1 - 1.5*iqr) | (features_df[col] > q3 + 1.5*iqr)).sum()
    print(f"{col}: {outliers} outliers")
```

**After Clipping**: Outliers should be reduced significantly.

### Train/Test Split Validation

**Critical Checks**:

1. **No Data Leakage**: Verify test dates are after train dates
```python
assert train_df['date'].max() < test_df['date'].min()
```

2. **Chronological Order**: Verify no overlap
```python
train_dates = set(train_df['date'].unique())
test_dates = set(test_df['date'].unique())
assert len(train_dates & test_dates) == 0
```

3. **Split Ratio**: Verify 80/20 split
```python
total_samples = len(train_df) + len(test_df)
train_ratio = len(train_df) / total_samples
assert 0.79 < train_ratio < 0.81  # Allow small tolerance
```

### Normalization Validation

**Check**: Verify scaler was fitted on train only

```python
# Training set statistics
train_mean = train_df[feature_cols].mean()
train_std = train_df[feature_cols].std()

# Test set should use training statistics (not its own)
# Test mean/std will NOT be exactly 0/1, but should be close
test_mean = test_df[feature_cols].mean()
test_std = test_df[feature_cols].std()

print(f"Train mean range: [{train_mean.min():.3f}, {train_mean.max():.3f}]")
print(f"Train std range: [{train_std.min():.3f}, {train_std.max():.3f}]")
```

---

## Output Structure

### Directory Organization

```
data/
├── raw/
│   ├── weekly_prices.parquet      # Raw price data (long format)
│   └── features_raw.parquet       # Engineered features (before preprocessing)
└── processed/
    ├── train.parquet               # Preprocessed training data
    ├── test.parquet                # Preprocessed test data
    ├── scaler.pkl                  # Fitted StandardScaler (for inference)
    └── metadata.json               # Pipeline metadata

results/
├── price_data_overview.png         # Price and returns visualization
├── feature_distributions.png       # Normalized feature distributions
└── feature_correlation_matrix.png # Correlation matrix (hierarchically clustered)
```

### Metadata Structure

**File**: `data/processed/metadata.json`

```json
{
    "feature_cols": ["return_1w", "rsi_14d", ...],
    "removed_features": ["feature_x", "feature_y"],
    "train_start": "2020-01-03T00:00:00",
    "train_end": "2023-06-30T00:00:00",
    "test_start": "2023-07-07T00:00:00",
    "test_end": "2024-12-27T00:00:00",
    "n_tickers": 8,
    "tickers": ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NFLX", "NVDA", "TSLA"]
}
```

**Usage**: Load metadata for RL environment setup:

```python
import json
with open('data/processed/metadata.json', 'r') as f:
    metadata = json.load(f)
    
feature_cols = metadata['feature_cols']
tickers = metadata['tickers']
```

### Scaler Persistence

**File**: `data/processed/scaler.pkl`

Save and load scaler for inference:

```python
import pickle

# Save (done automatically in pipeline)
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Load (for inference)
with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
```

**Important**: Use the same scaler for all inference to maintain consistency with training.

---

## Best Practices

### Time-Series Data Handling

1. **Chronological Splitting Only**
   - Never use random splits for time-series data
   - Always split by date to maintain temporal order
   - Reserve most recent data for testing

2. **No Future Data Leakage**
   - Ensure features use only past data
   - Rolling windows should not include future values
   - Calendar features are safe (they're deterministic)
   - **Calmar Ratio Fix**: Uses `max_dd.rolling(period).min()` instead of global minimum to prevent data leakage

3. **Lookback Window Handling**
   - Features with rolling windows will have NaN for initial periods
   - Drop rows with insufficient history before training
   - Document minimum required history per feature

### Feature Engineering

1. **Feature Scaling Considerations**
   - Always normalize features before RL training
   - Use z-score scaling for most features
   - Calendar features (sin/cos) are already bounded [-1, 1] but should still be normalized

2. **Handling Missing Values**
   - Drop rows with >50% missing features (notebook-level filtering)
   - Fill remaining NaN with 0 before preprocessing (pipeline-level)
   - Forward-fill used for macro indicators during merging (sparse updates)
   - Document missing data patterns

3. **Return Calculation**
   - Use log returns (`ln(P_t / P_{t-n})`) for all price-based returns
   - Log returns provide time-additive property and are standard in quantitative finance
   - Volume features use percentage change (volume is not price-like)

3. **Feature Selection Strategies**
   - Start with 50-100 most important features
   - Use correlation filtering to remove redundancy
   - Consider PCA only if feature count > 500
   - Use feature importance from trained models to guide selection

### Preprocessing

1. **Fit Scalers on Training Data Only**
   - Critical for preventing data leakage
   - Test set statistics should not influence preprocessing
   - Save scaler for consistent inference

2. **Validate on Separate Set**
   - Use validation set (if created) for hyperparameter tuning
   - Test set reserved for final evaluation only
   - Never refit scalers on validation/test data

3. **Test on Completely Unseen Data**
   - Test set should represent true out-of-sample performance
   - No overlap between train and test dates
   - Document any data quality issues in test set

### Code Organization

1. **All Utility Functions Consolidated in `utile.py`**
   - Reusable utilities for data fetching, feature engineering, preprocessing
   - Clear function signatures with type hints
   - Comprehensive docstrings
   - All functions organized in sections: Data Fetching, Feature Engineering, Preprocessing

2. **Notebooks for Orchestration**
   - Use notebooks to chain pipeline steps
   - Include visualizations and quality checks
   - Document decisions and parameter choices

3. **Clear Separation of Concerns**
   - Data fetching: `utile.py` (fetch_price_data, prepare_price_dataframe, fetch_macro_data)
   - Feature engineering: `utile.py` (calculate_*_features functions, engineer_all_features)
   - Preprocessing: `utile.py` (preprocess_pipeline, remove_outliers, normalize_features, etc.)
   - Execution: `notebooks/01_data_preparation.ipynb`

### Reproducibility

1. **Version Control**
   - Commit `requirements.txt` with exact versions
   - Document API key setup (without exposing keys)
   - Version control preprocessing parameters

2. **Random Seeds**
   - Set random seeds for any stochastic operations
   - Document seed values in metadata

3. **Data Provenance**
   - Record data source URLs/dates
   - Document any manual data modifications
   - Save raw data before any transformations

---

## Troubleshooting

### Common Issues

#### 1. Multi-Index DataFrame Handling

**Problem**: `KeyError` when accessing columns in multi-index DataFrame

**Solution**: Use `prepare_price_dataframe()` to convert to long format:

```python
from utile import prepare_price_dataframe
price_df = prepare_price_dataframe(price_data)
```

**Prevention**: Always use `prepare_price_dataframe()` after `fetch_price_data()`.

#### 2. Missing Adjusted Close Column

**Problem**: Warning "No close price column found for {ticker}"

**Solution**: 
- Verify `auto_adjust=False` in `fetch_price_data()`
- Check if ticker symbol is valid
- Some tickers may not have adjusted close data

**Debug**:
```python
print(price_data.columns)  # Check available columns
print(price_data[ticker].columns)  # Check ticker-specific columns
```

#### 3. API Key Errors

**Problem**: Empty DataFrame returned from `fetch_macro_data()`

**Solutions**:
1. Check `api_keys.json` exists and contains `"fred"` key
2. Verify API key is valid: `https://fred.stlouisfed.org/docs/api/api_key.html`
3. Set environment variable: `export FRED_API_KEY="your_key"`
4. Check `fredapi` is installed: `pip install fredapi`

**Note**: Pipeline continues without macro data if FRED API is unavailable.

#### 4. Feature Calculation Errors

**Problem**: `NaN` values in calculated features

**Causes**:
- Insufficient historical data for rolling windows
- Division by zero (handled with `+ 1e-10` in code)
- Missing input data (price, volume)

**Solution**:
```python
# Check missing data percentage
missing = features_df[feature_cols].isna().sum() / len(features_df)
print(missing[missing > 0.1])  # Features with >10% missing

# Drop rows with insufficient history
features_df_clean = features_df.dropna(subset=feature_cols)
```

#### 5. Memory Issues with Large Datasets

**Problem**: Out of memory errors during feature calculation

**Solutions**:
1. Process tickers in batches:
```python
for ticker_batch in chunks(tickers, 5):
    batch_data = fetch_price_data(ticker_batch, ...)
    # Process batch
```

2. Use PCA for dimensionality reduction:
```python
preprocessed = preprocess_pipeline(..., use_pca=True, pca_variance_threshold=0.95)
```

3. Save intermediate results:
```python
features_df.to_parquet('data/raw/features_raw.parquet')
```

#### 6. Normalization Issues

**Problem**: Test set has mean/std far from 0/1

**Cause**: Scaler was fitted on wrong dataset

**Solution**: Verify scaler was fitted on training data only:
```python
# Check scaler was fitted correctly
assert scaler is not None
train_mean_check = scaler.mean_  # Should match train_df statistics
```

#### 7. Data Leakage Detection

**Problem**: Suspiciously good test performance

**Check**: Verify no overlap between train and test:
```python
train_dates = set(train_df['date'].unique())
test_dates = set(test_df['date'].unique())
overlap = train_dates & test_dates
assert len(overlap) == 0, f"Data leakage detected: {overlap}"
```

### Debugging Tips

1. **Incremental Testing**: Test each pipeline step independently
2. **Small Dataset First**: Use 1-2 tickers and short date range for initial testing
3. **Visual Inspection**: Plot features to verify calculations
4. **Statistics Check**: Compare feature statistics before/after preprocessing
5. **Log Intermediate Results**: Save DataFrames at each step for inspection

---

## References

### Implementation Files

- **All Functions**: `utile.py`
  
  **Data Fetching Section**:
  - `fetch_price_data()` - Yahoo Finance data acquisition
  - `fetch_macro_data()` - FRED macroeconomic data
  - `prepare_price_dataframe()` - Multi-index to long format conversion
  - `_load_api_key_from_file()` - API key loading utility

  **Feature Engineering Section**:
  - `calculate_momentum_features()` - Momentum and trend indicators
  - `calculate_volatility_features()` - Volatility and risk metrics
  - `calculate_volume_features()` - Volume and liquidity indicators
  - `calculate_risk_adjusted_features()` - Sharpe, Sortino, Calmar ratios
  - `calculate_drawdown_features()` - Drawdown and path risk metrics
  - `calculate_statistical_features()` - Distribution statistics
  - `calculate_calendar_features()` - Cyclical calendar encodings
  - `engineer_all_features()` - Complete feature engineering pipeline

  **Preprocessing Section**:
  - `remove_outliers()` - Outlier clipping
  - `remove_highly_correlated_features()` - Correlation filtering
  - `apply_pca_filtering()` - PCA dimensionality reduction
  - `split_data()` - Chronological train/test split
  - `normalize_features()` - Z-score normalization
  - `preprocess_pipeline()` - Complete preprocessing workflow

- **Main Execution**: `notebooks/01_data_preparation.ipynb`
  - Orchestrates complete pipeline
  - Includes visualizations and quality checks
  - Saves processed data and metadata

### External Resources

- **Yahoo Finance**: [yfinance Documentation](https://github.com/ranaroussi/yfinance)
- **FRED API**: [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- **Technical Analysis Library**: [ta Documentation](https://technical-analysis-library-in-python.readthedocs.io/)
- **scikit-learn**: [Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)

### Related Documentation

- Project README: `README.md`
- Requirements: `requirements.txt`
- API Keys Setup: `api_keys.json` (template)

---

## Appendix: Feature Reference Table

| Category | Feature Count (Original) | Feature Count (Final) | Key Features | Implementation Function |
|----------|-------------------------|---------------------|-------------|------------------------|
| Momentum & Trend | ~35 | ~25 | return_1w, return_4w, return_13w, return_26w (log returns), rsi_{7,14,21}d, roc_{1,4,13,26}w, macd, macd_signal, macd_histogram, stochastic_{7,14,21}w, price_to_sma_{4,8,12,20,26}w, price_to_ema_{4,8,12,20,26}w, momentum_{1,4,13,26}w, price_position_{4,13,26,52}w | `utile.py::calculate_momentum_features()` |
| Volatility & Risk | ~20 | ~15 | volatility_{4,13,26,52}w (log returns), realized_volatility_{4,13,26}w, atr_{7,14,26}d, bb_width, bb_position, bb_width_{13,20,26}w, bb_position_{13,20,26}w, downside_volatility_{4,13,26}w, upside_volatility_{4,13,26}w, volatility_ratio_{4,13,26}w, parkinson_volatility_{4,13,26}w | `utile.py::calculate_volatility_features()` |
| Volume & Liquidity | ~30 | ~22 | obv, obv_roc_{1,4,13,26}w, volume_roc_{1,4,13,26}w, relative_volume_{4,8,13,20,26}w, volume_momentum, volume_momentum_4w, volume_sma_ratio, volume_ema_ratio, vwap_distance, vwap_momentum (log returns), vwap_trend (log returns), volume_std, volume_cv, volume_price_divergence, volume_price_correlation, volume_acceleration, volume_trend_strength, volume_breakout, mfi_14d, chaikin_money_flow, volume_oscillator, volume_weighted_macd, volume_roc_momentum | `utile.py::calculate_volume_features()` |
| Risk-Adjusted | ~15 | ~12 | sharpe_{4,13,26,52}w (log returns), sortino_{4,13,26,52}w (log returns), calmar_{4,13,26,52}w (log returns, rolling window), correlation_to_qqq, beta_to_market, relative_return | `utile.py::calculate_risk_adjusted_features()` |
| Drawdown & Path Risk | ~25 | ~18 | current_drawdown, drawdown_duration, max_drawdown_{4,13,26,52}w, avg_drawdown_{4,13,26,52}w, drawdown_volatility_{4,13,26,52}w, recovery_factor_{4,13,26,52}w, ulcer_index_{13,26,52}w, pain_index_{13,26,52}w, mae_{4,13,26}w, mfe_{4,13,26}w | `utile.py::calculate_drawdown_features()` |
| Statistical | 6 | 6 | skew_{13,26}w, kurt_{13,26}w, autocorr_1w, autocorr_4w (all use log returns) | `utile.py::calculate_statistical_features()` |
| Calendar | 6 | 4 | sin_month, cos_month, sin_dow, cos_dow, sin_dom, cos_dom (sin_dow, cos_dow may be removed if constant for weekly data) | `utile.py::calculate_calendar_features()` |
| Macro | 5 | 5 | treasury_10y, fed_funds_rate, vix, unemployment_rate, gdp_growth | `utile.py::merge_macro_data()` |
| **Total** | **~160** | **~110** | | `utile.py::engineer_all_features()` |

**Notes**: 
- **Original count**: ~160 features before preprocessing
- **Final count**: ~110 features after constant feature removal and correlation filtering (threshold=0.9)
- **Log returns**: All return features use log returns (`ln(P_t / P_{t-n})`) instead of simple percentage returns
- **Constant features**: Features with zero variance (e.g., `sin_dow`, `cos_dow` for weekly data) are automatically removed

---

*Last Updated: 2025-01-14*
*Pipeline Version: 1.0*
