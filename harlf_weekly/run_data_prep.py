"""
Data Preparation Script for Multi-Hierarchical RL Portfolio System
Extracted from 01_data_prep.ipynb for command-line execution
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Data fetching
import yfinance as yf

# Preprocessing
from sklearn.preprocessing import StandardScaler

# Random seed
np.random.seed(42)

print("="*70)
print("MULTI-HIERARCHICAL RL PORTFOLIO SYSTEM - DATA PREPARATION")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
BENCHMARK = 'QQQ'
START_DATE = '2020-01-01'
END_DATE = '2025-11-04'

MACRO_TICKERS = {
    'treasury_10y': '^TNX',
    'vix': '^VIX',
}

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

DATA_DIR = Path('data_hierarchical')
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / 'technical').mkdir(exist_ok=True)
(DATA_DIR / 'sentiment').mkdir(exist_ok=True)
(DATA_DIR / 'macro').mkdir(exist_ok=True)

print(f"\n📊 Configuration:")
print(f"   Stocks: {', '.join(TICKERS)}")
print(f"   Benchmark: {BENCHMARK}")
print(f"   Date Range: {START_DATE} to {END_DATE}")
print(f"   Splits: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

# ============================================================================
# STEP 1: DOWNLOAD & ALIGN DATA
# ============================================================================

print("\n" + "="*70)
print("STEP 1: DOWNLOAD & ALIGN DATA")
print("="*70)

def download_stock_data(tickers, start_date, end_date):
    """Download daily stock data and resample to weekly (Friday close)."""
    print(f"\nDownloading data for {len(tickers)} tickers...")
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if df.empty:
                print(f"⚠️  No data for {ticker}")
                continue

            # Handle multi-level column index from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Debug: print columns before normalization (only for first ticker)
            if ticker == tickers[0]:
                print(f"  [DEBUG] Raw columns for {ticker}: {df.columns.tolist()}")

            # Ensure columns are strings and normalize
            df.columns = [str(col).lower().replace(' ', '_') for col in df.columns]

            # Debug: print columns after normalization (only for first ticker)
            if ticker == tickers[0]:
                print(f"  [DEBUG] Normalized columns: {df.columns.tolist()}")

            # Use adjusted close if available, otherwise use close
            adj_close_col = 'adj_close' if 'adj_close' in df.columns else 'close'

            # Resample to weekly (Friday close)
            weekly = pd.DataFrame({
                'open': df['open'].resample('W-FRI').first(),
                'high': df['high'].resample('W-FRI').max(),
                'low': df['low'].resample('W-FRI').min(),
                'close': df['close'].resample('W-FRI').last(),
                'volume': df['volume'].resample('W-FRI').sum(),
                'adj_close': df[adj_close_col].resample('W-FRI').last()
            })

            weekly = weekly.ffill()
            data[ticker] = weekly
            print(f"  ✅ {ticker}: {len(weekly)} weeks")

        except Exception as e:
            print(f"  ❌ {ticker}: {str(e)}")

    return data

# Download stock data
stock_data = download_stock_data(TICKERS + [BENCHMARK], START_DATE, END_DATE)
benchmark_data = stock_data.pop(BENCHMARK)

print(f"\n✅ Downloaded {len(stock_data)} stocks + benchmark")

# Align all stocks to common date range
def align_dataframes(data_dict):
    """Align all DataFrames to common date index (intersection)."""
    common_dates = None
    for ticker, df in data_dict.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    aligned = {}
    for ticker, df in data_dict.items():
        aligned[ticker] = df.loc[common_dates].copy()

    return aligned, common_dates

stock_data_aligned, common_dates = align_dataframes(stock_data)
benchmark_data_aligned = benchmark_data.loc[common_dates].copy()

print(f"\n✅ Aligned to {len(common_dates)} common weeks")
print(f"   Date range: {common_dates[0].date()} to {common_dates[-1].date()}")

# ============================================================================
# STEP 2: TECHNICAL FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 2: CALCULATE TECHNICAL FEATURES (20 indicators)")
print("="*70)

def calculate_technical_features(df, benchmark_df):
    """Calculate 20 technical features for a single stock."""
    features = pd.DataFrame(index=df.index)
    price = df['adj_close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # === TREND FOLLOWING (5 features) ===
    features['price_to_sma_4w'] = price / price.rolling(4).mean() - 1
    features['price_to_sma_8w'] = price / price.rolling(8).mean() - 1
    features['price_to_sma_12w'] = price / price.rolling(12).mean() - 1
    features['price_to_ema_8w'] = price / price.ewm(span=8).mean() - 1
    features['price_to_ema_12w'] = price / price.ewm(span=12).mean() - 1

    # === MOMENTUM (7 features) ===
    ema_12 = price.ewm(span=12).mean()
    ema_26 = price.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    features['macd_histogram'] = macd - signal

    returns = price.pct_change()
    features['return_lag_1w'] = returns.shift(1)
    features['return_lag_2w'] = returns.shift(2)
    features['return_lag_3w'] = returns.shift(3)

    delta = price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    features['rsi_14w'] = 100 - (100 / (1 + rs))

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    features['stochastic_14w'] = 100 * (price - low_14) / (high_14 - low_14 + 1e-10)

    features['roc_4w'] = price.pct_change(4)

    # === VOLATILITY (3 features) ===
    features['volatility_12w'] = returns.rolling(12).std()

    tr1 = high - low
    tr2 = abs(high - price.shift(1))
    tr3 = abs(low - price.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    features['atr_pct_14w'] = atr / price

    sma_20 = price.rolling(20).mean()
    std_20 = price.rolling(20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    features['bb_position_20w'] = (price - bb_lower) / (bb_upper - bb_lower + 1e-10)

    # === VOLUME (3 features) ===
    features['volume_ratio_20w'] = volume / volume.rolling(20).mean()

    typical_price = (high + low + price) / 3
    money_flow = typical_price * volume
    mf_pos = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    mf_neg = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mf_ratio = mf_pos / (mf_neg + 1e-10)
    features['mfi_14w'] = 100 - (100 / (1 + mf_ratio))

    obv = (np.sign(price.diff()) * volume).cumsum()
    features['obv_roc_4w'] = obv.pct_change(4)

    # === BENCHMARK RELATIONSHIP (2 features) ===
    bench_returns = benchmark_df['adj_close'].pct_change()
    features['bench_corr_12w'] = returns.rolling(12).corr(bench_returns)

    covariance = returns.rolling(12).cov(bench_returns)
    bench_variance = bench_returns.rolling(12).var()
    features['bench_beta_12w'] = covariance / (bench_variance + 1e-10)

    return features

print("\nCalculating technical features for each stock...")
technical_features = {}
for ticker, df in stock_data_aligned.items():
    tech_feat = calculate_technical_features(df, benchmark_data_aligned)
    tech_feat['ticker'] = ticker
    tech_feat['date'] = tech_feat.index
    technical_features[ticker] = tech_feat
    print(f"  ✅ {ticker}: {tech_feat.shape[1]-2} features")

technical_df = pd.concat(technical_features.values(), ignore_index=True)
print(f"\n✅ Technical features: {technical_df.shape[0]} rows × {technical_df.shape[1]} columns")

# ============================================================================
# STEP 3: SENTIMENT FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 3: CALCULATE SENTIMENT FEATURES (12 indicators)")
print("="*70)

def calculate_sentiment_features(df, all_stocks_df):
    """Calculate 12 sentiment features for a single stock."""
    features = pd.DataFrame(index=df.index)
    price = df['adj_close']
    volume = df['volume']
    returns = price.pct_change()

    # === MOMENTUM SIGNALS (3 features) ===
    features['momentum_2w'] = price.pct_change(2)
    features['momentum_4w'] = price.pct_change(4)
    features['momentum_6w'] = price.pct_change(6)

    # === TREND QUALITY (2 features) ===
    features['positive_weeks_pct'] = (returns > 0).rolling(12).mean()
    features['price_dispersion'] = np.nan  # Placeholder

    # === VOLUME SENTIMENT (2 features) ===
    vol_norm = volume / volume.rolling(20).mean()
    features['volume_sentiment'] = returns * vol_norm

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    features['volume_surge'] = (volume - vol_mean) / (vol_std + 1e-10)

    # === MARKET REGIME (2 features) ===
    vol = returns.rolling(12).std()
    vol_percentile = vol.rolling(52).rank(pct=True)
    features['vol_regime'] = vol_percentile

    cummax = price.cummax()
    features['drawdown'] = (price - cummax) / cummax

    # === FEAR/GREED PROXIES (3 features) ===
    vol_spike = vol / vol.rolling(52).mean()
    features['market_fear'] = vol_spike
    features['credit_sentiment'] = -features['drawdown']
    features['news_sentiment'] = features['momentum_4w'] * np.tanh(vol_norm - 1)

    return features

print("\nCalculating cross-sectional features...")
all_returns = pd.DataFrame({
    ticker: stock_data_aligned[ticker]['adj_close'].pct_change()
    for ticker in TICKERS
})
price_dispersion = all_returns.std(axis=1)

print("Calculating sentiment features for each stock...")
sentiment_features = {}
for ticker, df in stock_data_aligned.items():
    sent_feat = calculate_sentiment_features(df, stock_data_aligned)
    sent_feat['price_dispersion'] = price_dispersion
    sent_feat['ticker'] = ticker
    sent_feat['date'] = sent_feat.index
    sentiment_features[ticker] = sent_feat
    print(f"  ✅ {ticker}: {sent_feat.shape[1]-2} features")

sentiment_df = pd.concat(sentiment_features.values(), ignore_index=True)
print(f"\n✅ Sentiment features: {sentiment_df.shape[0]} rows × {sentiment_df.shape[1]} columns")

# ============================================================================
# STEP 4: MACRO FEATURES
# ============================================================================

print("\n" + "="*70)
print("STEP 4: CALCULATE MACRO FEATURES (12 indicators)")
print("="*70)

print("\nDownloading macro data...")
vix_data = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
treasury_10y_data = yf.download('^TNX', start=START_DATE, end=END_DATE, progress=False)
treasury_2y_data = yf.download('^IRX', start=START_DATE, end=END_DATE, progress=False)

# Extract Close column (handle both Series and DataFrame)
vix_close = vix_data['Close'] if 'Close' in vix_data.columns else vix_data.iloc[:, 0]
treasury_10y_close = treasury_10y_data['Close'] if 'Close' in treasury_10y_data.columns else treasury_10y_data.iloc[:, 0]
treasury_2y_close = treasury_2y_data['Close'] if 'Close' in treasury_2y_data.columns else treasury_2y_data.iloc[:, 0]

vix_weekly = vix_close.resample('W-FRI').last().ffill()
treasury_10y_weekly = treasury_10y_close.resample('W-FRI').last().ffill()
treasury_2y_weekly = treasury_2y_close.resample('W-FRI').last().ffill()

# Ensure Series (squeeze if DataFrame)
vix_weekly = vix_weekly.squeeze() if hasattr(vix_weekly, 'squeeze') else vix_weekly
treasury_10y_weekly = treasury_10y_weekly.squeeze() if hasattr(treasury_10y_weekly, 'squeeze') else treasury_10y_weekly
treasury_2y_weekly = treasury_2y_weekly.squeeze() if hasattr(treasury_2y_weekly, 'squeeze') else treasury_2y_weekly

vix_aligned = vix_weekly.reindex(common_dates).ffill()
treasury_10y_aligned = treasury_10y_weekly.reindex(common_dates).ffill()
treasury_2y_aligned = treasury_2y_weekly.reindex(common_dates).ffill()

print(f"  ✅ VIX: {len(vix_aligned)} weeks")
print(f"  ✅ 10Y Treasury: {len(treasury_10y_aligned)} weeks")
print(f"  ✅ 2Y Treasury: {len(treasury_2y_aligned)} weeks")

def calculate_macro_features(dates, vix, treasury_10y, treasury_2y):
    """Calculate 12 macro features (shared across portfolio)."""
    features = pd.DataFrame(index=dates)

    # === INTEREST RATE ENVIRONMENT (4 features) ===
    features['treasury_10y'] = treasury_10y
    features['fed_funds_rate'] = treasury_2y
    features['yield_curve_2_10'] = treasury_10y - treasury_2y
    features['real_yield_10y'] = treasury_10y - 2.0

    # === MARKET VOLATILITY & RISK (2 features) ===
    features['vix'] = vix
    features['vix_regime'] = vix.rolling(52).rank(pct=True)

    # === ECONOMIC HEALTH (2 features) ===
    features['credit_spread'] = (vix - vix.rolling(52).mean()) / vix.rolling(52).std()
    features['unemployment_rate'] = 4.0  # Placeholder

    # === CALENDAR EFFECTS (4 features) ===
    months = dates.month
    features['month_sin'] = np.sin(2 * np.pi * months / 12)
    features['month_cos'] = np.cos(2 * np.pi * months / 12)

    quarters = dates.quarter
    features['quarter_sin'] = np.sin(2 * np.pi * quarters / 4)
    features['quarter_cos'] = np.cos(2 * np.pi * quarters / 4)

    return features

print("\nCalculating macro features...")
macro_features = calculate_macro_features(
    common_dates, vix_aligned, treasury_10y_aligned, treasury_2y_aligned
)

macro_df_list = []
for ticker in TICKERS:
    macro_copy = macro_features.copy()
    macro_copy['ticker'] = ticker
    macro_copy['date'] = macro_copy.index
    macro_df_list.append(macro_copy)

macro_df = pd.concat(macro_df_list, ignore_index=True)
print(f"\n✅ Macro features: {macro_df.shape[0]} rows × {macro_df.shape[1]} columns")

# ============================================================================
# STEP 5: TRAIN/VAL/TEST SPLITS
# ============================================================================

print("\n" + "="*70)
print("STEP 5: CREATE TRAIN/VAL/TEST SPLITS (60/20/20)")
print("="*70)

def create_time_splits(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """Create time-based train/val/test splits."""
    unique_dates = sorted(df['date'].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[train_end_idx:val_end_idx]
    test_dates = unique_dates[val_end_idx:]

    train_df = df[df['date'].isin(train_dates)].copy()
    val_df = df[df['date'].isin(val_dates)].copy()
    test_df = df[df['date'].isin(test_dates)].copy()

    return train_df, val_df, test_df, (train_dates[0], train_dates[-1]), (val_dates[0], val_dates[-1]), (test_dates[0], test_dates[-1])

print("\nSplitting technical features...")
tech_train, tech_val, tech_test, tech_train_range, tech_val_range, tech_test_range = create_time_splits(
    technical_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)
print(f"  Train: {len(tech_train)} rows ({tech_train_range[0].date()} to {tech_train_range[1].date()})")
print(f"  Val:   {len(tech_val)} rows ({tech_val_range[0].date()} to {tech_val_range[1].date()})")
print(f"  Test:  {len(tech_test)} rows ({tech_test_range[0].date()} to {tech_test_range[1].date()})")

print("\nSplitting sentiment features...")
sent_train, sent_val, sent_test, _, _, _ = create_time_splits(sentiment_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(f"  Train: {len(sent_train)} rows")
print(f"  Val:   {len(sent_val)} rows")
print(f"  Test:  {len(sent_test)} rows")

print("\nSplitting macro features...")
macro_train, macro_val, macro_test, _, _, _ = create_time_splits(macro_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(f"  Train: {len(macro_train)} rows")
print(f"  Val:   {len(macro_val)} rows")
print(f"  Test:  {len(macro_test)} rows")

print("\nCalculating and splitting returns...")
returns_df = pd.DataFrame({
    ticker: stock_data_aligned[ticker]['adj_close'].pct_change()
    for ticker in TICKERS
})
returns_df['date'] = common_dates

returns_train, returns_val, returns_test, _, _, _ = create_time_splits(returns_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(f"  Train: {len(returns_train)} weeks × {len(TICKERS)} stocks")
print(f"  Val:   {len(returns_val)} weeks × {len(TICKERS)} stocks")
print(f"  Test:  {len(returns_test)} weeks × {len(TICKERS)} stocks")

# ============================================================================
# STEP 6: FEATURE NORMALIZATION
# ============================================================================

print("\n" + "="*70)
print("STEP 6: NORMALIZE FEATURES (fit on train only)")
print("="*70)

def normalize_features(train_df, val_df, test_df, exclude_cols=['date', 'ticker']):
    """Normalize features using StandardScaler (fit on train only)."""
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    scaler = StandardScaler()
    train_features_normalized = scaler.fit_transform(train_df[feature_cols])
    val_features_normalized = scaler.transform(val_df[feature_cols])
    test_features_normalized = scaler.transform(test_df[feature_cols])

    train_normalized = pd.DataFrame(train_features_normalized, columns=feature_cols, index=train_df.index)
    train_normalized['date'] = train_df['date'].values
    train_normalized['ticker'] = train_df['ticker'].values

    val_normalized = pd.DataFrame(val_features_normalized, columns=feature_cols, index=val_df.index)
    val_normalized['date'] = val_df['date'].values
    val_normalized['ticker'] = val_df['ticker'].values

    test_normalized = pd.DataFrame(test_features_normalized, columns=feature_cols, index=test_df.index)
    test_normalized['date'] = test_df['date'].values
    test_normalized['ticker'] = test_df['ticker'].values

    return train_normalized, val_normalized, test_normalized, scaler, feature_cols

print("\nFilling nulls before normalization...")
# Fill nulls with appropriate strategies
# For rolling features, forward fill then backward fill
feature_cols_tech = [col for col in tech_train.columns if col not in ['date', 'ticker']]
feature_cols_sent = [col for col in sent_train.columns if col not in ['date', 'ticker']]
feature_cols_macro = [col for col in macro_train.columns if col not in ['date', 'ticker']]

for df in [tech_train, tech_val, tech_test]:
    df[feature_cols_tech] = df[feature_cols_tech].ffill().bfill().fillna(0)

for df in [sent_train, sent_val, sent_test]:
    df[feature_cols_sent] = df[feature_cols_sent].ffill().bfill().fillna(0)

for df in [macro_train, macro_val, macro_test]:
    df[feature_cols_macro] = df[feature_cols_macro].ffill().bfill().fillna(0)

print("  ✅ Nulls filled (forward fill → backward fill → zeros)")

print("\nNormalizing technical features...")
tech_train_norm, tech_val_norm, tech_test_norm, tech_scaler, tech_feature_cols = normalize_features(
    tech_train, tech_val, tech_test
)
print(f"  ✅ {len(tech_feature_cols)} features normalized")

print("\nNormalizing sentiment features...")
sent_train_norm, sent_val_norm, sent_test_norm, sent_scaler, sent_feature_cols = normalize_features(
    sent_train, sent_val, sent_test
)
print(f"  ✅ {len(sent_feature_cols)} features normalized")

print("\nNormalizing macro features...")
macro_train_norm, macro_val_norm, macro_test_norm, macro_scaler, macro_feature_cols = normalize_features(
    macro_train, macro_val, macro_test
)
print(f"  ✅ {len(macro_feature_cols)} features normalized")

# ============================================================================
# STEP 7: SAVE & VALIDATE
# ============================================================================

print("\n" + "="*70)
print("STEP 7: SAVE DATA & VALIDATE")
print("="*70)

print("\nSaving technical features...")
tech_train_norm.to_csv(DATA_DIR / 'technical' / 'train.csv', index=False)
tech_val_norm.to_csv(DATA_DIR / 'technical' / 'val.csv', index=False)
tech_test_norm.to_csv(DATA_DIR / 'technical' / 'test.csv', index=False)
print(f"  ✅ Saved to {DATA_DIR / 'technical'}/")

print("\nSaving sentiment features...")
sent_train_norm.to_csv(DATA_DIR / 'sentiment' / 'train.csv', index=False)
sent_val_norm.to_csv(DATA_DIR / 'sentiment' / 'val.csv', index=False)
sent_test_norm.to_csv(DATA_DIR / 'sentiment' / 'test.csv', index=False)
print(f"  ✅ Saved to {DATA_DIR / 'sentiment'}/")

print("\nSaving macro features...")
macro_train_norm.to_csv(DATA_DIR / 'macro' / 'train.csv', index=False)
macro_val_norm.to_csv(DATA_DIR / 'macro' / 'val.csv', index=False)
macro_test_norm.to_csv(DATA_DIR / 'macro' / 'test.csv', index=False)
print(f"  ✅ Saved to {DATA_DIR / 'macro'}/")

print("\nSaving returns data...")
returns_train.to_csv(DATA_DIR / 'returns_train.csv', index=False)
returns_val.to_csv(DATA_DIR / 'returns_val.csv', index=False)
returns_test.to_csv(DATA_DIR / 'returns_test.csv', index=False)
print(f"  ✅ Saved to {DATA_DIR}/")

print("\nCreating metadata...")
metadata = {
    'tickers': TICKERS,
    'benchmark': BENCHMARK,
    'date_range': {
        'start': str(common_dates[0].date()),
        'end': str(common_dates[-1].date()),
        'total_weeks': len(common_dates)
    },
    'splits': {
        'train': {
            'ratio': TRAIN_RATIO,
            'start': str(tech_train_range[0].date()),
            'end': str(tech_train_range[1].date()),
            'weeks': len(tech_train) // len(TICKERS)
        },
        'val': {
            'ratio': VAL_RATIO,
            'start': str(tech_val_range[0].date()),
            'end': str(tech_val_range[1].date()),
            'weeks': len(tech_val) // len(TICKERS)
        },
        'test': {
            'ratio': TEST_RATIO,
            'start': str(tech_test_range[0].date()),
            'end': str(tech_test_range[1].date()),
            'weeks': len(tech_test) // len(TICKERS)
        }
    },
    'features': {
        'technical': {
            'count': len(tech_feature_cols),
            'names': tech_feature_cols
        },
        'sentiment': {
            'count': len(sent_feature_cols),
            'names': sent_feature_cols
        },
        'macro': {
            'count': len(macro_feature_cols),
            'names': macro_feature_cols
        }
    },
    'normalization': {
        'technical': {
            'means': tech_scaler.mean_.tolist(),
            'stds': tech_scaler.scale_.tolist(),
            'feature_list': tech_feature_cols
        },
        'sentiment': {
            'means': sent_scaler.mean_.tolist(),
            'stds': sent_scaler.scale_.tolist(),
            'feature_list': sent_feature_cols
        },
        'macro': {
            'means': macro_scaler.mean_.tolist(),
            'stds': macro_scaler.scale_.tolist(),
            'feature_list': macro_feature_cols
        }
    },
    'created_at': datetime.now().isoformat()
}

with open(DATA_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"  ✅ Metadata saved to metadata.json")

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "="*70)
print("FINAL VALIDATION")
print("="*70)

all_checks_passed = True

# Check 1: No null values (should be none after filling)
print("\n1. Checking for null values...")
for name, df in [
    ('Technical Train', tech_train_norm),
    ('Sentiment Train', sent_train_norm),
    ('Macro Train', macro_train_norm)
]:
    null_count = df.isnull().sum().sum()
    if null_count == 0:
        print(f"  ✅ {name}: No null values")
    else:
        print(f"  ❌ {name}: {null_count} null values found!")
        all_checks_passed = False

# Check 2: Feature counts
print("\n2. Checking feature counts...")
expected_counts = {'Technical': 20, 'Sentiment': 12, 'Macro': 12}
actual_counts = {
    'Technical': len(tech_feature_cols),
    'Sentiment': len(sent_feature_cols),
    'Macro': len(macro_feature_cols)
}

for name, expected in expected_counts.items():
    actual = actual_counts[name]
    if actual == expected:
        print(f"  ✅ {name}: {actual} features (matches specification)")
    else:
        print(f"  ❌ {name}: {actual} features (expected {expected})!")
        all_checks_passed = False

# Check 3: All tickers present
print("\n3. Checking ticker presence...")
for name, df in [
    ('Technical', tech_train_norm),
    ('Sentiment', sent_train_norm),
    ('Macro', macro_train_norm)
]:
    unique_tickers = df['ticker'].unique()
    if set(unique_tickers) == set(TICKERS):
        print(f"  ✅ {name}: All {len(TICKERS)} tickers present")
    else:
        print(f"  ❌ {name}: Missing tickers!")
        all_checks_passed = False

# Summary
print("\n" + "="*70)
if all_checks_passed:
    print("🎉 ALL VALIDATION CHECKS PASSED!")
else:
    print("⚠️  SOME VALIDATION CHECKS FAILED")
print("="*70)

print(f"\n📊 DATA PREPARATION SUMMARY")
print(f"   Total Weeks: {len(common_dates)}")
print(f"   Train: {metadata['splits']['train']['weeks']} weeks")
print(f"   Val:   {metadata['splits']['val']['weeks']} weeks")
print(f"   Test:  {metadata['splits']['test']['weeks']} weeks")
print(f"   Total Features: {sum([len(tech_feature_cols), len(sent_feature_cols), len(macro_feature_cols)])}")
print(f"\n✅ Data preparation complete! Ready for agent training.")
print("="*70)
