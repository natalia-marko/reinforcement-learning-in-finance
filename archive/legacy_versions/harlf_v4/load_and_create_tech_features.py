
"""
load_and_create_tech_features.py
Data pipeline: Download prices, calculate technical features, save data
- Keeps all timestamps
- Forward-fills short internal gaps
- Backfills leading NaNs (prices with first real price, volume with 0)
- NO normalization here (avoid leakage)
- Adds lagged return feature to avoid contemporaneous leakage
- Writes coverage file with original first real date per ticker
"""

import os
import json
import warnings
from datetime import datetime

from typing import Optional, Tuple

import numpy as np
import pandas as pd
# yfinance is used for downloading historical price data.  It is imported
# lazily inside `collect_price_data` so that the module can be imported
# without requiring this optional dependency at runtime.  If you intend
# to download data, please ensure yfinance is installed (e.g. via
# `pip install yfinance`).
try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None  # type: ignore

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from config import PORTFOLIO, DATA_CONFIG

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _first_valid_index_per_col(df: pd.DataFrame):
    out = {}
    for c in df.columns:
        s = df[c]
        idx = s.first_valid_index()
        out[c] = idx
    return out

def _backfill_leading_with_first_value(
    s: pd.Series, *, is_volume: bool = False, volume_leading_value: float = 0.0
) -> pd.Series:
    """Backfill only the *leading* NaNs (before the first valid observation)."""
    if s.notna().any():
        first_idx = s.first_valid_index()
        if first_idx is not None:
            first_pos = s.index.get_loc(first_idx)
            first_val = s.iloc[first_pos]
            s.iloc[:first_pos] = volume_leading_value if is_volume else first_val
    return s

# ---------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------
def collect_price_data(portfolio, start_date, end_date, interval):
    tickers = list(portfolio.keys())
    print(f"\nTickers: {tickers}")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"Interval: {interval}")

    print("\nDownloading price data (this may take 1-2 minutes)...")
    if yf is None:
        raise ImportError(
            "yfinance is not installed. Please install it with 'pip install yfinance' "
            "to enable data downloading."
        )

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker",
        auto_adjust=True,
        progress=False,
    )

    price_raw = pd.DataFrame()
    volume_raw = pd.DataFrame()
    failed = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                price_raw[ticker] = data["Close"]
                volume_raw[ticker] = data["Volume"]
            else:
                price_raw[ticker] = data[ticker]["Close"]
                volume_raw[ticker] = data[ticker]["Volume"]
            start_v = price_raw[ticker].dropna().iloc[0] if price_raw[ticker].notna().any() else np.nan
            end_v = price_raw[ticker].dropna().iloc[-1] if price_raw[ticker].notna().any() else np.nan
            print(f"  ✓ {ticker}: {len(price_raw[ticker])} periods, ${start_v:.2f} → ${end_v:.2f}")
        except (KeyError, AttributeError, IndexError) as e:
            print(f"  ❌ {ticker}: Failed to download - {e}")
            failed.append(ticker)

    if price_raw.shape[1] == 0:
        raise ValueError("No data downloaded! Check tickers/internet.")

    # Record first real date BEFORE any filling/backfilling
    first_real_dates = _first_valid_index_per_col(price_raw)

    print("\nCleaning data (no row drops; forward-fill internals; backfill leading only)...")
    initial_rows = len(price_raw)

    # 1) Forward-fill short internal gaps
    price_data = price_raw.ffill(limit=5)
    volume_data = volume_raw.ffill(limit=5)

    # 2) Backfill only leading NaNs to keep panel rectangular
    price_data = price_data.apply(_backfill_leading_with_first_value, is_volume=False)
    volume_data = volume_data.apply(_backfill_leading_with_first_value, is_volume=True, volume_leading_value=0.0)

    # 3) Tiny controlled backfill at the very start
    price_data = price_data.bfill(limit=2)
    volume_data = volume_data.bfill(limit=2)

    print(f"  Rows before cleaning: {initial_rows}")
    print(f"  Rows after cleaning:  {len(price_data)} (no rows dropped)")

    # Diagnostics
    cov_rows = []
    for t in price_data.columns:
        s = price_data[t]
        first_real = first_real_dates.get(t, None)
        cov_rows.append({
            'ticker': t,
            'first_real_date': str(first_real.date()) if first_real is not None else None,
            'non_null_periods_postclean': int(s.notna().sum())
        })
    cov_df = pd.DataFrame(cov_rows, columns=['ticker', 'first_real_date', 'non_null_periods_postclean'])
    print("\nPer-asset first real date (pre-backfill) and coverage:")
    print(cov_df.sort_values(['first_real_date', 'ticker']))

    print("\nCalculating log returns (column-wise; no global drop)...")
    log_returns = price_data.apply(lambda s: np.log(s / s.shift(1)))

    # Summary dates (any non-null)
    first_dates = [price_raw[c].first_valid_index() for c in price_raw.columns if price_raw[c].first_valid_index() is not None]
    last_dates  = [price_raw[c].last_valid_index()  for c in price_raw.columns if price_raw[c].last_valid_index()  is not None]
    overall_first = min(first_dates).date() if first_dates else None
    overall_last  = max(last_dates).date() if last_dates else None

    print(f"\n{'─'*80}")
    print("DATA COLLECTION SUMMARY:")
    print(f"{'─'*80}")
    print(f"  Successfully loaded: {price_data.shape[1]} assets")
    if failed:
        print(f"  Failed to load: {failed}")
    print(f"  Date range: {overall_first} to {overall_last}")
    print(f"  Total periods retained: {len(price_data)} (no drops)")
    print(f"{'─'*80}\n")

    return price_data, volume_data, log_returns, failed, cov_df

# ---------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------
def calculate_technical_features(prices, volume, returns, lookbacks):
    """Compute weekly technical features (unscaled)."""
    print("\n" + "=" * 80)
    print("CALCULATING TECHNICAL FEATURES")
    print("=" * 80)

    print(f"\nLookback periods:")
    print(f"  Short:  {lookbacks['short']} weeks (~{lookbacks['short']/4:.1f} months)")
    print(f"  Medium: {lookbacks['medium']} weeks (~{lookbacks['medium']/4:.1f} months)")
    print(f"  Long:   {lookbacks['long']} weeks (~{lookbacks['long']/4:.1f} months)")

    all_idx = prices.index.union(returns.index).union(volume.index)
    all_features = pd.DataFrame(index=all_idx)

    for asset in prices.columns:
        print(f"\nProcessing {asset}...", end=" ")
        asset_prices = prices[asset]
        asset_volume = volume[asset] if asset in volume.columns else pd.Series(0.0, index=prices.index)
        asset_returns = returns[asset]

        # Align series
        common_idx = asset_prices.index.intersection(asset_returns.index).intersection(asset_volume.index)
        asset_prices = asset_prices.loc[common_idx]
        asset_volume = asset_volume.loc[common_idx]
        asset_returns = asset_returns.loc[common_idx]

        features = pd.DataFrame(index=common_idx)

        # 1. Lagged basic return (to avoid contemporaneous info)
        features['log_return'] = asset_returns.shift(1)

        # 2. Moving averages (ratios to price)
        s = lookbacks['short']; m = lookbacks['medium']; l = lookbacks['long']
        sma_s = asset_prices.rolling(s).mean()
        sma_m = asset_prices.rolling(m).mean()
        sma_l = asset_prices.rolling(l).mean()
        features['sma_short_ratio'] = sma_s / asset_prices - 1
        features['sma_medium_ratio'] = sma_m / asset_prices - 1
        features['sma_long_ratio'] = sma_l / asset_prices - 1
        features['sma_cross'] = (sma_s - sma_m) / asset_prices

        # 3. Volatility (annualized weekly)
        features['volatility_short'] = asset_returns.rolling(s).std() * np.sqrt(52)
        features['volatility_medium'] = asset_returns.rolling(m).std() * np.sqrt(52)
        features['volatility_long'] = asset_returns.rolling(l).std() * np.sqrt(52)

        # 4. Momentum
        features['momentum_short']  = np.log(asset_prices / asset_prices.shift(s))
        features['momentum_medium'] = np.log(asset_prices / asset_prices.shift(m))
        features['momentum_long']   = np.log(asset_prices / asset_prices.shift(l))

        # 5. Risk-adjusted metrics (annualized)
        mean_r = asset_returns.rolling(l).mean() * 52
        vol_r  = asset_returns.rolling(l).std() * np.sqrt(52)
        features['sharpe'] = mean_r / (vol_r + 1e-10)

        downside = asset_returns.where(asset_returns < 0, 0)
        dstd = downside.rolling(l).std() * np.sqrt(52)
        features['sortino'] = mean_r / (dstd + 1e-10)

        # 6. Drawdown
        cumulative = (1 + asset_returns.fillna(0)).cumprod()
        rolling_max = cumulative.rolling(l, min_periods=1).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        features['max_drawdown'] = drawdown.rolling(s).min()
        features['calmar'] = mean_r / (abs(features['max_drawdown']) + 1e-10)

        # 7. RSI proxy
        delta = asset_returns
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        features['rsi'] = 100 - (100 / (1 + rs))

        # 8. Volume features
        vol_sma_s = asset_volume.rolling(s).mean()
        features['volume_ratio'] = asset_volume / (vol_sma_s + 1)
        features['volume_momentum'] = asset_volume / (asset_volume.shift(s) + 1)
        features['log_vol'] = np.log(asset_volume + 1)

        # VPT
        vpt = (asset_returns.fillna(0) * asset_volume.fillna(0)).cumsum()
        vpt_ma = vpt.rolling(m).mean()
        vpt_std = vpt.rolling(l).std()
        features['vpt'] = vpt / (vpt_std + 1)
        features['vpt_ma'] = (vpt - vpt_ma) / (vpt_std + 1)

        # OBV
        obv = (np.sign(asset_returns.fillna(0)) * asset_volume.fillna(0)).cumsum()
        obv_ma = obv.rolling(m).mean()
        obv_std = obv.rolling(l).std()
        features['obv'] = obv / (obv_std + 1)
        features['obv_ma'] = (obv - obv_ma) / (obv_std + 1)

        # MFI (close as typical)
        typical_price = asset_prices
        money_flow = typical_price * asset_volume
        pos_flow = money_flow.where(asset_returns > 0, 0).rolling(14).sum()
        neg_flow = money_flow.where(asset_returns < 0, 0).rolling(14).sum()
        money_ratio = pos_flow / (neg_flow + 1)
        features['mfi'] = 100 - (100 / (1 + money_ratio))

        # Clean & prefix
        features = features.replace([np.inf, -np.inf], 0.0)
        features = features.ffill().bfill().fillna(0.0)

        features.columns = [f"{asset}_" + c for c in features.columns]
        all_features = pd.concat([all_features, features], axis=1)

        print(f"✓ {len(features.columns)} features")


    print(f"\n{'─'*80}")
    print(f"TOTAL FEATURES: {len(all_features.columns)}")
    print(f"{'─'*80}\n")
    return all_features

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
def save_data(price_data, features_df, portfolio, failed, coverage_df, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    price_file = os.path.join(output_dir, "prepared_prices_weekly.csv")
    price_data.to_csv(price_file)
    print(f"\n✓ Prices saved: {price_file}\n  Shape: {price_data.shape}")

    features_file = os.path.join(output_dir, "prepared_features_weekly.csv")
    features_df.to_csv(features_file)
    print(f"\n✓ Features saved: {features_file}\n  Shape: {features_df.shape}")

    coverage_file = os.path.join(output_dir, "asset_first_valid.csv")
    coverage_df.to_csv(coverage_file, index=False)
    print(f"\n✓ Coverage saved: {coverage_file}")

    # Summary
    first_dates = [price_data[c].first_valid_index() for c in price_data.columns if price_data[c].first_valid_index() is not None]
    last_dates  = [price_data[c].last_valid_index()  for c in price_data.columns if price_data[c].last_valid_index()  is not None]
    start_date = min(first_dates).date() if first_dates else None
    end_date   = max(last_dates).date() if last_dates else None

    summary = {
        "data_frequency": DATA_CONFIG["frequency"],
        "start_date": str(start_date),
        "end_date": str(end_date),
        "n_periods": int(len(price_data)),
        "n_assets": int(len(price_data.columns)),
        "assets": list(price_data.columns),
        "failed_assets": failed,
        "n_features": int(len(features_df.columns)),
        "features_per_asset": int(len(features_df.columns) // max(1, len(price_data.columns))),
        "lookbacks": {
            "short": DATA_CONFIG["lookback_short"],
            "medium": DATA_CONFIG["lookback_medium"],
            "long": DATA_CONFIG["lookback_long"],
        },
        "created_at": str(datetime.now()),
    }

    summary_file = os.path.join(output_dir, "data_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved: {summary_file}")

    print(f"\n{'='*80}\nDATA PIPELINE COMPLETE!\n{'='*80}\n")
    print("Summary:")
    print(f"  Assets:        {summary['n_assets']} loaded, {len(failed)} failed")
    print(f"  Date range:    {summary['start_date']} to {summary['end_date']}")
    print(f"  Periods:       {summary['n_periods']}")
    print(f"  Total features:{summary['n_features']}")
    print(f"  Features/asset:{summary['features_per_asset']}")
    print("="*80 + "\n")
    return summary

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    start_date = DATA_CONFIG["start_date"]
    end_date = DATA_CONFIG["end_date"]
    interval = DATA_CONFIG["interval"]

    lookbacks = {
        "short": DATA_CONFIG["lookback_short"],
        "medium": DATA_CONFIG["lookback_medium"],
        "long": DATA_CONFIG["lookback_long"],
    }

    print("\n" + "=" * 80)
    print("DATA PIPELINE - LOAD AND CREATE TECHNICAL FEATURES")
    print("=" * 80)

    print(f"\nPortfolio: {len(PORTFOLIO)} assets")
    print(f"Date range: {start_date} to {end_date or 'today'}")
    print(f"Frequency: {DATA_CONFIG['frequency']}")

    # Step 1
    price_data, volume_data, log_returns, failed, coverage_df = collect_price_data(
        PORTFOLIO, start_date, end_date, interval
    )
    # Step 2
    features_df = calculate_technical_features(price_data, volume_data, log_returns, lookbacks)
    # Step 3
    _ = save_data(price_data.round(2), features_df.round(4), PORTFOLIO, failed, coverage_df)




#
# The definitions of `fit_normalizer_on_train`, `normalize_features`,
# `prepare_train_val_test_split`, `prepare_walkforward_window` and
# `validate_no_leakage` have been moved to the separate `utils.py` module.
# They are imported at the top of this file to avoid code duplication and
# potential inconsistencies.  If you need to use these functions here,
# please import them from `utils` as shown above.

