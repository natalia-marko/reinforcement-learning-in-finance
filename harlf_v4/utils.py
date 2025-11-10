"""
utils.py

Utility functions for preparing and normalising data for the RL trading
pipeline.  These helpers centralise the logic for clipping pre‑IPO backfill,
fitting feature normalisers and splitting time series into training,
validation and test windows.  They are designed to operate on a single
asset at a time and to avoid any form of data leakage by ensuring that
normalisation parameters are learned exclusively from training data.

The functions defined here are deliberately free of any global state such
as `ASSET_START_DATES` – instead they derive the valid date range from
the provided price series.  This makes them robust when working with
assets that have different listing histories.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def clip_to_valid_range(
    prices: pd.Series,
    features: pd.DataFrame,
    asset_name: str,
    lookback_long: int = 52,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Trim the leading portion of the time series that consists solely of
    back‑filled data.  The first valid observation in the price series is
    interpreted as the first traded price of the asset.  A further
    `lookback_long` periods are discarded to ensure that rolling window
    features (which depend on historic prices) are fully realised before
    the start of training.  This prevents inadvertently training on
    artificial values created by back‑filling.

    Args:
        prices: Univariate price series for a single asset.  The index
            should be of a datetime type.
        features: DataFrame of technical features aligned with `prices`.
        asset_name: Name of the asset (for logging only).
        lookback_long: The length of the longest rolling lookback used in
            feature engineering.  This many periods will be skipped after
            the first real observation to avoid including any features
            computed on back‑filled data.

    Returns:
        Tuple of `(trimmed_prices, trimmed_features)` where both are
        restricted to dates on or after the computed valid start date.
    """
    if prices.first_valid_index() is None:
        # No real data – return empty series/dataframe
        return prices.iloc[0:0], features.iloc[0:0]

    first_real_idx = prices.first_valid_index()
    # Add lookback_long periods to ensure features are fully realised
    valid_start = first_real_idx + pd.Timedelta(weeks=lookback_long)

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices index must be a DatetimeIndex for clipping.")

    mask = prices.index >= valid_start
    clipped_prices = prices.loc[mask]
    clipped_features = features.loc[mask]
    return clipped_prices, clipped_features


def fit_normalizer_on_train(
    train_features: pd.DataFrame,
    feature_columns: Optional[list[str]] = None
) -> StandardScaler:
    """
    Fit a `StandardScaler` on the training feature set only.  This function
    never inspects validation or test data, thereby avoiding data leakage.

    Args:
        train_features: DataFrame containing only the training portion of
            the feature matrix.
        feature_columns: Optional list of column names to normalise.  If
            omitted, all columns will be used.

    Returns:
        A fitted `StandardScaler` instance.
    """
    if feature_columns is None:
        feature_columns = train_features.columns.tolist()

    scaler = StandardScaler()
    scaler.fit(train_features[feature_columns])
    return scaler


def normalize_features(
    features: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Apply a previously fitted `StandardScaler` to a feature matrix.

    Args:
        features: DataFrame of features to be normalised.
        scaler: A `StandardScaler` previously returned from
            `fit_normalizer_on_train`.
        feature_columns: Optional list of column names to normalise.  If
            omitted, all columns will be used.

    Returns:
        A new DataFrame where the specified columns have been transformed.
    """
    if feature_columns is None:
        feature_columns = features.columns.tolist()

    normalized = features.copy()
    normalized[feature_columns] = scaler.transform(features[feature_columns])
    return normalized


def prepare_train_val_test_split(
    prices: pd.Series,
    features: pd.DataFrame,
    asset_name: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    lookback_long: int = 52,
    clip_pre_ipo: bool = True
) -> Tuple[
    Tuple[pd.Series, pd.DataFrame],  # train
    Tuple[pd.Series, pd.DataFrame],  # val
    Tuple[pd.Series, pd.DataFrame],  # test
    StandardScaler                    # scaler fitted on train only
]:
    """
    Split a time series into sequential train/validation/test segments and
    normalise the feature matrix using only the training segment.  Optionally
    discard leading back‑filled data based on the longest lookback.

    Args:
        prices: Univariate price series for a single asset.
        features: DataFrame of features aligned with `prices`.
        asset_name: Name of the asset (for logging only).
        train_ratio: Fraction of the data to allocate to the training set.
        val_ratio: Fraction of the data to allocate to the validation set.
        lookback_long: Length of the longest lookback for feature generation.
        clip_pre_ipo: If True, remove back‑filled data before splitting.

    Returns:
        A tuple containing `(train_prices, train_features_norm)`,
        `(val_prices, val_features_norm)`, `(test_prices, test_features_norm)`
        and the fitted scaler.
    """
    # Optionally remove pre‑IPO back‑filled portion
    if clip_pre_ipo:
        prices, features = clip_to_valid_range(
            prices, features, asset_name, lookback_long
        )

    n_total = len(prices)
    if n_total == 0:
        raise ValueError(f"No data available for asset {asset_name} after clipping.")

    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_prices = prices.iloc[:n_train]
    val_prices = prices.iloc[n_train:n_train + n_val]
    test_prices = prices.iloc[n_train + n_val:]

    train_features = features.iloc[:n_train]
    val_features = features.iloc[n_train:n_train + n_val]
    test_features = features.iloc[n_train + n_val:]

    # Fit normaliser on training data only
    scaler = fit_normalizer_on_train(train_features)

    # Apply transformation to all splits
    train_features_norm = normalize_features(train_features, scaler)
    val_features_norm = normalize_features(val_features, scaler)
    test_features_norm = normalize_features(test_features, scaler)

    return (
        (train_prices, train_features_norm),
        (val_prices, val_features_norm),
        (test_prices, test_features_norm),
        scaler,
    )


def prepare_walkforward_window(
    prices: pd.Series,
    features: pd.DataFrame,
    asset_name: str,
    train_start: int,
    train_end: int,
    val_end: int,
    test_start: int,
    test_end: int,
    lookback_long: int = 52,
    clip_pre_ipo: bool = True
) -> Tuple[
    Tuple[pd.Series, pd.DataFrame],
    Tuple[pd.Series, pd.DataFrame],
    Tuple[pd.Series, pd.DataFrame],
    StandardScaler
]:
    """
    Extract a single walk‑forward window from the full time series and
    normalise the features within that window using only the training
    portion.  The indices `train_start`, `train_end`, `val_end`, `test_start`
    and `test_end` are integer offsets into the original series.

    Args:
        prices: Full univariate price series.
        features: Full feature matrix aligned with `prices`.
        asset_name: Name of the asset.
        train_start: Inclusive index of training window start.
        train_end: Exclusive index of training window end.
        val_end: Exclusive index of validation window end.
        test_start: Inclusive index of test window start.
        test_end: Exclusive index of test window end.
        lookback_long: Longest lookback used to compute features.
        clip_pre_ipo: If True, warn if the training window overlaps the
            back‑filled region; the clipping should normally have been
            performed ahead of time.

    Returns:
        `(train_prices, train_features_norm)`, `(val_prices, val_features_norm)`,
        `(test_prices, test_features_norm)`, and the fitted scaler.
    """
    # Slice the windows
    train_prices = prices.iloc[train_start:train_end]
    val_prices = prices.iloc[train_end:val_end]
    test_prices = prices.iloc[test_start:test_end]

    train_features = features.iloc[train_start:train_end]
    val_features = features.iloc[train_end:val_end]
    test_features = features.iloc[test_start:test_end]

    # Optionally warn if training window starts before the valid range
    if clip_pre_ipo:
        first_real_idx = prices.first_valid_index()
        if first_real_idx is not None:
            valid_start = first_real_idx + pd.Timedelta(weeks=lookback_long)
            window_start_date = train_prices.index[0]
            if window_start_date < valid_start:
                warnings.warn(
                    f"Training window for {asset_name} begins before valid start {valid_start.date()}. "
                    f"This may include back‑filled data."
                )

    scaler = fit_normalizer_on_train(train_features)

    train_features_norm = normalize_features(train_features, scaler)
    val_features_norm = normalize_features(val_features, scaler)
    test_features_norm = normalize_features(test_features, scaler)

    return (
        (train_prices, train_features_norm),
        (val_prices, val_features_norm),
        (test_prices, test_features_norm),
        scaler,
    )


def validate_no_leakage(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> None:
    """
    Perform a heuristic check for data leakage between training and test
    feature sets by comparing their aggregate statistics.  After proper
    normalisation on training data only, the test features should have a
    mean close to zero but not identically zero.

    Args:
        train_features: Normalised training feature matrix.
        test_features: Normalised test feature matrix.

    Raises:
        Warning if the test mean is suspiciously close to zero, which may
        indicate that the scaler was fitted on combined data.
    """
    train_mean = train_features.mean().mean()
    train_std = train_features.std().mean()
    test_mean = test_features.mean().mean()
    test_std = test_features.std().mean()

    if abs(test_mean) < 0.01:
        warnings.warn(
            "Test features have mean very close to 0. "
            "This might indicate normalisation on combined data (LEAKAGE!)."
        )
    # Optionally log statistics
    print(
        f"Leakage check: train mean={train_mean:.4f}, train std={train_std:.4f}, "
        f"test mean={test_mean:.4f}, test std={test_std:.4f}"
    )


def load_or_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load data from files or create it if missing.
    Uses local imports to avoid circular dependencies.
    """
    price_file = config.DATA_CONFIG['price_file']
    features_file = config.DATA_CONFIG['features_file']

    if os.path.exists(price_file) and os.path.exists(features_file):
        prices = pd.read_csv(price_file, index_col=0, parse_dates=True)
        features = pd.read_csv(features_file, index_col=0, parse_dates=True)
        return prices, features

    # If files are missing, run the pipeline (local import to avoid circular dependency)
    from load_and_create_tech_features import (
        collect_price_data,
        calculate_technical_features,
        save_data,
    )
    
    start_date = config.DATA_CONFIG['start_date']
    end_date = config.DATA_CONFIG['end_date']
    interval = config.DATA_CONFIG['interval']

    lookbacks = {
        'short': config.DATA_CONFIG['lookback_short'],
        'medium': config.DATA_CONFIG['lookback_medium'],
        'long': config.DATA_CONFIG['lookback_long'],
    }

    price_data, volume_data, log_returns, failed, coverage_df = collect_price_data(
        config.PORTFOLIO, start_date, end_date, interval
    )
    features_df = calculate_technical_features(
        price_data, volume_data, log_returns, lookbacks
    )
    save_data(price_data, features_df, config.PORTFOLIO, failed, coverage_df)
    return price_data, features_df

def analyze_trade_frequency(env) -> dict:
    """
    Analyze trade frequency and patterns from a completed environment episode.
    
    Args:
        env: TechnicalEnv after completing an episode
        
    Returns:
        dict with trade frequency metrics and recommendations
    """
    n_trades = len(env.trade_history)
    n_periods = len(env.portfolio_history) - 1
    
    if n_periods == 0:
        return {'error': 'No periods to analyze'}
    
    turnover_rate = (n_trades / n_periods) * 100
    avg_holding = n_periods / max(n_trades, 1)
    
    # Calculate trade sizes
    trade_sizes = [abs(t['position_change']) for t in env.trade_history]
    avg_trade_size = np.mean(trade_sizes) if trade_sizes else 0
    
    # Estimate transaction cost impact
    total_costs = sum(t['cost'] for t in env.trade_history)
    total_return = (env.portfolio_history[-1] - env.portfolio_history[0]) / env.portfolio_history[0]
    cost_impact = (total_costs / env.portfolio_history[0]) / max(abs(total_return), 0.001)
    
    # Recommendations
    recommendations = []
    if turnover_rate > 30:
        recommendations.append("⚠️ High turnover: Increase composite_turnover_weight to 1.0-2.0")
        recommendations.append("⚠️ Consider increasing transaction_cost to 0.004-0.005")
    elif turnover_rate > 20:
        recommendations.append("⚠️ Moderate-high turnover: Consider composite_turnover_weight 0.7-1.0")
    elif turnover_rate < 5:
        recommendations.append("⚠️ Low activity: Model may be undertrading")
        recommendations.append("⚠️ Consider reducing composite_turnover_weight to 0.1-0.2")
    else:
        recommendations.append("✓ Trade frequency is reasonable")
    
    if cost_impact > 0.15:
        recommendations.append(f"⚠️ Transaction costs eating {cost_impact*100:.1f}% of returns")
    
    return {
        'n_trades': n_trades,
        'n_periods': n_periods,
        'turnover_rate': turnover_rate,
        'avg_holding_periods': avg_holding,
        'avg_trade_size': avg_trade_size,
        'total_transaction_costs': total_costs,
        'cost_impact_pct': cost_impact * 100,
        'recommendations': recommendations
    }


def print_trade_analysis(env):
    """Print formatted trade frequency analysis"""
    analysis = analyze_trade_frequency(env)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print("\n" + "="*80)
    print("TRADE FREQUENCY ANALYSIS")
    print("="*80)
    print(f"\nTrading Activity:")
    print(f"  Total trades:        {analysis['n_trades']}")
    print(f"  Total periods:       {analysis['n_periods']}")
    print(f"  Turnover rate:       {analysis['turnover_rate']:.1f}%")
    print(f"  Avg holding:         {analysis['avg_holding_periods']:.1f} periods")
    print(f"  Avg trade size:      {analysis['avg_trade_size']:.2%}")
    
    print(f"\nCost Analysis:")
    print(f"  Total costs:         ${analysis['total_transaction_costs']:.2f}")
    print(f"  Cost impact:         {analysis['cost_impact_pct']:.1f}% of returns")
    
    print(f"\nRecommendations:")
    for rec in analysis['recommendations']:
        print(f"  {rec}")
    print("="*80 + "\n")


def run_single_asset_demo(asset_name: str) -> None:
    """
    Complete demo workflow for a single asset.
    Uses local imports to avoid circular dependencies.
    """
    # Local imports to avoid circular dependencies
    from tech_env_module import TechnicalEnv
    from walkforward_module import walk_forward_validation
    
    prices, features = load_or_prepare_data()

    if asset_name not in prices.columns:
        raise ValueError(f"Asset {asset_name} not found in price data.")

    # Extract univariate price series and corresponding feature subset
    price_series = prices[asset_name]
    asset_feature_cols = [c for c in features.columns if c.startswith(f"{asset_name}_")]
    asset_features = features[asset_feature_cols]

    # Remove back‑filled history and split into train/val/test
    (train_p, train_f), (val_p, val_f), (test_p, test_f), scaler = prepare_train_val_test_split(
        price_series,
        asset_features,
        asset_name,
        train_ratio=0.70,
        val_ratio=0.15,
        lookback_long=config.DATA_CONFIG['lookback_long'],
        clip_pre_ipo=True,
    )

    # Build environments; features are already normalised
    train_env = TechnicalEnv(train_p, train_f, **config.ENV_CONFIG)
    val_env = TechnicalEnv(val_p, val_f, **config.ENV_CONFIG)
    test_env = TechnicalEnv(test_p, test_f, **config.ENV_CONFIG)

    # Perform walk‑forward validation on the full series using configured window lengths
    wf_results = walk_forward_validation(
        price_series,
        asset_features,
        asset_name,
        config.ENV_CONFIG,
        config.MODEL_CONFIG,
        config.WALKFORWARD_CONFIG,
        algorithm=config.MODEL_CONFIG['algorithm'],
        clip_pre_ipo=True,
    )

    print("\nWalk‑forward evaluation results:")
    print(json.dumps({k: v if not isinstance(v, pd.DataFrame) else v.to_dict(orient='list')
                      for k, v in wf_results.items()}, indent=2, default=str))

