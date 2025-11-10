"""
Utility functions for data preparation pipeline
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

from harlf.config import defaults as config


# Relaxed VIF threshold for RL applications
VIF_THRESHOLD = 20  # Changed from 5

# Critical features allowed higher VIF (up to 30)
CRITICAL_FEATURES = ['return_21d', 'rsi_14d']


def save_dataframe(df, filepath, index=True):
    """
    Save DataFrame to CSV with proper formatting

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    index : bool
        Whether to save index
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    config.log(f"Saved: {filepath}")


def load_dataframe(filepath, index_col=0, parse_dates=True):
    """
    Load DataFrame from CSV

    Parameters:
    -----------
    filepath : str or Path
        Input file path
    index_col : int or str
        Column to use as index
    parse_dates : bool
        Whether to parse dates

    Returns:
    --------
    pd.DataFrame
    """
    df = pd.read_csv(filepath, index_col=index_col, parse_dates=parse_dates)
    config.log(f"Loaded: {filepath} - Shape: {df.shape}")
    return df


def calculate_vif(df, max_vif=20.0):
    """
    Calculate Variance Inflation Factor and iteratively remove features with VIF > max_vif

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    max_vif : float
        Maximum acceptable VIF (default: 20.0 for RL applications)

    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values
    list
        List of features to keep
    """
    config.log("Calculating VIF...")
    features_df = df.copy()
    removed_features = []

    while True:
        # Add constant for VIF calculation
        X = add_constant(features_df, has_constant='add')

        # Calculate VIF for each feature
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif.sort_values('VIF', ascending=False)

        # Get max VIF (excluding constant)
        max_vif_value = vif[vif['feature'] != 'const']['VIF'].max()

        if max_vif_value < max_vif or len(features_df.columns) <= 1:
            break

        # Remove feature with highest VIF
        remove_feature = vif[vif['VIF'] == max_vif_value]['feature'].values[0]
        config.log(f"  Removing '{remove_feature}' (VIF: {max_vif_value:.2f})")
        features_df = features_df.drop(remove_feature, axis=1)
        removed_features.append(remove_feature)

    config.log(f"VIF analysis complete. Kept {len(features_df.columns)} features, removed {len(removed_features)}")

    return vif[vif['feature'] != 'const'], list(features_df.columns)


def calculate_vif_with_exceptions(df):
    """
    Calculate VIF with exceptions for critical RL state features.

    RL state representation tolerates more redundancy than regression.
    VIF < 20 threshold based on state-space design principles.

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe

    Returns:
    --------
    pd.DataFrame
        DataFrame with VIF values
    list
        List of features to keep
    """
    config.log("Calculating VIF with exceptions for critical features...")
    features_df = df.copy()
    removed_features = []

    while True:
        # Add constant for VIF calculation
        X = add_constant(features_df, has_constant='add')

        # Calculate VIF for each feature
        vif = pd.DataFrame()
        vif["feature"] = X.columns
        vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif = vif.sort_values('VIF', ascending=False)

        # Get max VIF (excluding constant and critical features)
        non_critical = vif[
            (vif['feature'] != 'const') &
            (~vif['feature'].isin(CRITICAL_FEATURES))
        ]

        if len(non_critical) == 0:
            break

        max_vif_value = non_critical['VIF'].max()

        # Check if we should stop
        if max_vif_value < VIF_THRESHOLD or len(features_df.columns) <= 1:
            break

        # Remove feature with highest VIF (non-critical only)
        remove_feature = non_critical.iloc[0]['feature']
        config.log(f"  Removing '{remove_feature}' (VIF: {max_vif_value:.2f})")
        features_df = features_df.drop(remove_feature, axis=1)
        removed_features.append(remove_feature)

    # Check and report critical features with high VIF
    for feature in CRITICAL_FEATURES:
        if feature in features_df.columns:
            feature_vif = vif[vif['feature'] == feature]['VIF'].values
            if len(feature_vif) > 0:
                vif_value = feature_vif[0]
                if vif_value > VIF_THRESHOLD and vif_value < 30:
                    config.log(f"⚠️ {feature} VIF={vif_value:.2f} exceeds {VIF_THRESHOLD} but kept (critical feature)")
                elif vif_value >= 30:
                    config.log(f"⚠️ WARNING: {feature} VIF={vif_value:.2f} is very high (>30)")

    config.log(f"VIF analysis complete. Kept {len(features_df.columns)} features, removed {len(removed_features)}")

    return vif[vif['feature'] != 'const'], list(features_df.columns)


def check_stationarity(series, name=None, significance_level=0.05):
    """
    Perform Augmented Dickey-Fuller test for stationarity

    Parameters:
    -----------
    series : pd.Series
        Time series to test
    name : str
        Name of the series
    significance_level : float
        Significance level for test

    Returns:
    --------
    dict
        Test results
    """
    # Drop NaN values
    series_clean = series.dropna()

    if len(series_clean) == 0:
        return {
            'name': name or 'Unknown',
            'stationary': False,
            'p_value': 1.0,
            'test_statistic': np.nan,
        }

    # Perform ADF test
    result = adfuller(series_clean, autolag='AIC')

    is_stationary = result[1] < significance_level

    return {
        'name': name or 'Unknown',
        'stationary': is_stationary,
        'p_value': result[1],
        'test_statistic': result[0],
        'critical_values': result[4],
    }


def check_correlation(df, max_correlation=0.90):
    """
    Find highly correlated features

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    max_correlation : float
        Maximum acceptable correlation

    Returns:
    --------
    list
        List of tuples (feature1, feature2, correlation)
    """
    config.log(f"Checking correlations (threshold: {max_correlation})...")

    corr_matrix = df.corr().abs()

    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    high_corr = []
    for column in upper.columns:
        high_corr_features = upper.index[upper[column] > max_correlation].tolist()
        for feature in high_corr_features:
            high_corr.append((column, feature, upper.loc[feature, column]))

    if high_corr:
        config.log(f"  Found {len(high_corr)} highly correlated feature pairs")
    else:
        config.log("  No highly correlated features found")

    return high_corr


def remove_highly_correlated_features(df, max_correlation=0.90):
    """
    Remove one feature from each highly correlated pair

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    max_correlation : float
        Maximum acceptable correlation

    Returns:
    --------
    pd.DataFrame
        DataFrame with reduced features
    list
        List of removed features
    """
    config.log("Removing highly correlated features...")

    features_df = df.copy()
    removed = []

    while True:
        corr_matrix = features_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find first feature with correlation > threshold
        to_drop = None
        max_corr_value = 0

        for column in upper.columns:
            max_col_corr = upper[column].max()
            if max_col_corr > max_correlation and max_col_corr > max_corr_value:
                to_drop = column
                max_corr_value = max_col_corr

        if to_drop is None:
            break

        config.log(f"  Removing '{to_drop}' (max correlation: {max_corr_value:.3f})")
        features_df = features_df.drop(to_drop, axis=1)
        removed.append(to_drop)

    config.log(f"Removed {len(removed)} features due to high correlation")

    return features_df, removed


def check_nan_values(df, threshold=0.05):
    """
    Check for NaN values and report features with high NaN percentage

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    threshold : float
        Threshold for NaN percentage to report

    Returns:
    --------
    pd.DataFrame
        DataFrame with NaN statistics
    """
    nan_stats = pd.DataFrame({
        'nan_count': df.isna().sum(),
        'nan_percentage': df.isna().sum() / len(df),
        'total_rows': len(df),
    })

    nan_stats = nan_stats[nan_stats['nan_count'] > 0].sort_values('nan_percentage', ascending=False)

    if len(nan_stats) > 0:
        config.log(f"Found NaN values in {len(nan_stats)} features:")
        for feature, row in nan_stats.iterrows():
            if row['nan_percentage'] > threshold:
                config.log(f"  WARNING: {feature}: {row['nan_count']} ({row['nan_percentage']:.2%})")
            else:
                config.log(f"  {feature}: {row['nan_count']} ({row['nan_percentage']:.2%})")
    else:
        config.log("No NaN values found")

    return nan_stats


def save_metadata(metadata, filepath):
    """
    Save metadata as JSON

    Parameters:
    -----------
    metadata : dict
        Metadata to save
    filepath : str or Path
        Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    config.log(f"Saved metadata: {filepath}")


def load_metadata(filepath):
    """
    Load metadata from JSON

    Parameters:
    -----------
    filepath : str or Path
        Input file path

    Returns:
    --------
    dict
        Loaded metadata
    """
    with open(filepath, 'r') as f:
        metadata = json.load(f)

    config.log(f"Loaded metadata: {filepath}")
    return metadata


def get_feature_definitions():
    """
    Get feature definitions for all 23 features

    Returns:
    --------
    dict
        Feature definitions
    """
    definitions = {
        # Technical features (8)
        'return_5d': {
            'description': '5-day price momentum (short-term)',
            'calculation': '(close / close.shift(5)) - 1',
            'lag': 1,
            'category': 'momentum'
        },
        'return_21d': {
            'description': '21-day price momentum (medium-term)',
            'calculation': '(close / close.shift(21)) - 1',
            'lag': 1,
            'category': 'momentum'
        },
        'price_to_sma_20d': {
            'description': 'Price relative to 20-day SMA',
            'calculation': 'close / close.rolling(20).mean()',
            'lag': 1,
            'category': 'trend'
        },
        'macd_histogram': {
            'description': 'MACD histogram (12-26-9)',
            'calculation': 'MACD_line - Signal_line',
            'lag': 1,
            'category': 'trend'
        },
        'rsi_14d': {
            'description': '14-day Relative Strength Index',
            'calculation': 'RSI calculation based on price changes',
            'lag': 1,
            'category': 'mean_reversion'
        },
        'bb_position_20d': {
            'description': 'Bollinger Band position',
            'calculation': '(price - lower_band) / (upper_band - lower_band)',
            'lag': 1,
            'category': 'mean_reversion'
        },
        'volatility_21d': {
            'description': '21-day rolling standard deviation of returns',
            'calculation': 'returns.rolling(21).std()',
            'lag': 1,
            'category': 'volatility'
        },
        'atr_pct_14d': {
            'description': '14-day Average True Range as % of price',
            'calculation': 'ATR / close * 100',
            'lag': 1,
            'category': 'volatility'
        },

        # Volume features (2)
        'volume_ratio_20d': {
            'description': 'Current volume relative to 20-day average',
            'calculation': 'volume / volume.rolling(20).mean()',
            'lag': 1,
            'category': 'volume'
        },
        'mfi_14d': {
            'description': '14-day Money Flow Index (volume-weighted RSI)',
            'calculation': 'MFI calculation based on typical price and volume',
            'lag': 1,
            'category': 'volume'
        },

        # Cross-asset features (3)
        'bench_correlation_60d': {
            'description': '60-day rolling correlation with QQQ',
            'calculation': 'returns.rolling(60).corr(qqq_returns)',
            'lag': 1,
            'category': 'cross_asset'
        },
        'bench_beta_60d': {
            'description': '60-day rolling beta vs QQQ',
            'calculation': 'covariance / variance',
            'lag': 1,
            'category': 'cross_asset'
        },
        'bench_relative_strength': {
            'description': '21-day stock return vs QQQ return',
            'calculation': 'stock_return_21d / qqq_return_21d',
            'lag': 1,
            'category': 'cross_asset'
        },

        # Macro features (5)
        'treasury_10y': {
            'description': '10-Year Treasury Yield',
            'calculation': 'FRED DGS10 series',
            'lag': 1,
            'category': 'macro'
        },
        'yield_curve_slope': {
            'description': '10Y - 2Y Treasury spread',
            'calculation': 'DGS10 - DGS2',
            'lag': 1,
            'category': 'macro'
        },
        'vix': {
            'description': 'CBOE Volatility Index',
            'calculation': 'VIX level from Yahoo Finance',
            'lag': 1,
            'category': 'macro'
        },
        'vix_change_21d': {
            'description': '21-day change in VIX',
            'calculation': 'vix - vix.shift(21)',
            'lag': 1,
            'category': 'macro'
        },

        # Feature interactions (3)
        'momentum_volatility_ratio': {
            'description': 'Risk-adjusted momentum',
            'calculation': 'return_21d / volatility_21d',
            'lag': 1,
            'category': 'interaction'
        },
        'volume_weighted_rsi': {
            'description': 'Volume-confirmed RSI',
            'calculation': '(rsi_14d * volume_ratio_20d) / 100',
            'lag': 1,
            'category': 'interaction'
        },
        'vol_regime_indicator': {
            'description': 'Volatility regime indicator',
            'calculation': '1 if volatility_21d > volatility_63d else 0',
            'lag': 1,
            'category': 'interaction'
        },

        # Regime features (2)
        'regime_cluster': {
            'description': 'K-means cluster (0-2) based on volatility and returns',
            'calculation': 'KMeans clustering on [return_21d, volatility_21d, volume_ratio_20d]',
            'lag': 0,
            'category': 'regime'
        },
        'regime_transition': {
            'description': 'Regime change indicator',
            'calculation': '1 if regime changed in last 5 days else 0',
            'lag': 1,
            'category': 'regime'
        },
    }

    return definitions


def print_summary(df, title="DataFrame Summary"):
    """
    Print summary statistics for DataFrame

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to summarize
    title : str
        Title for summary
    """
    config.log(f"\n{'='*60}")
    config.log(f"{title}")
    config.log(f"{'='*60}")
    config.log(f"Shape: {df.shape}")
    config.log(f"Date range: {df.index.min()} to {df.index.max()}")
    config.log(f"Columns: {list(df.columns)}")
    config.log(f"NaN values: {df.isna().sum().sum()}")
    config.log(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    config.log(f"{'='*60}\n")
