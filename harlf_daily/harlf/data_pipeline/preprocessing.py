"""
Preprocessing module for outlier handling and normalization
Phase 5: Outlier Handling
Phase 8: Normalization
"""

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import pickle

from harlf import config
from harlf import utils


def winsorize_features(train_df, val_df=None, test_df=None, limits=None):
    """
    Apply winsorization to features using train percentiles

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame, optional
        Validation data
    test_df : pd.DataFrame, optional
        Test data
    limits : tuple
        Winsorization limits (lower, upper percentiles)

    Returns:
    --------
    pd.DataFrame or tuple
        Winsorized dataframes
    dict
        Winsorization bounds
    """
    if limits is None:
        limits = config.WINSORIZE_LIMITS

    config.log(f"Applying winsorization with limits: {limits}")

    # Calculate bounds from training data
    train_lower = train_df.quantile(limits[0])
    train_upper = train_df.quantile(1 - limits[1])

    # Store bounds
    bounds = {
        'lower': train_lower.to_dict(),
        'upper': train_upper.to_dict(),
    }

    # Apply to train
    train_winsorized = train_df.clip(lower=train_lower, upper=train_upper, axis=1)

    config.log(f"  Winsorized training data: {train_winsorized.shape}")

    # Apply to val/test if provided
    results = [train_winsorized]

    if val_df is not None:
        val_winsorized = val_df.clip(lower=train_lower, upper=train_upper, axis=1)
        results.append(val_winsorized)
        config.log(f"  Winsorized validation data: {val_winsorized.shape}")

    if test_df is not None:
        test_winsorized = test_df.clip(lower=train_lower, upper=train_upper, axis=1)
        results.append(test_winsorized)
        config.log(f"  Winsorized test data: {test_winsorized.shape}")

    if len(results) == 1:
        return train_winsorized, bounds
    else:
        return tuple(results), bounds


def normalize_features(train_df, val_df=None, test_df=None):
    """
    Normalize features using StandardScaler fitted on training data
    IMPORTANT: Excludes target columns (forward_log_return, forward_return, close_price)
    from normalization - these must remain as raw values for the environment.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame, optional
        Validation data
    test_df : pd.DataFrame, optional
        Test data

    Returns:
    --------
    pd.DataFrame or tuple
        Normalized dataframes
    StandardScaler
        Fitted scaler
    """
    config.log("Normalizing features with StandardScaler...")

    # Identify which target columns are present in the dataframe
    target_cols = [col for col in config.TARGET_COLUMNS if col in train_df.columns]

    if target_cols:
        config.log(f"  Excluding target columns from normalization: {target_cols}")

        # Separate target columns from feature columns
        train_targets = train_df[target_cols].copy()
        train_features = train_df.drop(columns=target_cols)

        val_targets = val_df[target_cols].copy() if val_df is not None else None
        val_features = val_df.drop(columns=target_cols) if val_df is not None else None

        test_targets = test_df[target_cols].copy() if test_df is not None else None
        test_features = test_df.drop(columns=target_cols) if test_df is not None else None
    else:
        # No target columns present - normalize everything
        train_features = train_df
        val_features = val_df
        test_features = test_df
        train_targets = None
        val_targets = None
        test_targets = None

    # Fit scaler on training features only
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)

    # Convert back to DataFrame
    train_normalized = pd.DataFrame(
        train_scaled,
        index=train_features.index,
        columns=train_features.columns
    )

    # Add target columns back to training data (unscaled)
    if train_targets is not None:
        for col in target_cols:
            train_normalized[col] = train_targets[col]

    config.log(f"  Normalized training data: {train_normalized.shape}")
    config.log(f"    Features normalized: {len(train_features.columns)}")
    if target_cols:
        config.log(f"    Targets preserved (raw): {len(target_cols)}")

    # Apply to val/test if provided
    results = [train_normalized]

    if val_df is not None:
        val_scaled = scaler.transform(val_features)
        val_normalized = pd.DataFrame(
            val_scaled,
            index=val_features.index,
            columns=val_features.columns
        )
        # Add target columns back (unscaled)
        if val_targets is not None:
            for col in target_cols:
                val_normalized[col] = val_targets[col]
        results.append(val_normalized)
        config.log(f"  Normalized validation data: {val_normalized.shape}")

    if test_df is not None:
        test_scaled = scaler.transform(test_features)
        test_normalized = pd.DataFrame(
            test_scaled,
            index=test_features.index,
            columns=test_features.columns
        )
        # Add target columns back (unscaled)
        if test_targets is not None:
            for col in target_cols:
                test_normalized[col] = test_targets[col]
        results.append(test_normalized)
        config.log(f"  Normalized test data: {test_normalized.shape}")

    if len(results) == 1:
        return train_normalized, scaler
    else:
        return tuple(results), scaler


def create_global_scaler(features_df, fold_0_train_indices):
    """
    Fit scaler on FIRST fold's training data ONLY (prevents data leakage)

    CRITICAL: This scaler will be used for ALL folds to ensure consistent normalization
    across time. Per-fold normalization creates temporal information leakage.

    Parameters:
    -----------
    features_df : pd.DataFrame
        All features (with lagged features and targets)
    fold_0_train_indices : DatetimeIndex
        Indices for fold 0 training data

    Returns:
    --------
    StandardScaler
        Fitted global scaler
    list
        Feature column names
    """
    config.log("="*60)
    config.log("CREATING GLOBAL SCALER (from fold 0 training data)")
    config.log("="*60)

    # Extract fold 0 training data
    train_0 = features_df.loc[fold_0_train_indices].copy()

    # Identify feature columns (exclude targets and metadata)
    all_cols = train_0.columns.tolist()
    target_cols = config.TARGET_COLUMNS + ['ticker']
    feature_cols = [c for c in all_cols if c not in target_cols]

    config.log(f"  Total columns: {len(all_cols)}")
    config.log(f"  Feature columns to normalize: {len(feature_cols)}")
    config.log(f"  Target/meta columns (not normalized): {[c for c in target_cols if c in all_cols]}")

    # Fit scaler on features only
    scaler = StandardScaler()
    scaler.fit(train_0[feature_cols])

    config.log(f"  Scaler fitted on {len(train_0)} samples from fold 0 training data")
    config.log(f"  Feature means range: [{scaler.mean_.min():.3f}, {scaler.mean_.max():.3f}]")
    config.log(f"  Feature stds range: [{scaler.scale_.min():.3f}, {scaler.scale_.max():.3f}]")
    config.log("="*60)

    return scaler, feature_cols


def normalize_with_global_scaler(df, scaler, feature_cols):
    """
    Apply global scaler to features (targets remain raw)

    Parameters:
    -----------
    df : pd.DataFrame
        Data to normalize
    scaler : StandardScaler
        Pre-fitted global scaler
    feature_cols : list
        List of feature column names

    Returns:
    --------
    pd.DataFrame
        Normalized dataframe
    """
    # Separate features from other columns
    features = df[feature_cols].copy()
    other_cols = [c for c in df.columns if c not in feature_cols]

    # Normalize features
    features_scaled = scaler.transform(features)

    # Reconstruct dataframe
    df_normalized = pd.DataFrame(
        features_scaled,
        index=features.index,
        columns=feature_cols
    )

    # Add back targets and metadata (unnormalized)
    for col in other_cols:
        df_normalized[col] = df[col]

    return df_normalized


def handle_nan_values(df, strategy='drop', fill_value=0):
    """
    Handle NaN values in features
    Preserves ticker column if present

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    strategy : str
        Strategy for handling NaN: 'drop', 'fill', 'ffill'
    fill_value : float
        Value to use for filling (if strategy='fill')

    Returns:
    --------
    pd.DataFrame
        DataFrame with NaN values handled
    """
    # Separate ticker column if present
    has_ticker = 'ticker' in df.columns
    if has_ticker:
        ticker_col = df['ticker'].copy()
        features = df.drop('ticker', axis=1)
    else:
        features = df.copy()

    nan_count_before = features.isna().sum().sum()

    if nan_count_before == 0:
        config.log("No NaN values to handle")
        return df

    config.log(f"Handling {nan_count_before} NaN values with strategy: '{strategy}'")

    if strategy == 'drop':
        # Drop rows with any NaN
        features_clean = features.dropna()
        rows_dropped = len(features) - len(features_clean)
        config.log(f"  Dropped {rows_dropped} rows with NaN values")

        # Update ticker column to match
        if has_ticker:
            ticker_col = ticker_col.loc[features_clean.index]

    elif strategy == 'fill':
        # Fill with constant value
        features_clean = features.fillna(fill_value)
        config.log(f"  Filled NaN values with {fill_value}")

    elif strategy == 'ffill':
        # Forward fill
        features_clean = features.ffill()
        # If still NaN at beginning, backfill
        features_clean = features_clean.bfill()
        # If still NaN, fill with 0
        features_clean = features_clean.fillna(0)
        config.log("  Forward filled NaN values")

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Add ticker back if it was present
    if has_ticker:
        features_clean['ticker'] = ticker_col

    nan_count_after = features_clean.drop('ticker', axis=1).isna().sum().sum() if has_ticker else features_clean.isna().sum().sum()
    config.log(f"  Remaining NaN values: {nan_count_after}")

    return features_clean


def add_lag_to_features(df, lag=1):
    """
    Add lag to all features to prevent look-ahead bias
    Handles multi-ticker data by grouping by ticker before lagging

    Parameters:
    -----------
    df : pd.DataFrame
        Features dataframe
    lag : int
        Number of periods to lag

    Returns:
    --------
    pd.DataFrame
        Lagged features
    """
    config.log(f"Adding {lag}-period lag to features...")

    # Check if ticker column exists (multi-ticker data)
    if 'ticker' in df.columns:
        # Group by ticker and apply lag within each group
        lagged_groups = []
        for ticker, group in df.groupby('ticker'):
            # Separate features from ticker
            features = group.drop('ticker', axis=1)
            # Apply lag
            features_lagged = features.shift(lag)
            # Add ticker back
            features_lagged['ticker'] = ticker
            # Drop NaN rows from lagging
            features_lagged = features_lagged.dropna()
            lagged_groups.append(features_lagged)

        # Combine all groups
        df_lagged = pd.concat(lagged_groups, axis=0)
    else:
        # Single ticker case
        df_lagged = df.shift(lag)
        df_lagged = df_lagged.dropna()

    config.log(f"  Lagged features shape: {df_lagged.shape}")

    return df_lagged


def preprocess_fold(train_df, val_df, test_df, add_lag=False, nan_strategy='ffill'):
    """
    Complete preprocessing pipeline for one fold

    ✅ CRITICAL: Lagging is now done in feature_engineering.py BEFORE targets are added
    This function should NO LONGER lag features (add_lag=False by default)

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data (ALREADY LAGGED in feature engineering)
    val_df : pd.DataFrame
        Validation data (ALREADY LAGGED in feature engineering)
    test_df : pd.DataFrame
        Test data (ALREADY LAGGED in feature engineering)
    add_lag : bool
        DEPRECATED - lagging now done in feature_engineering.py (default False)
    nan_strategy : str
        Strategy for handling NaN values

    Returns:
    --------
    tuple
        (train, val, test, preprocessing_params)
    """
    config.log("\nPreprocessing fold...")

    if add_lag:
        config.log("  WARNING: add_lag=True is deprecated. Lagging is now done in feature_engineering.py")

    # Store ticker column if present
    train_ticker = train_df['ticker'] if 'ticker' in train_df.columns else None
    val_ticker = val_df['ticker'] if 'ticker' in val_df.columns else None
    test_ticker = test_df['ticker'] if 'ticker' in test_df.columns else None

    # Handle NaN values
    if 'ticker' in train_df.columns:
        # Keep features and ticker together
        train_df_clean = handle_nan_values(train_df, strategy=nan_strategy)
        val_df_clean = handle_nan_values(val_df, strategy=nan_strategy)
        test_df_clean = handle_nan_values(test_df, strategy=nan_strategy)
    else:
        train_df_clean = handle_nan_values(train_df, strategy=nan_strategy)
        val_df_clean = handle_nan_values(val_df, strategy=nan_strategy)
        test_df_clean = handle_nan_values(test_df, strategy=nan_strategy)

    # ✅ NO LAGGING HERE - already done in feature_engineering.py
    # Remove ticker for preprocessing
    train_df = train_df_clean.drop('ticker', axis=1) if 'ticker' in train_df_clean.columns else train_df_clean
    val_df = val_df_clean.drop('ticker', axis=1) if 'ticker' in val_df_clean.columns else val_df_clean
    test_df = test_df_clean.drop('ticker', axis=1) if 'ticker' in test_df_clean.columns else test_df_clean

    # Winsorization
    (train_winsorized, val_winsorized, test_winsorized), winsorization_bounds = winsorize_features(
        train_df, val_df, test_df
    )

    # Normalization
    (train_normalized, val_normalized, test_normalized), scaler = normalize_features(
        train_winsorized, val_winsorized, test_winsorized
    )

    # Add ticker column back
    if train_ticker is not None:
        train_normalized['ticker'] = train_ticker
        val_normalized['ticker'] = val_ticker
        test_normalized['ticker'] = test_ticker

    # Prepare preprocessing parameters
    preprocessing_params = {
        'winsorization': winsorization_bounds,
        'normalization': {
            'means': dict(zip(scaler.feature_names_in_, scaler.mean_)),
            'stds': dict(zip(scaler.feature_names_in_, scaler.scale_)),
        },
        'scaler_path': None,  # Will be set when saving
    }

    config.log("Preprocessing complete")

    return train_normalized, val_normalized, test_normalized, preprocessing_params, scaler


def preprocess_fold_with_global_scaler(train_df, val_df, test_df,
                                         global_scaler, feature_cols,
                                         winsorize=True, nan_strategy='ffill'):
    """
    Preprocess fold using GLOBAL scaler (NO per-fold fitting)

    ✅ CRITICAL: Uses pre-fitted global scaler to prevent temporal data leakage

    Parameters:
    -----------
    train_df, val_df, test_df : pd.DataFrame
        Data splits (ALREADY LAGGED in feature engineering)
    global_scaler : StandardScaler
        Pre-fitted global scaler from fold 0 training data
    feature_cols : list
        Feature column names
    winsorize : bool
        Whether to apply winsorization
    nan_strategy : str
        Strategy for handling NaN values

    Returns:
    --------
    tuple
        (train_normalized, val_normalized, test_normalized, winsorization_bounds)
    """
    config.log("\nPreprocessing fold with GLOBAL scaler...")

    # Store ticker column if present
    train_ticker = train_df['ticker'] if 'ticker' in train_df.columns else None
    val_ticker = val_df['ticker'] if 'ticker' in val_df.columns else None
    test_ticker = test_df['ticker'] if 'ticker' in test_df.columns else None

    # Handle NaN values
    train_df = handle_nan_values(train_df, strategy=nan_strategy)
    val_df = handle_nan_values(val_df, strategy=nan_strategy)
    test_df = handle_nan_values(test_df, strategy=nan_strategy)

    # Remove ticker for preprocessing
    train_df_no_ticker = train_df.drop('ticker', axis=1) if 'ticker' in train_df.columns else train_df
    val_df_no_ticker = val_df.drop('ticker', axis=1) if 'ticker' in val_df.columns else val_df
    test_df_no_ticker = test_df.drop('ticker', axis=1) if 'ticker' in test_df.columns else test_df

    # Separate features from targets
    train_features = train_df_no_ticker[feature_cols]
    val_features = val_df_no_ticker[feature_cols]
    test_features = test_df_no_ticker[feature_cols]

    # Winsorization (using training bounds)
    if winsorize:
        train_lower = train_features.quantile(config.WINSORIZE_LIMITS[0])
        train_upper = train_features.quantile(1 - config.WINSORIZE_LIMITS[1])

        train_features = train_features.clip(lower=train_lower, upper=train_upper, axis=1)
        val_features = val_features.clip(lower=train_lower, upper=train_upper, axis=1)
        test_features = test_features.clip(lower=train_lower, upper=train_upper, axis=1)

        winsorization_bounds = {'lower': train_lower.to_dict(), 'upper': train_upper.to_dict()}
        config.log(f"  Applied winsorization with limits: {config.WINSORIZE_LIMITS}")
    else:
        winsorization_bounds = None

    # ✅ Apply GLOBAL normalization (same scaler for ALL folds)
    train_features_scaled = pd.DataFrame(
        global_scaler.transform(train_features),
        index=train_features.index,
        columns=feature_cols
    )
    val_features_scaled = pd.DataFrame(
        global_scaler.transform(val_features),
        index=val_features.index,
        columns=feature_cols
    )
    test_features_scaled = pd.DataFrame(
        global_scaler.transform(test_features),
        index=test_features.index,
        columns=feature_cols
    )

    # Add back targets (unnormalized) and metadata
    target_cols = [c for c in train_df_no_ticker.columns if c not in feature_cols]
    for col in target_cols:
        train_features_scaled[col] = train_df_no_ticker[col]
        val_features_scaled[col] = val_df_no_ticker[col]
        test_features_scaled[col] = test_df_no_ticker[col]

    # Add ticker back
    if train_ticker is not None:
        train_features_scaled['ticker'] = train_ticker
        val_features_scaled['ticker'] = val_ticker
        test_features_scaled['ticker'] = test_ticker

    config.log(f"  Normalized using GLOBAL scaler (from fold 0)")
    config.log(f"  Train: {train_features_scaled.shape}")
    config.log(f"  Val: {val_features_scaled.shape}")
    config.log(f"  Test: {test_features_scaled.shape}")

    return train_features_scaled, val_features_scaled, test_features_scaled, winsorization_bounds


def save_scaler(scaler, fold_id):
    """
    Save scaler object for a fold

    Parameters:
    -----------
    scaler : StandardScaler
        Fitted scaler
    fold_id : int
        Fold ID

    Returns:
    --------
    str
        Path to saved scaler
    """
    scaler_path = config.get_fold_dir(fold_id) / f"scaler_fold_{fold_id}.pkl"

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    config.log(f"Saved scaler: {scaler_path}")

    return str(scaler_path)


def load_scaler(fold_id):
    """
    Load scaler object for a fold

    Parameters:
    -----------
    fold_id : int
        Fold ID

    Returns:
    --------
    StandardScaler
        Loaded scaler
    """
    scaler_path = config.get_fold_dir(fold_id) / f"scaler_fold_{fold_id}.pkl"

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    config.log(f"Loaded scaler: {scaler_path}")

    return scaler


def save_preprocessing_params(all_params, filepath):
    """
    Save preprocessing parameters for all folds

    Parameters:
    -----------
    all_params : dict
        Dictionary of preprocessing params per fold
    filepath : str or Path
        Output file path
    """
    utils.save_metadata(all_params, filepath)


def load_preprocessing_params(filepath):
    """
    Load preprocessing parameters for all folds

    Parameters:
    -----------
    filepath : str or Path
        Input file path

    Returns:
    --------
    dict
        Preprocessing parameters
    """
    return utils.load_metadata(filepath)


if __name__ == "__main__":
    # Example usage
    config.log("Preprocessing module loaded")
    config.log("Use preprocess_fold() for preprocessing individual folds")
