"""
Walk-forward validation module
Phase 6: Walk-Forward Validation Setup
"""

import pandas as pd
import numpy as np

from harlf import config
from harlf import utils
from . import preprocessing


def walk_forward_split(data, n_splits=None, train_size=None, val_size=None, test_size=None):
    """
    Custom walk-forward validation with expanding window

    Parameters:
    -----------
    data : pd.DataFrame
        Time series data (must have DatetimeIndex)
    n_splits : int
        Maximum number of splits
    train_size : int
        Initial training window size (in days)
    val_size : int
        Validation window size (in days)
    test_size : int
        Test window size / step size (in days)

    Returns:
    --------
    list
        List of fold dictionaries
    """
    # Use config defaults if not specified
    if n_splits is None:
        n_splits = config.N_SPLITS
    if train_size is None:
        train_size = config.INITIAL_TRAIN_SIZE
    if val_size is None:
        val_size = config.VAL_SIZE
    if test_size is None:
        test_size = config.TEST_SIZE

    config.log("="*60)
    config.log("PHASE 6: WALK-FORWARD VALIDATION SETUP")
    config.log("="*60)
    config.log(f"Total data points: {len(data)}")
    config.log(f"Initial training size: {train_size} days")
    config.log(f"Validation size: {val_size} days")
    config.log(f"Test size: {test_size} days")
    config.log(f"Max splits: {n_splits}")

    folds = []
    start_idx = 0

    for i in range(n_splits):
        # Calculate indices
        train_end = start_idx + train_size
        val_end = train_end + val_size
        test_end = val_end + test_size

        # Check if we have enough data
        if test_end > len(data):
            config.log(f"Stopping at fold {i}: insufficient data")
            break

        # Get date ranges
        train_idx = data.index[start_idx:train_end]
        val_idx = data.index[train_end:val_end]
        test_idx = data.index[val_end:test_end]

        fold = {
            'fold_id': i,
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'train_start': train_idx[0],
            'train_end': train_idx[-1],
            'val_start': val_idx[0],
            'val_end': val_idx[-1],
            'test_start': test_idx[0],
            'test_end': test_idx[-1],
        }

        folds.append(fold)

        # Move forward by test_size (rolling window)
        start_idx += test_size

    config.log(f"\nCreated {len(folds)} walk-forward folds")

    # Print summary
    if len(folds) > 0:
        config.log(f"\nFold summary:")
        config.log(f"  First fold: {folds[0]['train_start']} to {folds[0]['test_end']}")
        config.log(f"  Last fold: {folds[-1]['train_start']} to {folds[-1]['test_end']}")

    return folds


def save_fold_indices(folds, filepath):
    """
    Save fold indices to CSV

    Parameters:
    -----------
    folds : list
        List of fold dictionaries
    filepath : str or Path
        Output file path
    """
    fold_data = []

    for fold in folds:
        fold_data.append({
            'fold_id': fold['fold_id'],
            'train_start': fold['train_start'],
            'train_end': fold['train_end'],
            'val_start': fold['val_start'],
            'val_end': fold['val_end'],
            'test_start': fold['test_start'],
            'test_end': fold['test_end'],
            'train_size': len(fold['train']),
            'val_size': len(fold['val']),
            'test_size': len(fold['test']),
        })

    fold_df = pd.DataFrame(fold_data)
    utils.save_dataframe(fold_df, filepath, index=False)

    config.log(f"Saved fold indices: {filepath}")


def load_fold_indices(filepath):
    """
    Load fold indices from CSV

    Parameters:
    -----------
    filepath : str or Path
        Input file path

    Returns:
    --------
    pd.DataFrame
        Fold indices
    """
    fold_df = utils.load_dataframe(filepath, index_col=None)
    return fold_df


def save_fold_data(fold_id, train_df, val_df, test_df):
    """
    Save train/val/test data for a specific fold
    Splits data into features and targets to prevent normalization issues

    Features (normalized):
        - Technical indicators, volume, macro, etc. (for agent observation)

    Targets (raw, NOT normalized):
        - forward_log_return, forward_return, close_price (for environment rewards)

    Parameters:
    -----------
    fold_id : int
        Fold ID
    train_df : pd.DataFrame
        Training data
    val_df : pd.DataFrame
        Validation data
    test_df : pd.DataFrame
        Test data
    """
    fold_files = config.get_fold_files(fold_id)

    # Identify target columns that should NOT be normalized
    target_cols = [col for col in config.TARGET_COLUMNS if col in train_df.columns]

    # Also preserve ticker column if present
    meta_cols = ['ticker'] if 'ticker' in train_df.columns else []
    all_preserve_cols = target_cols + meta_cols

    # Split into features and targets for each dataset
    for split_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        if len(target_cols) > 0:
            # Extract target columns
            targets = df[all_preserve_cols].copy()

            # Extract feature columns (everything except targets)
            features = df.drop(columns=target_cols).copy()

            # Save features and targets separately
            utils.save_dataframe(features, fold_files[f'{split_name}_features'])
            utils.save_dataframe(targets, fold_files[f'{split_name}_targets'])

            config.log(f"  {split_name.capitalize()}: {features.shape[1]} features + {len(target_cols)} targets")
        else:
            # No target columns - save as features only
            utils.save_dataframe(df, fold_files[f'{split_name}_features'])
            config.log(f"  {split_name.capitalize()}: {df.shape[1]} columns (no targets)")

        # Also save combined file for backward compatibility
        utils.save_dataframe(df, fold_files[split_name])

    config.log(f"Saved fold {fold_id} data (split into features and targets)")


def load_fold_data(fold_id, separate_files=True):
    """
    Load train/val/test data for a specific fold

    Parameters:
    -----------
    fold_id : int
        Fold ID
    separate_files : bool
        If True, load from separate features/targets files and merge
        If False, load from combined legacy files

    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
        OR
        ((train_features, train_targets), (val_features, val_targets), (test_features, test_targets))
        if separate_files=True and return_separate=True
    """
    fold_files = config.get_fold_files(fold_id)

    if separate_files:
        # Load features and targets separately, then merge
        train_features = utils.load_dataframe(fold_files['train_features'])
        train_targets = utils.load_dataframe(fold_files['train_targets'])
        train_df = pd.concat([train_features, train_targets], axis=1)

        val_features = utils.load_dataframe(fold_files['val_features'])
        val_targets = utils.load_dataframe(fold_files['val_targets'])
        val_df = pd.concat([val_features, val_targets], axis=1)

        test_features = utils.load_dataframe(fold_files['test_features'])
        test_targets = utils.load_dataframe(fold_files['test_targets'])
        test_df = pd.concat([test_features, test_targets], axis=1)
    else:
        # Load from combined legacy files
        train_df = utils.load_dataframe(fold_files['train'])
        val_df = utils.load_dataframe(fold_files['val'])
        test_df = utils.load_dataframe(fold_files['test'])

    return train_df, val_df, test_df


def load_fold_data_separate(fold_id):
    """
    Load train/val/test data as separate features and targets

    Parameters:
    -----------
    fold_id : int
        Fold ID

    Returns:
    --------
    tuple of tuples
        ((train_features, train_targets),
         (val_features, val_targets),
         (test_features, test_targets))
    """
    fold_files = config.get_fold_files(fold_id)

    train_features = utils.load_dataframe(fold_files['train_features'])
    train_targets = utils.load_dataframe(fold_files['train_targets'])

    val_features = utils.load_dataframe(fold_files['val_features'])
    val_targets = utils.load_dataframe(fold_files['val_targets'])

    test_features = utils.load_dataframe(fold_files['test_features'])
    test_targets = utils.load_dataframe(fold_files['test_targets'])

    return ((train_features, train_targets),
            (val_features, val_targets),
            (test_features, test_targets))


def prepare_fold(fold_id, fold, features_df, preprocess=True):
    """
    Prepare data for a specific fold with preprocessing

    Parameters:
    -----------
    fold_id : int
        Fold ID
    fold : dict
        Fold information
    features_df : pd.DataFrame
        All features
    preprocess : bool
        Whether to apply preprocessing

    Returns:
    --------
    tuple
        (train, val, test, preprocessing_params, scaler)
    """
    config.log(f"\n{'='*60}")
    config.log(f"PREPARING FOLD {fold_id}")
    config.log(f"{'='*60}")
    config.log(f"Train: {fold['train_start']} to {fold['train_end']} ({len(fold['train'])} days)")
    config.log(f"Val:   {fold['val_start']} to {fold['val_end']} ({len(fold['val'])} days)")
    config.log(f"Test:  {fold['test_start']} to {fold['test_end']} ({len(fold['test'])} days)")

    # Extract data for this fold
    train_df = features_df.loc[fold['train']].copy()
    val_df = features_df.loc[fold['val']].copy()
    test_df = features_df.loc[fold['test']].copy()

    config.log(f"\nRaw data shapes:")
    config.log(f"  Train: {train_df.shape}")
    config.log(f"  Val:   {val_df.shape}")
    config.log(f"  Test:  {test_df.shape}")

    if preprocess:
        # Apply preprocessing
        train_processed, val_processed, test_processed, params, scaler = preprocessing.preprocess_fold(
            train_df, val_df, test_df
        )

        config.log(f"\nProcessed data shapes:")
        config.log(f"  Train: {train_processed.shape}")
        config.log(f"  Val:   {val_processed.shape}")
        config.log(f"  Test:  {test_processed.shape}")

        return train_processed, val_processed, test_processed, params, scaler
    else:
        return train_df, val_df, test_df, None, None


def process_all_folds(features_df, max_folds=None, save_data=True):
    """
    Process all folds with GLOBAL normalization (prevents data leakage)

    ✅ CRITICAL: Uses ONE global scaler fitted on fold 0 training data for ALL folds

    Parameters:
    -----------
    features_df : pd.DataFrame
        All features (ALREADY LAGGED in feature engineering)
    max_folds : int
        Maximum number of folds to process (None = all)
    save_data : bool
        Whether to save fold data to files

    Returns:
    --------
    list
        List of folds with metadata
    dict
        Preprocessing parameters for all folds
    StandardScaler
        Global scaler
    list
        Feature column names
    """
    config.log("\n" + "="*60)
    config.log("PROCESSING ALL WALK-FORWARD FOLDS WITH GLOBAL SCALER")
    config.log("="*60)

    # Create walk-forward splits
    folds = walk_forward_split(features_df)

    # Limit folds if specified
    if max_folds is not None:
        folds = folds[:max_folds]
        config.log(f"Limited to {max_folds} folds")

    # ✅ CRITICAL: Create GLOBAL scaler from fold 0 training data ONLY
    fold_0_train_indices = folds[0]['train']
    global_scaler, feature_cols = preprocessing.create_global_scaler(
        features_df, fold_0_train_indices
    )

    # Save global scaler
    import pickle
    global_scaler_path = config.WALK_FORWARD_DIR / 'global_scaler.pkl'
    with open(global_scaler_path, 'wb') as f:
        pickle.dump(global_scaler, f)
    config.log(f"Saved global scaler: {global_scaler_path}")

    # Save fold indices
    save_fold_indices(folds, config.METADATA_FILES['fold_indices'])

    # ✅ Process each fold using GLOBAL scaler
    all_preprocessing_params = {}

    for fold in folds:
        fold_id = fold['fold_id']

        config.log(f"\n{'='*60}")
        config.log(f"PROCESSING FOLD {fold_id}")
        config.log(f"{'='*60}")
        config.log(f"Train: {fold['train_start']} to {fold['train_end']} ({len(fold['train'])} days)")
        config.log(f"Val:   {fold['val_start']} to {fold['val_end']} ({len(fold['val'])} days)")
        config.log(f"Test:  {fold['test_start']} to {fold['test_end']} ({len(fold['test'])} days)")

        # Extract fold data
        train_df = features_df.loc[fold['train']].copy()
        val_df = features_df.loc[fold['val']].copy()
        test_df = features_df.loc[fold['test']].copy()

        config.log(f"\nRaw data shapes:")
        config.log(f"  Train: {train_df.shape}")
        config.log(f"  Val:   {val_df.shape}")
        config.log(f"  Test:  {test_df.shape}")

        # ✅ Preprocess using GLOBAL scaler (not per-fold scaler)
        train, val, test, winsorization_bounds = preprocessing.preprocess_fold_with_global_scaler(
            train_df, val_df, test_df,
            global_scaler, feature_cols,
            winsorize=True
        )

        # Store preprocessing params
        all_preprocessing_params[f'fold_{fold_id}'] = {
            'winsorization': winsorization_bounds,
            'global_scaler_path': str(global_scaler_path),
            'feature_cols': feature_cols,
        }

        # Save fold data
        if save_data:
            save_fold_data(fold_id, train, val, test)

    # Save preprocessing parameters
    preprocessing.save_preprocessing_params(
        all_preprocessing_params,
        config.METADATA_FILES['preprocessing_params']
    )

    config.log("\n" + "="*60)
    config.log(f"PROCESSED {len(folds)} FOLDS WITH GLOBAL NORMALIZATION")
    config.log("="*60)

    return folds, all_preprocessing_params, global_scaler, feature_cols


def get_fold_summary(folds):
    """
    Generate summary statistics for folds

    Parameters:
    -----------
    folds : list
        List of fold dictionaries

    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    summary = pd.DataFrame([
        {
            'fold_id': f['fold_id'],
            'train_size': len(f['train']),
            'val_size': len(f['val']),
            'test_size': len(f['test']),
            'train_start': f['train_start'],
            'test_end': f['test_end'],
            'total_days': len(f['train']) + len(f['val']) + len(f['test']),
        }
        for f in folds
    ])

    return summary


if __name__ == "__main__":
    config.log("Validation module loaded")
    config.log("Use walk_forward_split() to create folds")
    config.log("Use process_all_folds() to process and save all folds")
