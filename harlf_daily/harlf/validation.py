"""
Walk-forward validation module
Phase 6: Walk-Forward Validation Setup
"""

import pandas as pd
import numpy as np
import config
import utils
import preprocessing


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

    utils.save_dataframe(train_df, fold_files['train'])
    utils.save_dataframe(val_df, fold_files['val'])
    utils.save_dataframe(test_df, fold_files['test'])

    config.log(f"Saved fold {fold_id} data")


def load_fold_data(fold_id):
    """
    Load train/val/test data for a specific fold

    Parameters:
    -----------
    fold_id : int
        Fold ID

    Returns:
    --------
    tuple
        (train_df, val_df, test_df)
    """
    fold_files = config.get_fold_files(fold_id)

    train_df = utils.load_dataframe(fold_files['train'])
    val_df = utils.load_dataframe(fold_files['val'])
    test_df = utils.load_dataframe(fold_files['test'])

    return train_df, val_df, test_df


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
    Process all folds with feature engineering and preprocessing

    Parameters:
    -----------
    features_df : pd.DataFrame
        All features
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
    """
    config.log("\n" + "="*60)
    config.log("PROCESSING ALL WALK-FORWARD FOLDS")
    config.log("="*60)

    # Create walk-forward splits
    folds = walk_forward_split(features_df)

    # Limit folds if specified
    if max_folds is not None:
        folds = folds[:max_folds]
        config.log(f"Limited to {max_folds} folds")

    # Save fold indices
    save_fold_indices(folds, config.METADATA_FILES['fold_indices'])

    # Process each fold
    all_preprocessing_params = {}

    for fold in folds:
        fold_id = fold['fold_id']

        # Prepare fold with preprocessing
        train, val, test, params, scaler = prepare_fold(
            fold_id, fold, features_df, preprocess=True
        )

        # Save scaler
        if scaler is not None:
            scaler_path = preprocessing.save_scaler(scaler, fold_id)
            params['scaler_path'] = scaler_path

        # Store preprocessing params
        all_preprocessing_params[f'fold_{fold_id}'] = params

        # Save fold data
        if save_data:
            save_fold_data(fold_id, train, val, test)

    # Save preprocessing parameters
    preprocessing.save_preprocessing_params(
        all_preprocessing_params,
        config.METADATA_FILES['preprocessing_params']
    )

    config.log("\n" + "="*60)
    config.log(f"PROCESSED {len(folds)} FOLDS SUCCESSFULLY")
    config.log("="*60)

    return folds, all_preprocessing_params


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
