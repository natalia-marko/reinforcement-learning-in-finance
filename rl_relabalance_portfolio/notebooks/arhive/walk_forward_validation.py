"""
Walking Forward Validation for RL Portfolio Training.
Splits train data chronologically into folds for validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path


def create_walk_forward_folds(
    train_data: pd.DataFrame,
    n_folds: int = 5,
    min_train_size: Optional[int] = None,
    date_col: str = 'date'
) -> List[Dict[str, pd.DataFrame]]:
    """
    Create chronological folds for walking forward validation.

    Proper walk-forward validation ensures no data leakage:
    - Fold 0: train=[0:T0], val=[T0:T1]
    - Fold 1: train=[0:T1], val=[T1:T2]  (expands training, non-overlapping val)
    - Fold 2: train=[0:T2], val=[T2:T3]  (expands training, non-overlapping val)

    Each fold's validation set becomes part of the next fold's training set.

    Parameters:
    -----------
    train_data : pd.DataFrame
        Training data (will be split into folds)
    n_folds : int
        Number of folds to create (default: 5)
    min_train_size : Optional[int]
        Minimum number of periods in training fold (default: None = auto)
    date_col : str
        Date column name (default: 'date')

    Returns:
    --------
    List[Dict[str, pd.DataFrame]]
        List of fold dictionaries, each with 'train' and 'val' keys
    """
    # Get unique dates sorted
    dates = sorted(train_data[date_col].unique())
    n_periods = len(dates)

    # Calculate minimum train size (use 20% of data if not specified)
    if min_train_size is None:
        min_train_size = max(52, int(n_periods * 0.2))  # At least 1 year or 20%

    # Calculate validation size per fold
    # Reserve enough data for all validation folds
    remaining_periods = n_periods - min_train_size
    val_size_per_fold = remaining_periods // n_folds

    # Ensure at least 4 weeks per validation fold
    if val_size_per_fold < 4:
        val_size_per_fold = 4

    # Calculate fold boundaries
    # Each fold uses expanding training window and non-overlapping validation
    folds = []

    for fold_idx in range(n_folds):
        # Calculate split points for this fold
        val_start_idx = min_train_size + (fold_idx * val_size_per_fold)
        val_end_idx = val_start_idx + val_size_per_fold

        # Last fold gets all remaining data
        if fold_idx == n_folds - 1:
            val_end_idx = n_periods

        # Ensure we don't go beyond available data
        if val_start_idx >= n_periods:
            break
        if val_end_idx > n_periods:
            val_end_idx = n_periods

        # Training: from start to validation start (expanding window)
        train_dates = dates[:val_start_idx]
        # Validation: non-overlapping window
        val_dates = dates[val_start_idx:val_end_idx]

        # Skip if validation set is empty
        if len(val_dates) == 0:
            continue

        # Split data
        train_fold = train_data[train_data[date_col].isin(train_dates)].copy()
        val_fold = train_data[train_data[date_col].isin(val_dates)].copy()

        if len(train_fold) > 0 and len(val_fold) > 0:
            folds.append({
                'train': train_fold,
                'val': val_fold,
                'fold_idx': fold_idx,
                'train_start': train_fold[date_col].min(),
                'train_end': train_fold[date_col].max(),
                'val_start': val_fold[date_col].min(),
                'val_end': val_fold[date_col].max(),
                'n_train_periods': len(train_dates),
                'n_val_periods': len(val_dates)
            })

    return folds


def print_fold_summary(folds: List[Dict[str, pd.DataFrame]]):
    """Print summary of created folds."""
    print("=" * 80)
    print("WALKING FORWARD VALIDATION FOLDS")
    print("=" * 80)
    
    for fold in folds:
        print(f"\nFold {fold['fold_idx']}:")
        print(f"  Train: {fold['train_start'].date()} to {fold['train_end'].date()} ({fold['n_train_periods']} periods)")
        print(f"  Val:   {fold['val_start'].date()} to {fold['val_end'].date()} ({fold['n_val_periods']} periods)")
        print(f"  Train samples: {len(fold['train'])}")
        print(f"  Val samples:   {len(fold['val'])}")
    
    print("\n" + "=" * 80)

