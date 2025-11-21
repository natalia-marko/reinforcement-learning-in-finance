"""
Data validation utilities for ensuring data quality and alignment
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def validate_data_alignment(price_data: pd.DataFrame,
                           technical_features: pd.DataFrame,
                           sentiment_features: pd.DataFrame,
                           regime_indicators: Optional[pd.DataFrame] = None,
                           min_periods: int = 60) -> bool:
    """
    Validate all dataframes are aligned and have no critical issues
    
    Args:
        price_data: Price data DataFrame
        technical_features: Technical features DataFrame
        sentiment_features: Sentiment features DataFrame
        regime_indicators: Optional regime indicators DataFrame
        min_periods: Minimum number of periods required (default: 60 for training data)
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If data validation fails with detailed error messages
    """
    errors = []
    warnings = []
    
    # Check index alignment
    common_dates = price_data.index
    if not technical_features.index.equals(common_dates):
        errors.append("Technical features dates don't match price data")
        # Check intersection
        intersection = technical_features.index.intersection(common_dates)
        if len(intersection) < len(common_dates) * 0.9:
            errors.append(f"Only {len(intersection)}/{len(common_dates)} dates overlap")
    
    if not sentiment_features.index.equals(common_dates):
        errors.append("Sentiment features dates don't match price data")
        intersection = sentiment_features.index.intersection(common_dates)
        if len(intersection) < len(common_dates) * 0.9:
            errors.append(f"Only {len(intersection)}/{len(common_dates)} dates overlap")
    
    if regime_indicators is not None:
        if not regime_indicators.index.equals(common_dates):
            warnings.append("Regime indicators dates don't match price data")
    
    # Check for NaN values
    price_nans = price_data.isna().sum().sum()
    if price_nans > 0:
        errors.append(f"Price data contains {price_nans} NaN values")
    
    tech_nans = technical_features.isna().sum().sum()
    if tech_nans > 0:
        warnings.append(f"Technical features contain {tech_nans} NaN values")
    
    sent_nans = sentiment_features.isna().sum().sum()
    if sent_nans > 0:
        warnings.append(f"Sentiment features contain {sent_nans} NaN values")
    
    # Check for sufficient data
    if len(common_dates) < min_periods:
        errors.append(f"Insufficient data: only {len(common_dates)} periods (minimum {min_periods} required)")
    
    # Check for duplicate dates
    if price_data.index.duplicated().any():
        errors.append("Price data contains duplicate dates")
    
    if technical_features.index.duplicated().any():
        errors.append("Technical features contain duplicate dates")
    
    if sentiment_features.index.duplicated().any():
        errors.append("Sentiment features contain duplicate dates")
    
    # Check for negative or zero prices
    if (price_data <= 0).any().any():
        errors.append("Price data contains non-positive values")
    
    # Check column alignment
    if len(price_data.columns) != len(set(price_data.columns)):
        errors.append("Price data contains duplicate column names")
    
    # Report warnings
    if warnings:
        print("Data Validation Warnings:")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    # Report errors
    if errors:
        error_msg = "Data validation failed:\n"
        error_msg += "\n".join(f"  ✗ {e}" for e in errors)
        raise ValueError(error_msg)
    
    print("✓ Data validation passed")
    return True


def validate_training_data(train_data: Tuple, val_data: Tuple, 
                          test_data: Optional[Tuple] = None) -> bool:
    """
    Validate training, validation, and test data splits
    
    Args:
        train_data: Tuple of (prices, technical, sentiment, regime) for training
        val_data: Tuple of validation data
        test_data: Optional tuple of test data
        
    Returns:
        bool: True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    train_prices, train_tech, train_sent, train_regime = train_data
    val_prices, val_tech, val_sent, val_regime = val_data
    
    errors = []
    
    # Validate each split with appropriate minimum periods
    # Training data needs more periods, validation/test can be smaller
    try:
        validate_data_alignment(train_prices, train_tech, train_sent, train_regime, min_periods=60)
    except ValueError as e:
        errors.append(f"Training data: {e}")
    
    try:
        validate_data_alignment(val_prices, val_tech, val_sent, val_regime, min_periods=12)
    except ValueError as e:
        errors.append(f"Validation data: {e}")
    
    # Check for overlap between train and validation
    train_dates = set(train_prices.index)
    val_dates = set(val_prices.index)
    overlap = train_dates.intersection(val_dates)
    if overlap:
        errors.append(f"Train and validation sets overlap: {len(overlap)} dates")
    
    # Check test data if provided
    if test_data is not None:
        test_prices, test_tech, test_sent, test_regime = test_data
        try:
            validate_data_alignment(test_prices, test_tech, test_sent, test_regime, min_periods=6)
        except ValueError as e:
            errors.append(f"Test data: {e}")
        
        test_dates = set(test_prices.index)
        train_test_overlap = train_dates.intersection(test_dates)
        val_test_overlap = val_dates.intersection(test_dates)
        
        if train_test_overlap:
            errors.append(f"Train and test sets overlap: {len(train_test_overlap)} dates")
        if val_test_overlap:
            errors.append(f"Validation and test sets overlap: {len(val_test_overlap)} dates")
    
    if errors:
        error_msg = "Training data validation failed:\n"
        error_msg += "\n".join(f"  ✗ {e}" for e in errors)
        raise ValueError(error_msg)
    
    print("✓ Training data validation passed")
    return True

