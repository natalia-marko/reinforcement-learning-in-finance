"""
Main pipeline script to run complete data preparation
Executes all phases: Data Download -> Feature Engineering -> Walk-Forward Validation -> Preprocessing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

from harlf.config import defaults as config
from harlf.utils import io as utils
from harlf.data_pipeline import data_download
from harlf.data_pipeline import feature_engineering
from harlf.data_pipeline import validation
from harlf.data_pipeline import preprocessing


def run_full_pipeline(max_folds=None, skip_download=False):
    """
    Run complete data preparation pipeline

    Parameters:
    -----------
    max_folds : int
        Maximum number of folds to process (None = all)
    skip_download : bool
        Skip data download if raw data already exists
    """
    start_time = time.time()

    config.log("\n" + "="*80)
    config.log(" REINFORCEMENT LEARNING PORTFOLIO OPTIMIZATION")
    config.log(" DATA PREPARATION PIPELINE")
    config.log("="*80)

    # ========================================================================
    # PHASE 1: Check project structure
    # ========================================================================
    config.log("\n" + "="*80)
    config.log("PHASE 1: PROJECT STRUCTURE")
    config.log("="*80)

    for directory in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR,
                      config.WALK_FORWARD_DIR, config.METADATA_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        config.log(f"✓ {directory}")

    # ========================================================================
    # PHASE 2: Data Collection
    # ========================================================================
    if not skip_download or not config.RAW_STOCK_FILE.exists():
        config.log("\n" + "="*80)
        config.log("PHASE 2: DATA COLLECTION")
        config.log("="*80)

        # Download and process data
        aligned_data = data_download.main()
    else:
        config.log("\n" + "="*80)
        config.log("PHASE 2: LOADING EXISTING RAW DATA")
        config.log("="*80)

        # Load existing raw data
        raw_data = data_download.load_raw_data()

    # ========================================================================
    # PHASE 3: Feature Engineering
    # ========================================================================
    config.log("\n" + "="*80)
    config.log("PHASE 3: FEATURE ENGINEERING")
    config.log("="*80)

    # Load raw data
    raw_data = {
        'stock': utils.load_dataframe(config.RAW_STOCK_FILE),
        'benchmark': utils.load_dataframe(config.RAW_BENCHMARK_FILE),
        'macro': utils.load_dataframe(config.RAW_MACRO_FILE),
        'vix': utils.load_dataframe(config.RAW_VIX_FILE),
    }

    # Engineer features for all tickers
    features_df, clustering_models = feature_engineering.engineer_all_features(
        raw_data,
        fit_clustering=True
    )

    # Save full features
    utils.save_dataframe(features_df, config.PROCESSED_FEATURES_FILE)

    # Save feature definitions
    feature_defs = utils.get_feature_definitions()
    utils.save_metadata(feature_defs, config.METADATA_FILES['feature_definitions'])

    # Print feature summary
    config.log(f"\nFeature engineering complete:")
    config.log(f"  Total rows: {len(features_df)}")
    config.log(f"  Features per ticker: {len(features_df.columns) - 1}")  # -1 for ticker column
    config.log(f"  Tickers: {features_df['ticker'].unique()}")

    # Check for NaN values
    utils.check_nan_values(features_df)

    # ========================================================================
    # PHASE 4: Feature Validation (Correlation & VIF)
    # ========================================================================
    config.log("\n" + "="*80)
    config.log("PHASE 4: FEATURE VALIDATION")
    config.log("="*80)

    # Get one ticker's features for analysis
    sample_ticker = config.TICKERS[0]
    sample_features = features_df[features_df['ticker'] == sample_ticker].drop('ticker', axis=1)

    # Remove any remaining NaN
    sample_features = sample_features.dropna()

    # Check correlations
    high_corr = utils.check_correlation(sample_features, max_correlation=config.MAX_CORRELATION)

    if len(high_corr) > 0:
        config.log(f"\nWARNING: Found {len(high_corr)} highly correlated feature pairs:")
        for feat1, feat2, corr in high_corr[:5]:  # Show first 5
            config.log(f"  {feat1} <-> {feat2}: {corr:.3f}")

    # Calculate VIF with RL-optimized exceptions (optional - can be slow)
    config.log("\nCalculating VIF with RL-optimized thresholds (this may take a minute)...")
    try:
        vif_df, features_to_keep = utils.calculate_vif_with_exceptions(sample_features)
        config.log(f"\nVIF analysis complete:")
        config.log(f"  Features kept: {len(features_to_keep)}")
        config.log(f"  Features removed: {len(sample_features.columns) - len(features_to_keep)}")

        # Show top VIF values
        config.log("\nTop 5 VIF values:")
        for idx, row in vif_df.head(5).iterrows():
            config.log(f"  {row['feature']}: {row['VIF']:.2f}")
    except Exception as e:
        config.log(f"VIF calculation failed: {e}")
        config.log("Continuing with all features...")

    # ========================================================================
    # PHASE 6: Walk-Forward Validation & Preprocessing
    # ========================================================================
    config.log("\n" + "="*80)
    config.log("PHASE 6: WALK-FORWARD VALIDATION")
    config.log("="*80)

    # Process all folds (with GLOBAL normalization)
    folds, preprocessing_params, global_scaler, feature_cols = validation.process_all_folds(
        features_df,
        max_folds=max_folds,
        save_data=True
    )

    config.log(f"\n✅ Global scaler created from fold 0 training data")
    config.log(f"✅ Same normalization applied to ALL {len(folds)} folds")

    # Generate fold summary
    fold_summary = validation.get_fold_summary(folds)
    config.log("\nFold Summary:")
    config.log(fold_summary.to_string())

    # ========================================================================
    # SUMMARY
    # ========================================================================
    elapsed_time = time.time() - start_time

    config.log("\n" + "="*80)
    config.log("PIPELINE COMPLETE!")
    config.log("="*80)
    config.log(f"\nTotal time: {elapsed_time/60:.2f} minutes")
    config.log(f"\nGenerated files:")
    config.log(f"  Raw data: {config.RAW_DATA_DIR}")
    config.log(f"  Processed features: {config.PROCESSED_FEATURES_FILE}")
    config.log(f"  Walk-forward folds: {config.WALK_FORWARD_DIR}")
    config.log(f"  Metadata: {config.METADATA_DIR}")
    config.log(f"\nTotal folds: {len(folds)}")
    config.log(f"Features per ticker: {len(features_df.columns) - 1}")
    config.log(f"Tickers: {len(config.TICKERS)}")

    return {
        'features': features_df,
        'folds': folds,
        'preprocessing_params': preprocessing_params,
        'clustering_models': clustering_models,
    }


def quick_test():
    """
    Quick test with limited folds for testing
    """
    config.log("Running QUICK TEST with 5 folds...")
    return run_full_pipeline(max_folds=5, skip_download=False)


def main():
    """
    Main entry point - run full pipeline
    """
    return run_full_pipeline(max_folds=None, skip_download=False)


if __name__ == "__main__":
    # You can run either:
    # result = quick_test()  # Quick test with 5 folds
    result = main()  # Full pipeline with all folds
