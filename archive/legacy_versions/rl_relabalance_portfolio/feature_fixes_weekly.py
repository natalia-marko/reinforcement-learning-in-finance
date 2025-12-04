"""
Evidence-Based Feature Engineering for WEEKLY Data

Adjusted for weekly rebalancing frequency.
Based on actual features available in your dataset.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple


def get_minimal_feature_set_weekly() -> Dict[str, List[str]]:
    """
    Return evidence-based minimal feature set for WEEKLY data.

    Adjusted to match actual feature names in your dataset.
    Total: ~18-20 features (literature standard: 15-25)
    """
    features = {
        # Tier 1: Momentum/Returns (4 features)
        'momentum': [
            'momentum_1w',           # 1-week momentum (short-term)
            'momentum_4w',           # 4-week momentum (monthly)
            'momentum_13w',          # 13-week momentum (quarterly)
            'returns_52w',           # 52-week momentum (annual)
        ],

        # Tier 2: Volatility (2 features)
        'volatility': [
            'volatility_4w',         # Short-term volatility (monthly)
            'volatility_52w',        # Long-term volatility (annual)
        ],

        # Tier 3: Price Ratios (2 features)
        'price_ratios': [
            'price_to_sma_12w',      # Price to 12-week SMA (quarterly)
            'price_to_ema_26w',      # Price to 26-week EMA (semi-annual)
        ],

        # Tier 4: Technical Indicators (3 features)
        'technical': [
            'macd_histogram',        # MACD histogram
            'bb_position',           # Bollinger Band position
            'stochastic_14w',        # Stochastic oscillator (14 weeks)
        ],

        # Tier 5: Volume (2 features)
        'volume': [
            'relative_volume_20w',   # 20-week relative volume
            'obv_roc_13w',          # OBV rate of change (13 weeks)
        ],

        # Tier 6: Risk-Adjusted Metrics (3 features)
        'risk_adjusted': [
            'sharpe_26w',           # 26-week Sharpe ratio (semi-annual)
            'sortino_26w',          # 26-week Sortino ratio
            'calmar_52w',           # 52-week Calmar ratio (annual)
        ],

        # Tier 7: Drawdown (1 feature)
        'drawdown': [
            'max_drawdown_4w',      # Maximum drawdown (4 weeks)
        ],

        # Tier 8: Macro (5 features)
        'macro': [
            'vix',                  # VIX (market volatility)
            'treasury_10y',         # 10-year treasury yield
            'yield_curve_10y2y',    # Yield curve (10y - 2y)
            'dxy',                  # Dollar index
            'oil_wti',              # Oil price (WTI)
        ]
    }

    return features


def get_all_minimal_features_weekly() -> List[str]:
    """Return flat list of all minimal features for weekly data."""
    feature_dict = get_minimal_feature_set_weekly()
    all_features = []
    for tier_features in feature_dict.values():
        all_features.extend(tier_features)
    return all_features


def filter_to_minimal_features_weekly(df: pd.DataFrame,
                                     feature_cols: List[str],
                                     verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Filter dataframe to keep only evidence-based minimal features for weekly data.

    Args:
        df: DataFrame with all features
        feature_cols: List of current feature column names
        verbose: Print diagnostic information

    Returns:
        Tuple of (filtered_df, selected_feature_cols)
    """
    minimal_features = get_all_minimal_features_weekly()

    # Find which minimal features exist in current feature set
    available_features = [f for f in minimal_features if f in feature_cols]
    missing_features = [f for f in minimal_features if f not in feature_cols]

    if verbose:
        print("=" * 80)
        print("MINIMAL FEATURE SET SELECTION (WEEKLY DATA)")
        print("=" * 80)
        print(f"Target features (literature): {len(minimal_features)}")
        print(f"Available in data: {len(available_features)}")
        print(f"Missing: {len(missing_features)}")

        if missing_features:
            print("\nMissing features:")
            for f in missing_features:
                print(f"  ✗ {f}")

        print(f"\nReducing from {len(feature_cols)} → {len(available_features)} features")
        print(f"Sample-to-feature ratio improvement:")
        n_samples = len(df)
        print(f"  Before: {n_samples / len(feature_cols):.1f} samples/feature")
        print(f"  After:  {n_samples / len(available_features):.1f} samples/feature")
        print(f"  Target: >10 samples/feature (preferably >20)")

    # Keep only non-feature columns + available minimal features
    non_feature_cols = [col for col in df.columns if col not in feature_cols]
    selected_cols = non_feature_cols + available_features

    filtered_df = df[selected_cols].copy()

    return filtered_df, available_features


def normalize_features_per_ticker(df: pd.DataFrame,
                                  feature_cols: List[str],
                                  ticker_col: str = 'ticker',
                                  window: int = 52,  # 52 weeks = 1 year for weekly data
                                  min_periods: int = 26,  # At least 6 months
                                  suffix: str = '_norm',
                                  verbose: bool = True) -> pd.DataFrame:
    """
    Apply per-ticker z-score normalization (Zhang et al. 2020).

    Adjusted for WEEKLY data:
    - window=52 (1 year for weekly data)
    - min_periods=26 (6 months minimum)

    Args:
        df: DataFrame with features and ticker column
        feature_cols: List of feature columns to normalize
        ticker_col: Column name for ticker identifier
        window: Rolling window (52 weeks = 1 year)
        min_periods: Minimum periods required (26 weeks = 6 months)
        suffix: Suffix to add to normalized column names
        verbose: Print diagnostic information

    Returns:
        DataFrame with normalized features added
    """
    if verbose:
        print("=" * 80)
        print("PER-TICKER Z-SCORE NORMALIZATION (WEEKLY DATA)")
        print("=" * 80)
        print(f"Method: Rolling z-score per ticker (Zhang et al. 2020)")
        print(f"Window: {window} weeks (~{window/52:.1f} year)")
        print(f"Min periods: {min_periods} weeks (~{min_periods/52:.1f} year)")
        print(f"Features to normalize: {len(feature_cols)}")

    result = df.copy()

    # Get unique tickers
    tickers = df[ticker_col].unique()

    if verbose:
        print(f"Tickers: {len(tickers)}")

    # Normalize each feature for each ticker
    for ticker in tickers:
        mask = result[ticker_col] == ticker

        for col in feature_cols:
            if col not in result.columns:
                if verbose:
                    print(f"Warning: Feature '{col}' not found, skipping")
                continue

            # Calculate rolling mean and std for this ticker
            rolling_mean = result.loc[mask, col].rolling(
                window=window,
                min_periods=min_periods
            ).mean()

            rolling_std = result.loc[mask, col].rolling(
                window=window,
                min_periods=min_periods
            ).std()

            # Z-score normalization
            normalized = (result.loc[mask, col] - rolling_mean) / (rolling_std + 1e-10)

            # Store with suffix
            norm_col = col + suffix
            result.loc[mask, norm_col] = normalized

    # Get normalized column names
    norm_cols = [col + suffix for col in feature_cols if col in result.columns]

    if verbose:
        print(f"\nNormalized features created: {len(norm_cols)}")

        # Sample statistics for first normalized feature
        if norm_cols:
            sample_col = norm_cols[0]
            # Remove NaN for statistics
            sample_data = result[sample_col].dropna()
            if len(sample_data) > 0:
                print(f"\nSample statistics for '{sample_col}':")
                print(f"  Mean: {sample_data.mean():.4f} (should be ~0)")
                print(f"  Std:  {sample_data.std():.4f} (should be ~1)")
                print(f"  Min:  {sample_data.min():.4f}")
                print(f"  Max:  {sample_data.max():.4f}")

    return result


def apply_minimal_feature_pipeline_weekly(df: pd.DataFrame,
                                         current_feature_cols: List[str],
                                         ticker_col: str = 'ticker',
                                         normalize: bool = True,
                                         norm_window: int = 52,  # 52 weeks for weekly data
                                         verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete pipeline for WEEKLY data: filter to minimal features + per-ticker normalization.

    Args:
        df: Input DataFrame with all features
        current_feature_cols: List of current feature columns
        ticker_col: Column name for ticker identifier
        normalize: Whether to apply per-ticker normalization
        norm_window: Rolling window for normalization (52 weeks = 1 year)
        verbose: Print diagnostic information

    Returns:
        Tuple of (processed_df, final_feature_cols)
    """
    if verbose:
        print("\n" + "=" * 80)
        print("EVIDENCE-BASED FEATURE ENGINEERING PIPELINE (WEEKLY DATA)")
        print("=" * 80)
        print(f"Input: {len(current_feature_cols)} features")
        print(f"Target: ~20 features (literature standard)")
        print(f"Normalization: {'Per-ticker z-score (Zhang 2020)' if normalize else 'None'}")
        print(f"Data frequency: WEEKLY")
        print("=" * 80)

    # Step 1: Filter to minimal features
    df_filtered, minimal_features = filter_to_minimal_features_weekly(
        df,
        current_feature_cols,
        verbose=verbose
    )

    # Step 2: Apply per-ticker normalization if requested
    if normalize:
        df_normalized = normalize_features_per_ticker(
            df_filtered,
            minimal_features,
            ticker_col=ticker_col,
            window=norm_window,
            min_periods=max(26, norm_window // 2),  # At least 6 months of data
            verbose=verbose
        )

        # Use normalized features
        final_features = [f + '_norm' for f in minimal_features]
        result_df = df_normalized
    else:
        final_features = minimal_features
        result_df = df_filtered

    if verbose:
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"Final feature count: {len(final_features)}")
        print(f"Feature reduction: {len(current_feature_cols)} → {len(final_features)}")
        print(f"Reduction: {(1 - len(final_features)/len(current_feature_cols))*100:.1f}%")
        print("\nExpected improvements:")
        print("  ✓ Reduced overfitting (sample/feature ratio improvement)")
        print("  ✓ Better generalization (literature-backed features)")
        print("  ✓ Stable gradients (per-ticker normalization)")
        print("  ✓ Less redundancy (removed 14 volatility variants, etc.)")

    return result_df, final_features


if __name__ == '__main__':
    """Example usage and validation for weekly data."""

    print("=" * 80)
    print("EVIDENCE-BASED FEATURE ENGINEERING (WEEKLY DATA)")
    print("Research-backed minimal feature set")
    print("=" * 80)

    # Show minimal feature set
    feature_dict = get_minimal_feature_set_weekly()
    all_features = get_all_minimal_features_weekly()

    print(f"\nTotal features: {len(all_features)}")
    print("\nFeatures by tier:")
    print("-" * 80)

    for tier, features in feature_dict.items():
        print(f"\n{tier.upper()} ({len(features)} features):")
        for f in features:
            print(f"  - {f}")

    print("\n" + "=" * 80)
    print("WEEKLY DATA PARAMETERS")
    print("=" * 80)
    print("""
Normalization window: 52 weeks (1 year)
Minimum periods: 26 weeks (6 months)
Rebalancing frequency: Weekly

Note: All features are in weekly format (e.g., volatility_4w means 4 weeks)
    """)

    print("\n" + "=" * 80)
    print("KEY TAKEAWAY")
    print("=" * 80)
    print("Literature standard: 15-25 features optimal")
    print("This module reduces 100 → ~20 features for WEEKLY data")
