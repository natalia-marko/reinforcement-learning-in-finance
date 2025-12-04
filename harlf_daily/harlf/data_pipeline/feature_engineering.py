"""
Feature engineering module for creating 23 technical, volume, cross-asset, macro, and regime features
Phase 3: Feature Engineering (Reduced Set)
"""

import pandas as pd
import numpy as np
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import MFIIndicator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from harlf.config import defaults as config
from harlf.utils import io as utils


def calculate_momentum_features(close_prices):
    """
    Calculate momentum features

    Parameters:
    -----------
    close_prices : pd.Series or pd.DataFrame
        Closing prices

    Returns:
    --------
    pd.DataFrame
        Momentum features
    """
    features = pd.DataFrame(index=close_prices.index)

    # 1. 5-day momentum
    features['return_5d'] = close_prices.pct_change(periods=config.MOMENTUM_WINDOWS['short'])

    # 2. 21-day momentum
    features['return_21d'] = close_prices.pct_change(periods=config.MOMENTUM_WINDOWS['medium'])

    return features


def calculate_trend_features(close_prices):
    """
    Calculate trend features

    Parameters:
    -----------
    close_prices : pd.Series or pd.DataFrame
        Closing prices

    Returns:
    --------
    pd.DataFrame
        Trend features
    """
    features = pd.DataFrame(index=close_prices.index)

    # 3. Price to SMA ratio
    sma_20 = close_prices.rolling(window=config.TREND_WINDOWS['sma']).mean()
    features['price_to_sma_20d'] = close_prices / sma_20

    # 4. MACD histogram
    macd = MACD(
        close=close_prices,
        window_slow=config.TREND_WINDOWS['macd_slow'],
        window_fast=config.TREND_WINDOWS['macd_fast'],
        window_sign=config.TREND_WINDOWS['macd_signal'],
    )
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    features['macd_histogram'] = macd_line - signal_line

    return features


def calculate_mean_reversion_features(close_prices, high_prices, low_prices):
    """
    Calculate mean reversion features

    Parameters:
    -----------
    close_prices : pd.Series
        Closing prices
    high_prices : pd.Series
        High prices
    low_prices : pd.Series
        Low prices

    Returns:
    --------
    pd.DataFrame
        Mean reversion features
    """
    features = pd.DataFrame(index=close_prices.index)

    # 5. RSI
    rsi = RSIIndicator(close=close_prices, window=config.MEAN_REVERSION_WINDOWS['rsi'])
    features['rsi_14d'] = rsi.rsi()

    # 6. Bollinger Band position
    bb = BollingerBands(
        close=close_prices,
        window=config.MEAN_REVERSION_WINDOWS['bb'],
        window_dev=config.MEAN_REVERSION_WINDOWS['bb_std'],
    )
    upper_band = bb.bollinger_hband()
    lower_band = bb.bollinger_lband()

    # Position within bands: 0 = at lower band, 1 = at upper band
    band_width = upper_band - lower_band
    features['bb_position_20d'] = (close_prices - lower_band) / band_width.replace(0, np.nan)

    return features


def calculate_volatility_features(returns, high_prices, low_prices, close_prices):
    """
    Calculate volatility features

    Parameters:
    -----------
    returns : pd.Series
        Daily returns
    high_prices : pd.Series
        High prices
    low_prices : pd.Series
        Low prices
    close_prices : pd.Series
        Close prices

    Returns:
    --------
    pd.DataFrame
        Volatility features
    """
    features = pd.DataFrame(index=returns.index)

    # 7. Rolling volatility (21-day)
    features['volatility_21d'] = returns.rolling(window=config.VOLATILITY_WINDOWS['short']).std()

    # 8. ATR as percentage of price
    atr = AverageTrueRange(
        high=high_prices,
        low=low_prices,
        close=close_prices,
        window=config.VOLATILITY_WINDOWS['atr'],
    )
    features['atr_pct_14d'] = (atr.average_true_range() / close_prices) * 100

    # Also calculate 63-day volatility for regime indicator (used later)
    features['volatility_63d'] = returns.rolling(window=config.VOLATILITY_WINDOWS['long']).std()

    return features


def calculate_volume_features(volume, high_prices, low_prices, close_prices):
    """
    Calculate volume features

    Parameters:
    -----------
    volume : pd.Series
        Volume data
    high_prices : pd.Series
        High prices
    low_prices : pd.Series
        Low prices
    close_prices : pd.Series
        Close prices

    Returns:
    --------
    pd.DataFrame
        Volume features
    """
    features = pd.DataFrame(index=volume.index)

    # 9. Volume ratio
    volume_ma = volume.rolling(window=config.VOLUME_WINDOWS['ratio']).mean()
    features['volume_ratio_20d'] = volume / volume_ma.replace(0, np.nan)

    # 10. Money Flow Index
    mfi = MFIIndicator(
        high=high_prices,
        low=low_prices,
        close=close_prices,
        volume=volume,
        window=config.VOLUME_WINDOWS['mfi'],
    )
    features['mfi_14d'] = mfi.money_flow_index()

    return features


def calculate_cross_asset_features(stock_returns, bench_returns):
    """
    Calculate cross-asset features vs benchmark

    Parameters:
    -----------
    stock_returns : pd.Series
        Stock returns
    bench_returns : pd.Series
        Benchmark returns

    Returns:
    --------
    pd.DataFrame
        Cross-asset features
    """
    features = pd.DataFrame(index=stock_returns.index)

    window_corr = config.CROSS_ASSET_WINDOWS['correlation']
    window_beta = config.CROSS_ASSET_WINDOWS['beta']
    window_rs = config.CROSS_ASSET_WINDOWS['relative_strength']

    # 11. Rolling correlation with benchmark
    features['bench_correlation_60d'] = stock_returns.rolling(window=window_corr).corr(bench_returns)

    # 12. Rolling beta
    # Beta = Covariance(stock, bench) / Variance(bench)
    covariance = stock_returns.rolling(window=window_beta).cov(bench_returns)
    bench_variance = bench_returns.rolling(window=window_beta).var()
    features['bench_beta_60d'] = covariance / bench_variance.replace(0, np.nan)

    # 13. Relative strength (21-day)
    stock_return_21d = stock_returns.rolling(window=window_rs).sum()
    bench_return_21d = bench_returns.rolling(window=window_rs).sum()
    features['bench_relative_strength'] = stock_return_21d / bench_return_21d.replace(0, np.nan)

    return features


def calculate_macro_features(macro_data, vix_data):
    """
    Calculate macro features

    Parameters:
    -----------
    macro_data : pd.DataFrame
        Macro indicators (treasury yields, unemployment, etc.)
    vix_data : pd.Series
        VIX data

    Returns:
    --------
    pd.DataFrame
        Macro features
    """
    features = pd.DataFrame(index=macro_data.index)

    # 14. 10-Year Treasury Yield
    features['treasury_10y'] = macro_data['treasury_10y']

    # 15. Yield curve slope
    features['yield_curve_slope'] = macro_data['treasury_10y'] - macro_data['treasury_2y']

    # 16. VIX level
    features['vix'] = vix_data

    # 17. VIX change (21-day)
    features['vix_change_21d'] = vix_data - vix_data.shift(config.MACRO_WINDOWS['vix_change'])

    # 18. Unemployment rate (REMOVED - not useful for RL)
    # features['unemployment_rate'] = macro_data['unemployment_rate']

    return features


def calculate_interaction_features(features_df):
    """
    Calculate feature interaction features

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with existing features

    Returns:
    --------
    pd.DataFrame
        Interaction features
    """
    interactions = pd.DataFrame(index=features_df.index)

    # 19. Momentum-volatility ratio (risk-adjusted momentum)
    interactions['momentum_volatility_ratio'] = (
        features_df['return_21d'] / features_df['volatility_21d'].replace(0, np.nan)
    )

    # 20. Volume-weighted RSI
    interactions['volume_weighted_rsi'] = (
        (features_df['rsi_14d'] * features_df['volume_ratio_20d']) / 100
    )

    # 21. Volatility regime indicator
    interactions['vol_regime_indicator'] = (
        features_df['volatility_21d'] > features_df['volatility_63d']
    ).astype(int)

    return interactions


def calculate_regime_features(features_df, fit_clustering=True, kmeans_model=None, scaler=None):
    """
    Calculate regime features using K-means clustering

    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame with existing features
    fit_clustering : bool
        Whether to fit new clustering model (True for train, False for val/test)
    kmeans_model : KMeans
        Pre-fitted KMeans model (for val/test)
    scaler : StandardScaler
        Pre-fitted scaler (for val/test)

    Returns:
    --------
    pd.DataFrame
        Regime features
    KMeans
        Fitted KMeans model (or input model if fit_clustering=False)
    StandardScaler
        Fitted scaler (or input scaler if fit_clustering=False)
    """
    regime_features = pd.DataFrame(index=features_df.index)

    # Features for clustering
    clustering_cols = ['return_21d', 'volatility_21d', 'volume_ratio_20d']

    # Check if all required features exist
    if not all(col in features_df.columns for col in clustering_cols):
        config.log("  WARNING: Missing features for clustering, skipping regime features")
        regime_features['regime_cluster'] = 0
        regime_features['regime_transition'] = 0
        return regime_features, kmeans_model, scaler

    clustering_data = features_df[clustering_cols].copy()

    # Drop rows with NaN values
    valid_idx = clustering_data.dropna().index
    clustering_data_clean = clustering_data.loc[valid_idx]

    if len(clustering_data_clean) == 0:
        config.log("  WARNING: No valid data for clustering")
        regime_features['regime_cluster'] = 0
        regime_features['regime_transition'] = 0
        return regime_features, kmeans_model, scaler

    if fit_clustering:
        # Fit new models on training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(clustering_data_clean)

        kmeans_model = KMeans(
            n_clusters=config.REGIME_CLUSTERS,
            random_state=config.RANDOM_STATE,
            n_init=10
        )
        clusters = kmeans_model.fit_predict(X_scaled)

        config.log(f"  Fitted K-means clustering with {config.REGIME_CLUSTERS} clusters")
    else:
        # Use pre-fitted models for validation/test data
        if kmeans_model is None or scaler is None:
            raise ValueError("Must provide kmeans_model and scaler when fit_clustering=False")

        X_scaled = scaler.transform(clustering_data_clean)
        clusters = kmeans_model.predict(X_scaled)

    # 22. Regime cluster
    regime_features.loc[valid_idx, 'regime_cluster'] = clusters
    regime_features['regime_cluster'] = regime_features['regime_cluster'].ffill()
    regime_features['regime_cluster'] = regime_features['regime_cluster'].fillna(0).astype(int)

    # 23. Regime transition
    regime_features['regime_transition'] = (
        regime_features['regime_cluster'] != regime_features['regime_cluster'].shift(config.REGIME_TRANSITION_WINDOW)
    ).astype(int)

    return regime_features, kmeans_model, scaler


def add_forward_returns(features_df, raw_data, ticker):
    """
    Add forward-looking log returns for RL environment.

    CRITICAL: These are TARGETS for the environment, NOT features for the agent!
    They represent the actual returns that will be realized by holding assets.

    Uses LOG RETURNS for better mathematical properties:
    - Time-additive: sum(log_returns) = total_log_return
    - Symmetric: ±x% treated equally
    - More normal distribution
    - Better for portfolio optimization

    Parameters:
    -----------
    features_df : pd.DataFrame
        Existing features dataframe (ALREADY LAGGED)
    raw_data : dict
        Raw data dictionary containing stock prices
    ticker : str
        Ticker symbol

    Returns:
    --------
    pd.DataFrame
        Features with added 'forward_log_return' target column
    """
    # Get close prices for this ticker
    close_prices = raw_data['stock'][f'{ticker}_close']

    # ✅ Calculate 1-day forward log return (TARGET for environment)
    # log_return_t = ln(P_{t+1} / P_t) = ln(P_{t+1}) - ln(P_t)
    features_df['forward_log_return'] = np.log(
        close_prices.shift(-1) / close_prices
    )

    # ✅ Add simple return for reference/debugging (will be removed by VIF analysis anyway)
    features_df['forward_return'] = close_prices.pct_change().shift(-1)

    # ❌ REMOVED: close_price target (non-stationary, not useful for RL)
    # If needed for calculations, can reconstruct from returns:
    # price_t = price_0 * exp(cumsum(log_returns))

    # The last row will have NaN (no future data) - this is correct
    # The environment will not use this final step

    return features_df


def engineer_features_for_ticker(ticker, raw_data, benchmark_returns, macro_features,
                                   fit_clustering=True, kmeans_model=None, clustering_scaler=None):
    """
    Engineer all features for a single ticker

    Parameters:
    -----------
    ticker : str
        Ticker symbol
    raw_data : dict
        Dictionary with raw stock data
    benchmark_returns : pd.Series
        Benchmark returns
    macro_features : pd.DataFrame
        Macro features
    fit_clustering : bool
        Whether to fit clustering model
    kmeans_model : KMeans
        Pre-fitted KMeans model
    clustering_scaler : StandardScaler
        Pre-fitted clustering scaler

    Returns:
    --------
    pd.DataFrame
        All features for the ticker
    KMeans
        KMeans model
    StandardScaler
        Clustering scaler
    """
    config.log(f"Engineering features for {ticker}...")

    # Extract ticker data
    close = raw_data['stock'][f'{ticker}_close']
    high = raw_data['stock'][f'{ticker}_high']
    low = raw_data['stock'][f'{ticker}_low']
    volume = raw_data['stock'][f'{ticker}_volume']
    returns = raw_data['stock'][f'{ticker}_returns']

    # Calculate features by category
    momentum = calculate_momentum_features(close)
    trend = calculate_trend_features(close)
    mean_reversion = calculate_mean_reversion_features(close, high, low)
    volatility = calculate_volatility_features(returns, high, low, close)
    volume_features = calculate_volume_features(volume, high, low, close)
    cross_asset = calculate_cross_asset_features(returns, benchmark_returns)

    # Combine all features
    features = pd.concat([
        momentum,
        trend,
        mean_reversion,
        volatility,
        volume_features,
        cross_asset,
        macro_features,
    ], axis=1)

    # Calculate interaction features
    interactions = calculate_interaction_features(features)
    features = pd.concat([features, interactions], axis=1)

    # Calculate regime features
    regime, kmeans_model, clustering_scaler = calculate_regime_features(
        features,
        fit_clustering=fit_clustering,
        kmeans_model=kmeans_model,
        scaler=clustering_scaler
    )
    features = pd.concat([features, regime], axis=1)

    # Remove volatility_63d (only used for regime indicator)
    if 'volatility_63d' in features.columns:
        features = features.drop('volatility_63d', axis=1)

    # ✅ FIXED: Do NOT lag features here.
    # We assume the agent trades at Close of day t, using data up to Close of day t.
    # The target is the return from Close t to Close t+1.
    config.log(f"  Using features at time t to predict returns t->t+1...")
    features_lagged = features.copy()

    # ✅ Add forward returns WITHOUT lagging (these are targets, not features)
    # features_lagged[t-1] will predict forward_log_return[t→t+1]
    features_lagged = add_forward_returns(features_lagged, raw_data, ticker)

    # Add ticker column
    features_lagged['ticker'] = ticker

    # Drop NaN rows created by lagging
    features_lagged = features_lagged.dropna()

    config.log(f"  Generated {len(features_lagged.columns)-1} features + forward returns (lagged)")

    return features_lagged, kmeans_model, clustering_scaler


def engineer_all_features(raw_data, fit_clustering=True, clustering_models=None):
    """
    Engineer features for all tickers

    Parameters:
    -----------
    raw_data : dict
        Dictionary with all raw data
    fit_clustering : bool
        Whether to fit new clustering models
    clustering_models : dict
        Dictionary of pre-fitted clustering models per ticker

    Returns:
    --------
    pd.DataFrame
        Combined features for all tickers
    dict
        Dictionary of clustering models per ticker
    """
    config.log("="*60)
    config.log("PHASE 3: FEATURE ENGINEERING")
    config.log("="*60)

    # Calculate macro features once (same for all tickers)
    macro_features = calculate_macro_features(raw_data['macro'], raw_data['vix']['vix'])

    # Get benchmark returns
    benchmark_returns = raw_data['benchmark']['returns']

    # Initialize storage
    all_features = []
    new_clustering_models = {} if clustering_models is None else clustering_models

    # Process each ticker
    for ticker in config.TICKERS:
        # Get clustering models if available
        kmeans = new_clustering_models.get(ticker, {}).get('kmeans', None) if not fit_clustering else None
        scaler = new_clustering_models.get(ticker, {}).get('scaler', None) if not fit_clustering else None

        # Engineer features
        ticker_features, kmeans_model, clustering_scaler = engineer_features_for_ticker(
            ticker=ticker,
            raw_data=raw_data,
            benchmark_returns=benchmark_returns,
            macro_features=macro_features,
            fit_clustering=fit_clustering,
            kmeans_model=kmeans,
            clustering_scaler=scaler
        )

        # Store models
        if fit_clustering:
            new_clustering_models[ticker] = {
                'kmeans': kmeans_model,
                'scaler': clustering_scaler
            }

        all_features.append(ticker_features)

    # Combine all tickers
    combined_features = pd.concat(all_features, axis=0)

    config.log(f"\nTotal features engineered: {len(combined_features)}")
    config.log(f"Feature columns: {len(combined_features.columns)-1}")  # -1 for ticker column

    return combined_features, new_clustering_models


def main():
    """
    Main function to engineer features from raw data
    """
    # Load raw data
    raw_data = utils.load_dataframe(config.RAW_STOCK_FILE)
    benchmark_data = utils.load_dataframe(config.RAW_BENCHMARK_FILE)
    macro_data = utils.load_dataframe(config.RAW_MACRO_FILE)
    vix_data = utils.load_dataframe(config.RAW_VIX_FILE)

    # Prepare data dictionary
    data = {
        'stock': raw_data,
        'benchmark': benchmark_data,
        'macro': macro_data,
        'vix': vix_data,
    }

    # Engineer features
    features, clustering_models = engineer_all_features(data, fit_clustering=True)

    # Save features
    utils.save_dataframe(features, config.PROCESSED_FEATURES_FILE)

    # Save feature definitions
    feature_defs = utils.get_feature_definitions()
    utils.save_metadata(feature_defs, config.METADATA_FILES['feature_definitions'])

    config.log("="*60)
    config.log("FEATURE ENGINEERING COMPLETE")
    config.log("="*60)

    return features, clustering_models


if __name__ == "__main__":
    main()
