"""
Portfolio Management System V3.0 - Weekly Trading with Complete Configuration
============================================================================
Enhanced with weekly data aggregation, fixed configuration, and all audit issues resolved.

Author: System Architecture
Date: November 2024
Version: 3.0
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime, timedelta
import yfinance as yf
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# COMPLETE CONFIGURATION WITH ALL REQUIRED ATTRIBUTES
# ============================================================================

@dataclass
class SystemConfig:
    """Complete configuration with ALL required attributes for weekly trading."""
    
    # ========================================
    # ASSETS & DATA
    # ========================================
    tickers: List[str] = field(default_factory=lambda: [
        'NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG'
    ])
    
    # Dynamic date configuration
    @property
    def data_end_date(self) -> str:
        """Always use yesterday's date to ensure data availability."""
        return (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    @property
    def data_start_date(self) -> str:
        """Start date for data collection."""
        return '2015-01-01'
    
    # Data configuration
    use_log_returns: bool = True
    auto_adjust: bool = False
    data_frequency: str = 'weekly'  # Changed from daily to weekly
    
    # Data split ratios
    train_ratio: float = 0.60
    val_ratio: float = 0.15
    test_ratio: float = 0.25  # Increased for more robust out-of-sample evaluation
    
    # ========================================
    # NEURAL NETWORK ARCHITECTURE
    # ========================================
    activation: str = 'gelu'
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout_rate: float = 0.3  # Increased for better regularization
    use_attention: bool = True  # Multi-head attention for asset relationships (as per EXECUTIVE_SUMMARY)
    attention_heads: int = 8
    use_lstm: bool = True  # Temporal pattern recognition (as per EXECUTIVE_SUMMARY)
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    
    # ========================================
    # TRAINING HYPERPARAMETERS
    # ========================================
    learning_rate: float = 5e-4  # Reduced to prevent overfitting
    critic_lr_multiplier: float = 2.0
    exploration_noise: float = 0.4  # Starts at 40% (as per EXECUTIVE_SUMMARY)
    exploration_decay: float = 0.997  # OPTIMIZED: Slower decay maintains exploration longer (was 0.995)
    min_exploration: float = 0.05  # Minimum 5% exploration (as per EXECUTIVE_SUMMARY)
    buffer_size: int = 10000
    batch_size: int = 128  # Stable gradient estimates (as per EXECUTIVE_SUMMARY)
    gamma: float = 0.99
    n_epochs: int = 4  # Prevents overoptimization (as per EXECUTIVE_SUMMARY)
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.05  # OPTIMIZED: Increased for better exploration (was 0.03)
    max_grad_norm: float = 1.0  # OPTIMIZED: Less aggressive clipping allows learning (was 0.5)
    
    # Training control
    max_episodes: int = 1000
    trajectory_length: int = 2048
    lr_decay_factor: float = 0.5
    lr_patience: int = 10
    
    # ========================================
    # WALK-FORWARD VALIDATION (Adjusted for weekly)
    # ========================================
    n_folds: int = 10
    train_window_weeks: int = 104  # 2 years in weeks (was days)
    val_window_weeks: int = 13     # 3 months in weeks
    test_window_weeks: int = 13    # 3 months in weeks
    step_weeks: int = 4             # Monthly retraining
    
    # ========================================
    # FEATURES (Adjusted for weekly)
    # ========================================
    feature_lookbacks: List[int] = field(default_factory=lambda: [
        1, 2, 4, 8, 13, 26  # Weekly lookbacks: 1w, 2w, 1m, 3m, 6m, 1y
    ])
    use_technical_indicators: bool = True
    use_market_regime: bool = True
    use_relative_features: bool = True
    normalize_features: bool = True
    
    # ========================================
    # PORTFOLIO ENVIRONMENT
    # ========================================
    initial_capital: float = 100_000
    max_position_size: float = 0.40  # OPTIMIZED: 40% max position allows higher concentration (was 0.30)
    min_position_size: float = 0.0
    transaction_cost_bps: float = 2.5  # Realistic ETF/index transaction costs
    slippage_bps: float = 1.5  # Minimal slippage for weekly rebalancing
    stop_loss_threshold: float = 0.40  # 40% drawdown triggers stop-loss (no longer used after STEP 4)
    max_leverage: float = 1.0
    
    # ========================================
    # REWARDS (Adjusted for weekly)
    # ========================================
    reward_type: str = 'sharpe'
    risk_free_rate: float = 0.04
    reward_lookback: int = 26  # 26 weeks (6 months) for stable reward signal
    reward_scaling: float = 1.0  # Reduced to prevent reward explosion
    penalty_drawdown: bool = True
    
    # ========================================
    # MONITORING & LOGGING
    # ========================================
    log_every_n_episodes: int = 10
    save_every_n_episodes: int = 100
    validate_every_n_episodes: int = 50  # Every 50 episodes to reduce contamination (as per EXECUTIVE_SUMMARY)
    checkpoint_frequency: int = 5
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 50  # Increased from 20 for more stable early stopping
    patience: int = 50
    min_delta: float = 0.01
    
    # ========================================
    # REPRODUCIBILITY
    # ========================================
    random_seed: int = 42  # Default seed for reproducibility
    
    # ========================================
    # PATHS
    # ========================================
    data_dir: Path = Path('data')
    model_dir: Path = Path('models')
    log_dir: Path = Path('logs')
    results_dir: Path = Path('results')
    
    def __post_init__(self):
        """Create directories and validate configuration."""
        for dir_path in [self.data_dir, self.model_dir, self.log_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        assert abs(total_ratio - 1.0) < 1e-6, f"Split ratios must sum to 1, got {total_ratio}"
        assert len(self.hidden_dims) >= 2, "Need at least 2 hidden layers"
        assert 0 < self.learning_rate < 1, "Learning rate out of range"
        assert 0 < self.gamma <= 1, "Gamma must be in (0, 1]"
        assert 0 < self.clip_epsilon < 1, "Clip epsilon out of range"
        assert 0 <= self.dropout_rate < 1, "Dropout rate out of range"
        assert 0 <= self.min_position_size <= self.max_position_size <= 1, "Invalid position constraints"


# ============================================================================
# WEEKLY DATA PIPELINE
# ============================================================================

class DataPipeline:
    """Data pipeline with weekly aggregation and complete feature engineering."""
    
    def __init__(self, config: SystemConfig, use_selected_features: bool = True):
        self.config = config
        self.logger = self._setup_logger()
        self.scaler = None
        self.use_selected_features = use_selected_features
        self.selected_features = None
        
        # Load selected features if requested
        if self.use_selected_features:
            self.selected_features = self._load_selected_features()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.config.log_dir / 'data_pipeline.log')
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def _load_selected_features(self) -> Optional[List[str]]:
        """
        Load selected features from JSON file.
        
        Returns:
            List of selected feature names, or None if file not found
        """
        selected_features_path = self.config.data_dir / 'selected_features.json'
        
        if not selected_features_path.exists():
            self.logger.warning(f"Selected features file not found at {selected_features_path}. Using all features.")
            return None
        
        try:
            with open(selected_features_path, 'r') as f:
                selected_features = json.load(f)
            
            if not isinstance(selected_features, list):
                self.logger.warning(f"Invalid format in {selected_features_path}. Using all features.")
                return None
            
            self.logger.info(f"Loaded {len(selected_features)} selected features from {selected_features_path}")
            return selected_features
        
        except Exception as e:
            self.logger.warning(f"Error loading selected features: {e}. Using all features.")
            return None
    
    def load_data(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load and aggregate data to weekly frequency.
        
        Returns:
            DataFrame with weekly OHLCV data and log returns
        """
        cache_file = self.config.data_dir / 'weekly_market_data_cache.parquet'
        
        # Check cache
        if use_cache and cache_file.exists():
            cache_date = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if (datetime.now() - cache_date).days < 7:  # Weekly cache validity
                self.logger.info(f"Loading cached weekly data from {cache_file}")
                return pd.read_parquet(cache_file)
        
        self.logger.info(f"Downloading daily data and converting to weekly...")
        self.logger.info(f"Period: {self.config.data_start_date} to {self.config.data_end_date}")
        
        # Download daily data first
        processed_data = []
        
        for ticker in self.config.tickers:
            self.logger.info(f"Processing {ticker}...")
            try:
                # Download daily data
                stock = yf.Ticker(ticker)
                daily_data = stock.history(
                    start=self.config.data_start_date,
                    end=self.config.data_end_date,
                    auto_adjust=self.config.auto_adjust,
                    actions=True
                )
                
                if daily_data.empty:
                    self.logger.warning(f"No data for {ticker}, skipping...")
                    continue
                
                # Convert to weekly data
                weekly_data = self._aggregate_to_weekly(daily_data)
                weekly_data['ticker'] = ticker
                
                # Calculate log returns
                weekly_data['log_returns'] = np.log(
                    weekly_data['close'] / weekly_data['close'].shift(1)
                )
                
                # Additional weekly metrics
                weekly_data['high_low_ratio'] = np.log(weekly_data['high'] / weekly_data['low'])
                weekly_data['close_open_ratio'] = np.log(weekly_data['close'] / weekly_data['open'])
                weekly_data['volume_change'] = np.log(
                    weekly_data['volume'] / weekly_data['volume'].shift(1).fillna(weekly_data['volume'])
                )
                
                processed_data.append(weekly_data)
                
            except Exception as e:
                self.logger.error(f"Error processing {ticker}: {str(e)}")
                continue
        
        if not processed_data:
            raise ValueError("Failed to download any ticker data!")
        
        # Combine all tickers
        combined_data = pd.concat(processed_data, ignore_index=True)
        
        # Ensure column names are lowercase
        combined_data.columns = [col.lower() for col in combined_data.columns]
        
        # Save cache
        combined_data.to_parquet(cache_file)
        self.logger.info(f"Cached weekly data saved to {cache_file}")
        
        # Log summary
        self.logger.info(f"Loaded {len(combined_data)} weekly observations")
        self.logger.info(f"Date range: {combined_data['date'].min()} to {combined_data['date'].max()}")
        self.logger.info(f"Tickers: {', '.join(combined_data['ticker'].unique())}")
        
        return combined_data
    
    def _aggregate_to_weekly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate daily data to weekly frequency.
        
        Args:
            daily_data: Daily OHLCV data
            
        Returns:
            Weekly aggregated data
        """
        # Reset index to have date as column
        daily_data = daily_data.reset_index()
        
        # Set date as index for resampling
        daily_data.set_index('Date', inplace=True)
        
        # Resample to weekly frequency (Friday close)
        weekly_agg = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }
        
        # Handle dividends and splits if present
        if 'Dividends' in daily_data.columns:
            weekly_agg['Dividends'] = 'sum'
        if 'Stock Splits' in daily_data.columns:
            weekly_agg['Stock Splits'] = 'prod'  # Multiply for splits
        
        weekly_data = daily_data.resample('W-FRI').agg(weekly_agg)
        
        # Remove weeks with no trading (holidays)
        weekly_data = weekly_data.dropna(subset=['Close'])
        
        # Reset index to have date as column
        weekly_data.reset_index(inplace=True)
        weekly_data.rename(columns={'Date': 'date'}, inplace=True)
        
        # Rename columns to lowercase
        weekly_data.columns = [col.lower() for col in weekly_data.columns]
        
        return weekly_data
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for weekly data.
        
        Args:
            df: Weekly OHLCV data
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Engineering features for weekly data...")
        
        # Sort by date and ticker
        df = df.sort_values(['ticker', 'date'])
        
        # Add technical indicators (adjusted for weekly)
        df = self._add_technical_indicators_weekly(df)
        
        # Add rolling statistics (adjusted for weekly)
        df = self._add_rolling_statistics_weekly(df)
        
        # Add regime features
        if self.config.use_market_regime:
            df = self._add_regime_features_weekly(df)
        
        # Add relative features
        if self.config.use_relative_features:
            df = self._add_relative_features(df)
        
        # Validate features (this should handle all NaN values)
        df = self._validate_features(df)
        
        # Final check: drop rows only if ALL feature columns are NaN (shouldn't happen after validation)
        feature_cols = [col for col in df.columns 
                       if col not in ['date', 'ticker', 'open', 'high', 'low', 
                                     'close', 'volume', 'dividends', 'stock splits']]
        
        if len(feature_cols) > 0:
            # Only drop rows where ALL features are NaN
            rows_with_all_nan_features = df[feature_cols].isna().all(axis=1)
            if rows_with_all_nan_features.any():
                initial_rows = len(df)
                df = df[~rows_with_all_nan_features]
                dropped_rows = initial_rows - len(df)
                if dropped_rows > 0:
                    self.logger.warning(f"Dropped {dropped_rows} rows where all features were NaN")
        
        # Verify no NaN values remain in feature columns
        remaining_nans = df[feature_cols].isna().sum().sum() if len(feature_cols) > 0 else 0
        if remaining_nans > 0:
            self.logger.warning(f"Warning: {remaining_nans} NaN values still remain after validation")
            df[feature_cols] = df[feature_cols].fillna(0)
        
        # Filter to selected features if specified
        if self.use_selected_features and self.selected_features is not None:
            # Find which selected features actually exist in the dataframe
            available_selected = [f for f in self.selected_features if f in feature_cols]
            missing_selected = [f for f in self.selected_features if f not in feature_cols]
            
            if missing_selected:
                self.logger.warning(f"Selected features not found in data: {missing_selected}")
            
            if available_selected:
                # Keep only selected features (plus non-feature columns)
                cols_to_keep = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                               'dividends', 'stock splits'] + available_selected
                cols_to_keep = [col for col in cols_to_keep if col in df.columns]
                
                # Drop features that are not selected
                features_to_drop = [f for f in feature_cols if f not in available_selected]
                if features_to_drop:
                    df = df.drop(columns=features_to_drop)
                    self.logger.info(f"Filtered features: kept {len(available_selected)} selected features, "
                                   f"removed {len(features_to_drop)} features")
                    feature_cols = available_selected
                else:
                    self.logger.info(f"All {len(feature_cols)} features are in selected set")
            else:
                self.logger.warning("No selected features found in data. Using all features.")
        
        # Log feature count
        self.logger.info(f"Final feature count: {len(feature_cols)} features")
        
        return df
    
    def _add_technical_indicators_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators adjusted for weekly data."""
        
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_df = df[ticker_mask].copy()
            
            close_prices = ticker_df['close']
            high_prices = ticker_df['high']
            low_prices = ticker_df['low']
            volume = ticker_df['volume']
            
            # Price-based features with weekly lookbacks
            for lookback in self.config.feature_lookbacks:
                # Use min_periods to reduce initial NaNs (require at least 30% of window)
                min_periods = max(1, int(lookback * 0.3))
                
                # Returns
                df.loc[ticker_mask, f'returns_{lookback}w'] = close_prices.pct_change(lookback)
                
                # Moving averages
                df.loc[ticker_mask, f'sma_{lookback}w'] = close_prices.rolling(lookback, min_periods=min_periods).mean()
                sma_values = df.loc[ticker_mask, f'sma_{lookback}w']
                df.loc[ticker_mask, f'price_to_sma_{lookback}w'] = (
                    close_prices / (sma_values + 1e-10)
                )
                
                # Volatility (weekly)
                returns = close_prices.pct_change()
                df.loc[ticker_mask, f'volatility_{lookback}w'] = (
                    returns.rolling(lookback, min_periods=min_periods).std() * np.sqrt(52)  # Annualized weekly vol
                )
                
                # Volume features
                volume_ma = volume.rolling(lookback, min_periods=min_periods).mean()
                df.loc[ticker_mask, f'volume_ratio_{lookback}w'] = (
                    volume / (volume_ma + 1e-10)
                )
            
            # RSI (14 weeks is standard for weekly) - use min_periods
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
            rs = gain / (loss + 1e-10)
            df.loc[ticker_mask, 'rsi_14w'] = 100 - (100 / (1 + rs))
            
            # MACD (adjusted for weekly: 12, 26, 9) - EWM doesn't need min_periods but we'll handle NaNs
            exp1 = close_prices.ewm(span=12, adjust=False, min_periods=4).mean()
            exp2 = close_prices.ewm(span=26, adjust=False, min_periods=8).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False, min_periods=3).mean()
            df.loc[ticker_mask, 'macd'] = macd
            df.loc[ticker_mask, 'macd_signal'] = signal
            df.loc[ticker_mask, 'macd_diff'] = macd - signal
            
            # Bollinger Bands (20 weeks) - use min_periods
            sma = close_prices.rolling(window=20, min_periods=7).mean()
            std = close_prices.rolling(window=20, min_periods=7).std()
            df.loc[ticker_mask, 'bb_upper'] = sma + (std * 2)
            df.loc[ticker_mask, 'bb_lower'] = sma - (std * 2)
            df.loc[ticker_mask, 'bb_width'] = (
                (df.loc[ticker_mask, 'bb_upper'] - df.loc[ticker_mask, 'bb_lower']) / (sma + 1e-10)
            )
            df.loc[ticker_mask, 'bb_position'] = (
                (close_prices - df.loc[ticker_mask, 'bb_lower']) / 
                (df.loc[ticker_mask, 'bb_upper'] - df.loc[ticker_mask, 'bb_lower'] + 1e-10)
            )
            
            # ATR (Average True Range) for weekly - use min_periods
            high_low = high_prices - low_prices
            high_close = np.abs(high_prices - close_prices.shift())
            low_close = np.abs(low_prices - close_prices.shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df.loc[ticker_mask, 'atr_14w'] = true_range.rolling(14, min_periods=5).mean()
            
        return df
    
    def _add_rolling_statistics_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for weekly data."""
        
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_df = df[ticker_mask].copy()
            
            if 'log_returns' in ticker_df.columns:
                returns = ticker_df['log_returns']
            else:
                returns = ticker_df['close'].pct_change()
            
            # Rolling statistics with weekly windows
            for window in [4, 8, 13, 26]:  # 1m, 2m, 3m, 6m in weeks
                # Use min_periods to reduce initial NaNs (require at least 30% of window)
                min_periods = max(2, int(window * 0.3))
                
                # Sharpe ratio (annualized for weekly)
                roll_mean = returns.rolling(window, min_periods=min_periods).mean()
                roll_std = returns.rolling(window, min_periods=min_periods).std()
                df.loc[ticker_mask, f'sharpe_{window}w'] = (
                    (roll_mean - self.config.risk_free_rate/52) / (roll_std + 1e-10) * np.sqrt(52)
                )
                
                # Skewness and kurtosis
                df.loc[ticker_mask, f'skew_{window}w'] = returns.rolling(window, min_periods=min_periods).skew()
                df.loc[ticker_mask, f'kurt_{window}w'] = returns.rolling(window, min_periods=min_periods).kurt()
                
                # Max drawdown
                cum_returns = (1 + returns).cumprod()
                rolling_max = cum_returns.rolling(window, min_periods=1).max()
                drawdown = (cum_returns - rolling_max) / (rolling_max + 1e-10)
                df.loc[ticker_mask, f'max_dd_{window}w'] = drawdown.rolling(window, min_periods=min_periods).min()
        
        return df
    
    def _add_regime_features_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime indicators for weekly data."""
        
        # Calculate market return (equal-weighted)
        market_returns = df.groupby('date')['log_returns'].mean()
        df['market_return'] = df['date'].map(market_returns)
        
        # Market volatility (annualized weekly)
        market_vol = market_returns.rolling(window=8, min_periods=4).std() * np.sqrt(52)
        df['market_volatility'] = df['date'].map(market_vol)
        
        # Volatility regimes
        vol_quantiles = market_vol.quantile([0.33, 0.67])
        df['vol_regime_low'] = (df['market_volatility'] < vol_quantiles.iloc[0]).astype(float)
        df['vol_regime_medium'] = (
            (df['market_volatility'] >= vol_quantiles.iloc[0]) & 
            (df['market_volatility'] <= vol_quantiles.iloc[1])
        ).astype(float)
        df['vol_regime_high'] = (df['market_volatility'] > vol_quantiles.iloc[1]).astype(float)
        
        # Trend regime (13-week trend)
        market_cum_returns = market_returns.rolling(window=13, min_periods=4).sum()
        df['trend_up'] = df['date'].map(lambda x: 1.0 if market_cum_returns.get(x, 0) > 0 else 0.0)
        df['trend_strength'] = df['date'].map(market_cum_returns)
        
        # Calculate beta to market
        for ticker in df['ticker'].unique():
            ticker_mask = df['ticker'] == ticker
            ticker_df = df[ticker_mask].copy()
            
            beta_values = []
            for i in range(len(ticker_df)):
                # Calculate correlation over last 13 weeks
                window_start = max(0, i - 13)
                window_data = ticker_df.iloc[window_start:i+1]
                
                if len(window_data) > 2:
                    ticker_rets = window_data['log_returns']
                    market_rets = window_data['market_return']
                    
                    if ticker_rets.std() > 0 and market_rets.std() > 0:
                        corr = ticker_rets.corr(market_rets)
                        beta = corr * (ticker_rets.std() / market_rets.std())
                        beta_values.append(beta if not pd.isna(beta) else 1.0)
                    else:
                        beta_values.append(1.0)
                else:
                    beta_values.append(1.0)
            
            df.loc[ticker_mask, 'beta_to_market'] = beta_values
        
        # Drop temporary market_return column
        df = df.drop(columns=['market_return'])
        
        return df
    
    def _add_relative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional relative features."""
        
        for date in df['date'].unique():
            date_mask = df['date'] == date
            date_df = df[date_mask].copy()
            
            # Relative returns
            if 'log_returns' in date_df.columns:
                returns = date_df['log_returns']
                df.loc[date_mask, 'relative_return'] = returns - returns.mean()
                df.loc[date_mask, 'return_rank'] = returns.rank(pct=True)
            
            # Relative volume
            if 'volume' in date_df.columns:
                volume = date_df['volume']
                df.loc[date_mask, 'relative_volume'] = volume / volume.mean()
                df.loc[date_mask, 'volume_rank'] = volume.rank(pct=True)
            
            # Relative RSI
            if 'rsi_14w' in date_df.columns:
                rsi = date_df['rsi_14w']
                df.loc[date_mask, 'relative_rsi'] = rsi - rsi.mean()
        
        return df
    
    def _validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean features."""
        
        # Get feature columns (exclude metadata columns)
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 
                       'dividends', 'stock splits']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Check for NaN values
        nan_counts = df[feature_cols].isna().sum()
        if nan_counts.any():
            nan_cols = nan_counts[nan_counts > 0].index.tolist()
            self.logger.warning(f"Found NaN values in {len(nan_cols)} feature columns")
            
            # Forward fill then backward fill within each ticker group
            for col in feature_cols:
                if df[col].isna().any():
                    df[col] = df.groupby('ticker')[col].ffill().bfill()
            
            # For any remaining NaNs, fill with 0 (safer than dropping all rows)
            remaining_nans = df[feature_cols].isna().sum().sum()
            if remaining_nans > 0:
                self.logger.warning(f"Filling {remaining_nans} remaining NaN values with 0")
                df[feature_cols] = df[feature_cols].fillna(0)
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_feature_cols = [col for col in numeric_cols if col in feature_cols]
        
        if len(numeric_feature_cols) > 0:
            inf_mask = np.isinf(df[numeric_feature_cols].values)
            if inf_mask.any():
                self.logger.warning("Found infinite values, replacing with NaN then filling...")
                df[numeric_feature_cols] = df[numeric_feature_cols].replace([np.inf, -np.inf], np.nan)
                df[numeric_feature_cols] = df[numeric_feature_cols].fillna(0)
        
        # Clip extreme values (outliers) - only for feature columns
        for col in numeric_feature_cols:
            if df[col].notna().sum() > 0:  # Only clip if there's data
                percentile_99 = df[col].quantile(0.99)
                percentile_1 = df[col].quantile(0.01)
                if pd.notna(percentile_99) and pd.notna(percentile_1):
                    df[col] = df[col].clip(lower=percentile_1, upper=percentile_99)
        
        return df
    
    def normalize_splits(self, splits: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Normalize features across train/val/test splits."""
        from sklearn.preprocessing import RobustScaler
        
        # Validate that splits are not empty
        if splits['train'].empty:
            error_msg = (
                "Cannot normalize features: training split is empty. "
                f"Train: {len(splits['train'])} rows, "
                f"Val: {len(splits['val'])} rows, "
                f"Test: {len(splits['test'])} rows. "
                "This usually means all data was dropped during feature engineering. "
                "Check feature engineering logic and data quality."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        feature_cols = [col for col in splits['train'].columns 
                       if col not in ['date', 'ticker', 'open', 'high', 'low', 
                                     'close', 'volume', 'dividends', 'stock splits']]
        
        if len(feature_cols) == 0:
            error_msg = "No feature columns found for normalization."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Fit scaler on training data only
        self.scaler = RobustScaler()
        splits['train'][feature_cols] = self.scaler.fit_transform(
            splits['train'][feature_cols].fillna(0)
        )
        
        # Transform validation and test data
        if not splits['val'].empty:
            splits['val'][feature_cols] = self.scaler.transform(
                splits['val'][feature_cols].fillna(0)
            )
        
        if not splits['test'].empty:
            splits['test'][feature_cols] = self.scaler.transform(
                splits['test'][feature_cols].fillna(0)
            )
        
        self.logger.info("Features normalized using RobustScaler")
        
        return splits
    
    def create_train_val_test_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create temporal splits with no overlap for weekly data."""
        self.logger.info("\nCreating train/val/test splits for weekly data...")
        
        # Validate input data
        if df.empty:
            error_msg = (
                "Cannot create splits: input DataFrame is empty. "
                "This usually means all data was dropped during feature engineering. "
                "Check feature engineering logic and data quality."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        dates = sorted(df['date'].unique())
        n_dates = len(dates)
        
        if n_dates == 0:
            error_msg = "Cannot create splits: no dates found in DataFrame."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Calculate split points
        train_end_idx = int(n_dates * self.config.train_ratio)
        val_end_idx = int(n_dates * (self.config.train_ratio + self.config.val_ratio))
        
        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]
        test_dates = dates[val_end_idx:]
        
        # Verify no overlap
        assert len(set(train_dates) & set(val_dates)) == 0, "Train-Val overlap!"
        assert len(set(val_dates) & set(test_dates)) == 0, "Val-Test overlap!"
        assert len(set(train_dates) & set(test_dates)) == 0, "Train-Test overlap!"
        
        splits = {
            'train': df[df['date'].isin(train_dates)].copy(),
            'val': df[df['date'].isin(val_dates)].copy(),
            'test': df[df['date'].isin(test_dates)].copy()
        }
        
        # Log split information
        for split_name, split_df in splits.items():
            if not split_df.empty:
                n_weeks = split_df['date'].nunique()
                self.logger.info(
                    f"  {split_name}: {len(split_df)} rows, "
                    f"{split_df['date'].min().date()} to {split_df['date'].max().date()}, "
                    f"{n_weeks} weeks"
                )
            else:
                self.logger.warning(f"  {split_name}: EMPTY (0 rows)")
        
        # Warn if train split is empty
        if splits['train'].empty:
            self.logger.warning(
                f"Training split is empty! Input had {len(df)} rows and {n_dates} dates. "
                f"Split indices: train_end={train_end_idx}, val_end={val_end_idx}, total={n_dates}"
            )
        
        return splits


# ============================================================================
# BUY-AND-HOLD BENCHMARK CALCULATION
# ============================================================================

def calculate_buy_hold_performance(test_data: pd.DataFrame, config: SystemConfig) -> dict:
    """
    Calculate buy-and-hold performance using actual historical prices.
    
    True buy-and-hold strategy:
    1. Start with equal weights (1/n_assets) at first date
    2. Buy positions at initial prices
    3. Hold without rebalancing - weights drift as prices change
    4. Calculate portfolio value each week using actual prices
    5. No transaction costs, no rebalancing
    
    Args:
        test_data: DataFrame with test period data (must have 'date', 'ticker', 'close')
        config: SystemConfig
    
    Returns:
        dict with performance metrics:
            - total_return: Total return over test period
            - annual_return: Annualized return
            - sharpe: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown
            - portfolio_values: List of portfolio values over time
            - returns: Array of weekly returns
    """
    
    # Get test dates sorted
    test_dates = sorted(test_data['date'].unique())
    n_assets = len(config.tickers)
    
    if len(test_dates) < 2:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': [config.initial_capital],
            'returns': np.array([])
        }
    
    # Get initial prices (first date)
    initial_date = test_dates[0]
    initial_data = test_data[test_data['date'] == initial_date]
    
    # Get initial prices for all assets
    initial_prices = {}
    for ticker in config.tickers:
        ticker_data = initial_data[initial_data['ticker'] == ticker]
        if ticker_data.empty:
            raise ValueError(f"No data for {ticker} on initial date {initial_date}")
        initial_prices[ticker] = ticker_data['close'].values[0]
    
    # Initialize portfolio: buy equal weights at initial prices
    initial_value = config.initial_capital
    equal_weight = 1.0 / n_assets
    
    # Calculate number of shares for each asset (buy-and-hold: fixed shares)
    shares = {}
    for ticker in config.tickers:
        shares[ticker] = (initial_value * equal_weight) / initial_prices[ticker]
    
    # Track portfolio values and returns over time
    portfolio_values = [initial_value]
    returns = []
    dates_series = [initial_date]
    
    # Calculate portfolio value at each subsequent date using actual prices
    for i in range(1, len(test_dates)):
        current_date = test_dates[i]
        dates_series.append(current_date)
        
        # Get current prices
        current_data = test_data[test_data['date'] == current_date]
        current_prices = {}
        for ticker in config.tickers:
            ticker_data = current_data[current_data['ticker'] == ticker]
            if not ticker_data.empty:
                current_prices[ticker] = ticker_data['close'].values[0]
            else:
                # If price missing, use previous price (hold position)
                # For first missing price, use initial price
                if ticker in initial_prices:
                    current_prices[ticker] = initial_prices[ticker]
                else:
                    current_prices[ticker] = 0.0
        
        # Calculate current portfolio value: sum(shares * current_price)
        current_portfolio_value = 0.0
        for ticker in config.tickers:
            if ticker in shares and ticker in current_prices:
                current_portfolio_value += shares[ticker] * current_prices[ticker]
        
        portfolio_values.append(current_portfolio_value)
        
        # Calculate weekly return
        if i > 0:
            prev_value = portfolio_values[i - 1]
            if prev_value > 0:
                weekly_return = (current_portfolio_value / prev_value) - 1
            else:
                weekly_return = 0.0
            returns.append(weekly_return)
    
    # Calculate metrics
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': portfolio_values,
            'returns': returns_array
        }
    
    # Total return from actual portfolio values
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value) - 1
    
    # Annualized return using actual compounding
    n_weeks = len(returns_array)
    if n_weeks > 0:
        annual_return = (final_value / initial_value) ** (52 / n_weeks) - 1
    else:
        annual_return = 0.0
    
    # Sharpe ratio (annualized for weekly returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    weekly_rf = config.risk_free_rate / 52
    sharpe = (mean_return - weekly_rf) / (std_return + 1e-8) * np.sqrt(52)
    
    # Max drawdown from actual portfolio values
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / (peak + 1e-10)
    max_drawdown = abs(drawdown.min())
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values,
        'returns': returns_array,
        'dates': pd.to_datetime(dates_series)
    }


def calculate_qqq_performance(test_data: pd.DataFrame, config: SystemConfig) -> dict:
    """
    Calculate QQQ (Invesco QQQ Trust) benchmark performance.
    
    QQQ is a popular tech ETF that tracks the NASDAQ-100 index.
    This function downloads QQQ historical data and calculates performance
    over the same test period.
    
    Args:
        test_data: DataFrame with test period data (used to get date range)
        config: SystemConfig
    
    Returns:
        dict with performance metrics (same format as buy-and-hold):
            - total_return: Total return over test period
            - annual_return: Annualized return
            - sharpe: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown
            - portfolio_values: List of portfolio values over time
            - returns: Array of weekly returns
    """
    import yfinance as yf
    
    # Get test date range
    test_dates = sorted(test_data['date'].unique())
    if len(test_dates) < 2:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': [config.initial_capital],
            'returns': np.array([])
        }
    
    start_date = test_dates[0]
    end_date = test_dates[-1]
    
    # Download QQQ data for the test period
    try:
        qqq = yf.Ticker('QQQ')
        qqq_data = qqq.history(
            start=start_date,
            end=end_date + pd.Timedelta(days=7),  # Add buffer to ensure we get end date
            interval='1wk'  # Weekly data
        )
        
        if qqq_data.empty:
            raise ValueError("No QQQ data downloaded")
        
        # Reset index to have date as column
        qqq_data = qqq_data.reset_index()
        qqq_data['date'] = pd.to_datetime(qqq_data['Date']).dt.date
        
        # Get close prices
        qqq_data = qqq_data[['date', 'Close']].copy()
        qqq_data.columns = ['date', 'close']
        qqq_data = qqq_data.sort_values('date')
        
    except Exception as e:
        logging.warning(f"Error downloading QQQ data: {e}. Returning zero performance.")
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': [config.initial_capital],
            'returns': np.array([])
        }
    
    # Align QQQ dates with test dates (use closest available dates)
    # For each test date, find the closest QQQ date
    aligned_prices = {}
    for test_date in test_dates:
        # Convert test_date to date object
        if isinstance(test_date, pd.Timestamp):
            test_date_obj = test_date.date()
        elif isinstance(test_date, datetime):
            test_date_obj = test_date.date()
        else:
            test_date_obj = pd.to_datetime(test_date).date()
        
        # Calculate date differences (both are date objects, so subtraction gives timedelta)
        date_diffs = []
        for qqq_date in qqq_data['date']:
            diff = abs((qqq_date - test_date_obj).days)
            date_diffs.append(diff)
        
        date_diffs = pd.Series(date_diffs, index=qqq_data.index)
        closest_idx = date_diffs.idxmin()
        min_diff = date_diffs[closest_idx]
        
        if min_diff <= 7:  # Within a week
            aligned_prices[test_date] = qqq_data.loc[closest_idx, 'close']
        else:
            # If no close date, use forward fill from previous
            if len(aligned_prices) > 0:
                aligned_prices[test_date] = list(aligned_prices.values())[-1]
            else:
                aligned_prices[test_date] = qqq_data.iloc[0]['close']
    
    # Calculate portfolio values (buy QQQ at first date, hold)
    initial_value = config.initial_capital
    initial_price = aligned_prices[test_dates[0]]
    
    if initial_price <= 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': [initial_value],
            'returns': np.array([])
        }
    
    # Number of shares (buy-and-hold)
    shares = initial_value / initial_price
    
    # Calculate portfolio values over time
    portfolio_values = [initial_value]
    returns = []
    dates_series = [test_dates[0]]
    
    for i in range(1, len(test_dates)):
        current_date = test_dates[i]
        dates_series.append(current_date)
        
        current_price = aligned_prices[current_date]
        current_value = shares * current_price
        
        portfolio_values.append(current_value)
        
        # Calculate weekly return
        prev_value = portfolio_values[i - 1]
        if prev_value > 0:
            weekly_return = (current_value / prev_value) - 1
        else:
            weekly_return = 0.0
        returns.append(weekly_return)
    
    # Calculate metrics
    returns_array = np.array(returns)
    
    if len(returns_array) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe': 0.0,
            'max_drawdown': 0.0,
            'portfolio_values': portfolio_values,
            'returns': returns_array
        }
    
    # Total return
    final_value = portfolio_values[-1]
    total_return = (final_value / initial_value) - 1
    
    # Annualized return
    n_weeks = len(returns_array)
    if n_weeks > 0:
        annual_return = (final_value / initial_value) ** (52 / n_weeks) - 1
    else:
        annual_return = 0.0
    
    # Sharpe ratio (annualized for weekly returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    weekly_rf = config.risk_free_rate / 52
    sharpe = (mean_return - weekly_rf) / (std_return + 1e-8) * np.sqrt(52)
    
    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (np.array(portfolio_values) - peak) / (peak + 1e-10)
    max_drawdown = abs(drawdown.min())
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values,
        'returns': returns_array,
        'dates': pd.to_datetime(dates_series)
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline for weekly trading system."""
    print("\n" + "="*70)
    print("PORTFOLIO MANAGEMENT SYSTEM V3.0 - WEEKLY TRADING")
    print("="*70)
    
    # Initialize configuration
    config = SystemConfig()
    
    print(f"\nConfiguration:")
    print(f"  Data: {config.data_start_date} to {config.data_end_date}")
    print(f"  Frequency: {config.data_frequency}")
    print(f"  Returns type: {'Log returns' if config.use_log_returns else 'Simple returns'}")
    print(f"  Tickers: {', '.join(config.tickers)}")
    print(f"  Network: {config.hidden_dims}")
    print(f"  Activation: {config.activation}")
    
    # Initialize data pipeline
    print("\n1. INITIALIZING DATA PIPELINE...")
    pipeline = DataPipeline(config)  # Fixed: using correct class name
    
    # Load weekly market data
    print("\n2. LOADING WEEKLY DATA...")
    raw_data = pipeline.load_data(use_cache=True)
    
    # Display sample of raw data
    print("\n3. RAW WEEKLY DATA SAMPLE:")
    print(raw_data.head(10))
    print(f"\nShape: {raw_data.shape}")
    print(f"Columns: {raw_data.columns.tolist()}")
    
    # Engineer features
    print("\n4. ENGINEERING FEATURES FOR WEEKLY DATA...")
    featured_data = pipeline.engineer_features(raw_data)
    
    # Display feature statistics
    print("\n5. FEATURE STATISTICS:")
    feature_cols = [col for col in featured_data.columns 
                   if col not in ['date', 'ticker', 'open', 'high', 'low', 
                                 'close', 'volume', 'dividends', 'stock splits']]
    
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Sample features: {feature_cols[:15]}")
    
    # Create splits
    print("\n6. CREATING TRAIN/VAL/TEST SPLITS...")
    splits = pipeline.create_train_val_test_splits(featured_data)
    
    # Normalize splits
    print("\n7. NORMALIZING FEATURES...")
    splits = pipeline.normalize_splits(splits)
    
    # Display summary
    print("\n" + "="*70)
    print("DATA PREPARATION COMPLETE")
    print("="*70)
    
    # Save processed data
    output_file = config.data_dir / 'processed_weekly_data.parquet'
    featured_data.to_parquet(output_file)
    print(f"\nProcessed weekly data saved to: {output_file}")
    
    # Save splits
    for split_name, split_df in splits.items():
        split_file = config.data_dir / f'{split_name}_weekly_data.parquet'
        split_df.to_parquet(split_file)
        print(f"{split_name} data saved to: {split_file}")
    
    # Display final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total trading weeks: {featured_data['date'].nunique()}")
    print(f"Total data points: {len(featured_data)}")
    print(f"Features per observation: {len(feature_cols)}")
    
    if config.use_log_returns:
        # Display weekly return statistics
        print("\nWeekly Return Statistics by Ticker:")
        for ticker in config.tickers:
            ticker_data = raw_data[raw_data['ticker'] == ticker]
            if 'log_returns' in ticker_data.columns:
                log_ret = ticker_data['log_returns']
                weekly_mean = log_ret.mean()
                annual_mean = weekly_mean * 52
                weekly_vol = log_ret.std()
                annual_vol = weekly_vol * np.sqrt(52)
                sharpe = annual_mean / annual_vol if annual_vol > 0 else 0
                
                print(f"  {ticker}:")
                print(f"    Annual return: {annual_mean:.2%}")
                print(f"    Annual volatility: {annual_vol:.2%}")
                print(f"    Sharpe ratio: {sharpe:.2f}")
    
    return featured_data, splits


if __name__ == '__main__':
    featured_data, splits = main()
