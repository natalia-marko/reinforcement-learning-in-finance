"""
Data Preparation Module - WEEKLY (Leak-Safe)
============================================

Key points:
1) Compute technicals on WEEKLY data (resample first, then indicators)
2) Robust dedup of (date, ticker) before any pivoting
3) No leak-prone global price/return matrices
4) Save prices & returns **per split** only
5) Macro/calendar weekly with staleness features

Outputs
-------
- data/raw_technical.csv
- data/macro_calendar.csv
- data/train/technical.csv
- data/train/macro_calendar.csv
- data/val/technical.csv
- data/val/macro_calendar.csv
- data/test/technical.csv
- data/test/macro_calendar.csv
- data/train/prices.csv
- data/train/log_returns.csv
- data/train/ew_returns.csv
- data/val/prices.csv
- data/val/log_returns.csv
- data/val/ew_returns.csv
- data/test/prices.csv
- data/test/log_returns.csv
- data/test/ew_returns.csv
- data/technical_indicators_weekly.csv   (normalized, with split column)
- data/macro_weekly.csv                  (normalized, with split column)
- data/metadata.json
"""

import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DataPreparator:
    """Prepares and saves all datasets for the RL pipeline."""

    def __init__(
        self,
        tickers: List[str],
        benchmark: str,
        start_date: str,
        end_date: str,
        train_end: str,
        val_end: str,
        output_dir: str = 'data',
        # Technical feature config (WEEKLY periods)
        sma_windows: List[int] = [4, 8, 12],  # Weeks
        ema_windows: List[int] = [8, 12],     # Weeks
        return_lags: int = 3,                 # Weeks
        rsi_period: int = 14,                 # Weeks
        macd_fast: int = 12,                  # Weeks
        macd_slow: int = 26,                  # Weeks
        macd_signal: int = 9,                 # Weeks
        stochastic_period: int = 14,          # Weeks
        bb_period: int = 20,                  # Weeks
        atr_period: int = 14,                 # Weeks
        # Macro config
        macro_forward_fill_limit: int = 2     # Weeks
    ):
        """Initialize data preparator with configuration."""
        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.train_end = pd.to_datetime(train_end)
        self.val_end = pd.to_datetime(val_end)
        self.output_dir = Path(output_dir)

        # Technical features config (all in WEEKS)
        self.sma_windows = sma_windows
        self.ema_windows = ema_windows
        self.return_lags = return_lags
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.stochastic_period = stochastic_period
        self.bb_period = bb_period
        self.atr_period = atr_period

        # Macro config
        self.macro_forward_fill_limit = macro_forward_fill_limit

        # Columns that should NOT be normalized
        self.non_norm_cols = [
            'ticker',  # Categorical
            # Binary flags (already 0 or 1)
            'is_month_end', 'is_quarter_end', 'is_year_end',
            # Cyclical encodings (already in [-1, 1])
            'month_sin', 'month_cos', 'week_sin', 'week_cos',
            # Categorical
            'month', 'week_of_year',
            # Time-tracking (absolute, not relative) - EXACT names used below
            'days_since_pceinflation_update', 'days_since_unemployment_update'
        ]

        # Create output directories
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
        (self.output_dir / 'test').mkdir(exist_ok=True)

        # Storage
        self.raw_technical: Optional[pd.DataFrame] = None
        self.macro_calendar: Optional[pd.DataFrame] = None
        self.normalization_params: Dict = {}

        # Cached benchmark weekly (close/return)
        self.bench_weekly: Optional[pd.DataFrame] = None

        print(f"\n{'='*70}")
        print("DATA PREPARATION - WEEKLY (Leak-Safe)")
        print(f"{'='*70}")
        print(f"Tickers: {tickers}")
        print(f"Benchmark: {benchmark}")
        print(f"Date range: {start_date} to {end_date}")
        print(f"Train: {start_date} to {train_end}")
        print(f"Val: {train_end} to {val_end}")
        print(f"Test: {val_end} to {end_date}")
        print(f"Output: {self.output_dir}")
        print(f"Frequency: WEEKLY (Friday close)")
        print(f"{'='*70}\n")

    def prepare_all(self):
        """Run complete data preparation pipeline."""
        # Step 1: Prepare technical features (WEEKLY)
        print("\n[1/6] Preparing WEEKLY technical features...")
        self.prepare_technical_features()

        # Step 1.5: Align tickers to common date range
        print("\n[1.5/6] Aligning tickers to common date range...")
        self.align_tickers()

        # Step 2: Prepare macro + calendar features (WEEKLY)
        print("\n[2/6] Preparing WEEKLY macro and calendar features...")
        self.prepare_macro_calendar()

        # Step 3: Split data
        print("\n[3/6] Splitting data by train/val/test...")
        splits = self.split_data()

        # Step 4: Normalize data
        print("\n[4/6] Normalizing splits (fit on train only)...")
        normalized_splits = self.normalize_splits(splits)

        # Step 5: Save everything
        print("\n[5/6] Saving datasets and metadata...")
        self.save_all(normalized_splits)

        print(f"\n{'='*70}")
        print("✓ DATA PREPARATION COMPLETE")
        print(f"{'='*70}")
        print(f"Raw technical: {self.output_dir / 'raw_technical.csv'}")
        print(f"Macro calendar: {self.output_dir / 'macro_calendar.csv'}")
        print(f"Normalized splits: {self.output_dir / 'train|val|test'}")
        print(f"Per-split prices/returns: train|val|test/prices.csv & log_returns.csv")
        print(f"Metadata: {self.output_dir / 'metadata.json'}")
        print(f"Also: technical_indicators_weekly.csv & macro_weekly.csv (normalized, split-tagged)")
        print(f"{'='*70}\n")

    def prepare_technical_features(self):
        """Load assets and create technical features on WEEKLY data (resample first)."""
        raw_path = self.output_dir / 'raw_technical.csv'
        if raw_path.exists():
            self.raw_technical = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            print("  ✓ Loaded cached technical data")
            return self.raw_technical

        all_data = []

        # Download benchmark
        print(f"  Downloading benchmark {self.benchmark}...")
        bench_weekly = self._download_and_resample(self.benchmark, is_benchmark=True)
        self.bench_weekly = bench_weekly

        for ticker in self.tickers:
            print(f"  Processing {ticker}...")
            try:
                weekly_df = self._download_and_resample(ticker)
                if weekly_df is None or weekly_df.empty:
                    print(f"    ⚠ No data for {ticker}, skipping")
                    continue

                weekly_df = self._calculate_technical_indicators_weekly(weekly_df)

                # Relative to benchmark
                if bench_weekly is not None:
                    common_dates = weekly_df.index.intersection(bench_weekly.index)
                    if len(common_dates) > 12:
                        asset_rets = weekly_df.loc[common_dates, 'return']
                        bench_rets = bench_weekly.loc[common_dates, 'return']
                        weekly_df.loc[common_dates, 'bench_corr_12w'] = asset_rets.rolling(12).corr(bench_rets)
                        cov = asset_rets.rolling(12).cov(bench_rets)
                        var = bench_rets.rolling(12).var()
                        weekly_df.loc[common_dates, 'bench_beta_12w'] = cov / (var + 1e-8)

                weekly_df['ticker'] = ticker
                all_data.append(weekly_df)

            except Exception as e:
                print(f"    ⚠ Error processing {ticker}: {e}")
                continue

        if not all_data:
            raise ValueError("No data successfully downloaded for any ticker!")

        self.raw_technical = pd.concat(all_data, axis=0)

        # Filter to range & sort
        self.raw_technical = self.raw_technical[
            (self.raw_technical.index >= self.start_date) &
            (self.raw_technical.index <= self.end_date)
        ].sort_index()

        print(f"\n  ✓ Created WEEKLY technical features: {self.raw_technical.shape}")
        print(f"  Features ({len(self.raw_technical.columns)} total):")
        for col in sorted(self.raw_technical.columns):
            print(f"    - {col}")

        print(f"\n  Checking ticker coverage...")
        for ticker in self.tickers:
            ticker_data = self.raw_technical[self.raw_technical['ticker'] == ticker]
            if len(ticker_data) > 0:
                start = ticker_data.index.min()
                end = ticker_data.index.max()
                weeks = len(ticker_data)
                print(f"    {ticker}: {weeks} weeks ({start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')})")

        self.raw_technical.to_csv(raw_path)
        return self.raw_technical

    def align_tickers(self):
        """Align all tickers to common date range (latest common start, earliest common end)."""
        if self.raw_technical is None:
            raise ValueError("Must call prepare_technical_features first")

        ticker_ranges = {}
        for ticker in self.tickers:
            ticker_data = self.raw_technical[self.raw_technical['ticker'] == ticker]
            if len(ticker_data) > 0:
                ticker_ranges[ticker] = {
                    'start': ticker_data.index.min(),
                    'end': ticker_data.index.max(),
                    'weeks': len(ticker_data)
                }

        latest_start = max(info['start'] for info in ticker_ranges.values())
        earliest_end = min(info['end'] for info in ticker_ranges.values())

        print(f"\n  Ticker date ranges:")
        for ticker, info in ticker_ranges.items():
            print(f"    {ticker}: {info['start'].strftime('%Y-%m-%d')} to {info['end'].strftime('%Y-%m-%d')} ({info['weeks']} weeks)")

        print(f"\n  Common coverage: {latest_start.strftime('%Y-%m-%d')} to {earliest_end.strftime('%Y-%m-%d')}")

        before_rows = len(self.raw_technical)
        self.raw_technical = self.raw_technical[
            (self.raw_technical.index >= latest_start) &
            (self.raw_technical.index <= earliest_end)
        ]
        after_rows = len(self.raw_technical)

        if before_rows - after_rows > 0:
            print(f"  ⚠ Dropped {before_rows - after_rows} rows to align tickers")

        weeks_per_ticker = self.raw_technical.groupby('ticker').size()
        if weeks_per_ticker.nunique() == 1:
            print(f"  ✓ All tickers aligned: {weeks_per_ticker.iloc[0]} weeks each")
        else:
            print(f"  ⚠ Tickers still have different week counts:")
            for ticker, count in weeks_per_ticker.items():
                print(f"    {ticker}: {count} weeks")

        return self.raw_technical

    def _download_and_resample(self, ticker: str, is_benchmark: bool = False) -> Optional[pd.DataFrame]:
        """Download daily data and resample to WEEKLY."""
        try:
            warmup = pd.Timedelta(days=365)  # 1 year warmup
            df = yf.download(
                ticker,
                start=self.start_date - warmup,
                end=self.end_date,
                progress=False
            )
            if df.empty:
                return None

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

            weekly = pd.DataFrame()
            weekly['open'] = df['Open'].resample('W-FRI').first()
            weekly['high'] = df['High'].resample('W-FRI').max()
            weekly['low'] = df['Low'].resample('W-FRI').min()
            weekly['close'] = df[close_col].resample('W-FRI').last()
            weekly['volume'] = df['Volume'].resample('W-FRI').sum()

            weekly['return'] = np.log(weekly['close'] / weekly['close'].shift(1))
            weekly = weekly.dropna()
            return weekly

        except Exception as e:
            if not is_benchmark:
                raise e
            return None

    def _calculate_technical_indicators_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on WEEKLY data."""
        data = df.copy()
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        returns = data['return']

        # TREND
        for window in self.sma_windows:
            sma = close.rolling(window=window).mean()
            data[f'price_to_sma_{window}w'] = close / (sma + 1e-10)

        for window in self.ema_windows:
            ema = close.ewm(span=window, adjust=False).mean()
            data[f'price_to_ema_{window}w'] = close / (ema + 1e-10)

        ema_fast = close.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        data['macd_hist'] = macd_line - macd_signal

        # MOMENTUM
        for lag in range(1, self.return_lags + 1):
            data[f'return_lag_{lag}w'] = returns.shift(lag)

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        data['rsi'] = 100 - (100 / (1 + rs))

        low_n = low.rolling(window=self.stochastic_period).min()
        high_n = high.rolling(window=self.stochastic_period).max()
        data['stoch_k'] = 100 * (close - low_n) / (high_n - low_n + 1e-10)

        data['roc_4w'] = close.pct_change(periods=4) * 100

        # VOLATILITY
        data['volatility'] = returns.rolling(window=12).std() * np.sqrt(52)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        data['atr_pct'] = (atr / (close + 1e-10)) * 100

        bb_middle = close.rolling(window=self.bb_period).mean()
        bb_std = close.rolling(window=self.bb_period).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        data['bb_position'] = (close - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # VOLUME
        vol_ma = volume.rolling(window=20).mean()
        data['volume_ratio'] = volume / (vol_ma + 1e-10)

        typical_price = (high + low + close) / 3
        raw_money_flow = typical_price * volume
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        positive_mf = positive_flow.rolling(window=14).sum()
        negative_mf = negative_flow.rolling(window=14).sum()
        money_ratio = positive_mf / (negative_mf + 1e-10)
        data['mfi'] = 100 - (100 / (1 + money_ratio))

        obv = (np.sign(returns) * volume).cumsum()
        data['obv_roc_4w'] = obv.pct_change(periods=4) * 100

        return data

    def prepare_macro_calendar(self):
        """Load macro indicators and create calendar features (WEEKLY)."""
        macro_path = self.output_dir / 'macro_calendar.csv'
        if macro_path.exists():
            self.macro_calendar = pd.read_csv(macro_path, index_col=0, parse_dates=True)
            print("  ✓ Loaded cached macro calendar")
            return self.macro_calendar

        print("  Fetching macro indicators...")
        macro_daily = self._get_macro_indicators()

        print("  Creating calendar features...")
        calendar_daily = self._build_calendar_features()

        combined_daily = pd.concat([macro_daily, calendar_daily], axis=1)

        print("  Resampling to weekly...")
        self.macro_calendar = self._resample_macro_to_weekly(combined_daily)

        self.macro_calendar = self.macro_calendar[
            (self.macro_calendar.index >= self.start_date) &
            (self.macro_calendar.index <= self.end_date)
        ]

        print(f"\n  ✓ Created WEEKLY macro/calendar features: {self.macro_calendar.shape}")
        print(f"  Features ({len(self.macro_calendar.columns)} total):")
        for col in sorted(self.macro_calendar.columns):
            print(f"    - {col}")

        self.macro_calendar.to_csv(macro_path)
        return self.macro_calendar

    def _get_macro_indicators(self) -> pd.DataFrame:
        """Fetch macro indicators (daily), optimized for tech/semis; resampled later."""
        warmup = pd.Timedelta(days=365)
        start = self.start_date - warmup
        end = self.end_date

        macro_data = {}

        # Market indicators
        print("    Downloading market indicators...")
        try:
            vxn = yf.download('^VXN', start=start, end=end, progress=False)
            if not vxn.empty:
                if isinstance(vxn.columns, pd.MultiIndex):
                    vxn.columns = vxn.columns.get_level_values(0)
                macro_data['nasdaq_vol'] = vxn['Close'].squeeze()
        except Exception as e:
            print(f"      ⚠ VXN: {e}")

        try:
            tnx = yf.download('^TNX', start=start, end=end, progress=False)
            if not tnx.empty:
                if isinstance(tnx.columns, pd.MultiIndex):
                    tnx.columns = tnx.columns.get_level_values(0)
                macro_data['treasury_10y'] = tnx['Close'].squeeze()
        except Exception as e:
            print(f"      ⚠ TNX: {e}")

        try:
            dxy = yf.download('DX-Y.NYB', start=start, end=end, progress=False)
            if not dxy.empty:
                if isinstance(dxy.columns, pd.MultiIndex):
                    dxy.columns = dxy.columns.get_level_values(0)
                macro_data['dollar_index'] = dxy['Close'].squeeze()
        except Exception as e:
            print(f"      ⚠ DXY: {e}")

        try:
            smh = yf.download('SMH', start=start, end=end, progress=False)
            if not smh.empty:
                if isinstance(smh.columns, pd.MultiIndex):
                    smh.columns = smh.columns.get_level_values(0)
                macro_data['semiconductor_etf'] = smh['Close'].squeeze()
        except Exception as e:
            print(f"      ⚠ SMH: {e}")

        # FRED
        print("    Downloading FRED economic data...")
        fred_series = {
            'DFF': 'fed_funds_rate',
            'UNRATE': 'unemployment',
            'PCEPI': 'pce_inflation',
            'T10Y2Y': 'yield_curve',
            'BAMLH0A0HYM2': 'credit_spread',
            'TEDRATE': 'ted_spread',
        }

        for series_id, col_name in fred_series.items():
            try:
                series = pdr.get_data_fred([series_id], start=start, end=end)
                if not series.empty:
                    macro_data[col_name] = series[series_id]
            except Exception as e:
                print(f"      ⚠ {series_id}: {e}")

        if not macro_data:
            return pd.DataFrame(index=pd.date_range(start, end, freq='D'))

        df = pd.DataFrame(macro_data)

        # Forward fill with limits
        high_freq = ['nasdaq_vol', 'treasury_10y', 'dollar_index', 'semiconductor_etf',
                     'fed_funds_rate', 'yield_curve', 'credit_spread', 'ted_spread']
        low_freq = ['unemployment', 'pce_inflation']

        for col in df.columns:
            if col in high_freq:
                df[col] = df[col].ffill(limit=7)   # ~1 week
            elif col in low_freq:
                df[col] = df[col].ffill(limit=35)  # ~5 weeks

        # Regime indicators (z-scores)
        if 'nasdaq_vol' in df.columns:
            vol_ma = df['nasdaq_vol'].rolling(252, min_periods=60).mean()
            vol_std = df['nasdaq_vol'].rolling(252, min_periods=60).std()
            df['vol_regime'] = (df['nasdaq_vol'] - vol_ma) / (vol_std + 1e-8)

        if 'credit_spread' in df.columns:
            spread_ma = df['credit_spread'].rolling(252, min_periods=60).mean()
            spread_std = df['credit_spread'].rolling(252, min_periods=60).std()
            df['credit_regime'] = (df['credit_spread'] - spread_ma) / (spread_std + 1e-8)

        return df

    def _resample_macro_to_weekly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resample macro data to weekly and add weekly-specific features."""
        weekly = df.resample('W-FRI').last()

        # Staleness tracking for low-frequency indicators
        low_freq_cols = ['unemployment', 'pce_inflation']
        name_map = {
            'unemployment': 'days_since_unemployment_update',
            'pce_inflation': 'days_since_pceinflation_update'
        }

        for col in low_freq_cols:
            if col in weekly.columns:
                actual_updates = (~weekly[col].isnull()) & (weekly[col] != weekly[col].shift(1))
                days_since = []
                update_dates = weekly.index[actual_updates]
                for date in weekly.index:
                    if len(update_dates) == 0 or date < update_dates[0]:
                        days_since.append(0)
                    else:
                        recent = update_dates[update_dates <= date]
                        days_since.append((date - recent[-1]).days if len(recent) > 0 else 0)
                weekly[name_map[col]] = days_since

        # FFill remaining NaNs with limits
        for col in weekly.columns:
            if col in ['unemployment', 'pce_inflation']:
                weekly[col] = weekly[col].ffill(limit=8)
            elif col not in ['week_of_year', 'week_sin', 'week_cos']:
                weekly[col] = weekly[col].ffill(limit=4)

        # Weekly calendar
        dates = weekly.index
        weekly['week_of_year'] = dates.isocalendar().week
        weekly['week_sin'] = np.sin(2 * np.pi * weekly['week_of_year'] / 52)
        weekly['week_cos'] = np.cos(2 * np.pi * weekly['week_of_year'] / 52)

        return weekly

    def _build_calendar_features(self) -> pd.DataFrame:
        """Create calendar features (daily, resampled later)."""
        dates = pd.date_range(self.start_date - pd.Timedelta(days=365),
                              self.end_date, freq='D')
        calendar = pd.DataFrame(index=dates)
        calendar['month'] = dates.month
        calendar['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
        calendar['month_cos'] = np.cos(2 * np.pi * dates.month / 12)
        calendar['is_month_end'] = dates.is_month_end.astype(int)
        calendar['is_quarter_end'] = dates.is_quarter_end.astype(int)
        calendar['is_year_end'] = dates.is_year_end.astype(int)
        return calendar

    def split_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split by train/val/test."""
        splits = {}
        tech_train = self.raw_technical[self.raw_technical.index <= self.train_end].copy()
        macro_train = self.macro_calendar[self.macro_calendar.index <= self.train_end].copy()
        splits['train'] = (tech_train, macro_train)

        tech_val = self.raw_technical[
            (self.raw_technical.index > self.train_end) &
            (self.raw_technical.index <= self.val_end)
        ].copy()
        macro_val = self.macro_calendar[
            (self.macro_calendar.index > self.train_end) &
            (self.macro_calendar.index <= self.val_end)
        ].copy()
        splits['val'] = (tech_val, macro_val)

        tech_test = self.raw_technical[self.raw_technical.index > self.val_end].copy()
        macro_test = self.macro_calendar[self.macro_calendar.index > self.val_end].copy()
        splits['test'] = (tech_test, macro_test)

        for split_name in ['train', 'val', 'test']:
            tech, macro = splits[split_name]
            print(f"  {split_name}: technical={tech.shape}, macro={macro.shape}")
            if split_name == 'train' and tech.empty:
                raise ValueError("Train split is empty! Check your date ranges.")
        return splits

    def normalize_splits(
        self,
        splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """Normalize using TRAIN statistics only (no data leakage)."""
        train_tech, train_macro = splits['train']

        # Technical: drop NaN rows (indicator warmups)
        print("\n  Checking technical data for NaNs...")
        if train_tech.isnull().sum().sum() > 0:
            print("  ⚠ NaNs in train technical - dropping rows")
            train_tech = train_tech.dropna()
            splits['train'] = (train_tech, train_macro)
        else:
            print("  ✓ No NaNs in train technical")

        # Macro: impute (don't drop)
        print("\n  Checking macro data for NaNs...")
        if train_macro.isnull().sum().sum() > 0:
            print("  ⚠ NaNs in train macro - imputing (ffill/bfill/mean)")
            train_macro = train_macro.ffill(limit=12).bfill(limit=4).fillna(train_macro.mean())
            if train_macro.isnull().sum().sum() > 0:
                cols_with_nans = train_macro.columns[train_macro.isnull().any()].tolist()
                print(f"    Dropping cols with remaining NaNs: {cols_with_nans}")
                train_macro = train_macro.drop(columns=cols_with_nans)
            splits['train'] = (train_tech, train_macro)
        else:
            print("  ✓ No NaNs in train macro")

        # Clean val/test by same rules
        for split_name in ['val', 'test']:
            split_tech, split_macro = splits[split_name]
            if split_tech.isnull().sum().sum() > 0:
                print(f"  ⚠ NaNs in {split_name} technical - dropping rows")
                split_tech = split_tech.dropna()
            if split_macro.isnull().sum().sum() > 0:
                print(f"  ⚠ NaNs in {split_name} macro - imputing")
                split_macro = split_macro.ffill(limit=12).bfill(limit=4).fillna(train_macro.mean())
                cols_to_drop = [c for c in split_macro.columns if c not in train_macro.columns]
                if cols_to_drop:
                    split_macro = split_macro.drop(columns=cols_to_drop)
            splits[split_name] = (split_tech, split_macro)

        # Determine columns to normalize
        train_tech, train_macro = splits['train']
        tech_cols = [c for c in train_tech.columns if c not in self.non_norm_cols]
        macro_cols = [c for c in train_macro.columns if c not in self.non_norm_cols]

        print(f"\n  Normalizing {len(tech_cols)} technical features")
        print(f"  Normalizing {len(macro_cols)} macro features")
        print(f"  Excluding {len(self.non_norm_cols)} non-normalizable columns")

        tech_means = train_tech[tech_cols].mean()
        tech_stds = train_tech[tech_cols].std()
        macro_means = train_macro[macro_cols].mean()
        macro_stds = train_macro[macro_cols].std()

        self.normalization_params = {
            'technical': {
                'mean': {k: float(v) for k, v in tech_means.items()},
                'std': {k: float(v) for k, v in tech_stds.items()},
                'columns': tech_cols
            },
            'macro': {
                'mean': {k: float(v) for k, v in macro_means.items()},
                'std': {k: float(v) for k, v in macro_stds.items()},
                'columns': macro_cols
            }
        }

        normalized = {}
        for split_name, (tech_df, macro_df) in splits.items():
            print(f"  Applying normalization to {split_name}...")
            tech_norm = tech_df.copy()
            for c in tech_cols:
                if c in tech_norm:
                    tech_norm[c] = (tech_df[c] - tech_means[c]) / (tech_stds[c] + 1e-8)
            macro_norm = macro_df.copy()
            for c in macro_cols:
                if c in macro_norm:
                    macro_norm[c] = (macro_df[c] - macro_means[c]) / (macro_stds[c] + 1e-8)
            normalized[split_name] = (tech_norm, macro_norm)

        print("\n  Final NaN check...")
        for split_name, (tech, macro) in normalized.items():
            print(f"  ✓ {split_name}: {tech.isnull().sum().sum()} tech NaNs, {macro.isnull().sum().sum()} macro NaNs")
        return normalized

    def _split_label(self, idx: pd.DatetimeIndex) -> pd.Series:
        """Split labels for a DatetimeIndex."""
        lab = pd.Series(index=idx, dtype="string")
        lab.loc[idx <= self.train_end] = "train"
        lab.loc[(idx > self.train_end) & (idx <= self.val_end)] = "val"
        lab.loc[idx > self.val_end] = "test"
        return lab

    def save_all(self, normalized_splits: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]):
        """Save raw/normalized data + per-split prices/returns + metadata (no leakage)."""
        # Raw caches
        self.raw_technical.to_csv(self.output_dir / 'raw_technical.csv')
        self.macro_calendar.to_csv(self.output_dir / 'macro_calendar.csv')
        print("  ✓ Saved raw data")

        # Normalized per-split technical + macro
        for split_name, (tech, macro) in normalized_splits.items():
            split_dir = self.output_dir / split_name
            tech.round(4).to_csv(split_dir / 'technical.csv')
            macro.round(4).to_csv(split_dir / 'macro_calendar.csv')
            print(f"  ✓ Saved {split_name} split (technical + macro)")

        # Concatenated normalized (with split column) for tech/macro
        def _cat_norm(kind: str) -> pd.DataFrame:
            parts = []
            for split_name, (tech_df, macro_df) in normalized_splits.items():
                df = tech_df if kind == "tech" else macro_df
                tmp = df.copy()
                tmp.insert(0, "split", split_name)
                parts.append(tmp)
            return pd.concat(parts, axis=0).sort_index()

        technical_norm_all = _cat_norm("tech")
        macro_norm_all = _cat_norm("macro")
        technical_norm_all.to_csv(self.output_dir / 'technical_indicators_weekly.csv')
        macro_norm_all.to_csv(self.output_dir / 'macro_weekly.csv')
        print("  ✓ Saved technical_indicators_weekly.csv and macro_weekly.csv (normalized, split-tagged)")

        # Build weekly wide PRICES and LOG RETURNS once, then slice by split
        df_rt = self.raw_technical.copy()
        if df_rt.index.name is None:
            df_rt.index.name = 'date'
        df_rt = df_rt.reset_index()

        keep_cols = ['date', 'ticker', 'close', 'return']
        missing = [c for c in keep_cols if c not in df_rt.columns]
        if missing:
            raise ValueError(f"Missing expected columns for prices/returns: {missing}")

        # Deduplicate (date, ticker) robustly
        df_rt = (
            df_rt[keep_cols]
            .sort_values(['date', 'ticker'])
            .groupby(['date', 'ticker'], as_index=False, sort=False)
            .last()
        )

        prices_wide = pd.pivot_table(
            df_rt, index='date', columns='ticker', values='close', aggfunc='last'
        ).sort_index()

        returns_wide = pd.pivot_table(
            df_rt, index='date', columns='ticker', values='return', aggfunc='last'
        ).sort_index()

        # Append benchmark to both, if available
        if self.bench_weekly is not None and not self.bench_weekly.empty:
            bench_p = self.bench_weekly['close'].rename(self.benchmark)
            bench_r = self.bench_weekly['return'].rename(self.benchmark)
            prices_wide = prices_wide.join(bench_p, how='outer')
            returns_wide = returns_wide.join(bench_r, how='outer')

        # Align span to technical window (post-alignment)
        min_idx = max(prices_wide.index.min(), self.raw_technical.index.min())
        max_idx = min(prices_wide.index.max(), self.raw_technical.index.max())
        prices_wide = prices_wide.loc[(prices_wide.index >= min_idx) & (prices_wide.index <= max_idx)]
        returns_wide = returns_wide.reindex(prices_wide.index)

        # Helper: mask by split boundaries
        def _mask(idx, split: str):
            if split == 'train':
                return idx <= self.train_end
            elif split == 'val':
                return (idx > self.train_end) & (idx <= self.val_end)
            return idx > self.val_end

        # Save per-split prices/log_returns/ew_returns (no global combined files)
        for split_name in ['train', 'val', 'test']:
            m = _mask(prices_wide.index, split_name)
            split_dir = self.output_dir / split_name
            split_dir.mkdir(exist_ok=True)

            prices_split = prices_wide.loc[m].copy()
            prices_split.to_csv(split_dir / 'prices.csv')

            returns_split = returns_wide.loc[m].copy()
            returns_split.to_csv(split_dir / 'log_returns.csv')

            asset_cols = [t for t in self.tickers if t in returns_split.columns]
            ew = returns_split[asset_cols].mean(axis=1).rename('ew_log_ret')

            if self.benchmark in returns_split.columns:
                qqq = returns_split[self.benchmark].rename('qqq_log_ret')
            elif self.bench_weekly is not None and not self.bench_weekly.empty:
                qqq = self.bench_weekly['return'].reindex(returns_split.index).rename('qqq_log_ret')
            else:
                qqq = pd.Series(index=returns_split.index, dtype=float, name='qqq_log_ret')

            ew_df = pd.concat([ew, qqq], axis=1).dropna()
            ew_df['excess_ew_over_qqq'] = ew_df['ew_log_ret'] - ew_df['qqq_log_ret']
            ew_df.to_csv(split_dir / 'ew_returns.csv')

            print(f"  ✓ Saved {split_name}/prices.csv, {split_name}/log_returns.csv, {split_name}/ew_returns.csv")

        # Metadata
        metadata = {
            'tickers': self.tickers,
            'benchmark': self.benchmark,
            'frequency': 'weekly',
            'dates': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d'),
                'train_end': self.train_end.strftime('%Y-%m-%d'),
                'val_end': self.val_end.strftime('%Y-%m-%d')
            },
            'features': {
                'technical': list(self.raw_technical.columns),
                'macro_calendar': list(self.macro_calendar.columns),
                'technical_count': int(len(self.raw_technical.columns)),
                'macro_count': int(len(self.macro_calendar.columns))
            },
            'normalization': self.normalization_params,
            'created_at': datetime.now().isoformat()
        }
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print("  ✓ Saved metadata")

        print("\n  Split-scoped market matrices written:")
        for s in ['train', 'val', 'test']:
            print(f"   - {s}/prices.csv, {s}/log_returns.csv, {s}/ew_returns.csv")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Prepare WEEKLY data for RL pipeline (leak-safe)')
    parser.add_argument('--tickers', nargs='+', default=['NVDA', 'MU', 'AMD', 'ASML', 'MSFT', 'GOOG', 'AI'])
    parser.add_argument('--benchmark', default='QQQ')
    parser.add_argument('--start', default='2018-01-01')
    parser.add_argument('--end', default='2025-10-01')
    parser.add_argument('--train-end', default='2023-12-31')
    parser.add_argument('--val-end', default='2024-06-30')
    parser.add_argument('--output-dir', default='data')
    args = parser.parse_args()

    preparator = DataPreparator(
        tickers=args.tickers,
        benchmark=args.benchmark,
        start_date=args.start,
        end_date=args.end,
        train_end=args.train_end,
        val_end=args.val_end,
        output_dir=args.output_dir
    )
    preparator.prepare_all()


if __name__ == '__main__':
    main()
