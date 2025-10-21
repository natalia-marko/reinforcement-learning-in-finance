

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas_datareader.data as web
except ImportError:
    web = None

try:
    import gymnasium as gym
except ImportError:
    gym = None


def load_price_data(
    tickers: List[str],
    start: str | _dt.date,
    end: Optional[str | _dt.date] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download daily price data for a list of tickers."""
    if yf is None:
        raise ImportError("yfinance required. Install: pip install yfinance")

    if end is None:
        end = _dt.date.today()

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    frames = []
    successful_tickers = []
    for ticker in tickers:
        try:
            if ticker in data.columns.get_level_values(0) and not data[ticker].empty:
                df = data[ticker].copy()
                df.columns = [c.lower() for c in df.columns]
                
                if 'adj_close' not in df.columns and 'close' in df.columns:
                    df['adj_close'] = df['close']
                
                df['ticker'] = ticker
                frames.append(df)
                successful_tickers.append(ticker)
            else:
                print(f"Warning: No data available for {ticker}")
        except Exception as e:
            print(f"Warning: Failed to process {ticker}: {e}")
    
    if not frames:
        raise ValueError("No tickers could be downloaded successfully")
    
    all_df = pd.concat(frames)
    all_df.index.name = 'date'
    return all_df.reset_index().set_index(['date', 'ticker'])


def resample_to_weekly(prices: pd.DataFrame, how: str = 'last') -> pd.DataFrame:
    """Aggregate daily prices to weekly frequency (ending Friday)."""
    if not isinstance(prices.index, pd.MultiIndex):
        raise ValueError("prices must have MultiIndex: ['date','ticker']")

    frames = []
    for ticker, df in prices.groupby(level='ticker'):
        df = df.reset_index(level='ticker', drop=True)
        
        ohlc = {}
        if 'open' in df.columns:
            ohlc['open'] = 'first' if how == 'last' else 'mean'
        if 'high' in df.columns:
            ohlc['high'] = 'max' if how == 'last' else 'mean'
        if 'low' in df.columns:
            ohlc['low'] = 'min' if how == 'last' else 'mean'
        if 'close' in df.columns:
            ohlc['close'] = 'last' if how == 'last' else 'mean'
        if 'adj_close' in df.columns:
            ohlc['adj_close'] = 'last' if how == 'last' else 'mean'
        if 'volume' in df.columns:
            ohlc['volume'] = 'sum'
            
        weekly = df.resample('W-FRI').agg(ohlc)
        weekly['ticker'] = ticker
        frames.append(weekly)
    
    weekly_df = pd.concat(frames)
    weekly_df.index.name = 'date'
    return weekly_df.reset_index().set_index(['date', 'ticker'])


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns of a price series."""
    return np.log(prices / prices.shift(1))


def compute_technical_indicators(
    weekly_prices: pd.DataFrame,
    windows: List[int] | None = None,
    include_returns: bool = True,
    include_lags: int = 3,
    benchmark_data: pd.DataFrame = None,
    sector_data: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute 35 technical indicators with LAGGED relative performance.
    """
    if windows is None:
        windows = [4, 8, 12]
    
    frames = []
    
    for ticker, df in weekly_prices.groupby(level='ticker'):
        df = df.reset_index(level='ticker', drop=True).copy()
        feat = pd.DataFrame(index=df.index)
        
        # SMA ratios and volatility
        for w in windows:
            sma = df['adj_close'].rolling(window=w).mean()
            feat[f'sma_{w}_ratio'] = df['adj_close'] / sma
            
            returns = df['adj_close'].pct_change()
            feat[f'vol_{w}'] = returns.rolling(window=w).std()
            
            vol_sma = feat[f'vol_{w}'].rolling(window=w).mean()
            feat[f'vol_sma_{w}_ratio'] = feat[f'vol_{w}'] / (vol_sma + 1e-8)
        
        # RSI
        delta = df['adj_close'].diff()
        up = delta.clip(lower=0).rolling(window=14).mean()
        down = -delta.clip(upper=0).rolling(window=14).mean()
        rsi = 100 - 100 / (1 + up / (down + 1e-9))
        feat['rsi'] = rsi / 100
        
        # Volume-Price Trend
        if 'volume' in df.columns:
            vpt = (df['volume'] * df['adj_close'].pct_change()).cumsum()
            feat['vpt'] = vpt / (vpt.rolling(window=52).std() + 1e-8)
        
        # On-Balance Volume
        if 'volume' in df.columns:
            obv = (df['volume'] * np.sign(df['adj_close'].diff())).cumsum()
            feat['obv'] = obv / (obv.rolling(window=52).std() + 1e-8)
        
        # MACD
        ema_12 = df['adj_close'].ewm(span=12).mean()
        ema_26 = df['adj_close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        feat['macd'] = macd / df['adj_close']
        feat['macd_signal'] = signal / df['adj_close']
        feat['macd_histogram'] = (macd - signal) / df['adj_close']
        
        # Momentum
        for w in windows:
            feat[f'momentum_{w}'] = df['adj_close'].pct_change(w)
        
        # Bollinger Bands
        bb_window = 20
        sma_20 = df['adj_close'].rolling(window=bb_window).mean()
        std_20 = df['adj_close'].rolling(window=bb_window).std()
        
        feat['bb_upper'] = (sma_20 + 2 * std_20) / df['adj_close']
        feat['bb_lower'] = (sma_20 - 2 * std_20) / df['adj_close']
        feat['bb_width'] = (2 * std_20) / (sma_20 + 1e-8)
        feat['bb_pct_b'] = (df['adj_close'] - (sma_20 - 2 * std_20)) / (4 * std_20 + 1e-8)
        
        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['adj_close'].shift())
            low_close = np.abs(df['low'] - df['adj_close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            feat['atr_normalized'] = atr / (df['adj_close'] + 1e-8)
        else:
            feat['atr_normalized'] = 0.0
        
        # Long-term trend
        sma_26 = df['adj_close'].rolling(window=26).mean()
        sma_52 = df['adj_close'].rolling(window=52).mean()
        
        feat['sma_26_ratio'] = df['adj_close'] / (sma_26 + 1e-8)
        feat['sma_52_ratio'] = df['adj_close'] / (sma_52 + 1e-8)
        
        rolling_high_52 = df['adj_close'].rolling(window=52).max()
        feat['dist_from_52w_high'] = (df['adj_close'] - rolling_high_52) / (rolling_high_52 + 1e-8)
        
        # Volume regime
        if 'volume' in df.columns:
            volume_sma_26 = df['volume'].rolling(window=26).mean()
            feat['volume_sma_ratio'] = df['volume'] / (volume_sma_26 + 1e-8)
            feat['volume_spike'] = (df['volume'] > 2 * volume_sma_26).astype(float)
        else:
            feat['volume_sma_ratio'] = 0.0
            feat['volume_spike'] = 0.0
        
        # Placeholders for relative performance (filled later)
        feat['return_vs_qqq'] = 0.0
        feat['return_vs_xlk'] = 0.0
        
        # Returns and lags
        if include_returns:
            returns = compute_log_returns(df['adj_close'])
            feat['ret'] = returns
            for i in range(1, include_lags + 1):
                feat[f'ret_lag_{i}'] = returns.shift(i)
        
        feat['ticker'] = ticker
        frames.append(feat)
    
    tech_df = pd.concat(frames)
    tech_df.index.name = 'date'
    tech_df = tech_df.reset_index().set_index(['date', 'ticker'])
    
    # FIX: LAGGED relative performance
    if benchmark_data is not None and not benchmark_data.empty:
        try:
            if isinstance(benchmark_data.index, pd.MultiIndex):
                qqq_series = benchmark_data.xs('QQQ', level='ticker')['adj_close']
            else:
                qqq_series = benchmark_data['adj_close']
            
            qqq_returns = compute_log_returns(qqq_series)
            
            if not isinstance(qqq_returns.index, pd.DatetimeIndex):
                qqq_returns.index = pd.to_datetime(qqq_returns.index)
            
            for ticker in tech_df.index.get_level_values('ticker').unique():
                ticker_data = tech_df.xs(ticker, level='ticker')
                aligned_qqq = qqq_returns.reindex(ticker_data.index, method='ffill')
                
                if 'ret' in ticker_data.columns:
                    ticker_returns_lagged = ticker_data['ret'].shift(1)
                    aligned_qqq_lagged = aligned_qqq.shift(1)
                    
                    tech_df.loc[(slice(None), ticker), 'return_vs_qqq'] = (
                        ticker_returns_lagged - aligned_qqq_lagged
                    ).values
        except Exception as e:
            print(f"  Warning: QQQ relative performance failed: {str(e)[:50]}")
    
    if sector_data is not None and not sector_data.empty:
        try:
            if isinstance(sector_data.index, pd.MultiIndex):
                xlk_series = sector_data.xs('XLK', level='ticker')['adj_close']
            else:
                xlk_series = sector_data['adj_close']
            
            xlk_returns = compute_log_returns(xlk_series)
            
            if not isinstance(xlk_returns.index, pd.DatetimeIndex):
                xlk_returns.index = pd.to_datetime(xlk_returns.index)
            
            for ticker in tech_df.index.get_level_values('ticker').unique():
                ticker_data = tech_df.xs(ticker, level='ticker')
                aligned_xlk = xlk_returns.reindex(ticker_data.index, method='ffill')
                
                if 'ret' in ticker_data.columns:
                    ticker_returns_lagged = ticker_data['ret'].shift(1)
                    aligned_xlk_lagged = aligned_xlk.shift(1)
                    
                    tech_df.loc[(slice(None), ticker), 'return_vs_xlk'] = (
                        ticker_returns_lagged - aligned_xlk_lagged
                    ).values
        except Exception as e:
            print(f"  Warning: XLK relative performance failed: {str(e)[:50]}")
    
    return tech_df.dropna()


def compute_technical_features_orchestrator(
    start: str,
    end: str,
    tickers: List[str],
    windows: List[int] | None = None,
    include_returns: bool = True,
    include_lags: int = 3,
) -> pd.DataFrame:
    """Orchestrator to compute all technical features with benchmarks."""
    print(f"Computing enhanced technical indicators for {len(tickers)} tickers...")
    
    all_prices = load_price_data(tickers, start, end)
    weekly = resample_to_weekly(all_prices)
    
    # Download QQQ benchmark
    try:
        qqq_data = load_price_data(['QQQ'], start, end)
        qqq_weekly = resample_to_weekly(qqq_data)
        qqq_weekly = qqq_weekly[qqq_weekly.index.get_level_values('ticker') == 'QQQ']
        print("  QQQ benchmark loaded")
    except Exception as e:
        print(f"  Warning: QQQ failed ({str(e)[:40]})")
        qqq_weekly = None
    
    # Download XLK sector
    try:
        xlk_data = load_price_data(['XLK'], start, end)
        xlk_weekly = resample_to_weekly(xlk_data)
        xlk_weekly = xlk_weekly[xlk_weekly.index.get_level_values('ticker') == 'XLK']
        print("  XLK sector loaded")
    except Exception as e:
        print(f"  Warning: XLK failed ({str(e)[:40]})")
        xlk_weekly = None
    
    tech_features = compute_technical_indicators(
        weekly,
        windows=windows,
        include_returns=include_returns,
        include_lags=include_lags,
        benchmark_data=qqq_weekly,
        sector_data=xlk_weekly
    )
    
    print(f"  Computed: {len(tech_features.columns)} features")
    return tech_features


def get_macro_indicators(start, end):
    """
    Fetch 15 continuous macro features (NO BINARY FLAGS).
    
    Features:
    - Raw FRED (5): VIX, Yield_Curve, HY_Spread, TED_Spread, Term_Spread
    - Market ratios (4): Tech_Leadership, Growth_Value, Risk_On_Off, Credit_Risk
    - Continuous regimes (3): VIX_Regime, Credit_Regime, Liquidity_Regime
    - Momentum (3): Curve_Momentum, Tech_Momentum, Risk_Momentum
    """
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.today()
    
    print(f"\nFetching macro indicators (continuous only)...")
    
    # FRED indicators
    fred_indicators = {
        'VIXCLS': 'VIX',
        'T10Y2Y': 'Yield_Curve',
        'BAMLH0A0HYM2': 'HY_Spread',
        'TEDRATE': 'TED_Spread',
        'T10Y3M': 'Term_Spread',
    }
    
    fred_data = {}
    for code, name in fred_indicators.items():
        try:
            series = web.DataReader(code, 'fred', start_dt, end_dt)
            series.columns = [name]
            fred_data[name] = series
            print(f"  {name}")
        except Exception as e:
            print(f"  Warning: {name} unavailable ({str(e)[:30]})")
    
    if not fred_data:
        print("Warning: No FRED data fetched!")
        macro_df = pd.DataFrame()
    else:
        macro_df = pd.concat(fred_data.values(), axis=1)
        macro_df = macro_df.resample('W-FRI').last()
        macro_df = macro_df.ffill(limit=4)  # Max 4 weeks
    
    # Market indicators
    print("\nFetching market indicators...")
    
    market_tickers = {
        'QQQ': 'NASDAQ', 'SPY': 'S&P 500',
        'XLK': 'Tech', 'XLV': 'Healthcare',
        'TLT': 'Treasury', 'HYG': 'HY Bonds',
    }
    
    market_data = {}
    for ticker, desc in market_tickers.items():
        try:
            data = yf.download(ticker, start=start_dt, end=end_dt, progress=False, auto_adjust=True)
            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    series = data[('Close', ticker)] if ('Close', ticker) in data.columns else data.iloc[:, 0]
                else:
                    series = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
                
                if isinstance(series, pd.Series) and len(series) > 0:
                    market_data[ticker] = series
                    print(f"  {ticker} ({desc})")
        except Exception as e:
            print(f"  Warning: {ticker} unavailable ({str(e)[:30]})")
    
    if market_data:
        market_df = pd.DataFrame(market_data)
        if not isinstance(market_df.index, pd.DatetimeIndex):
            market_df.index = pd.to_datetime(market_df.index)
        
        market_df = market_df.resample('W-FRI').last()
        market_df = market_df.ffill(limit=2)  # Max 2 weeks
        
        # Compute ratios
        if 'QQQ' in market_df.columns and 'SPY' in market_df.columns:
            macro_df['Tech_Leadership'] = market_df['QQQ'] / market_df['SPY']
        if 'XLK' in market_df.columns and 'XLV' in market_df.columns:
            macro_df['Growth_Value'] = market_df['XLK'] / market_df['XLV']
        if 'QQQ' in market_df.columns and 'TLT' in market_df.columns:
            macro_df['Risk_On_Off'] = market_df['QQQ'] / market_df['TLT']
        if 'HYG' in market_df.columns and 'TLT' in market_df.columns:
            macro_df['Credit_Risk'] = market_df['HYG'] / market_df['TLT']
    
    # Continuous regimes (NO binary flags)
    print("\nComputing continuous regimes...")
    df = macro_df.copy()
    
    if 'VIX' in df.columns:
        vix_ma = df['VIX'].rolling(52, min_periods=20).mean()
        vix_std = df['VIX'].rolling(52, min_periods=20).std()
        df['VIX_Regime'] = (df['VIX'] - vix_ma) / (vix_std + 1e-8)
    
    if 'HY_Spread' in df.columns:
        hy_ma = df['HY_Spread'].rolling(52, min_periods=20).mean()
        hy_std = df['HY_Spread'].rolling(52, min_periods=20).std()
        df['Credit_Regime'] = (df['HY_Spread'] - hy_ma) / (hy_std + 1e-8)
    
    if 'TED_Spread' in df.columns:
        ted_ma = df['TED_Spread'].rolling(52, min_periods=20).mean()
        ted_std = df['TED_Spread'].rolling(52, min_periods=20).std()
        df['Liquidity_Regime'] = (df['TED_Spread'] - ted_ma) / (ted_std + 1e-8)
    
    # Momentum
    if 'Yield_Curve' in df.columns:
        df['Curve_Momentum'] = df['Yield_Curve'].diff(4)
    if 'Tech_Leadership' in df.columns:
        df['Tech_Momentum'] = df['Tech_Leadership'].pct_change(4, fill_method=None) * 100
    if 'Risk_On_Off' in df.columns:
        df['Risk_Momentum'] = df['Risk_On_Off'].pct_change(4, fill_method=None) * 100
    
    # Select final 15 features
    final_features = [
        'VIX', 'Yield_Curve', 'HY_Spread', 'TED_Spread', 'Term_Spread',
        'Tech_Leadership', 'Growth_Value', 'Risk_On_Off', 'Credit_Risk',
        'VIX_Regime', 'Credit_Regime', 'Liquidity_Regime',
        'Curve_Momentum', 'Tech_Momentum', 'Risk_Momentum'
    ]
    
    df = df[[f for f in final_features if f in df.columns]].dropna(how='all')
    
    print(f"\nMacro features: {len(df.columns)} columns (all continuous)")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Rows: {len(df)}")
    
    return df


def build_calendar_frame(index, prefix='cal__'):
    """Build 4 calendar features for seasonality."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    features = pd.DataFrame(index=index)

    # Year cycle (sin/cos)
    woy = index.isocalendar().week.astype(int)
    woy = np.clip(woy, 1, 52)
    features[f"{prefix}year_sin"] = np.sin(2 * np.pi * woy / 52)
    features[f"{prefix}year_cos"] = np.cos(2 * np.pi * woy / 52)

    # Quarter end
    quarter_end_months = [3, 6, 9, 12]
    features[f"{prefix}quarterend"] = index.month.isin(quarter_end_months).astype(float)

    # Tax season
    features[f"{prefix}tax_season"] = index.month.isin([1, 2, 3, 4]).astype(float)

    return features.astype('float32')


def split_data(
    data: pd.DataFrame,
    train_start: str,
    train_end: str,
    val_end: str,
    test_end: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Split data into train/val/test."""
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)
    if test_end is not None:
        test_end = pd.to_datetime(test_end)
    else:
        test_end = data.index.get_level_values('date').max()

    mask_train = (data.index.get_level_values('date') >= train_start) & (
        data.index.get_level_values('date') <= train_end
    )
    mask_val = (data.index.get_level_values('date') > train_end) & (
        data.index.get_level_values('date') <= val_end
    )
    mask_test = (data.index.get_level_values('date') > val_end) & (
        data.index.get_level_values('date') <= test_end
    )

    return {
        'train': data.loc[mask_train],
        'val': data.loc[mask_val],
        'test': data.loc[mask_test],
    }


def normalise_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Normalise to zero mean and unit variance."""
    means = df.mean(skipna=True)
    stds = df.std(skipna=True) + 1e-8
    norm_df = (df - means) / stds
    return norm_df, means, stds


def apply_normalisation(df: pd.DataFrame, means: pd.Series, stds: pd.Series) -> pd.DataFrame:
    """Apply previously computed normalisation."""
    return (df - means) / (stds + 1e-8)


class PortfolioEnvWeekly(gym.Env):
    """Gymnasium environment for weekly portfolio rebalancing."""
    metadata = {'render_modes': ['human']}

    def __init__(self, features: np.ndarray, returns: np.ndarray, cost_rate: float = 0.0005, seed: Optional[int] = None) -> None:
        super().__init__()
        
        if gym is None:
            raise ImportError("gymnasium required. Install: pip install gymnasium")
        
        self.features = features
        self.returns = returns
        self.cost_rate = cost_rate
        self.n_steps, self.n_features, self.n_assets = features.shape
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_features, self.n_assets), dtype=np.float32)
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets, dtype=np.float32)
        self.current_step = 0
        self._rng = np.random.RandomState(seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.current_step = 0
        self.weights = np.array([1.0 / self.n_assets] * self.n_assets, dtype=np.float32)
        if seed is not None:
            self._rng.seed(seed)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, 0, 1)
        if action.sum() == 0:
            action = np.ones_like(action) / len(action)
        weights = action / action.sum()
        
        turnover = np.sum(np.abs(weights - self.weights))
        cost = self.cost_rate * turnover
        
        asset_ret = self.returns[self.current_step]
        port_ret = np.dot(weights, asset_ret)
        reward = port_ret - cost
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= self.n_steps
        obs = self._get_obs() if not done else None
        info = {'portfolio_return': port_ret, 'transaction_cost': cost}
        return obs, reward, done, False, info

    def _get_obs(self) -> np.ndarray:
        return self.features[self.current_step]

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Weights: {self.weights}")


class BaseRLAgent:
    """Base RL agent using Stable-Baselines3."""

    def __init__(self, algo='PPO', policy_kwargs=None, **algo_kwargs):
        from stable_baselines3 import PPO, SAC

        self.algo_name = algo.upper()
        if self.algo_name == 'PPO':
            self.Algo = PPO
        elif self.algo_name == 'SAC':
            self.Algo = SAC
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
            
        self.policy_kwargs = policy_kwargs or {}
        self.algo_kwargs = algo_kwargs
        self.model = None

    def train(self, env: PortfolioEnvWeekly, timesteps=50_000, verbose=0, n_envs=1):
        from stable_baselines3.common.vec_env import DummyVecEnv
        
        if n_envs > 1:
            print(f"  Warning: Parallel envs disabled for stability. Training with single env.")
        
        vec_env = DummyVecEnv([lambda: env])
        
        model_kwargs = {
            'policy_kwargs': self.policy_kwargs,
            'verbose': verbose,
            **self.algo_kwargs,
        }
        
        if self.algo_name == 'PPO':
            model_kwargs.setdefault('n_steps', 2048)
            model_kwargs.setdefault('batch_size', 64)
            model_kwargs.setdefault('n_epochs', 10)
        
        self.model = self.Algo('MlpPolicy', vec_env, **model_kwargs)
        self.model.learn(total_timesteps=timesteps)

    def predict(self, obs):
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")
        action, _states = self.model.predict(obs, deterministic=True)
        return action


class TechnicalAgent(BaseRLAgent):
    def __init__(self, **kwargs):
        super().__init__(algo='PPO', **kwargs)


class SentimentAgent(BaseRLAgent):
    def __init__(self, **kwargs):
        super().__init__(algo='PPO', **kwargs)


class SuperAgent(BaseRLAgent):
    def __init__(self, **kwargs):
        super().__init__(algo='SAC', **kwargs)


class MetaAgent:
    """Meta-agent that blends two base agents."""
    def __init__(self):
        self.alpha: Optional[float] = None

    def fit(self, returns: pd.Series, tech_actions: np.ndarray, sent_actions: np.ndarray) -> float:
        best_alpha = 0.5
        best_sharpe = -np.inf
        alphas = np.linspace(0, 1, 21)
        for a in alphas:
            combined = a * tech_actions + (1 - a) * sent_actions
            combined = combined / combined.sum(axis=1, keepdims=True)
            port_ret = (combined * returns.values.reshape(-1, 1)).sum(axis=1)
            sharpe = np.mean(port_ret) / (np.std(port_ret) + 1e-9)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_alpha = a
        self.alpha = best_alpha
        return best_alpha

    def predict(self, tech_action: np.ndarray, sent_action: np.ndarray) -> np.ndarray:
        if self.alpha is None:
            raise RuntimeError("MetaAgent not fitted. Call .fit() first.")
        combined = self.alpha * tech_action + (1 - self.alpha) * sent_action
        combined = np.clip(combined, 0, None)
        return combined / combined.sum()


def create_env_for_asset(
    asset: str,
    split_name: str,
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    returns: pd.Series,
    cost_rate: float = 0.0005,
) -> PortfolioEnvWeekly:
    """Build environment for single asset."""
    feats = datasets[asset][split_name]
    obs = feats.values.astype(np.float32)
    obs = obs.reshape((len(feats), obs.shape[1], 1))
    
    rets = returns.xs(asset, level='ticker').loc[feats.index]
    rets_array = rets.values.astype(np.float32).reshape((len(rets), 1))
    
    env = PortfolioEnvWeekly(features=obs, returns=rets_array, cost_rate=cost_rate)
    return env


def create_env_for_portfolio(
    split_name: str,
    datasets: Dict[str, Dict[str, pd.DataFrame]],
    returns: pd.Series,
    cost_rate: float = 0.0005,
) -> PortfolioEnvWeekly:
    """Build environment for multiple assets."""
    tickers = list(datasets.keys())
    if len(tickers) == 0:
        raise ValueError("datasets is empty")
    
    common_cols = set(datasets[tickers[0]][split_name].columns)
    for t in tickers[1:]:
        common_cols &= set(datasets[t][split_name].columns)
    common_cols = sorted(common_cols)
    
    if not common_cols:
        raise ValueError("No common feature columns across assets")
    
    common_dates = None
    for t in tickers:
        idx = datasets[t][split_name].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    
    if common_dates is None or len(common_dates) == 0:
        raise ValueError("No common dates across assets")
    
    common_dates = common_dates.sort_values()
    
    feature_list: List[np.ndarray] = []
    return_list: List[np.ndarray] = []
    for t in tickers:
        df = datasets[t][split_name].loc[common_dates, common_cols]
        feature_list.append(df.values)
        r = returns.xs(t, level='ticker').loc[common_dates]
        return_list.append(r.values)
    
    features = np.stack(feature_list, axis=2).astype(np.float32)
    returns_matrix = np.column_stack(return_list).astype(np.float32)
    
    env = PortfolioEnvWeekly(features=features, returns=returns_matrix, cost_rate=cost_rate)
    return env


@dataclass
class NormalisationStats:
    means: pd.Series
    stds: pd.Series


def prepare_datasets(
    tickers: List[str],
    benchmark: str,
    start: str,
    end: str,
    train_end: str,
    val_end: str,
    windows: List[int] | None = None,
    include_lags: int = 3,
    include_returns: bool = True,
    macro: bool = True,
    calendar: bool = True,
    ipo_dates: dict[str, str] | None = None,
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, NormalisationStats], pd.DataFrame]:
    """Prepare complete datasets with validation."""
    # Input validation
    if not tickers:
        raise ValueError("tickers list cannot be empty")
    if not benchmark:
        raise ValueError("benchmark cannot be empty")
    
    start_dt = pd.to_datetime(start)
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    end_dt = pd.to_datetime(end)
    
    if not (start_dt < train_end_dt < val_end_dt < end_dt):
        raise ValueError("Dates must be ordered: start < train_end < val_end < end")
    
    print(f"Downloading price data for {len(tickers)} tickers + benchmark...")
    all_prices = load_price_data(tickers + [benchmark], start, end)
    
    print("Resampling to weekly frequency...")
    weekly = resample_to_weekly(all_prices)
    
    print("Computing log returns...")
    returns_frames = []
    for ticker in weekly.index.get_level_values('ticker').unique():
        ticker_data = weekly.xs(ticker, level='ticker')
        ticker_returns = compute_log_returns(ticker_data['adj_close'])
        ticker_returns.name = 'ret'
        ticker_returns.index.name = 'date'
        ticker_returns = ticker_returns.reset_index()
        ticker_returns['ticker'] = ticker
        returns_frames.append(ticker_returns)
    
    weekly_returns = pd.concat(returns_frames)
    weekly_returns = weekly_returns.set_index(['date', 'ticker'])['ret']
    
    if macro:
        print("Fetching macro data...")
        macro_data = get_core_macro_indicators(start, end)
    else:
        macro_data = pd.DataFrame()
    
    if calendar:
        print("Building calendar features...")
        unique_dates = weekly.index.get_level_values('date').unique()
        calendar_data = build_calendar_frame(unique_dates)
    else:
        calendar_data = pd.DataFrame()

    print("Computing technical indicators...")
    tech = compute_technical_features_orchestrator(start, end, tickers, windows, include_returns, include_lags)
    
    available_tickers = tech.index.get_level_values('ticker').unique()
    print(f"Successfully downloaded data for: {list(available_tickers)}")
    
    tickers_with_data = [t for t in tickers if t in available_tickers]
    if len(tickers_with_data) < len(tickers):
        missing_tickers = [t for t in tickers if t not in available_tickers]
        print(f"Warning: Missing data for tickers: {missing_tickers}")
    
    if not tickers_with_data:
        raise ValueError("No tickers have valid data")
    
    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
    norms: Dict[str, NormalisationStats] = {}
    
    print("Preparing datasets for each ticker...")
    for ticker in tickers_with_data:
        if ipo_dates and ticker in ipo_dates:
            first_date = pd.to_datetime(ipo_dates[ticker])
        else:
            ticker_data = weekly.xs(ticker, level='ticker')
            first_date = ticker_data[ticker_data['close'].notna()].index.min()

        buffer = max(windows or [4, 8, 12])
        first_date += pd.Timedelta(weeks=buffer)

        feats_ticker = tech.xs(ticker, level='ticker')
        feats_ticker = feats_ticker[feats_ticker.index >= first_date]

        features = feats_ticker.copy()
        if not macro_data.empty:
            features = features.join(macro_data, on='date', how='left')
        
        if not calendar_data.empty:
            features = features.join(calendar_data, on='date', how='left')
        
        bm_ret = weekly_returns.xs(benchmark, level='ticker').rename('benchmark_ret')
        features = features.join(bm_ret, on='date', how='left')
        features = features.dropna()
        
        split_features = split_data(features, start, train_end, val_end, end)
        
        normed_splits = {}
        for split_name, split_df in split_features.items():
            if split_name == 'train':
                normed, means, stds = normalise_features(split_df)
                normed_splits[split_name] = normed
                norm_stats = NormalisationStats(means=means, stds=stds)
            else:
                normed_splits[split_name] = apply_normalisation(
                    split_df, norm_stats.means, norm_stats.stds
                )
        
        datasets[ticker] = normed_splits
        norms[ticker] = norm_stats
    
    print(f"\nSuccessfully prepared datasets for {len(tickers_with_data)} tickers")
    return datasets, norms, weekly_returns