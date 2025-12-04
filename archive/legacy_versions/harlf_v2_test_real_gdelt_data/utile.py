import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import pipeline
import gymnasium as gym
from gymnasium import spaces
import pickle


def collect_price_data(ASSETS, TRAIN_START, TEST_END):
    """
    Collect historical price data using yfinance
    Returns log returns as specified in the paper
    """
    print("Collecting price data...")
    
    tickers = list(ASSETS.keys())
    
    # Download data with group_by ticker to get all price types
    data = yf.download(tickers, start=TRAIN_START, end=TEST_END, 
                       group_by='ticker', auto_adjust=False)
    
    # Try to use Adj Close first, fallback to Close if not available
    price_data = pd.DataFrame()
    
    for ticker in tickers:
        try:
            # Try Adj Close first
            if 'Adj Close' in data[ticker].columns:
                price_data[ticker] = data[ticker]['Adj Close']
                print(f"Using Adj Close for {ticker}")
            else:
                # Fallback to Close
                price_data[ticker] = data[ticker]['Close']
                print(f"Using Close for {ticker} (Adj Close not available)")
        except KeyError:
            # Handle case where ticker data is not available
            print(f"Warning: No data available for {ticker}")
            continue
    
    # Handle missing data
    price_data = price_data.ffill().bfill().round(2)
    # Calculate log returns and resample to monthly
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    monthly_prices = price_data.resample('ME').last().round(2)
    monthly_log_returns = np.log(monthly_prices / monthly_prices.shift(1)).dropna().round(2)
    
    print(f"Data collected: {monthly_log_returns.shape}")
    
    return price_data, monthly_prices, monthly_log_returns


def calculate_technical_indicators(prices, returns):
    """
    Calculate technical indicators used in HARLF
    """
    print("Calculating indicators...")
    
    indicators = pd.DataFrame(index=returns.index)
    rolling_window = 21  # ~1 month
    
    for asset in prices.columns:
        asset_returns = returns[asset].dropna()
        
        # Rolling metrics
        mean_return = asset_returns.rolling(rolling_window).mean() * 252
        volatility = asset_returns.rolling(rolling_window).std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = np.where(volatility != 0, mean_return / volatility, 0)
        
        # Sortino ratio
        downside_returns = asset_returns[asset_returns < 0]
        downside_vol = downside_returns.rolling(rolling_window).std() * np.sqrt(252)
        downside_vol_aligned = downside_vol.reindex(asset_returns.index).fillna(volatility)
        sortino_ratio = np.where(downside_vol_aligned != 0, mean_return / downside_vol_aligned, 0)
        
        # Maximum drawdown
        cumulative = (1 + asset_returns).cumprod()
        rolling_max = cumulative.rolling(rolling_window).max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.rolling(rolling_window).min()
        
        # Calmar ratio
        max_dd_abs = abs(max_drawdown)
        calmar_ratio = np.where(max_dd_abs != 0, mean_return / max_dd_abs, 0)
        
        # Store indicators
        indicators[f'{asset}_sharpe'] = pd.Series(sharpe_ratio, index=returns.index)
        indicators[f'{asset}_sortino'] = pd.Series(sortino_ratio, index=returns.index)
        indicators[f'{asset}_calmar'] = pd.Series(calmar_ratio, index=returns.index)
        indicators[f'{asset}_volatility'] = volatility
        indicators[f'{asset}_max_drawdown'] = abs(max_drawdown)
    
    # Correlation features
    correlation_features = pd.DataFrame(index=returns.index)
    for i, date in enumerate(returns.index[21:]):
        period_returns = returns.loc[returns.index[i]:date]
        if len(period_returns) >= rolling_window:
            corr_matrix = period_returns.tail(rolling_window).corr()
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            for j, val in enumerate(corr_values):
                correlation_features.loc[date, f'corr_{j}'] = val
    
    # Combine and normalize
    all_indicators = pd.concat([indicators, correlation_features], axis=1)
    
    # Min-max normalization
    min_vals = all_indicators.min()
    max_vals = all_indicators.max()
    range_vals = max_vals - min_vals
    normalized_indicators = all_indicators.copy()
    
    for col in all_indicators.columns:
        if range_vals[col] != 0:
            normalized_indicators[col] = (all_indicators[col] - min_vals[col]) / range_vals[col]
        else:
            normalized_indicators[col] = 0.5
    
    normalized_indicators = normalized_indicators.fillna(0)
    
    print(f"Indicators calculated: {normalized_indicators.shape}")
    
    return normalized_indicators.round(4)


def load_real_gdelt_sentiment(assets, date_range):
    """
    Load real GDELT v2tone sentiment data with proper date alignment
    """
    print("Loading real GDELT sentiment data...")
    
    try:
        # Load GDELT data
        gdelt_data = pd.read_csv('v2tone_2015_2025.csv')
        reddit_data = pd.read_csv('reddit_monthly_sentiment.csv')
        
        # Convert date columns to datetime
        gdelt_data['month'] = pd.to_datetime(gdelt_data['month'])
        reddit_data['month'] = pd.to_datetime(reddit_data['month'])
        
        # Create sentiment data DataFrame
        sentiment_data = pd.DataFrame(index=date_range, columns=assets.keys())
        
        # Process each asset
        for ticker in assets.keys():
            ticker_lower = ticker.lower()
            ticker_upper = ticker.upper()
            
            # Get GDELT data for this ticker
            gdelt_ticker = gdelt_data[gdelt_data['ticker'] == ticker_lower]
            reddit_ticker = reddit_data[reddit_data['ticker'] == ticker_upper]
            
            if not gdelt_ticker.empty:
                gdelt_ticker = gdelt_ticker.set_index('month')['weighted_tone']
                # Normalize GDELT tone from [-10, 10] to [-1, 1]
                gdelt_ticker = gdelt_ticker / 10.0
                gdelt_ticker = np.clip(gdelt_ticker, -1, 1)
            
            if not reddit_ticker.empty:
                reddit_ticker = reddit_ticker.set_index('month')['sent_ew_weighted']
            
            # Align with our date range (month-end dates)
            for date in date_range:
                # Convert month-end date to month-start for GDELT lookup
                month_start = date.replace(day=1)
                
                # Try GDELT data first
                if not gdelt_ticker.empty and month_start in gdelt_ticker.index:
                    sentiment_data.loc[date, ticker] = gdelt_ticker.loc[month_start]
                # Try Reddit data as fallback
                elif not reddit_ticker.empty and month_start in reddit_ticker.index:
                    sentiment_data.loc[date, ticker] = reddit_ticker.loc[month_start]
                else:
                    sentiment_data.loc[date, ticker] = 0  # Neutral if no data
        
        # Fill any remaining NaN values with 0
        sentiment_data = sentiment_data.fillna(0)
        
        print(f"Sentiment data loaded: {sentiment_data.shape}")
        print(f"Non-zero values: {(sentiment_data != 0).sum().sum()}")
        
        return sentiment_data
        
    except Exception as e:
        print(f"Error loading sentiment data: {e}")
        # Fallback to neutral sentiment
        return pd.DataFrame(0, index=date_range, columns=assets.keys())

def collect_sentiment_data(assets, technical_indicators):
    """
    Load real GDELT + Reddit sentiment data
    """
    print("Collecting real sentiment data...")
    date_range = technical_indicators.index
    return load_real_gdelt_sentiment(assets, date_range)



# Cell 5: NLP Features Creation
def create_technical_features(technical_indicators):
    """
    Create features for TECHNICAL AGENT (pure quantitative analysis)
    Uses all technical indicators: Sharpe, Sortino, Calmar ratios, volatility, correlations
    """
    print("Creating technical features for numerical agent...")
    
    features = technical_indicators.copy()
    
    # Normalize to [0,1] range
    min_vals = features.min()
    max_vals = features.max()
    range_vals = max_vals - min_vals
    
    normalized_features = features.copy()
    for col in features.columns:
        if range_vals[col] != 0:
            normalized_features[col] = (features[col] - min_vals[col]) / range_vals[col]
        else:
            normalized_features[col] = 0.5  # Neutral for constant features
    
    normalized_features = normalized_features.fillna(0)
    print(f"Technical features shape: {normalized_features.shape}")
    return normalized_features

def create_sentiment_features(sentiment_data, monthly_returns=None, monthly_prices=None, mode='sentiment_logret'):
    """
    Create features for SENTIMENT AGENT (news/social media driven)
    
    Modes:
    - 'sentiment_logret': Sentiment + Log Returns (recommended)
    - 'sentiment_momentum': Sentiment + Price Momentum  
    - 'sentiment_only': Pure sentiment signals
    """
    print(f"Creating sentiment features (mode: {mode})...")
    
    if mode == 'sentiment_logret' and monthly_returns is not None:
        # Option 1: Sentiment + Log Returns
        log_returns = np.log(1 + monthly_returns).fillna(0)
        common_dates = sentiment_data.index.intersection(log_returns.index)
        
        sentiment_aligned = sentiment_data.loc[common_dates]
        log_returns_aligned = log_returns.loc[common_dates]
        
        features = pd.concat([sentiment_aligned, log_returns_aligned], axis=1)
        features.columns = [f'sent_{col}' for col in sentiment_aligned.columns] + [f'logret_{col}' for col in log_returns_aligned.columns]
        
    elif mode == 'sentiment_momentum' and monthly_prices is not None:
        # Option 2: Sentiment + Price Momentum
        price_momentum = monthly_prices.pct_change(1).fillna(0)
        common_dates = sentiment_data.index.intersection(price_momentum.index)
        
        sentiment_aligned = sentiment_data.loc[common_dates]
        momentum_aligned = price_momentum.loc[common_dates]
        
        features = pd.concat([sentiment_aligned, momentum_aligned], axis=1)
        features.columns = [f'sent_{col}' for col in sentiment_aligned.columns] + [f'momentum_{col}' for col in momentum_aligned.columns]
        
    elif mode == 'sentiment_only':
        # Option 3: Pure sentiment
        features = sentiment_data.copy()
        
    else:
        raise ValueError(f"Invalid mode '{mode}' or missing required data")
    
    # Normalize to [0,1] range
    min_vals = features.min()
    max_vals = features.max()
    range_vals = max_vals - min_vals
    
    normalized_features = features.copy()
    for col in features.columns:
        if range_vals[col] != 0:
            normalized_features[col] = (features[col] - min_vals[col]) / range_vals[col]
        else:
            normalized_features[col] = 0.5  # Neutral for constant features
    
    normalized_features = normalized_features.fillna(0)
    print(f"Sentiment features shape: {normalized_features.shape}")
    return normalized_features

def create_conditional_sentiment_features(sentiment_data, monthly_returns, correlation_threshold=0.1):
    """
    Create sentiment features that only include assets where sentiment correlates with returns
    This creates a smarter sentiment agent that focuses on meaningful signals
    """
    print(f"Creating conditional sentiment features (threshold: {correlation_threshold})...")
    
    # Calculate correlations between sentiment and returns
    correlations = {}
    significant_assets = []
    
    for asset in sentiment_data.columns:
        if asset in monthly_returns.columns:
            # Calculate correlation, handling NaN values
            valid_mask = ~(sentiment_data[asset].isna() | monthly_returns[asset].isna())
            if valid_mask.sum() > 10:  # Need at least 10 valid observations
                corr = sentiment_data[asset][valid_mask].corr(monthly_returns[asset][valid_mask])
                if not pd.isna(corr) and abs(corr) > correlation_threshold:
                    correlations[asset] = corr
                    significant_assets.append(asset)
    
    print(f"Assets with significant sentiment-return correlation:")
    print(f"  Count: {len(significant_assets)}/{len(sentiment_data.columns)}")
    print(f"  Assets: {significant_assets}")
    for asset in significant_assets:
        print(f"    {asset}: {correlations[asset]:.4f}")
    
    if not significant_assets:
        print("Warning: No assets with significant correlations found. Using all sentiment data.")
        significant_assets = list(sentiment_data.columns)
    
    # Create features using only significant assets
    # Option 1: Sentiment + Log Returns for significant assets only
    log_returns = np.log(1 + monthly_returns).fillna(0)
    
    # Align dates
    common_dates = sentiment_data.index.intersection(log_returns.index)
    sentiment_aligned = sentiment_data.loc[common_dates]
    log_returns_aligned = log_returns.loc[common_dates]
    
    # Select only significant assets
    sentiment_significant = sentiment_aligned[significant_assets]
    log_returns_significant = log_returns_aligned[significant_assets]
    
    # Combine features
    features = pd.concat([sentiment_significant, log_returns_significant], axis=1)
    features.columns = [f'sent_{col}' for col in sentiment_significant.columns] + [f'logret_{col}' for col in log_returns_significant.columns]
    
    # Normalize to [0,1] range
    min_vals = features.min()
    max_vals = features.max()
    range_vals = max_vals - min_vals
    
    normalized_features = features.copy()
    for col in features.columns:
        if range_vals[col] != 0:
            normalized_features[col] = (features[col] - min_vals[col]) / range_vals[col]
        else:
            normalized_features[col] = 0.5  # Neutral for constant features
    
    normalized_features = normalized_features.fillna(0)
    
    print(f"Conditional sentiment features shape: {normalized_features.shape}")
    print(f"Feature structure: 0-{len(significant_assets)-1} = Sentiment (significant assets), {len(significant_assets)}-{2*len(significant_assets)-1} = Log returns (significant assets)")
    
    return normalized_features, significant_assets, correlations

# Legacy function for backward compatibility
def create_nlp_features(sentiment_data, monthly_returns):
    """
    Legacy function - use create_sentiment_features() instead
    """
    return create_sentiment_features(sentiment_data, monthly_returns, mode='sentiment_logret')




# Cell 6: HARLF Portfolio Environment
class HARLFPortfolioEnv(gym.Env):
    """
    Custom Portfolio Environment for HARLF
    Implements the exact specifications from the paper
    """
    
    def __init__(self, price_data, features, sentiment_features=None, 
                 train_period=True, alpha1=20.0, alpha2=0.2, alpha3=0.05):
        super(HARLFPortfolioEnv, self).__init__()
        
        self.price_data = price_data
        self.features = features
        self.sentiment_features = sentiment_features
        self.train_period = train_period
        
        # Reward function parameters (from paper)
        self.alpha1 = alpha1  # ROI weight
        self.alpha2 = alpha2  # Max Drawdown penalty
        self.alpha3 = alpha3  # Volatility penalty
        
        # Portfolio constraints from paper
        self.n_assets = len(price_data.columns)
        self.initial_capital = 100000  # $100k initial capital
        
        # Action space: continuous portfolio weights [0,1] that sum to 1
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        
        # Observation space: normalized features
        if sentiment_features is not None:
            obs_dim = features.shape[1] + sentiment_features.shape[1]
        else:
            obs_dim = features.shape[1]
            
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Initialize state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        
        # Align dates - ensure all data sources have same dates
        self.common_dates = self.features.index
        if self.sentiment_features is not None:
            self.common_dates = self.features.index.intersection(
                self.sentiment_features.index
            )
        # Also ensure price data alignment
        self.common_dates = self.common_dates.intersection(self.price_data.index)
        
        print(f"Environment initialized with {len(self.common_dates)} time steps")
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation vector"""
        if self.current_step >= len(self.common_dates):
            return np.zeros(self.observation_space.shape[0])
        
        current_date = self.common_dates[self.current_step]
        
        # Get technical features
        tech_features = self.features.loc[current_date].values
        
        # Get sentiment features if available
        if self.sentiment_features is not None:
            if current_date in self.sentiment_features.index:
                sent_features = self.sentiment_features.loc[current_date].values
                observation = np.concatenate([tech_features, sent_features])
            else:
                sent_features = np.zeros(self.sentiment_features.shape[1])
                observation = np.concatenate([tech_features, sent_features])
        else:
            observation = tech_features
        
        # Handle NaN values
        observation = np.nan_to_num(observation, nan=0.0)
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.current_step >= len(self.common_dates) - 1:
            return self._get_observation(), 0, True, True, {}
        
        # Normalize action to ensure weights sum to 1
        action = np.clip(action, 0, 1)
        weights = action / (action.sum() + 1e-8)
        
        # Get current and next period dates
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[self.current_step + 1]
        
        # Calculate portfolio return using log returns
        if current_date in self.price_data.index and next_date in self.price_data.index:
            log_returns = np.log(self.price_data.loc[next_date] / self.price_data.loc[current_date])
            log_returns = log_returns.fillna(0)
            portfolio_log_return = np.sum(weights * log_returns)
            portfolio_return = np.exp(portfolio_log_return) - 1
        else:
            portfolio_return = 0
        
        # Update portfolio value
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate reward using HARLF reward function
        reward = self._calculate_reward(weights, portfolio_return, old_value)
        
        # Update state
        self.weights = weights
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.common_dates) - 1
        terminated = done
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'weights': weights
        }
    
    def _calculate_reward(self, weights, portfolio_return, old_value):
        """
        Calculate reward using improved HARLF formula:
        Reward = α1 * ROI - α2 * MDD - α3 * σ
        With better scaling and stability
        """
        # ROI component (amplified to make positive returns more rewarding)
        roi = portfolio_return
        
        # Maximum Drawdown component (only penalize significant drawdowns)
        if len(self.portfolio_history) >= 3:
            portfolio_series = pd.Series(self.portfolio_history)
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
            # Only penalize drawdowns > 1%
            max_drawdown = max(0, max_drawdown - 0.01)
        else:
            max_drawdown = 0
        
        # Volatility component (only penalize excessive volatility)
        if len(self.portfolio_history) >= 10:
            recent_returns = pd.Series(self.portfolio_history[-10:]).pct_change().dropna()
            if len(recent_returns) > 0:
                volatility = recent_returns.std()
                # Only penalize volatility > 5%
                volatility = max(0, volatility - 0.05)
            else:
                volatility = 0
        else:
            volatility = 0
        
        # Calculate final reward with improved scaling
        reward = self.alpha1 * roi - self.alpha2 * max_drawdown - self.alpha3 * volatility
        
        # Add positive bias to encourage exploration and make rewards more positive
        reward += 0.01
        
        return reward
    
    def get_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        if len(self.portfolio_history) <= 1:
            return {}
        
        # Convert to returns
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        # Calculate metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = abs(drawdown.min())
        
        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': self.portfolio_value
        }
    

from pathlib import Path
from stable_baselines3 import PPO, SAC, DDPG, TD3

def load_base_agents(model_dir: Path | str = "models") -> dict[str, object]:
    """
    Recreate the `base_agents` dict without relying on the broken pickle.
    Looks at the zip files in `models/` and loads them with SB3.
    """
    model_dir = Path(model_dir)
    agent_paths = sorted(model_dir.glob("base_agent_*_*.zip"))
    agents: dict[str, object] = {}

    for p in agent_paths:
        name = p.stem.replace("base_agent_", "")
        if "PPO" in name:
            agent = PPO.load(p)
        elif "SAC" in name:
            agent = SAC.load(p)
        elif "DDPG" in name:
            agent = DDPG.load(p)
        elif "TD3" in name:
            agent = TD3.load(p)
        else:
            raise ValueError(f"Unknown agent type in filename {p.name}")

        agents[name] = agent
    return agents