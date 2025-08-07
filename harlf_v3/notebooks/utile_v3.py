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
    price_data = price_data.fillna(method='ffill').fillna(method='bfill').round(2)
    
    # Calculate log returns and resample to monthly
    log_returns = np.log(price_data / price_data.shift(1)).dropna()
    monthly_prices = price_data.resample('M').last().round(2)
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






# Cell 5: NLP Features Creation
def create_nlp_features(sentiment_data, technical_indicators):
    """
    Create NLP-driven observation vectors as described in Section 3.2
    Combines volatility and sentiment scores
    """
    print("Creating NLP-driven features...")
    
    # Ensure both datasets have the same index
    common_dates = sentiment_data.index.intersection(technical_indicators.index)
    sentiment_data = sentiment_data.loc[common_dates]
    technical_indicators_aligned = technical_indicators.loc[common_dates]
    
    nlp_features = pd.DataFrame(index=common_dates)
    
    # Extract volatility features from technical indicators
    volatility_cols = [col for col in technical_indicators_aligned.columns if 'volatility' in col]
    
    for date in common_dates:
        feature_vector = []
        
        # Add volatility vector
        vol_values = technical_indicators_aligned.loc[date, volatility_cols].values
        feature_vector.extend(vol_values)
        
        # Add sentiment score vector
        sentiment_values = sentiment_data.loc[date].values
        feature_vector.extend(sentiment_values)
        
        # Store combined feature vector
        for i, val in enumerate(feature_vector):
            nlp_features.loc[date, f'nlp_feature_{i}'] = val
    
    # Normalize features
    min_vals = nlp_features.min()
    max_vals = nlp_features.max()
    range_vals = max_vals - min_vals
    
    normalized_features = nlp_features.copy()
    for col in nlp_features.columns:
        if range_vals[col] != 0:
            normalized_features[col] = (nlp_features[col] - min_vals[col]) / range_vals[col]
        else:
            normalized_features[col] = 0.5
    
    normalized_features = normalized_features.fillna(0)
    print(f"NLP features shape: {normalized_features.shape}")
    return normalized_features




# Cell 6: HARLF Portfolio Environment
class HARLFPortfolioEnv(gym.Env):
    """
    Custom Portfolio Environment for HARLF
    Implements the exact specifications from the paper
    """
    
    def __init__(self, price_data, features, sentiment_features=None, 
                 train_period=True, alpha1=1.0, alpha2=2.0, alpha3=0.5):
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
        Calculate reward using HARLF formula:
        Reward = α1 * ROI - α2 * MDD - α3 * σ
        """
        # ROI component
        roi = portfolio_return
        
        # Maximum Drawdown component
        if len(self.portfolio_history) >= 2:
            portfolio_series = pd.Series(self.portfolio_history)
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        else:
            max_drawdown = 0
        
        # Volatility component (using recent portfolio returns)
        if len(self.portfolio_history) >= 21:
            recent_returns = pd.Series(self.portfolio_history[-21:]).pct_change().dropna()
            volatility = recent_returns.std()
        else:
            volatility = 0
        
        # Calculate final reward
        reward = self.alpha1 * roi - self.alpha2 * max_drawdown - self.alpha3 * volatility
        
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

    # names were stored previously in base_agent_names.pkl, we can reconstruct them from filenames
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