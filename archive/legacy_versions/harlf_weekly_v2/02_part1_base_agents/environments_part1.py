"""
Multi-Hierarchical RL System - Base Layer Environments (Part 1)
================================================================

Defines two separate environments:
1. TechnicalAgentEnv: Uses only technical indicators
2. SentimentAgentEnv: Uses only sentiment indicators

Both environments:
- Output portfolio weights (long-only via softmax)
- Reward is Sharpe ratio
- Weekly rebalancing
- Proper handling of NaN and edge cases
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Optional


class BasePortfolioEnv(gym.Env):
    """
    Base environment for portfolio management.
    Both technical and sentiment agents inherit from this.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        tickers: list,
        feature_cols: list,
        rolling_vol_window: int = 12,
        softmax_temperature: float = 5.0,
        random_start: bool = False,
    ):
        """
        Initialize base portfolio environment.
        
        Parameters
        ----------
        features_df : pd.DataFrame
            Long-format features with 'ticker' column
        returns_df : pd.DataFrame
            Wide-format returns (columns = tickers)
        tickers : list
            List of tickers
        feature_cols : list
            List of feature column names
        rolling_vol_window : int
            Window for EMA volatility calculation
        softmax_temperature : float
            Temperature for action -> weights conversion
        random_start : bool
            If True, start episodes at random positions
        """
        super().__init__()
        
        self.features_df = features_df
        self.returns_df = returns_df
        self.tickers = tickers
        self.feature_cols = feature_cols
        self.n_assets = len(tickers)
        self.n_features = len(feature_cols)
        self.temperature = softmax_temperature
        self.random_start = random_start
        
        # Validate data
        assert 'ticker' in features_df.columns, "features_df must have 'ticker' column"
        assert set(tickers).issubset(returns_df.columns), "All tickers must be in returns_df"
        
        # Action space: continuous values for each asset
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: features for all assets
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n_assets * self.n_features,),
            dtype=np.float32
        )
        
        # Dates available
        self.dates = sorted(features_df.index.unique())
        
        # Volatility estimation params
        self.W = rolling_vol_window
        self.alpha = 2.0 / (self.W + 1.0)
        self.start_idx = max(self.W, 1)
        
        # Episode state
        self._t = None
        self._weights = None
        self._ema_mean = 0.0
        self._ema_var = 0.0
        self._ema_initialized = False
        
    def _get_obs(self):
        """Get current observation."""
        date = self.dates[self._t]
        
        # Get features for all tickers on this date
        date_features = self.features_df.loc[[date]]
        
        # Pivot to matrix: rows=tickers, cols=features
        feature_matrix = (
            date_features[date_features['ticker'].isin(self.tickers)]
            .set_index('ticker')[self.feature_cols]
            .reindex(self.tickers)
            .fillna(0.0)
            .values
            .astype(np.float32)
        )
        
        return feature_matrix.flatten()
    
    def _softmax(self, x):
        """Convert actions to portfolio weights."""
        x = np.asarray(x, dtype=np.float32) * self.temperature
        x = x - np.max(x)
        e = np.exp(x)
        return e / (e.sum() + 1e-12)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Validate dataset size
        min_required = self.start_idx + 2
        if len(self.dates) < min_required:
            raise ValueError(
                f"Dataset too small: {len(self.dates)} dates, need {min_required}"
            )
        
        # Starting position
        if self.random_start:
            max_start = len(self.dates) - 2
            if max_start > self.start_idx:
                self._t = np.random.randint(self.start_idx, max_start)
            else:
                self._t = self.start_idx
        else:
            self._t = self.start_idx
        
        # Initialize weights (equal weight)
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        
        # Initialize EMA
        self._ema_mean = 0.0
        self._ema_var = 1e-3  # Small but non-zero!
        self._ema_initialized = False
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step."""
        # Convert action to weights
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        w = self._softmax(action)
        self._weights = w
        
        # Check if future return exists
        if self._t + 1 >= len(self.dates):
            return self._get_obs(), 0.0, True, False, {}
        
        # Get next period's returns
        next_date = self.dates[self._t + 1]
        r_vec = self.returns_df.loc[next_date, self.tickers].values.astype(np.float32)
        
        # Portfolio return
        port_log_r = float(np.dot(w, r_vec))
        
        # Update EMA of mean and variance
        if not self._ema_initialized:
            self._ema_mean = port_log_r
            self._ema_var = 1e-3  # Small initial variance
            self._ema_initialized = True
        else:
            delta = port_log_r - self._ema_mean
            self._ema_mean += self.alpha * delta
            self._ema_var = (1.0 - self.alpha) * (self._ema_var + self.alpha * delta * delta)
        
        # Calculate reward (Sharpe-like, bounded)
        running_vol = max(math.sqrt(max(self._ema_var, 0.0)), 1e-4)
        reward = np.clip(port_log_r / running_vol * math.sqrt(52.0), -10.0, 10.0)
        
        # Advance time
        self._t += 1
        terminated = self._t >= (len(self.dates) - 1)
        
        info = {
            'port_log_r': port_log_r,
            'running_vol': running_vol,
            'weights': w.copy()
        }
        
        return self._get_obs(), float(reward), terminated, False, info
    
    def run_full_pass(self, model):
        """Run full episode and return Sharpe ratio."""
        obs, _ = self.reset()
        returns = []
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.step(action)
            if 'port_log_r' in info:
                returns.append(info['port_log_r'])
        
        returns = np.array(returns)
        if len(returns) < 2 or returns.std() < 1e-12:
            return 0.0
        
        return float((returns.mean() / returns.std()) * math.sqrt(52.0))


class TechnicalAgentEnv(BasePortfolioEnv):
    """Environment for Technical Agent using only technical indicators."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize Technical Agent environment.
        
        Parameters
        ----------
        data_dir : str
            Path to data directory
        split : str
            'train', 'val', or 'test'
        **kwargs
            Additional arguments for BasePortfolioEnv
        """
        data_dir = Path(data_dir)
        
        # Load technical features
        tech_path = data_dir / 'technical' / f'{split}.csv'
        features_df = pd.read_csv(tech_path, index_col=0, parse_dates=True)
        
        # Load returns
        ret_path = data_dir / f'returns_{split}.csv'
        returns_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        
        # Load metadata
        with open(data_dir / 'metadata.json', 'r') as f:
            import json
            metadata = json.load(f)
        
        tickers = metadata['tickers']
        # Use only normalized indicator features, not base columns (open, high, low, close, volume, return)
        feature_cols = metadata.get('technical_indicator_features', 
                                     [c for c in metadata['technical_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
        feature_cols = [c for c in feature_cols if c in features_df.columns]
        
        super().__init__(
            features_df=features_df,
            returns_df=returns_df,
            tickers=tickers,
            feature_cols=feature_cols,
            **kwargs
        )
        
        self.agent_type = 'technical'
        print(f"TechnicalAgentEnv ({split}): {len(self.dates)} dates, {self.n_assets} assets, {self.n_features} features")


class SentimentAgentEnv(BasePortfolioEnv):
    """Environment for Sentiment Agent using only sentiment indicators."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        **kwargs
    ):
        """
        Initialize Sentiment Agent environment.
        
        Parameters
        ----------
        data_dir : str
            Path to data directory
        split : str
            'train', 'val', or 'test'
        **kwargs
            Additional arguments for BasePortfolioEnv
        """
        data_dir = Path(data_dir)
        
        # Load sentiment features
        sent_path = data_dir / 'sentiment' / f'{split}.csv'
        features_df = pd.read_csv(sent_path, index_col=0, parse_dates=True)
        
        # Load returns
        ret_path = data_dir / f'returns_{split}.csv'
        returns_df = pd.read_csv(ret_path, index_col=0, parse_dates=True)
        
        # Load metadata
        with open(data_dir / 'metadata.json', 'r') as f:
            import json
            metadata = json.load(f)
        
        tickers = metadata['tickers']
        # Use only normalized indicator features, not base columns (open, high, low, close, volume, return)
        feature_cols = metadata.get('sentiment_indicator_features',
                                     [c for c in metadata['sentiment_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
        feature_cols = [c for c in feature_cols if c in features_df.columns]
        
        super().__init__(
            features_df=features_df,
            returns_df=returns_df,
            tickers=tickers,
            feature_cols=feature_cols,
            **kwargs
        )
        
        self.agent_type = 'sentiment'
        print(f"SentimentAgentEnv ({split}): {len(self.dates)} dates, {self.n_assets} assets, {self.n_features} features")


# Validation callback for early stopping
from stable_baselines3.common.callbacks import BaseCallback

class ValidationCallback(BaseCallback):
    """
    Callback for validation-based early stopping.
    Evaluates on validation set and saves best model.
    """
    
    def __init__(
        self,
        val_env,
        eval_freq: int = 5000,
        patience: int = 5,
        save_path: Optional[str] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.save_path = save_path
        self.best_sharpe = -np.inf
        self.no_improve = 0
        
    def _on_step(self):
        if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
            return True
        
        # Evaluate on validation set
        val_sharpe = self.val_env.run_full_pass(self.model)
        
        if self.verbose:
            print(f"[Step {self.num_timesteps:,}] Val Sharpe: {val_sharpe:.3f} (Best: {self.best_sharpe:.3f})")
        
        # Check for improvement
        if val_sharpe > self.best_sharpe + 1e-6:
            self.best_sharpe = val_sharpe
            self.no_improve = 0
            
            if self.save_path:
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"  ✓ New best model saved -> {self.save_path}")
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                if self.verbose:
                    print(f"  ⛔ Early stopping (no improvement for {self.patience} evaluations)")
                return False
        
        return True


def create_env(agent_type: str, data_dir: str, split: str, **kwargs):
    """
    Factory function to create appropriate environment.
    
    Parameters
    ----------
    agent_type : str
        'technical' or 'sentiment'
    data_dir : str
        Path to data directory
    split : str
        'train', 'val', or 'test'
    **kwargs
        Additional environment parameters
    
    Returns
    -------
    env : BasePortfolioEnv
        Initialized environment
    """
    if agent_type == 'technical':
        return TechnicalAgentEnv(data_dir, split, **kwargs)
    elif agent_type == 'sentiment':
        return SentimentAgentEnv(data_dir, split, **kwargs)
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")


if __name__ == '__main__':
    # Test environments
    print("Testing Technical Agent Environment...")
    try:
        env = TechnicalAgentEnv('data_hierarchical', split='train')
        obs, _ = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  ✓ Technical environment works!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\nTesting Sentiment Agent Environment...")
    try:
        env = SentimentAgentEnv('data_hierarchical', split='train')
        obs, _ = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        print(f"  ✓ Sentiment environment works!")
    except Exception as e:
        print(f"  ✗ Error: {e}")
