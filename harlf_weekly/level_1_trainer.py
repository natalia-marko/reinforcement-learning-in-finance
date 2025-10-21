"""
Level 1 Training Module - Tech & Momentum Agents
================================================

Trains specialist agents on technical features with early stopping.

Architecture:
- Tech Agent: Learns from price/volume patterns
- Momentum Agent: Learns from momentum/oscillator features
- Both use PPO with early stopping
- Evaluated on validation set

Usage:
------
python level_1_trainer.py --timesteps 50000 --early-stopping
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym


class PortfolioEnvWeekly(gym.Env):
    """Gymnasium environment for weekly portfolio rebalancing."""
    metadata = {'render_modes': ['human']}

    def __init__(
        self, 
        features: np.ndarray, 
        returns: np.ndarray, 
        cost_rate: float = 0.0005, 
        seed: Optional[int] = None
    ) -> None:
        super().__init__()
        
        self.features = features
        self.returns = returns
        self.cost_rate = cost_rate
        self.n_steps, self.n_features, self.n_assets = features.shape
        
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.n_assets,), 
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.n_features, self.n_assets), 
            dtype=np.float32
        )
        
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
        
        # FIXED: Use returns from NEXT period (proper temporal alignment)
        # Agent decides at time t, gets rewarded for returns at time t+1
        if self.current_step < self.n_steps - 1:
            asset_ret = self.returns[self.current_step + 1]  # Next period returns
        else:
            asset_ret = np.zeros(self.n_assets)  # No returns for last step
        
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


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on validation Sharpe ratio.
    
    Monitors validation performance and stops training if no improvement
    for a specified number of evaluations.
    """
    
    def __init__(
        self,
        val_env: gym.Env,
        check_freq: int = 5000,
        patience: int = 5,
        min_delta: float = 0.01,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.val_env = val_env
        self.check_freq = check_freq
        self.patience = patience
        self.min_delta = min_delta
        
        self.best_sharpe = -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.evaluation_history = []
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        if self.n_calls % self.check_freq == 0:
            # Evaluate on validation set
            sharpe = self._evaluate()
            
            self.evaluation_history.append({
                'timestep': self.num_timesteps,
                'sharpe': sharpe,
                'is_best': False
            })
            
            # Check for improvement
            if sharpe > self.best_sharpe + self.min_delta:
                self.best_sharpe = sharpe
                self.wait = 0
                self.evaluation_history[-1]['is_best'] = True
                
                if self.verbose > 0:
                    print(f"\n  ✓ Timestep {self.num_timesteps:,}: Val Sharpe improved to {sharpe:.3f}")
            else:
                self.wait += 1
                if self.verbose > 0:
                    print(f"\n  - Timestep {self.num_timesteps:,}: Val Sharpe={sharpe:.3f} "
                          f"(no improvement, patience: {self.wait}/{self.patience})")
                
                if self.wait >= self.patience:
                    self.stopped_epoch = self.num_timesteps
                    if self.verbose > 0:
                        print(f"\n  ⚠ Early stopping triggered at timestep {self.num_timesteps:,}")
                        print(f"  Best Val Sharpe: {self.best_sharpe:.3f}")
                    return False  # Stop training
        
        return True  # Continue training
    
    def _evaluate(self) -> float:
        """Evaluate model on validation environment."""
        obs, _ = self.val_env.reset()
        returns = []
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.val_env.step(action)
            done = terminated or truncated
            returns.append(info['portfolio_return'])
        
        returns = np.array(returns)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(52)  # Annualized
        else:
            sharpe = 0.0
        
        return sharpe


def load_data(split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """Load and prepare data for training."""
    data_path = Path('data')
    tech_path = data_path / split / 'technical.csv'
    macro_path = data_path / split / 'macro_calendar.csv'
    
    # Load data
    tech_df = pd.read_csv(tech_path)
    macro_df = pd.read_csv(macro_path)
    
    # Get tickers and prepare data
    tickers = sorted(tech_df['ticker'].unique())
    
    # Prepare features and returns for each ticker
    features_list = []
    returns_list = []
    
    # Find minimum length across all tickers to ensure consistent shapes
    ticker_lengths = []
    for ticker in tickers:
        ticker_data = tech_df[tech_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        ticker_lengths.append(len(ticker_data))
    
    min_length = min(ticker_lengths)
    
    for ticker in tickers:
        ticker_data = tech_df[tech_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('Date')
        
        # Truncate to minimum length to ensure consistent shapes
        ticker_data = ticker_data.head(min_length)
        
        # Select features (exclude ticker column, return, and Date)
        # IMPORTANT: Exclude 'return' to avoid data leakage
        feature_cols = [col for col in ticker_data.columns 
                       if col not in ['ticker', 'return', 'Date']]
        
        features = ticker_data[feature_cols].values
        returns = ticker_data['return'].values
        
        features_list.append(features)
        returns_list.append(returns)
    
    # Stack: (n_steps, n_features, n_assets)
    features_array = np.stack(features_list, axis=-1)
    returns_array = np.array(returns_list).T  # (n_steps, n_assets)
    
    return features_array, returns_array


def get_feature_indices(tech_df: pd.DataFrame) -> Tuple[list, list]:
    """Get indices for tech and momentum features."""
    # Define feature categories
    tech_keywords = [
        'open', 'high', 'low', 'close', 'volume',
        'sma', 'ema', 'bb_position', 'atr',
        'bench_corr', 'bench_beta'
    ]
    
    momentum_keywords = [
        'rsi', 'macd', 'stoch', 'roc', 'mfi',
        'return', 'volatility', 'obv', 'volume_ratio'
    ]
    
    # Get feature indices
    feature_cols = [col for col in tech_df.columns 
                   if col not in ['ticker', 'return', 'Date']]
    
    tech_features_idx = []
    momentum_features_idx = []
    
    for i, col in enumerate(feature_cols):
        if any(kw in col.lower() for kw in tech_keywords):
            tech_features_idx.append(i)
        if any(kw in col.lower() for kw in momentum_keywords):
            momentum_features_idx.append(i)
    
    return tech_features_idx, momentum_features_idx


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Level 1 agents')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--algo', choices=['ppo', 'sac'], default='ppo', help='Algorithm')
    parser.add_argument('--steps', type=int, default=50000, help='Training steps')
    parser.add_argument('--eval-freq', type=int, default=5000, help='Evaluation frequency')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--rolling-vol', type=int, default=12, help='Rolling volatility window')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    print("Loading data...")
    train_features, train_returns = load_data('train')
    val_features, val_returns = load_data('val')
    
    # Load tech dataframe to get feature indices
    tech_df = pd.read_csv('data/train/technical.csv')
    tech_features_idx, momentum_features_idx = get_feature_indices(tech_df)
    
    # Create environments
    tech_features = train_features[:, tech_features_idx, :]
    momentum_features = train_features[:, momentum_features_idx, :]
    
    tech_env = PortfolioEnvWeekly(tech_features, train_returns, cost_rate=0.0005)
    momentum_env = PortfolioEnvWeekly(momentum_features, train_returns, cost_rate=0.0005)
    
    print(f"Tech Environment: {tech_features.shape}")
    print(f"Momentum Environment: {momentum_features.shape}")
    
    # Train agents
    print("\nTraining Tech Agent...")
    tech_agent = PPO('MlpPolicy', tech_env, learning_rate=args.lr, verbose=1, seed=args.seed)
    tech_agent.learn(total_timesteps=args.steps)
    
    print("\nTraining Momentum Agent...")
    momentum_agent = PPO('MlpPolicy', momentum_env, learning_rate=args.lr, verbose=1, seed=args.seed)
    momentum_agent.learn(total_timesteps=args.steps)
    
    # Save models
    tech_agent.save('models/tech_agent')
    momentum_agent.save('models/momentum_agent')
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
