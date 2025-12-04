"""
Simple Baseline Reward Approaches

This provides truly simple baseline rewards for comparison:
1. Simple Return - Just the log return itself (no risk adjustment)
2. Simple Sharpe - Sharpe calculated from all historical returns (not EMA)

These are the simplest possible reward functions to establish a baseline.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import json
import math

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.callbacks import BaseCallback


class SimpleReturnEnv(gym.Env):
    """Environment with simple log return as reward (no risk adjustment)."""
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        tickers: list,
        feature_cols: list,
        softmax_temperature: float = 3.0,
        random_start: bool = False,
    ):
        super().__init__()
        
        self.features_df = features_df
        self.returns_df = returns_df
        self.tickers = tickers
        self.feature_cols = feature_cols
        self.temperature = softmax_temperature
        self.random_start = random_start
        
        # Get unique dates
        self.dates = sorted(self.features_df.index.unique())
        self.n_assets = len(tickers)
        self.n_features = len(feature_cols)
        
        # Action space: continuous weights for each asset
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: features for each asset
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets * self.n_features,),
            dtype=np.float32
        )
        
        self._t = 0
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._episode_returns = []
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax with temperature."""
        x_scaled = x / self.temperature
        e_x = np.exp(x_scaled - np.max(x_scaled))
        return e_x / e_x.sum()
    
    def _get_obs(self):
        """Get observation for current timestep."""
        current_date = self.dates[self._t]
        obs_list = []
        
        for ticker in self.tickers:
            ticker_data = self.features_df.loc[current_date, self.feature_cols]
            obs_list.append(ticker_data.values.astype(np.float32))
        
        obs = np.concatenate(obs_list)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        if self.random_start and len(self.dates) > 50:
            max_start = len(self.dates) - 50
            self._t = np.random.randint(0, max_start)
        else:
            self._t = 0
        
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._episode_returns = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step with simple return reward."""
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
        
        # Simple reward: just the return itself (scaled for training stability)
        reward = np.clip(port_log_r * math.sqrt(52.0), -10.0, 10.0)
        
        self._episode_returns.append(port_log_r)
        
        # Move to next timestep
        self._t += 1
        obs = self._get_obs()
        terminated = (self._t + 1 >= len(self.dates))
        
        return obs, reward, terminated, False, {}
    
    def run_full_pass(self, model):
        """Run full episode and compute Sharpe ratio."""
        obs, _ = self.reset()
        done = False
        returns = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.step(action)
            if len(self._episode_returns) > 0:
                returns.append(self._episode_returns[-1])
        
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        return float((returns.mean() / returns.std()) * math.sqrt(52))


class SimpleSharpeEnv(gym.Env):
    """Environment with simple Sharpe ratio (using all historical returns)."""
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        tickers: list,
        feature_cols: list,
        softmax_temperature: float = 3.0,
        random_start: bool = False,
    ):
        super().__init__()
        
        self.features_df = features_df
        self.returns_df = returns_df
        self.tickers = tickers
        self.feature_cols = feature_cols
        self.temperature = softmax_temperature
        self.random_start = random_start
        
        # Get unique dates
        self.dates = sorted(self.features_df.index.unique())
        self.n_assets = len(tickers)
        self.n_features = len(feature_cols)
        
        # Action space: continuous weights for each asset
        self.action_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: features for each asset
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.n_assets * self.n_features,),
            dtype=np.float32
        )
        
        self._t = 0
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._episode_returns = []
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Apply softmax with temperature."""
        x_scaled = x / self.temperature
        e_x = np.exp(x_scaled - np.max(x_scaled))
        return e_x / e_x.sum()
    
    def _get_obs(self):
        """Get observation for current timestep."""
        current_date = self.dates[self._t]
        obs_list = []
        
        for ticker in self.tickers:
            ticker_data = self.features_df.loc[current_date, self.feature_cols]
            obs_list.append(ticker_data.values.astype(np.float32))
        
        obs = np.concatenate(obs_list)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        if self.random_start and len(self.dates) > 50:
            max_start = len(self.dates) - 50
            self._t = np.random.randint(0, max_start)
        else:
            self._t = 0
        
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._episode_returns = []
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step with simple Sharpe reward."""
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
        self._episode_returns.append(port_log_r)
        
        # Simple Sharpe: calculate from all historical returns so far
        if len(self._episode_returns) >= 2:
            returns_arr = np.array(self._episode_returns)
            mean_ret = returns_arr.mean()
            std_ret = max(returns_arr.std(), 1e-6)
            sharpe = (mean_ret / std_ret) * math.sqrt(52.0)
            reward = np.clip(sharpe, -10.0, 10.0)
        else:
            # Not enough data yet
            reward = 0.0
        
        # Move to next timestep
        self._t += 1
        obs = self._get_obs()
        terminated = (self._t + 1 >= len(self.dates))
        
        return obs, reward, terminated, False, {}
    
    def run_full_pass(self, model):
        """Run full episode and compute Sharpe ratio."""
        obs, _ = self.reset()
        done = False
        returns = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = self.step(action)
            if len(self._episode_returns) > 0:
                returns.append(self._episode_returns[-1])
        
        if len(returns) < 2:
            return 0.0
        
        returns = np.array(returns)
        return float((returns.mean() / returns.std()) * math.sqrt(52))


def create_simple_return_env(data_dir: str, split: str, agent_type: str, **kwargs):
    """Create environment with simple return reward."""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path('../..') / data_dir
    
    # Load data
    if agent_type == 'technical':
        features_path = data_dir / 'technical' / f'{split}.csv'
    else:
        features_path = data_dir / 'sentiment' / f'{split}.csv'
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(data_dir / f'returns_{split}.csv', index_col=0, parse_dates=True)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    tickers = metadata['tickers']
    if agent_type == 'technical':
        feature_cols = metadata.get('technical_indicator_features', 
                                     [c for c in metadata['technical_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    else:
        feature_cols = metadata.get('sentiment_indicator_features',
                                     [c for c in metadata['sentiment_features']
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    return SimpleReturnEnv(
        features_df=features_df,
        returns_df=returns_df,
        tickers=tickers,
        feature_cols=feature_cols,
        **kwargs
    )


def create_simple_sharpe_env(data_dir: str, split: str, agent_type: str, **kwargs):
    """Create environment with simple Sharpe reward."""
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = Path('../..') / data_dir
    
    # Load data
    if agent_type == 'technical':
        features_path = data_dir / 'technical' / f'{split}.csv'
    else:
        features_path = data_dir / 'sentiment' / f'{split}.csv'
    
    features_df = pd.read_csv(features_path, index_col=0, parse_dates=True)
    returns_df = pd.read_csv(data_dir / f'returns_{split}.csv', index_col=0, parse_dates=True)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    tickers = metadata['tickers']
    if agent_type == 'technical':
        feature_cols = metadata.get('technical_indicator_features', 
                                     [c for c in metadata['technical_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    else:
        feature_cols = metadata.get('sentiment_indicator_features',
                                     [c for c in metadata['sentiment_features']
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    return SimpleSharpeEnv(
        features_df=features_df,
        returns_df=returns_df,
        tickers=tickers,
        feature_cols=feature_cols,
        **kwargs
    )


class ValidationCallback(BaseCallback):
    """Validation callback for early stopping."""
    
    def __init__(self, val_env, eval_freq: int = 5000, patience: int = 5,
                 save_path: str = None, verbose: int = 1):
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
        
        val_sharpe = self.val_env.run_full_pass(self.model)
        
        if self.verbose:
            print(f"[Step {self.num_timesteps:,}] Val Sharpe: {val_sharpe:.3f} (Best: {self.best_sharpe:.3f})")
        
        if val_sharpe > self.best_sharpe + 1e-6:
            self.best_sharpe = val_sharpe
            self.no_improve = 0
            
            if self.save_path:
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"  New best model saved")
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                if self.verbose:
                    print(f"  Early stopping")
                return False
        
        return True


def train_simple_baseline(agent_type: str, algorithm: str, config: dict, 
                         reward_type: str = 'simple_return', verbose: bool = True):
    """
    Train agent with simple baseline reward.
    
    Parameters
    ----------
    agent_type : str
        'technical' or 'sentiment'
    algorithm : str
        'PPO', 'SAC', or 'A2C'
    config : dict
        Training configuration
    reward_type : str
        'simple_return' or 'simple_sharpe'
    verbose : bool
        Print progress
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {agent_type.upper()} Agent with {algorithm} ({reward_type.upper()})")
        print(f"{'='*70}")
    
    # Create environments
    if reward_type == 'simple_return':
        create_env = create_simple_return_env
    else:
        create_env = create_simple_sharpe_env
    
    train_env = create_env(
        config['data_dir'], 'train', agent_type,
        softmax_temperature=config['softmax_temperature'],
        random_start=True
    )
    val_env = create_env(
        config['data_dir'], 'val', agent_type,
        softmax_temperature=config['softmax_temperature']
    )
    test_env = create_env(
        config['data_dir'], 'test', agent_type,
        softmax_temperature=config['softmax_temperature']
    )
    
    if verbose:
        print(f"{reward_type.capitalize()}Env ({agent_type}, train): {len(train_env.dates)} dates, "
              f"{train_env.n_assets} assets, {train_env.n_features} features")
    
    # Create model
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', train_env, 
                   learning_rate=config['learning_rate'],
                   gamma=config['gamma'],
                   verbose=0)
    elif algorithm == 'SAC':
        model = SAC('MlpPolicy', train_env,
                   learning_rate=config['learning_rate'],
                   gamma=config['gamma'],
                   verbose=0)
    else:  # A2C
        model = A2C('MlpPolicy', train_env,
                   learning_rate=config['learning_rate'],
                   gamma=config['gamma'],
                   verbose=0)
    
    # Train
    save_path = Path(config['models_dir']) / agent_type / 'simple_baseline' / f"{reward_type}_{agent_type}_{algorithm.lower()}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    callback = ValidationCallback(
        val_env=val_env,
        eval_freq=config['eval_freq'],
        patience=config['patience'],
        save_path=str(save_path),
        verbose=1 if verbose else 0
    )
    
    model.learn(
        total_timesteps=config['total_steps'],
        callback=callback,
        progress_bar=False
    )
    
    # Load best model and evaluate
    model = model.__class__.load(save_path)
    
    train_sharpe = train_env.run_full_pass(model)
    val_sharpe = val_env.run_full_pass(model)
    test_sharpe = test_env.run_full_pass(model)
    
    if verbose:
        print(f"\n{agent_type.upper()} {algorithm} Results:")
        print(f"  Train Sharpe: {train_sharpe:.3f}")
        print(f"  Val Sharpe:   {val_sharpe:.3f}")
        print(f"  Test Sharpe:  {test_sharpe:.3f}")
    
    return {
        'agent_type': agent_type,
        'algorithm': algorithm,
        'reward_type': reward_type,
        'train_sharpe': train_sharpe,
        'val_sharpe': val_sharpe,
        'test_sharpe': test_sharpe,
        'model_path': str(save_path)
    }


def main():
    """
    Main function to train both approaches and compare.
    """
    
    print("="*70)
    print("SIMPLE BASELINE REWARD APPROACHES")
    print("="*70)
    
    # Configuration
    BASE_CONFIG = {
        'data_dir': '../data_hierarchical',
        'models_dir': 'models',  # Base models directory
        'total_steps': 300_000,
        'eval_freq': 5_000,
        'patience': 5,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'softmax_temperature': 3.0,
        'seed': 42,
    }
    
    Path(BASE_CONFIG['models_dir']).mkdir(parents=True, exist_ok=True)
    
    ALGORITHMS = ['PPO', 'SAC', 'A2C']
    
    # Train Technical Agent with Simple Return
    print("\n" + "="*70)
    print("TRAINING TECHNICAL AGENT (Simple Return)")
    print("="*70)
    
    tech_simple_return_results = []
    for algo in ALGORITHMS:
        result = train_simple_baseline('technical', algo, BASE_CONFIG, 
                                      reward_type='simple_return', verbose=True)
        tech_simple_return_results.append(result)
    
    # Train Technical Agent with Simple Sharpe
    print("\n" + "="*70)
    print("TRAINING TECHNICAL AGENT (Simple Sharpe)")
    print("="*70)
    
    tech_simple_sharpe_results = []
    for algo in ALGORITHMS:
        result = train_simple_baseline('technical', algo, BASE_CONFIG,
                                      reward_type='simple_sharpe', verbose=True)
        tech_simple_sharpe_results.append(result)
    
    # Results summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nSimple Return Results:")
    for r in tech_simple_return_results:
        print(f"  {r['algorithm']}: Train={r['train_sharpe']:.3f}, "
              f"Val={r['val_sharpe']:.3f}, Test={r['test_sharpe']:.3f}")
    
    print("\nSimple Sharpe Results:")
    for r in tech_simple_sharpe_results:
        print(f"  {r['algorithm']}: Train={r['train_sharpe']:.3f}, "
              f"Val={r['val_sharpe']:.3f}, Test={r['test_sharpe']:.3f}")
    
    # Save results
    results = {
        'simple_return': tech_simple_return_results,
        'simple_sharpe': tech_simple_sharpe_results,
    }
    
    results_path = Path(BASE_CONFIG['models_dir']) / 'simple_baseline_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    main()

