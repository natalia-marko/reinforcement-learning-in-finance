"""
Multi-Objective Reward Approach

Combines multiple objectives with explicit penalties:
- Return maximization
- Volatility control
- Diversification (concentration penalty)
- Turnover minimization (transaction cost proxy)

This is more practical for real-world trading but requires hyperparameter tuning.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from pathlib import Path
import json
import math

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch as th


class MultiObjectiveEnv(gym.Env):
    """
    Portfolio environment with multi-objective reward function.
    
    Reward = Return - λ1*Volatility - λ2*Concentration - λ3*Turnover
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        tickers: list,
        feature_cols: list,
        softmax_temperature: float = 3.0,
        # Reward weight
        return_scale: float = 10.0,
        # Penalty weights
        volatility_penalty: float = 0.1,
        concentration_penalty: float = 0.5,
        turnover_penalty: float = 0.01,
        # Rolling window for volatility
        vol_window: int = 12,
        random_start: bool = False,
    ):
        """
        Initialize Multi-Objective environment.
        
        Parameters
        ----------
        return_scale : float
            Scale factor for return reward (higher = more emphasis on returns)
        volatility_penalty : float
            Weight for volatility penalty (higher = more penalty)
        concentration_penalty : float
            Weight for concentration penalty (penalizes max weight)
        turnover_penalty : float
            Weight for turnover penalty (transaction cost proxy)
        vol_window : int
            Window for rolling volatility calculation
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
        
        # Reward and penalty weights
        self.return_scale = return_scale
        self.lambda_vol = volatility_penalty
        self.lambda_conc = concentration_penalty
        self.lambda_turn = turnover_penalty
        
        # Volatility tracking
        self.vol_window = vol_window
        self.alpha_vol = 2.0 / (vol_window + 1.0)
        
        # Validate
        assert 'ticker' in features_df.columns
        assert set(tickers).issubset(returns_df.columns)
        
        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_assets * self.n_features,), dtype=np.float32
        )
        
        self.dates = sorted(features_df.index.unique())
        
        # Episode state
        self._t = None
        self._weights = None
        self._prev_weights = None
        
        # For volatility penalty
        self._mean_return = 0.0
        self._ema_var = 1e-6
        self._n_steps = 0
        
    def _get_obs(self):
        """Get current observation."""
        date = self.dates[self._t]
        date_features = self.features_df.loc[[date]]
        
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
        
        # Starting position
        if self.random_start and len(self.dates) > 10:
            max_start = len(self.dates) - 5
            self._t = np.random.randint(1, max_start)
        else:
            self._t = 1
        
        # Initialize weights (equal weight)
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        self._prev_weights = self._weights.copy()
        
        # Reset tracking
        self._mean_return = 0.0
        self._ema_var = 1e-6
        self._n_steps = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step with multi-objective reward."""
        # Convert action to weights
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        w = self._softmax(action)
        
        # Check if future return exists
        if self._t + 1 >= len(self.dates):
            return self._get_obs(), 0.0, True, False, {}
        
        # Get next period's returns
        next_date = self.dates[self._t + 1]
        r_vec = self.returns_df.loc[next_date, self.tickers].values.astype(np.float32)
        
        # Portfolio return (log return)
        port_log_r = float(np.dot(w, r_vec))
        
        # Update volatility tracking (EMA)
        self._n_steps += 1
        delta = port_log_r - self._mean_return
        self._mean_return += self.alpha_vol * delta
        self._ema_var = (1.0 - self.alpha_vol) * self._ema_var + self.alpha_vol * delta**2
        
        # === OBJECTIVE 1: Return (maximize) ===
        reward_return = port_log_r * self.return_scale
        
        # === PENALTY 1: Volatility (penalize high volatility) ===
        instantaneous_vol = abs(port_log_r - self._mean_return)
        penalty_volatility = -self.lambda_vol * instantaneous_vol
        
        # === PENALTY 2: Concentration (penalize large single positions) ===
        max_weight = np.max(w)
        # Penalize if any weight > 30%
        if max_weight > 0.3:
            penalty_concentration = -self.lambda_conc * (max_weight - 0.3)
        else:
            penalty_concentration = 0.0
        
        # === PENALTY 3: Turnover (penalize large changes in weights) ===
        turnover = np.sum(np.abs(w - self._prev_weights))
        penalty_turnover = -self.lambda_turn * turnover
        
        # === COMBINED REWARD ===
        reward = reward_return + penalty_volatility + penalty_concentration + penalty_turnover
        reward = np.clip(reward, -10.0, 10.0)
        
        # Update state
        self._prev_weights = self._weights.copy()
        self._weights = w
        self._t += 1
        terminated = self._t >= (len(self.dates) - 1)
        
        info = {
            'port_log_r': port_log_r,
            'reward_return': reward_return,
            'penalty_volatility': penalty_volatility,
            'penalty_concentration': penalty_concentration,
            'penalty_turnover': penalty_turnover,
            'max_weight': max_weight,
            'turnover': turnover,
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
        
        return float((returns.mean() / returns.std()) * math.sqrt(52))


def create_multi_objective_env(data_dir: str, split: str, agent_type: str, **kwargs):
    """Create environment with Multi-Objective reward."""
    # Fix path - go up one level from choosing_the_best_reward subdirectory
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
    
    # Get indicator features only
    if agent_type == 'technical':
        feature_cols = metadata.get('technical_indicator_features',
                                     [c for c in metadata['technical_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    else:
        feature_cols = metadata.get('sentiment_indicator_features',
                                     [c for c in metadata['sentiment_features'] 
                                      if c not in ['open', 'high', 'low', 'close', 'volume', 'return']])
    
    feature_cols = [c for c in feature_cols if c in features_df.columns]
    
    env = MultiObjectiveEnv(
        features_df=features_df,
        returns_df=returns_df,
        tickers=tickers,
        feature_cols=feature_cols,
        **kwargs
    )
    
    print(f"MultiObjective-{agent_type.capitalize()}Env ({split}): {len(env.dates)} dates, {env.n_assets} assets, {env.n_features} features")
    
    return env


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


def train_multi_objective(agent_type: str, algorithm: str, config: dict, verbose: bool = True):
    """Train agent with Multi-Objective reward."""
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {agent_type.upper()} Agent with {algorithm} (Multi-Objective)")
        print(f"{'='*70}")
    
    # Create environments
    train_env = create_multi_objective_env(
        config['data_dir'], 'train', agent_type,
        softmax_temperature=config['softmax_temperature'],
        return_scale=config['return_scale'],
        volatility_penalty=config['volatility_penalty'],
        concentration_penalty=config['concentration_penalty'],
        turnover_penalty=config['turnover_penalty'],
        vol_window=config['vol_window'],
        random_start=True
    )
    val_env = create_multi_objective_env(
        config['data_dir'], 'val', agent_type,
        softmax_temperature=config['softmax_temperature'],
        return_scale=config['return_scale'],
        volatility_penalty=config['volatility_penalty'],
        concentration_penalty=config['concentration_penalty'],
        turnover_penalty=config['turnover_penalty'],
        vol_window=config['vol_window']
    )
    test_env = create_multi_objective_env(
        config['data_dir'], 'test', agent_type,
        softmax_temperature=config['softmax_temperature'],
        return_scale=config['return_scale'],
        volatility_penalty=config['volatility_penalty'],
        concentration_penalty=config['concentration_penalty'],
        turnover_penalty=config['turnover_penalty'],
        vol_window=config['vol_window']
    )
    
    vec_env = DummyVecEnv([lambda: train_env])
    
    # Create model
    model_kwargs = {
        'policy': 'MlpPolicy',
        'env': vec_env,
        'learning_rate': config['learning_rate'],
        'gamma': config['gamma'],
        'seed': config['seed'],
        'verbose': 0 if not verbose else 1,
    }
    
    if algorithm == 'PPO':
        model = PPO(
            **model_kwargs,
            n_steps=2048,
            batch_size=512,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256, 256]),
        )
    elif algorithm == 'SAC':
        model = SAC(
            **model_kwargs,
            buffer_size=200_000,
            batch_size=1024,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=[256, 256], qf=[256, 256]))
        )
    elif algorithm == 'A2C':
        model = A2C(
            **model_kwargs,
            n_steps=5,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256, 256])
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Training
    save_path = Path(config['models_dir']) / agent_type / 'multi_objective' / f"multi_obj_{agent_type}_{algorithm.lower()}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    callback = ValidationCallback(
        val_env=val_env,
        eval_freq=config['eval_freq'],
        patience=config['patience'],
        save_path=str(save_path),
        verbose=1 if verbose else 0
    )
    
    model.learn(total_timesteps=config['total_steps'], callback=callback)
    
    # Load best and evaluate
    if save_path.exists():
        if algorithm == 'PPO':
            model = PPO.load(str(save_path), env=vec_env)
        elif algorithm == 'SAC':
            model = SAC.load(str(save_path), env=vec_env)
        elif algorithm == 'A2C':
            model = A2C.load(str(save_path), env=vec_env)
    
    train_sharpe = train_env.run_full_pass(model)
    val_sharpe = val_env.run_full_pass(model)
    test_sharpe = test_env.run_full_pass(model)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Train Sharpe: {train_sharpe:.3f}")
        print(f"  Val Sharpe:   {val_sharpe:.3f}")
        print(f"  Test Sharpe:  {test_sharpe:.3f}")
    
    return {
        'agent_type': agent_type,
        'algorithm': algorithm,
        'reward_type': 'multi_objective',
        'train_sharpe': train_sharpe,
        'val_sharpe': val_sharpe,
        'test_sharpe': test_sharpe,
        'model_path': str(save_path)
    }


def main():
    """Main training and evaluation."""
    
    print("="*70)
    print("MULTI-OBJECTIVE REWARD APPROACH")
    print("="*70)
    
    # Base Configuration
    BASE_CONFIG = {
        'data_dir': '../../data_hierarchical',
        'models_dir': '../../models',  # Base models directory
        'total_steps': 300_000,
        'eval_freq': 5_000,
        'patience': 5,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'softmax_temperature': 3.0,
        'vol_window': 12,
        'seed': 42,
    }
    
    # Optimized penalty configurations based on best performance
    TECHNICAL_CONFIG = {
        **BASE_CONFIG,
        'return_scale': 8.0,            # Optimized for best performance
        'volatility_penalty': 0.050,    # Reduced penalty for better returns
        'concentration_penalty': 0.250, # Moderate diversification requirement
        'turnover_penalty': 0.005,      # Minimal trading penalty
    }
    
    SENTIMENT_CONFIG = {
        **BASE_CONFIG,
        'return_scale': 3.0,            # Much less aggressive (sentiment is noisier)
        'volatility_penalty': 0.2,      # Strong penalty for stability
        'concentration_penalty': 1.0,   # Force diversification
        'turnover_penalty': 0.02,       # Minimize rebalancing
    }
    
    # Default config (for backward compatibility)
    # Create models directories for both agents
    Path(BASE_CONFIG['models_dir']).mkdir(parents=True, exist_ok=True)
    
    ALGORITHMS = ['PPO', 'SAC', 'A2C']
    
    # Train Technical Agent
    print("\n" + "="*70)
    print("TRAINING TECHNICAL AGENT (Multi-Objective)")
    print("="*70)
    print(f"Return Scale: {TECHNICAL_CONFIG['return_scale']}")
    print(f"Penalties: Vol={TECHNICAL_CONFIG['volatility_penalty']}, Conc={TECHNICAL_CONFIG['concentration_penalty']}, Turn={TECHNICAL_CONFIG['turnover_penalty']}")
    
    tech_results = []
    for algo in ALGORITHMS:
        result = train_multi_objective('technical', algo, TECHNICAL_CONFIG, verbose=True)
        tech_results.append(result)
    
    tech_df = pd.DataFrame(tech_results).sort_values('val_sharpe', ascending=False)
    
    print("\n" + "="*70)
    print("TECHNICAL AGENT RESULTS")
    print("="*70)
    print(tech_df.to_string(index=False))
    
    best_tech = tech_df.iloc[0].to_dict()
    print(f"\nBest: {best_tech['algorithm']} (Test Sharpe: {best_tech['test_sharpe']:.3f})")
    
    # Train Sentiment Agent
    print("\n" + "="*70)
    print("TRAINING SENTIMENT AGENT (Multi-Objective)")
    print("="*70)
    print(f"Return Scale: {SENTIMENT_CONFIG['return_scale']}")
    print(f"Penalties: Vol={SENTIMENT_CONFIG['volatility_penalty']}, Conc={SENTIMENT_CONFIG['concentration_penalty']}, Turn={SENTIMENT_CONFIG['turnover_penalty']}")
    
    sent_results = []
    for algo in ALGORITHMS:
        result = train_multi_objective('sentiment', algo, SENTIMENT_CONFIG, verbose=True)
        sent_results.append(result)
    
    sent_df = pd.DataFrame(sent_results).sort_values('val_sharpe', ascending=False)
    
    print("\n" + "="*70)
    print("SENTIMENT AGENT RESULTS")
    print("="*70)
    print(sent_df.to_string(index=False))
    
    best_sent = sent_df.iloc[0].to_dict()
    print(f"\nBest: {best_sent['algorithm']} (Test Sharpe: {best_sent['test_sharpe']:.3f})")
    
    # Save results
    results = {
        'approach': 'multi_objective',
        'technical': best_tech,
        'sentiment': best_sent,
        'configs': {
            'technical': TECHNICAL_CONFIG,
            'sentiment': SENTIMENT_CONFIG,
        },
        'all_results': {
            'technical': tech_results,
            'sentiment': sent_results
        }
    }
    
    results_path = Path(BASE_CONFIG['models_dir']) / 'multi_objective_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*70)
    print("MULTI-OBJECTIVE TRAINING COMPLETE")
    print("="*70)
    print(f"\nTechnical: {best_tech['algorithm']} - Test Sharpe: {best_tech['test_sharpe']:.3f}")
    print(f"Sentiment: {best_sent['algorithm']} - Test Sharpe: {best_sent['test_sharpe']:.3f}")
    
    return results


if __name__ == '__main__':
    results = main()


if __name__ == '__main__':
    results = main()

