"""
Portfolio Environments
======================
All environment code in one file!
Your original logic is preserved exactly.

CHANGES IN THIS VERSION:
- Added transaction_cost parameter to environment
- Transaction costs applied to portfolio returns
- Better handling of edge cases
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from stable_baselines3.common.callbacks import BaseCallback

from utils import load_data, compute_sharpe
from rewards import create_reward


# ============================================================================
# Portfolio Environment (combines all your environments)
# ============================================================================

class PortfolioEnv(gym.Env):
    """
    Portfolio management environment.
    Works with any reward function!
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, features_df, returns_df, tickers, feature_cols,
                 reward_function, softmax_temperature=3.0, random_start=False,
                 transaction_cost=0.0):
        """
        Initialize environment.
        
        Parameters
        ----------
        features_df : DataFrame
            Features with 'ticker' column
        returns_df : DataFrame
            Returns (wide format)
        tickers : list
            List of tickers
        feature_cols : list
            Feature column names
        reward_function : Reward object
            Reward function (from rewards.py)
        softmax_temperature : float
            Temperature for softmax
        random_start : bool
            Random starting position
        transaction_cost : float
            Transaction cost per unit turnover (e.g., 0.001 = 10 bps)
        """
        
        super().__init__()
        
        self.features_df = features_df
        self.returns_df = returns_df
        self.tickers = tickers
        self.feature_cols = feature_cols
        self.n_assets = len(tickers)
        self.n_features = len(feature_cols)
        self.reward_fn = reward_function
        self.temperature = softmax_temperature
        self.random_start = random_start
        self.transaction_cost = transaction_cost  # NEW
        
        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_assets * self.n_features,), dtype=np.float32
        )
        
        self.dates = sorted(features_df.index.unique())
        self._t = None
        self._weights = None
        self._total_transaction_cost = 0.0  # Track cumulative costs
    
    def _get_obs(self):
        """Get observation."""
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
        """Convert actions to weights."""
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
        
        # Initialize weights
        self._weights = np.ones(self.n_assets, dtype=np.float32) / self.n_assets
        
        # Reset tracking
        self._total_transaction_cost = 0.0
        
        # Reset reward function
        self.reward_fn.reset()
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step."""
        # Convert action to weights
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        w = self._softmax(action)
        
        # Check if done
        if self._t + 1 >= len(self.dates):
            return self._get_obs(), 0.0, True, False, {}
        
        # Get returns
        next_date = self.dates[self._t + 1]
        r_vec = self.returns_df.loc[next_date, self.tickers].values.astype(np.float32)
        
        # Portfolio return (before costs)
        port_return = float(np.dot(w, r_vec))
        
        # Apply transaction costs
        if self.transaction_cost > 0:
            turnover = np.sum(np.abs(w - self._weights))
            cost = self.transaction_cost * turnover
            port_return -= cost
            self._total_transaction_cost += cost
        
        # Compute reward
        # Multi-objective needs weights, others don't
        if hasattr(self.reward_fn, 'compute') and 'weights' in self.reward_fn.compute.__code__.co_varnames:
            reward = self.reward_fn.compute(port_return, w)
        else:
            reward = self.reward_fn.compute(port_return)
        
        # Update state
        self._weights = w
        self._t += 1
        terminated = self._t >= (len(self.dates) - 1)
        
        info = {
            'port_log_r': port_return, 
            'weights': w.copy(),
            'turnover': np.sum(np.abs(w - self._weights)) if self.transaction_cost > 0 else 0.0,
            'transaction_cost': cost if self.transaction_cost > 0 else 0.0,
            'total_transaction_cost': self._total_transaction_cost
        }
        
        return self._get_obs(), float(reward), terminated, False, info
    
    def run_full_pass(self, model):
        """
        Run full episode and compute Sharpe.
        
        Parameters
        ----------
        model : RL model
            Trained model
        
        Returns
        -------
        float : Sharpe ratio
        """
        
        obs, _ = self.reset()
        returns = []
        total_turnover = 0.0
        total_cost = 0.0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.step(action)
            if 'port_log_r' in info:
                returns.append(info['port_log_r'])
                if 'turnover' in info:
                    total_turnover += info['turnover']
                if 'transaction_cost' in info:
                    total_cost += info['transaction_cost']
        
        sharpe = compute_sharpe(returns)
        
        # Store additional metrics for reference
        self._last_eval_metrics = {
            'sharpe': sharpe,
            'total_return': sum(returns),
            'avg_turnover': total_turnover / len(returns) if returns else 0,
            'total_transaction_cost': total_cost
        }
        
        return sharpe


# ============================================================================
# Validation Callback (single source of truth!)
# ============================================================================

class ValidationCallback(BaseCallback):
    """Callback for validation and early stopping."""
    
    def __init__(self, val_env, eval_freq=5000, patience=10, 
                 save_path=None, verbose=1):
        """
        Initialize callback.
        
        Parameters
        ----------
        val_env : PortfolioEnv
            Validation environment
        eval_freq : int
            Evaluate every N steps
        patience : int
            Early stopping patience
        save_path : str
            Path to save best model
        verbose : int
            0=silent, 1=normal
        """
        
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.save_path = save_path
        self.best_sharpe = -np.inf
        self.no_improve = 0
        self.eval_history = []  # NEW: Track evaluation history
    
    def _on_step(self):
        """Called after each step."""
        
        if self.eval_freq <= 0 or (self.n_calls % self.eval_freq) != 0:
            return True
        
        # Evaluate
        val_sharpe = self.val_env.run_full_pass(self.model)
        
        # Store history
        self.eval_history.append({
            'step': self.num_timesteps,
            'val_sharpe': val_sharpe,
            'best_sharpe': self.best_sharpe
        })
        
        if self.verbose:
            print(f"\n[Step {self.num_timesteps:,}] Val Sharpe: {val_sharpe:.3f} "
                  f"(Best: {self.best_sharpe:.3f})")
            
            # Print transaction cost info if available
            if hasattr(self.val_env, '_last_eval_metrics'):
                metrics = self.val_env._last_eval_metrics
                if metrics.get('total_transaction_cost', 0) > 0:
                    print(f"  Avg Turnover: {metrics['avg_turnover']:.3f}, "
                          f"Total Cost: {metrics['total_transaction_cost']:.5f}")
        
        # Check improvement
        if val_sharpe > self.best_sharpe + 1e-6:
            self.best_sharpe = val_sharpe
            self.no_improve = 0
            
            if self.save_path:
                self.model.save(self.save_path)
                if self.verbose:
                    print(f"  ✓ New best model saved")
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                if self.verbose:
                    print(f"  ✗ Early stopping (no improvement for {self.patience} evaluations)")
                return False
        
        return True


# ============================================================================
# Helper to create environments easily
# ============================================================================

def create_env(data_dir, agent_type, split, reward_type, reward_kwargs=None,
               softmax_temperature=3.0, random_start=False, transaction_cost=0.0):
    """
    Create environment (simple wrapper).
    
    Parameters
    ----------
    data_dir : str
        Data directory
    agent_type : str
        'technical' or 'sentiment'
    split : str
        'train', 'val', or 'test'
    reward_type : str
        'ema_sharpe', 'differential_sharpe', 'multi_objective', etc.
    reward_kwargs : dict, optional
        Reward parameters
    softmax_temperature : float
        Softmax temperature
    random_start : bool
        Random starting position
    transaction_cost : float
        Transaction cost per unit turnover (NEW)
    
    Returns
    -------
    PortfolioEnv
    
    Examples
    --------
    >>> env = create_env('data_hierarchical', 'technical', 'train', 
    ...                  'ema_sharpe', transaction_cost=0.001)
    """
    
    if reward_kwargs is None:
        reward_kwargs = {}
    
    # Load data
    features_df, returns_df, tickers, feature_cols = load_data(data_dir, agent_type, split)
    
    # Create reward
    reward_fn = create_reward(reward_type, **reward_kwargs)
    
    # Create environment
    env = PortfolioEnv(
        features_df=features_df,
        returns_df=returns_df,
        tickers=tickers,
        feature_cols=feature_cols,
        reward_function=reward_fn,
        softmax_temperature=softmax_temperature,
        random_start=random_start,
        transaction_cost=transaction_cost  # NEW
    )
    
    return env


if __name__ == '__main__':
    print("Environment loaded!")
    print("\n⚠️  REFACTORED VERSION - Key changes:")
    print("  - Added transaction_cost parameter")
    print("  - Transaction costs applied to returns")
    print("  - Track turnover and costs in info dict")
    print("\nYou can create any environment with:")
    print("  env = create_env('data_hierarchical', 'technical', 'train',")
    print("                   'ema_sharpe', transaction_cost=0.0025)")