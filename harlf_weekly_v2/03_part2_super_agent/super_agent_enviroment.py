"""
Multi-Hierarchical RL System - Super Agent Environment (Part 2)
================================================================

The Super Agent sits above Technical and Sentiment agents and learns:
1. WHEN to trust each agent (regime-dependent)
2. HOW MUCH to trust each agent (dynamic weighting)
3. HOW to blend their recommendations (ensemble strategy)

Input Features:
- Technical agent's recommended weights (N assets)
- Sentiment agent's recommended weights (N assets)
- Recent performance metrics (both agents)
- Disagreement/consensus metrics
- Market context

Output:
- Final portfolio weights (blended)

The Super Agent learns to beat both individual agents by:
- Trusting technical in trending markets
- Trusting sentiment in volatile markets
- Blending when both agree
- Overriding when both are wrong
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

from stable_baselines3 import PPO, SAC, A2C


class SuperAgentEnv(gym.Env):
    """
    Super Agent Environment - Hierarchical Layer 2
    
    Combines Technical and Sentiment agents' recommendations.
    Learns optimal ensemble strategy.
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        technical_agent_path: str,
        sentiment_agent_path: str,
        returns_df: pd.DataFrame,
        tickers: list,
        technical_features_df: pd.DataFrame = None,
        sentiment_features_df: pd.DataFrame = None,
        performance_window: int = 4,  # Weeks to track agent performance
        rolling_vol_window: int = 12,
        random_start: bool = False,
        include_context: bool = True,  # Include market context features
    ):
        """
        Initialize Super Agent environment.
        
        Parameters
        ----------
        technical_agent_path : str
            Path to trained technical agent (.zip)
        sentiment_agent_path : str
            Path to trained sentiment agent (.zip)
        returns_df : pd.DataFrame
            Wide-format returns (columns = tickers)
        tickers : list
            List of tickers
        technical_features_df : pd.DataFrame, optional
            Technical features (for generating recommendations)
        sentiment_features_df : pd.DataFrame, optional
            Sentiment features (for generating recommendations)
        performance_window : int
            Window to calculate agent performance
        rolling_vol_window : int
            Window for volatility in reward
        random_start : bool
            Random episode start
        include_context : bool
            Include market context in observations
        """
        super().__init__()
        
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.returns_df = returns_df
        self.performance_window = performance_window
        self.include_context = include_context
        self.random_start = random_start
        
        # Feature DataFrames
        self.technical_features_df = technical_features_df
        self.sentiment_features_df = sentiment_features_df
        
        # Load base agents
        print(f"Loading base agents...")
        self.technical_agent = self._load_agent(technical_agent_path)
        self.sentiment_agent = self._load_agent(sentiment_agent_path)
        print(f"  ✓ Technical agent loaded")
        print(f"  ✓ Sentiment agent loaded")
        
        # Dates available
        self.dates = sorted(returns_df.index)
        
        # Observation space components:
        # 1. Technical weights (N)
        # 2. Sentiment weights (N)
        # 3. Recent tech performance (performance_window)
        # 4. Recent sent performance (performance_window)
        # 5. Disagreement metrics (2: magnitude, direction)
        # 6. Recent volatility (1)
        # 7. Market context (optional, 4 features)
        
        obs_size = (
            self.n_assets +  # tech weights
            self.n_assets +  # sent weights
            self.performance_window +  # tech recent returns
            self.performance_window +  # sent recent returns
            2 +  # disagreement metrics
            1    # recent volatility
        )
        
        if include_context:
            obs_size += 4  # market context features
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: blend weights
        # 2 values (tech_logit, sent_logit) -> softmax -> blend weights
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Volatility estimation params
        self.W = rolling_vol_window
        self.alpha = 2.0 / (self.W + 1.0)
        self.start_idx = max(self.W, performance_window, 1)
        
        # Episode state
        self._t = None
        self._tech_returns_history = []
        self._sent_returns_history = []
        self._super_returns_history = []
        self._ema_mean = 0.0
        self._ema_var = 0.0
        self._ema_initialized = False
        
        print(f"SuperAgentEnv initialized:")
        print(f"  Assets: {self.n_assets}")
        print(f"  Dates: {len(self.dates)}")
        print(f"  Observation size: {obs_size}")
        print(f"  Start index: {self.start_idx}")
    
    def _load_agent(self, path: str):
        """Load a trained agent."""
        path = str(path)
        
        # Try each algorithm
        for algo_class in [PPO, SAC, A2C]:
            try:
                return algo_class.load(path)
            except:
                continue
        
        raise ValueError(f"Could not load agent from {path}")
    
    def _get_agent_weights(self, agent, features: np.ndarray) -> np.ndarray:
        """
        Get portfolio weights from an agent given features.
        
        Parameters
        ----------
        agent : trained RL agent
            Technical or sentiment agent
        features : np.ndarray
            Feature vector for the agent
        
        Returns
        -------
        weights : np.ndarray
            Portfolio weights from the agent
        """
        # Get action from agent (deterministic)
        action, _ = agent.predict(features, deterministic=True)
        
        # Convert action to weights (softmax with temperature=5.0)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = action * 5.0  # temperature
        action = action - np.max(action)
        e = np.exp(action)
        weights = e / (e.sum() + 1e-12)
        
        return weights
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation for super agent."""
        date = self.dates[self._t]
        
        # Get base agent recommendations
        tech_weights = self._get_technical_weights(date)
        sent_weights = self._get_sentiment_weights(date)
        
        # Calculate recent performance
        tech_perf = self._calculate_recent_performance('technical')
        sent_perf = self._calculate_recent_performance('sentiment')
        
        # Calculate disagreement
        disagreement_mag = np.abs(tech_weights - sent_weights).mean()
        disagreement_dir = np.sign(tech_weights - sent_weights).mean()
        
        # Recent volatility
        if len(self._super_returns_history) >= 4:
            recent_vol = np.std(self._super_returns_history[-4:]) * np.sqrt(52)
        else:
            recent_vol = 0.2  # Default
        
        # Combine observation
        obs = np.concatenate([
            tech_weights,
            sent_weights,
            tech_perf,
            sent_perf,
            [disagreement_mag, disagreement_dir],
            [recent_vol]
        ])
        
        # Add market context if enabled
        if self.include_context:
            context = self._get_market_context(date)
            obs = np.concatenate([obs, context])
        
        return obs.astype(np.float32)
    
    def _get_technical_weights(self, date: pd.Timestamp) -> np.ndarray:
        """Get technical agent's recommended weights for this date."""
        if self.technical_features_df is not None:
            date_features = self.technical_features_df.loc[[date]]
            
            # Pivot to matrix: rows=tickers, cols=features
            feature_cols = [c for c in date_features.columns 
                          if c not in ['ticker', 'open', 'high', 'low', 'close', 'volume', 'return']]
            
            feature_matrix = (
                date_features[date_features['ticker'].isin(self.tickers)]
                .set_index('ticker')[feature_cols]
                .reindex(self.tickers)
                .fillna(0.0)
                .values
                .astype(np.float32)
                .flatten()
            )
            
            weights = self._get_agent_weights(self.technical_agent, feature_matrix)
        else:
            # Fallback: equal weights
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def _get_sentiment_weights(self, date: pd.Timestamp) -> np.ndarray:
        """Get sentiment agent's recommended weights for this date."""
        if self.sentiment_features_df is not None:
            date_features = self.sentiment_features_df.loc[[date]]
            
            feature_cols = [c for c in date_features.columns 
                          if c not in ['ticker', 'open', 'high', 'low', 'close', 'volume', 'return']]
            
            feature_matrix = (
                date_features[date_features['ticker'].isin(self.tickers)]
                .set_index('ticker')[feature_cols]
                .reindex(self.tickers)
                .fillna(0.0)
                .values
                .astype(np.float32)
                .flatten()
            )
            
            weights = self._get_agent_weights(self.sentiment_agent, feature_matrix)
        else:
            # Fallback: equal weights
            weights = np.ones(self.n_assets) / self.n_assets
        
        return weights
    
    def _calculate_recent_performance(self, agent_type: str) -> np.ndarray:
        """Calculate recent performance of an agent."""
        if agent_type == 'technical':
            history = self._tech_returns_history
        else:
            history = self._sent_returns_history
        
        # Return last N returns (padded with zeros if needed)
        if len(history) >= self.performance_window:
            recent = history[-self.performance_window:]
        else:
            recent = [0.0] * (self.performance_window - len(history)) + history
        
        return np.array(recent, dtype=np.float32)
    
    def _get_market_context(self, date: pd.Timestamp) -> np.ndarray:
        """Get market context features."""
        # Simple context: recent market returns, volatility, trend, momentum
        if len(self._super_returns_history) >= 4:
            recent_returns = self._super_returns_history[-4:]
            
            avg_return = np.mean(recent_returns)
            volatility = np.std(recent_returns) * np.sqrt(52)
            trend = 1.0 if len(recent_returns) >= 2 and recent_returns[-1] > recent_returns[0] else -1.0
            momentum = sum(1 if r > 0 else -1 for r in recent_returns) / len(recent_returns)
        else:
            avg_return = 0.0
            volatility = 0.2
            trend = 0.0
            momentum = 0.0
        
        return np.array([avg_return, volatility, trend, momentum], dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Starting position
        if self.random_start:
            max_start = len(self.dates) - 10
            if max_start > self.start_idx:
                self._t = np.random.randint(self.start_idx, max_start)
            else:
                self._t = self.start_idx
        else:
            self._t = self.start_idx
        
        # Reset histories
        self._tech_returns_history = []
        self._sent_returns_history = []
        self._super_returns_history = []
        
        # Initialize EMA
        self._ema_mean = 0.0
        self._ema_var = 1e-3
        self._ema_initialized = False
        
        return self._get_obs(), {}
    
    def step(self, action):
        """Execute one step."""
        # Convert action to blend weights
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Softmax to get blend weights
        action = action - np.max(action)
        e = np.exp(action)
        blend_weights = e / (e.sum() + 1e-12)
        
        # Get base agent recommendations
        date = self.dates[self._t]
        tech_weights = self._get_technical_weights(date)
        sent_weights = self._get_sentiment_weights(date)
        
        # Blend portfolios
        final_weights = blend_weights[0] * tech_weights + blend_weights[1] * sent_weights
        
        # Ensure weights sum to 1
        final_weights = final_weights / (final_weights.sum() + 1e-12)
        
        # Check if next return exists
        if self._t + 1 >= len(self.dates):
            return self._get_obs(), 0.0, True, False, {}
        
        # Get next period's returns
        next_date = self.dates[self._t + 1]
        r_vec = self.returns_df.loc[next_date, self.tickers].values.astype(np.float32)
        
        # Calculate returns for each strategy
        tech_return = float(np.dot(tech_weights, r_vec))
        sent_return = float(np.dot(sent_weights, r_vec))
        super_return = float(np.dot(final_weights, r_vec))
        
        # Store performance
        self._tech_returns_history.append(tech_return)
        self._sent_returns_history.append(sent_return)
        self._super_returns_history.append(super_return)
        
        # Update EMA
        if not self._ema_initialized:
            self._ema_mean = super_return
            self._ema_var = 1e-3
            self._ema_initialized = True
        else:
            delta = super_return - self._ema_mean
            self._ema_mean += self.alpha * delta
            self._ema_var = (1.0 - self.alpha) * (self._ema_var + self.alpha * delta * delta)
        
        # Calculate reward (Sharpe-like)
        running_vol = max(math.sqrt(max(self._ema_var, 0.0)), 1e-4)
        reward = np.clip(super_return / running_vol * math.sqrt(52.0), -10.0, 10.0)
        
        # Advance time
        self._t += 1
        terminated = self._t >= (len(self.dates) - 1)
        
        info = {
            'super_return': super_return,
            'tech_return': tech_return,
            'sent_return': sent_return,
            'blend_weights': blend_weights.copy(),
            'final_weights': final_weights.copy(),
            'running_vol': running_vol
        }
        
        return self._get_obs(), float(reward), terminated, False, info
    
    def evaluate_full_episode(self, model, split_name: str = 'test') -> Dict:
        """
        Evaluate super agent on full episode and compare to base agents.
        
        Returns
        -------
        results : dict
            Performance metrics for super, tech, and sent agents
        """
        obs, _ = self.reset()
        
        tech_returns = []
        sent_returns = []
        super_returns = []
        blend_weights_history = []
        
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, info = self.step(action)
            
            tech_returns.append(info['tech_return'])
            sent_returns.append(info['sent_return'])
            super_returns.append(info['super_return'])
            blend_weights_history.append(info['blend_weights'])
        
        # Convert to arrays
        tech_returns = np.array(tech_returns)
        sent_returns = np.array(sent_returns)
        super_returns = np.array(super_returns)
        blend_weights_history = np.array(blend_weights_history)
        
        # Calculate metrics
        def calc_metrics(returns):
            if len(returns) < 2 or returns.std() < 1e-12:
                return {'sharpe': 0.0, 'total_return': 0.0, 'volatility': 0.0}
            return {
                'sharpe': float((returns.mean() / returns.std()) * math.sqrt(52.0)),
                'total_return': float(returns.sum()),
                'volatility': float(returns.std() * math.sqrt(52.0)),
                'mean_return': float(returns.mean())
            }
        
        results = {
            'super': calc_metrics(super_returns),
            'technical': calc_metrics(tech_returns),
            'sentiment': calc_metrics(sent_returns),
            'blend_weights': {
                'mean_tech': float(blend_weights_history[:, 0].mean()),
                'mean_sent': float(blend_weights_history[:, 1].mean()),
                'std_tech': float(blend_weights_history[:, 0].std()),
                'std_sent': float(blend_weights_history[:, 1].std())
            },
            'improvement': {
                'over_tech': float(super_returns.mean() - tech_returns.mean()),
                'over_sent': float(super_returns.mean() - sent_returns.mean()),
                'over_best': float(super_returns.mean() - max(tech_returns.mean(), sent_returns.mean()))
            }
        }
        
        return results


def create_super_agent_env(
    data_dir: str,
    models_dir: str,
    split: str = 'train',
    **kwargs
) -> SuperAgentEnv:
    """
    Factory function to create Super Agent environment.
    
    Parameters
    ----------
    data_dir : str
        Path to data directory (e.g., 'data_hierarchical_enhanced')
    models_dir : str
        Path to models directory with trained base agents
    split : str
        'train', 'val', or 'test'
    **kwargs
        Additional environment parameters
    
    Returns
    -------
    env : SuperAgentEnv
        Initialized super agent environment
    """
    data_dir = Path(data_dir)
    models_dir = Path(models_dir)
    
    # Load returns
    returns_df = pd.read_csv(data_dir / f'returns_{split}.csv', index_col=0, parse_dates=True)
    
    # Load features
    tech_features = pd.read_csv(data_dir / 'technical' / f'{split}.csv', index_col=0, parse_dates=True)
    sent_features = pd.read_csv(data_dir / 'sentiment' / f'{split}.csv', index_col=0, parse_dates=True)
    
    # Load metadata
    with open(data_dir / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    tickers = metadata['tickers']
    
    # Find best base agents
    with open(models_dir / 'best_models_part1.json', 'r') as f:
        best_models = json.load(f)
    
    tech_model_path = models_dir / Path(best_models['technical']['model_path']).name
    sent_model_path = models_dir / Path(best_models['sentiment']['model_path']).name
    
    # Create environment
    env = SuperAgentEnv(
        technical_agent_path=str(tech_model_path),
        sentiment_agent_path=str(sent_model_path),
        returns_df=returns_df,
        tickers=tickers,
        technical_features_df=tech_features,
        sentiment_features_df=sent_features,
        **kwargs
    )
    
    return env


if __name__ == '__main__':
    # Test super agent environment
    print("Testing Super Agent Environment...")
    
    try:
        env = create_super_agent_env(
            data_dir='data_hierarchical_enhanced',
            models_dir='models_part1',
            split='train'
        )
        
        obs, _ = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  Action space: {env.action_space}")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(f"  Step successful!")
        print(f"    Reward: {reward:.3f}")
        print(f"    Blend weights: {info['blend_weights']}")
        
        print("  ✓ Super Agent environment works!")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()