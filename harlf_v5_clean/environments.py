"""
Portfolio environment classes for reinforcement learning

This module contains all environment classes used for training:
- Base agents (technical and sentiment-based)
- Super agent (combines base agents)
- Meta agent (refines super agent)
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler


class BasePortfolioEnv(gym.Env):
    """Base class for portfolio environments with common functionality"""
    
    def get_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        
        max_drawdown = 0.0
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }


class SentimentEnv(BasePortfolioEnv):
    """Sentiment-based portfolio environment with proper action space"""
    
    def __init__(self, price_data, sentiment_features, 
                 initial_capital=100_000, transaction_cost=0.002,
                 max_position=0.30, verbose=0):
        super().__init__()
        
        self.price_data = price_data
        self.sentiment_features = sentiment_features
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Action space normalized to [-1, 1] for RL algorithms
        # Will be rescaled to [0, max_position] in step function
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(sentiment_features.shape[1],), dtype=np.float32
        )
        
        self.common_dates = sentiment_features.index.intersection(price_data.index)
        if len(self.common_dates) < 2:
            raise ValueError(f"Only {len(self.common_dates)} overlapping dates!")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        if self.current_step >= len(self.common_dates):
            self.current_step = len(self.common_dates) - 1
        
        current_date = self.common_dates[self.current_step]
        obs = self.sentiment_features.loc[current_date].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)
    
    def step(self, action):
        # Rescale action from [-1, 1] to [0, 1]
        action = (action + 1.0) / 2.0
        action = np.clip(action, 0, 1)
        
        # Normalize to sum to 1 with max position constraint
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self.n_assets) / self.n_assets
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        try:
            price_ratio = self.price_data.loc[next_date] / self.price_data.loc[current_date]
            price_ratio = price_ratio.fillna(1.0)
            price_ratio = np.clip(price_ratio.values, 0.5, 2.0)
            asset_returns = price_ratio - 1.0
        except Exception as e:
            print(f"Warning: Error calculating returns: {e}")
            asset_returns = np.zeros(self.n_assets)
        
        portfolio_return = np.sum(weights * asset_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        portfolio_return = np.clip(portfolio_return, -0.50, 0.50)
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Log return as reward (more stable for RL)
        reward = np.log(1 + portfolio_return + 1e-8)
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights.copy()
        }
        
        return self._get_observation(), reward, done, False, info


class TechnicalEnv(BasePortfolioEnv):
    """Technical indicator-based portfolio environment with proper action space"""
    
    def __init__(self, price_data, technical_features, 
                 initial_capital=100_000, transaction_cost=0.002,
                 max_position=0.30, verbose=0):
        super().__init__()
        
        self.price_data = price_data
        self.technical_features = technical_features
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # Action space normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(technical_features.shape[1],), dtype=np.float32
        )
        
        self.common_dates = technical_features.index.intersection(price_data.index)
        if len(self.common_dates) < 2:
            raise ValueError(f"Only {len(self.common_dates)} overlapping dates!")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        if self.current_step >= len(self.common_dates):
            self.current_step = len(self.common_dates) - 1
        
        current_date = self.common_dates[self.current_step]
        obs = self.technical_features.loc[current_date].values
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs.astype(np.float32)
    
    def step(self, action):
        # Rescale action from [-1, 1] to [0, 1]
        action = (action + 1.0) / 2.0
        action = np.clip(action, 0, 1)
        
        # Normalize to sum to 1 with max position constraint
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(self.n_assets) / self.n_assets
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        try:
            price_ratio = self.price_data.loc[next_date] / self.price_data.loc[current_date]
            price_ratio = price_ratio.fillna(1.0)
            price_ratio = np.clip(price_ratio.values, 0.5, 2.0)
            asset_returns = price_ratio - 1.0
        except Exception as e:
            print(f"Warning: Error calculating returns: {e}")
            asset_returns = np.zeros(self.n_assets)
        
        portfolio_return = np.sum(weights * asset_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        portfolio_return = np.clip(portfolio_return, -0.50, 0.50)
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Log return as reward
        reward = np.log(1 + portfolio_return + 1e-8)
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights.copy()
        }
        
        return self._get_observation(), reward, done, False, info



class SuperAgentEnv(gym.Env):
    """
    Super Agent: Sees BOTH base agent weights AND raw features
    
    Fixes the information bottleneck problem!
    """
    
    def __init__(self, price_data, base_agents, technical_features, 
                 sentiment_features, **config):
        super().__init__()
        
        self.price_data = price_data
        self.base_agents = base_agents
        self.technical_features = technical_features
        self.sentiment_features = sentiment_features
        
        self.n_assets = len(price_data.columns)
        self.n_base_agents = len(base_agents)
        
        # Config
        self.initial_capital = 100_000
        self.transaction_cost = config.get('transaction_cost', 0.002)
        self.max_position = config.get('max_position', 0.30)
        self.max_turnover = config.get('max_turnover', 0.25)
        self.constraint_penalty = config.get('constraint_penalty', 50.0)
        
        # Reward parameters
        self.alpha_returns = config.get('alpha_returns', 1.0)
        self.alpha_mdd = config.get('alpha_mdd', 1.0)
        self.alpha_vol = config.get('alpha_vol', 0.5)
        self.alpha_concentration = config.get('alpha_concentration', 0.3)
        
        self.obs_scaler = None
        self.normalize = True
        self.common_dates = price_data.index
        
        # Action space
        self.action_space = spaces.Box(
            low=0.0, high=self.max_position,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        # OBSERVATION SPACE (base weights + raw features)
        n_tech_features = technical_features.shape[1]
        n_sent_features = sentiment_features.shape[1]
        obs_dim = (self.n_base_agents * self.n_assets +  # Base agent weights
                   n_tech_features +                      # Technical features
                   n_sent_features)                       # Sentiment features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"   Super Agent observation size: {obs_dim}")
        print(f"      - Base weights: {self.n_base_agents * self.n_assets}")
        print(f"      - Technical:    {n_tech_features}")
        print(f"      - Sentiment:    {n_sent_features}")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        for agent in self.base_agents.values():
            agent.reset(seed)
        
        if self.normalize and self.obs_scaler is None:
            self._fit_scaler()
        
        self._update_base_agents()
        
        return self._get_observation(), {}
    
    def _fit_scaler(self):
        obs_samples = []
        n_samples = min(len(self.common_dates), 200)
        
        for step in range(n_samples):
            self.current_step = step
            self._update_base_agents()
            obs = self._get_observation_raw()
            obs_samples.append(obs)
        
        self.obs_scaler = StandardScaler()
        self.obs_scaler.fit(obs_samples)
        self.current_step = 0
    
    def _get_observation_raw(self):
        """Include base weights + raw features"""
        
        # Get base agent weights
        agent_weights = [agent.weights for agent in self.base_agents.values()]
        obs_weights = np.concatenate(agent_weights)
        
        # Get raw features for current date
        current_date = self.common_dates[self.current_step]
        
        tech_obs = np.zeros(self.technical_features.shape[1])
        sent_obs = np.zeros(self.sentiment_features.shape[1])
        
        if current_date in self.technical_features.index:
            tech_obs = self.technical_features.loc[current_date].values
            tech_obs = np.nan_to_num(tech_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        if current_date in self.sentiment_features.index:
            sent_obs = self.sentiment_features.loc[current_date].values
            sent_obs = np.nan_to_num(sent_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine: weights + technical + sentiment
        obs = np.concatenate([obs_weights, tech_obs, sent_obs])
        
        return obs
    
    def _get_observation(self):
        obs = self._get_observation_raw()
        
        if self.normalize and self.obs_scaler is not None:
            obs = self.obs_scaler.transform(obs.reshape(1, -1))[0]
            obs = np.clip(obs, -10, 10)
        
        return obs.astype(np.float32)
    
    def _apply_constraints(self, weights):
        violation_penalty = 0.0
        
        max_weight = np.max(weights)
        if max_weight > self.max_position:
            violation_penalty += self.constraint_penalty * (max_weight - self.max_position)
        
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum()
        
        turnover = np.sum(np.abs(weights - self.weights))
        if turnover > self.max_turnover:
            violation_penalty += self.constraint_penalty * (turnover - self.max_turnover)
            scale = self.max_turnover / turnover
            weights = self.weights + scale * (weights - self.weights)
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
        
        return weights, violation_penalty
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        
        weights, constraint_penalty = self._apply_constraints(weights)
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        log_returns = np.log(
            self.price_data.loc[next_date] / self.price_data.loc[current_date]
        )
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        arithmetic_returns = np.exp(log_returns) - 1
        
        portfolio_return = np.sum(weights * arithmetic_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(portfolio_return)
        
        portfolio_series = pd.Series(self.portfolio_history)
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        current_mdd = abs(drawdown.iloc[-1])
        
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.0
        hhi = np.sum(weights ** 2)
        
        reward = (
            self.alpha_returns * np.log(1 + portfolio_return + 1e-8) -
            self.alpha_mdd * current_mdd -
            self.alpha_vol * volatility -
            self.alpha_concentration * hhi -
            constraint_penalty
        )
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        if not done:
            self._update_base_agents()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "turnover": turnover,
            "hhi": hhi
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _update_base_agents(self):
        if self.current_step >= len(self.common_dates):
            return
        
        current_date = self.common_dates[self.current_step]
        
        for agent_name, agent in self.base_agents.items():
            obs = agent.get_observation(current_date)
            agent.predict(obs, deterministic=True)
    
    def get_portfolio_metrics(self):
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)
        
        max_drawdown = 0.0
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(12) if len(returns) > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }
    
    def get_observation(self, date):
        """Get observation for a specific date (used by MetaAgent)"""
        if date in self.common_dates:
            self.current_step = self.common_dates.get_loc(date)
            return self._get_observation()
        return None


class MetaAgentEnv(gym.Env):
    """Meta Agent: Adjusts Super Agent based on market regimes"""
    
    def __init__(self, price_data, super_agent, regime_indicators=None, **config):
        super().__init__()
        
        self.price_data = price_data
        self.super_agent = super_agent
        self.regime_indicators = regime_indicators
        self.n_assets = len(price_data.columns)
        
        self.initial_capital = 100_000
        self.transaction_cost = config.get('transaction_cost', 0.002)
        self.max_position = config.get('max_position', 0.30)
        self.max_turnover = config.get('max_turnover', 0.25)
        self.constraint_penalty = config.get('constraint_penalty', 50.0)
        
        self.alpha_returns = config.get('alpha_returns', 1.0)
        self.alpha_mdd = config.get('alpha_mdd', 1.0)
        self.alpha_vol = config.get('alpha_vol', 0.5)
        self.alpha_concentration = config.get('alpha_concentration', 0.3)
        
        self.obs_scaler = None
        self.normalize = True
        
        self.common_dates = price_data.index
        if regime_indicators is not None:
            self.regime_indicators = regime_indicators.loc[self.common_dates]
        
        self.action_space = spaces.Box(
            low=0.0, high=self.max_position,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        obs_dim = self.n_assets
        if regime_indicators is not None:
            obs_dim += regime_indicators.shape[1]
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        self.super_agent.reset(seed)
        
        if self.normalize and self.obs_scaler is None:
            self._fit_scaler()
        
        self._update_super_agent()
        
        return self._get_observation(), {}
    
    def _fit_scaler(self):
        obs_samples = []
        n_samples = min(len(self.common_dates), 200)
        
        for step in range(n_samples):
            self.current_step = step
            self._update_super_agent()
            obs = self._get_observation_raw()
            obs_samples.append(obs)
        
        self.obs_scaler = StandardScaler()
        self.obs_scaler.fit(obs_samples)
        self.current_step = 0
    
    def _get_observation_raw(self):
        super_weights = self.super_agent.weights
        obs = super_weights.copy()
        
        if self.regime_indicators is not None and self.current_step < len(self.common_dates):
            current_date = self.common_dates[self.current_step]
            if current_date in self.regime_indicators.index:
                regime_obs = self.regime_indicators.loc[current_date].values
                regime_obs = np.nan_to_num(regime_obs, nan=0.0)
                obs = np.concatenate([obs, regime_obs])
        
        return obs
    
    def _get_observation(self):
        obs = self._get_observation_raw()
        
        if self.normalize and self.obs_scaler is not None:
            obs = self.obs_scaler.transform(obs.reshape(1, -1))[0]
            obs = np.clip(obs, -10, 10)
        
        obs = np.nan_to_num(obs, nan=0.0)
        return obs.astype(np.float32)
    
    def _apply_constraints(self, weights):
        violation_penalty = 0.0
        
        max_weight = np.max(weights)
        if max_weight > self.max_position:
            violation_penalty += self.constraint_penalty * (max_weight - self.max_position)
        
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum()
        
        turnover = np.sum(np.abs(weights - self.weights))
        if turnover > self.max_turnover:
            violation_penalty += self.constraint_penalty * (turnover - self.max_turnover)
            scale = self.max_turnover / turnover
            weights = self.weights + scale * (weights - self.weights)
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
        
        return weights, violation_penalty
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        
        weights, constraint_penalty = self._apply_constraints(weights)
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        log_returns = np.log(
            self.price_data.loc[next_date] / self.price_data.loc[current_date]
        )
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        arithmetic_returns = np.exp(log_returns) - 1
        
        portfolio_return = np.sum(weights * arithmetic_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(portfolio_return)
        
        portfolio_series = pd.Series(self.portfolio_history)
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        current_mdd = abs(drawdown.iloc[-1])
        
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.0
        hhi = np.sum(weights ** 2)
        
        reward = (
            self.alpha_returns * np.log(1 + portfolio_return + 1e-8) -
            self.alpha_mdd * current_mdd -
            self.alpha_vol * volatility -
            self.alpha_concentration * hhi -
            constraint_penalty
        )
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        if not done:
            self._update_super_agent()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "turnover": turnover,
            "hhi": hhi
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _update_super_agent(self):
        if self.current_step >= len(self.common_dates):
            return
        
        current_date = self.common_dates[self.current_step]
        super_obs = self.super_agent.get_observation(current_date)
        self.super_agent.predict(super_obs, deterministic=True)
    
    def get_portfolio_metrics(self):
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)
        
        max_drawdown = 0.0
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(12) if len(returns) > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }

