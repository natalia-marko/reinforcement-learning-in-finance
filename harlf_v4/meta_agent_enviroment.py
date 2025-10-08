import gym
from gym import spaces
import numpy as np
import pandas as pd


class MetaAgentEnv(gym.Env):
    """
    Meta agent is the top-level coordinator that observes all agents
    and makes the final portfolio decision.
    Observation: Tech + Sentiment features + Agent weights + Market regime indicators
    """
    def __init__(self, price_data, features, sentiment_features, 
                 sentiment_agent, technical_agent, super_agent, 
                 regime_indicators=None, initial_capital=100000, 
                 alpha_returns=1.0, alpha_mdd=0.5, alpha_vol=0.5, exploration_bias=0.001,
                 # Backward compatibility with old names
                 alpha1=None, alpha2=None, alpha3=None, **kwargs):
        super().__init__()
        self.price_data = price_data
        self.features = features
        self.sentiment_features = sentiment_features
        self.sentiment_agent = sentiment_agent
        self.technical_agent = technical_agent
        self.super_agent = super_agent
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        
        # Reward function parameters (support both old and new naming)
        self.alpha_returns = alpha1 if alpha1 is not None else alpha_returns
        self.alpha_mdd = alpha2 if alpha2 is not None else alpha_mdd
        self.alpha_vol = alpha3 if alpha3 is not None else alpha_vol
        # Keep old names for backward compatibility
        self.alpha1 = self.alpha_returns
        self.alpha2 = self.alpha_mdd
        self.alpha3 = self.alpha_vol
        self.exploration_bias = exploration_bias
        
        # Regime indicators (bull/bear market signals)
        self.regime_indicators = regime_indicators
        
        # Observation includes: tech + sentiment features + all agent weights + regime
        obs_dim = features.shape[1] + sentiment_features.shape[1] + 3*self.n_assets
        if regime_indicators is not None:
            obs_dim += regime_indicators.shape[1]
        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Find common dates across all data sources
        self.common_dates = (features.index
                             .intersection(sentiment_features.index)
                             .intersection(price_data.index))
        
        self.reset()
    
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.log_returns_history = []
        self.returns_history = []
        
        # Reset all sub-agents
        self.sentiment_agent.reset(seed)
        self.technical_agent.reset(seed)
        self.super_agent.reset(seed)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        current_date = self.common_dates[self.current_step]
        
        # Get technical and sentiment features
        tech_obs = self.features.loc[current_date].values
        sent_obs = self.sentiment_features.loc[current_date].values
        
        # Get weights from all agents
        sentiment_weights = self.sentiment_agent.weights
        technical_weights = self.technical_agent.weights
        super_weights = self.super_agent.weights
        
        # Concatenate features and agent weights
        obs = np.concatenate([tech_obs, sent_obs, 
                              sentiment_weights, technical_weights, super_weights])
        
        # Add regime indicators if available
        if self.regime_indicators is not None:
            if current_date in self.regime_indicators.index:
                regime_obs = self.regime_indicators.loc[current_date].values
                regime_obs = np.nan_to_num(regime_obs, nan=0.0)
                obs = np.concatenate([obs, regime_obs])
        
        obs = np.nan_to_num(obs, nan=0.0)
        return obs.astype(np.float32)
    
    def step(self, action):
        # Normalize action to valid portfolio weights
        action = np.clip(action, 0, 1)
        total = action.sum()
        if total < 1e-6:
            weights = np.ones(self.n_assets) / self.n_assets
        else:
            weights = action / total
        
        # Get current and next dates
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        # Calculate returns using meta agent's final decision
        log_returns = np.log(self.price_data.loc[next_date] / self.price_data.loc[current_date])
        log_returns = log_returns.fillna(0)
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate portfolio return correctly
        arithmetic_returns = np.exp(log_returns) - 1
        portfolio_return = np.sum(weights * arithmetic_returns)
        portfolio_log_return = np.log(1 + portfolio_return)
        
        # Store for reward calculation
        self.log_returns_history.append(portfolio_log_return)
        self.returns_history.append(portfolio_return)
        
        # Update portfolio
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Calculate complex reward: alpha1*log_returns - alpha2*mdd - alpha3*vol + exploration_bias
        # Maximum drawdown
        if len(self.portfolio_history) > 1:
            portfolio_series = pd.Series(self.portfolio_history)
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            current_mdd = abs(drawdown.iloc[-1])
        else:
            current_mdd = 0.0
        
        # Volatility
        if len(self.returns_history) > 1:
            volatility = np.std(self.returns_history)
        else:
            volatility = 0.0
        
        # Composite reward
        reward = (self.alpha1 * portfolio_log_return - 
                 self.alpha2 * current_mdd - 
                 self.alpha3 * volatility + 
                 self.exploration_bias)
        
        # Store weights
        self.weights = weights
        self.current_step += 1
        
        # Update all sub-agents to keep them in sync
        # Each agent makes its own recommendation based on its observations
        self._update_sub_agents()
        
        done = self.current_step >= len(self.common_dates) - 1
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "sentiment_weights": self.sentiment_agent.weights,
            "technical_weights": self.technical_agent.weights,
            "super_weights": self.super_agent.weights
        }
        
        obs = self._get_observation()
        return obs, reward, done, False, info
    
    def _update_sub_agents(self):
        """Update all sub-agents with their respective observations"""
        if self.current_step >= len(self.common_dates):
            return
            
        current_date = self.common_dates[self.current_step]
        
        # Update sentiment agent
        sent_obs = self.sentiment_features.loc[current_date].values
        sent_obs = np.nan_to_num(sent_obs, nan=0.0).astype(np.float32)
        sent_action, _ = self.sentiment_agent.predict(sent_obs, deterministic=True)
        sent_action = np.clip(sent_action, 0, 1)
        if sent_action.sum() > 1e-6:
            self.sentiment_agent.weights = sent_action / sent_action.sum()
        
        # Update technical agent
        tech_obs = self.features.loc[current_date].values
        tech_obs = np.nan_to_num(tech_obs, nan=0.0).astype(np.float32)
        tech_action, _ = self.technical_agent.predict(tech_obs, deterministic=True)
        tech_action = np.clip(tech_action, 0, 1)
        if tech_action.sum() > 1e-6:
            self.technical_agent.weights = tech_action / tech_action.sum()
        
        # Update super agent
        super_obs = np.concatenate([self.sentiment_agent.weights, 
                                    self.technical_agent.weights]).astype(np.float32)
        super_action, _ = self.super_agent.predict(super_obs, deterministic=True)
        super_action = np.clip(super_action, 0, 1)
        if super_action.sum() > 1e-6:
            self.super_agent.weights = super_action / super_action.sum()
    
    def get_portfolio_metrics(self):
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)
        else:
            sharpe_ratio = 0
        
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = abs(drawdown.min())
        volatility = returns.std() * np.sqrt(12) if len(returns) > 0 else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value
        }