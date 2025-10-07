import gym
from gym import spaces
import numpy as np
import pandas as pd


class SuperAgentEnv(gym.Env):
    """
    Super agent combines recommendations from sentiment and technical agents.
    Action: Blending weights between the two base agents' recommendations.
    Observation: Base agent weights + market regime indicators
    """
    def __init__(self, price_data, sentiment_agent, technical_agent, regime_indicators=None,
                 initial_capital=100000, alpha1=1.0, alpha2=0.5, alpha3=0.5, 
                 exploration_bias=0.01, **kwargs):
        super().__init__()
        self.price_data = price_data
        self.sentiment_agent = sentiment_agent
        self.technical_agent = technical_agent
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        
        # Reward function parameters
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.exploration_bias = exploration_bias
        
        # Regime indicators (bull/bear market signals)
        self.regime_indicators = regime_indicators
        
        # Action: portfolio weights (blend of base agent recommendations)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation: weights from both base agents + regime indicators
        obs_dim = 2 * self.n_assets  # Base agent weights
        if regime_indicators is not None:
            obs_dim += regime_indicators.shape[1]  # Add regime features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Use common dates from both agents
        self.common_dates = sentiment_agent.env.common_dates.intersection(
            technical_agent.env.common_dates
        )
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
        
        # Reset base agents
        self.sentiment_agent.reset(seed)
        self.technical_agent.reset(seed)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        # Get current recommendations from base agents
        sentiment_weights = self.sentiment_agent.weights
        technical_weights = self.technical_agent.weights
        
        obs = np.concatenate([sentiment_weights, technical_weights])
        
        # Add regime indicators if available
        if self.regime_indicators is not None and self.current_step < len(self.common_dates):
            current_date = self.common_dates[self.current_step]
            if current_date in self.regime_indicators.index:
                regime_obs = self.regime_indicators.loc[current_date].values
                regime_obs = np.nan_to_num(regime_obs, nan=0.0)
                obs = np.concatenate([obs, regime_obs])
        
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
        
        # Calculate returns using super agent's weights
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
        
        # Update base agents to keep them in sync
        # Note: They take their own actions, but we use super agent's decision
        sentiment_obs = self.sentiment_agent.env.observations[self.current_step]
        sentiment_action, _ = self.sentiment_agent.predict(sentiment_obs, deterministic=True)
        self.sentiment_agent.env.weights = sentiment_action / sentiment_action.sum()
        
        technical_obs = self.technical_agent.env.observations[self.current_step]
        technical_action, _ = self.technical_agent.predict(technical_obs, deterministic=True)
        self.technical_agent.env.weights = technical_action / technical_action.sum()
        
        done = self.current_step >= len(self.common_dates) - 1
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "sentiment_weights": self.sentiment_agent.weights,
            "technical_weights": self.technical_agent.weights
        }
        
        obs = self._get_observation()
        return obs, reward, done, False, info
    
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