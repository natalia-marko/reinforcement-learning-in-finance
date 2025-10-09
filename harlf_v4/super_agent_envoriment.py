import gym
from gym import spaces
import numpy as np
import pandas as pd


class SuperAgentEnv(gym.Env):
    """
    Super agent combines recommendations from sentiment and technical agents.
    Simplified: Super agent observes base agent weights and makes direct allocation decisions.
    """
    def __init__(self, price_data, sentiment_agent, technical_agent, regime_indicators=None,
                 initial_capital=100000, transaction_cost=0.01,
                 alpha_returns=1.0, alpha_mdd=0.5, alpha_vol=0.5, 
                 exploration_bias=0.01, 
                 # Backward compatibility with old names
                 alpha1=None, alpha2=None, alpha3=None, **kwargs):
        super().__init__()
        self.price_data = price_data
        self.sentiment_agent = sentiment_agent
        self.technical_agent = technical_agent
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # Reward function parameters (support both old and new naming)
        self.alpha_returns = alpha1 if alpha1 is not None else alpha_returns
        self.alpha_mdd = alpha2 if alpha2 is not None else alpha_mdd
        self.alpha_vol = alpha3 if alpha3 is not None else alpha_vol
        # Keep old names for backward compatibility
        self.alpha1 = self.alpha_returns
        self.alpha2 = self.alpha_mdd
        self.alpha3 = self.alpha_vol
        self.exploration_bias = exploration_bias
        
        # Use price_data dates as the authoritative timeline
        # The base agents will be queried at each step regardless of their training period
        self.common_dates = price_data.index
        
        # Align regime indicators with common dates
        if regime_indicators is not None:
            self.regime_indicators = regime_indicators.loc[self.common_dates]
        else:
            self.regime_indicators = None
        
        # Action: portfolio weights (blend of base agent recommendations)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32)
        
        # Observation space: Sentiment + Technical weights + optional regime indicators
        obs_dim = 2 * self.n_assets  # Sentiment + Technical weights
        if self.regime_indicators is not None:
            obs_dim += self.regime_indicators.shape[1]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()
    
    def reset(self, seed=None, options=None):
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
        
        # Apply transaction costs
        prev_weights = self.weights.copy()
        transaction_cost_penalty = self.transaction_cost * np.sum(np.abs(weights - prev_weights))
        portfolio_return -= transaction_cost_penalty
        
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
        done = self.current_step >= len(self.common_dates) - 1
        
        # Update base agents for NEXT step (proper temporal alignment)
        # This maintains agent state synchronized with environment progression
        # Key: We update agents AFTER transition to prepare for next observation
        if not done:
            self._update_base_agents()
        
        
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "sentiment_weights": self.sentiment_agent.weights,
            "technical_weights": self.technical_agent.weights,
            "transaction_cost": transaction_cost_penalty
        }
        
        obs = self._get_observation()
        return obs, reward, done, False, info
    
    def _update_base_agents(self):
        """
        Update base agents with current step observations.
        
        This is called AFTER each step transition to keep base agents
        synchronized with the environment state. Base agents provide
        recommendations based on their current observations, which the
        super agent then uses to make portfolio decisions.
        
        Temporal alignment:
        - After step t→t+1 transition
        - Update agents with features at time t+1
        - These updated weights will be used in next observation
        """
        if self.current_step >= len(self.common_dates):
            return
            
        current_date = self.common_dates[self.current_step]
        
        # Update sentiment agent with current sentiment features
        if hasattr(self.sentiment_agent, 'env'):
            sent_env = self.sentiment_agent.env
            if hasattr(sent_env, 'sentiment_features') and current_date in sent_env.sentiment_features.index:
                sent_obs = sent_env.sentiment_features.loc[current_date].values
                sent_obs = np.nan_to_num(sent_obs, nan=0.0).astype(np.float32)
                self.sentiment_agent.predict(sent_obs, deterministic=True)
        
        # Update technical agent with current technical features
        if hasattr(self.technical_agent, 'env'):
            tech_env = self.technical_agent.env
            if hasattr(tech_env, 'features') and current_date in tech_env.features.index:
                tech_obs = tech_env.features.loc[current_date].values
                tech_obs = np.nan_to_num(tech_obs, nan=0.0).astype(np.float32)
                self.technical_agent.predict(tech_obs, deterministic=True)
    
    def get_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
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
        # FIXED: Added win_rate to metrics
        win_rate = np.mean(np.array(self.returns_history) > 0) if self.returns_history else 0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }