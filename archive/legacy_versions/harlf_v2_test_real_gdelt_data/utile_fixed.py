import gym
from gym import spaces
import numpy as np
import pandas as pd
import logging

class HARLFPortfolioEnv(gym.Env):
    """
    Improved Custom Portfolio Environment for HARLF.
    Implements recommendations for robustness, flexibility, and realism.
    """

    def __init__(
        self,
        price_data,
        features,
        sentiment_features=None,
        train_period=True,
        alpha1=20.0,
        alpha2=0.2,
        alpha3=0.05,
        initial_capital=100000,
        transaction_cost=0.0,
        reward_bias=0.01,
        risk_free_rate=0.0,
        logger=None
    ):
        super(HARLFPortfolioEnv, self).__init__()

        self.price_data = price_data
        self.features = features
        self.sentiment_features = sentiment_features
        self.train_period = train_period

        # Reward function parameters (now configurable)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.reward_bias = reward_bias
        self.risk_free_rate = risk_free_rate

        # Portfolio constraints and transaction cost
        self.n_assets = len(price_data.columns)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

        # Logging
        self.logger = logger or logging.getLogger(__name__)

        # Action space: continuous portfolio weights [0,1] that sum to 1
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Observation space: normalized features
        if sentiment_features is not None:
            obs_dim = features.shape[1] + sentiment_features.shape[1]
        else:
            obs_dim = features.shape[1]

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Initialize state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]

        # Align dates - ensure all data sources have same dates
        self.common_dates = self.features.index
        if self.sentiment_features is not None:
            self.common_dates = self.features.index.intersection(
                self.sentiment_features.index
            )
        self.common_dates = self.common_dates.intersection(self.price_data.index)

        # Warn if significant data lost during alignment
        expected_steps = min(len(self.features), len(self.price_data))
        if len(self.common_dates) < 0.8 * expected_steps:
            self.logger.warning(
                f"Significant data loss during date alignment: {len(self.common_dates)} steps out of expected {expected_steps}."
            )

        self.logger.info(f"Environment initialized with {len(self.common_dates)} time steps")

    def reset(self, seed=None):
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]

        return self._get_observation(), {}

    def _get_observation(self):
        """Get current observation vector with smarter imputation for NaNs."""
        if self.current_step >= len(self.common_dates):
            return np.zeros(self.observation_space.shape[0])

        current_date = self.common_dates[self.current_step]

        # Get technical features
        tech_features = self.features.loc[current_date].values

        # Get sentiment features if available
        if self.sentiment_features is not None:
            if current_date in self.sentiment_features.index:
                sent_features = self.sentiment_features.loc[current_date].values
                observation = np.concatenate([tech_features, sent_features])
            else:
                sent_features = np.nan * np.ones(self.sentiment_features.shape[1])
                observation = np.concatenate([tech_features, sent_features])
        else:
            observation = tech_features

        # Smarter imputation: fill NaNs with feature means, fallback to zero
        nan_mask = np.isnan(observation)
        if np.any(nan_mask):
            mean_vals = np.nanmean(observation)
            observation[nan_mask] = mean_vals if not np.isnan(mean_vals) else 0.0

        return observation.astype(np.float32)

    def step(self, action):
        """Execute one step in the environment with improved validation and transaction cost modeling."""
        # Validate action shape
        if not isinstance(action, np.ndarray) or action.shape != (self.n_assets,):
            raise ValueError(f"Action shape {action.shape} does not match number of assets {self.n_assets}")

        if self.current_step >= len(self.common_dates) - 1:
            # Episode is done
            return self._get_observation(), 0, True, {}, {}

        # Clip and normalize action; handle edge cases
        action = np.clip(action, 0, 1)
        total = action.sum()
        if total < 1e-6:
            # Fallback to equal weights if sum is (near) zero
            weights = np.ones(self.n_assets) / self.n_assets
            self.logger.warning("Action sum is zero; reverting to equal weights.")
        else:
            weights = action / total

        # Transaction cost calculation (proportional to weight change)
        weight_change = np.abs(weights - self.weights)
        cost = self.transaction_cost * self.portfolio_value * weight_change.sum()

        # Get current and next period dates
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[self.current_step + 1]

        # Calculate portfolio return using log returns
        if current_date in self.price_data.index and next_date in self.price_data.index:
            log_returns = np.log(self.price_data.loc[next_date] / self.price_data.loc[current_date])
            log_returns = log_returns.fillna(0)
            portfolio_log_return = np.sum(weights * log_returns)
            portfolio_return = np.exp(portfolio_log_return) - 1
        else:
            portfolio_return = 0

        # Update portfolio value with transaction cost
        old_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_value -= cost
        self.portfolio_history.append(self.portfolio_value)

        # Calculate reward using improved HARLF reward function
        reward = self._calculate_reward(weights, portfolio_return, old_value, cost)

        # Update state
        self.weights = weights
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= len(self.common_dates) - 1
        terminated = done

        # No truncated flag, for code clarity
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'weights': weights,
            'transaction_cost': cost
        }

        return self._get_observation(), reward, terminated, info

    def _calculate_reward(self, weights, portfolio_return, old_value, cost):
        """
        Calculate reward using improved HARLF formula:
        Reward = α1 * ROI - α2 * MDD - α3 * σ - transaction_cost
        With better scaling and stability
        """
        # ROI component
        roi = portfolio_return

        # Maximum Drawdown component
        if len(self.portfolio_history) >= 3:
            portfolio_series = pd.Series(self.portfolio_history)
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
            max_drawdown = max(0, max_drawdown - 0.01)
        else:
            max_drawdown = 0

        # Volatility component
        if len(self.portfolio_history) >= 10:
            recent_returns = pd.Series(self.portfolio_history[-10:]).pct_change().dropna()
            if len(recent_returns) > 0:
                volatility = recent_returns.std()
                volatility = max(0, volatility - 0.05)
            else:
                volatility = 0
        else:
            volatility = 0

        # Final reward with configurable bias and transaction cost penalty
        reward = (
            self.alpha1 * roi
            - self.alpha2 * max_drawdown
            - self.alpha3 * volatility
            - cost / self.initial_capital  # scale transaction cost
            + self.reward_bias
        )

        return reward

    def get_portfolio_metrics(self):
        """Calculate portfolio performance metrics with risk-free rate for Sharpe ratio."""
        if len(self.portfolio_history) <= 1:
            return {}

        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()

        # Calculate metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital

        # Sharpe ratio: mean excess return over risk-free rate
        if len(returns) > 0 and returns.std() > 0:
            excess_returns = returns - self.risk_free_rate / 252  # daily rf rate
            sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Volatility
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'final_value': self.portfolio_value
        }