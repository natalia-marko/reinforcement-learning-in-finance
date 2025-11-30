import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from core.config import *
import os
import cloudpickle

class PortfolioEnv(gym.Env):
    """
    Portfolio Environment for RL-based Asset Allocation

    Key Features:
    1. Multi-scale feature support (momentum, volatility, technical indicators)
    2. Correlation-aware state space (cross-asset relationships)
    3. Clean reward: Sharpe ratio - transaction costs
    4. Configurable constraints (position limits, minimum turnover)
    """

    def __init__(self, features_df, prices_df=None, tickers=TICKERS,
                 rebalance_period=REBALANCE_PERIOD, initial_balance=INITIAL_BALANCE,
                 transaction_cost=TOTAL_COST_BPS, use_correlation=True,
                 max_position_size=0.4, min_turnover=0.01):
        super().__init__()

        self.features_df = features_df
        self.prices_df = prices_df
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rebalance_period = rebalance_period
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.use_correlation = use_correlation
        self.max_position_size = max_position_size
        self.min_turnover = min_turnover

        self.ret_cols = [f'{t}_simple_ret' for t in tickers]

        if not all(col in features_df.columns for col in self.ret_cols):
            raise ValueError(f"Missing return columns in features. Expected: {self.ret_cols}")

        self.feat_cols = [c for c in features_df.columns if c not in self.ret_cols]

        self.data_matrix = features_df[self.feat_cols].values

        if prices_df is not None:
            aligned_prices = prices_df.reindex(features_df.index)
            self.return_matrix = aligned_prices[tickers].pct_change().fillna(0).values
        else:
            print("⚠️ WARNING: prices_df not provided to PortfolioEnv. Using shifted returns from features (Potential Leakage!)")
            self.return_matrix = features_df[self.ret_cols].values

        if use_correlation and prices_df is not None:
            self.correlation_features = self._calculate_correlation_features()
        else:
            self.correlation_features = None

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )

        n_market_features = len(self.feat_cols)
        n_correlation_features = self.n_assets * 2 if use_correlation else 0
        n_portfolio_features = self.n_assets + 2  # Only mean, std of recent returns

        self.obs_shape = (n_market_features + n_correlation_features + n_portfolio_features,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )

        self.returns_history = deque(maxlen=52)
        self.weights_history = deque(maxlen=52)
        self.last_reward = 0.0  # For early termination

    def _calculate_correlation_features(self, window=52):
        if self.prices_df is None:
            return None

        returns = self.prices_df[self.tickers].pct_change()
        features = []

        for i in range(len(returns)):
            if i < window:
                features.append(np.zeros(self.n_assets * 2))
            else:
                window_returns = returns.iloc[i-window:i]
                corr_matrix = window_returns.corr()

                avg_correlations = []
                betas = []

                market_returns = window_returns.mean(axis=1)
                market_var = market_returns.var()

                for ticker in self.tickers:
                    other_corrs = corr_matrix[ticker].drop(ticker)
                    avg_correlations.append(other_corrs.mean())

                    cov = window_returns[ticker].cov(market_returns)
                    beta = cov / (market_var + 1e-8)
                    betas.append(beta)

                features.append(np.array(avg_correlations + betas))

        return np.array(features)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.balance]
        self.returns_history.clear()
        self.weights_history.clear()
        self.episode_returns = []
        self.recent_returns = deque(maxlen=4)
        self.recent_volatility = 0.01
        self.last_reward = 0.0

        return self._get_observation(), {}

    def _get_observation(self):
        idx = min(self.current_step, len(self.features_df) - 1)
        market_obs = self.data_matrix[idx]

        if self.correlation_features is not None:
            corr_obs = self.correlation_features[idx]
        else:
            corr_obs = np.array([])

        portfolio_stats = np.array([
            np.mean(list(self.recent_returns)) if self.recent_returns else 0,
            np.std(list(self.recent_returns)) if len(self.recent_returns) > 1 else 0.01,
        ])

        obs = np.concatenate([
            market_obs,
            corr_obs,
            self.current_weights,
            portfolio_stats
        ])

        return obs.astype(np.float32)

    def _softmax_with_constraints(self, x):
        e_x = np.exp(x - np.max(x))
        weights = e_x / e_x.sum()
        weights = np.clip(weights, 0, 1)
        weights = weights / weights.sum()

        for _ in range(10):
            if np.all(weights <= self.max_position_size + 1e-6):
                break
            weights = np.clip(weights, 0, self.max_position_size)
            current_sum = weights.sum()
            if current_sum >= 1.0:
                break
            residual = 1.0 - current_sum
            uncapped_mask = weights < self.max_position_size - 1e-6
            if not np.any(uncapped_mask):
                break
            weights[uncapped_mask] += residual * (weights[uncapped_mask] / weights[uncapped_mask].sum())

        weights = weights / weights.sum()
        return weights

    def _calculate_reward(self, period_returns, turnover):
        if len(period_returns) == 0:
            return 0.0

        mean_return = np.mean(period_returns)
        risk_free_rate = 0.05 / 52
        excess_return = mean_return - risk_free_rate
        cost_penalty = turnover * 0.3

        return (excess_return * 100) - cost_penalty

    def step(self, action):
        target_weights = self._softmax_with_constraints(action)
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        if turnover < self.min_turnover:
            target_weights = self.current_weights
            turnover = 0.0

        transaction_cost = self.balance * turnover * self.transaction_cost

        balance_after_cost = self.balance - transaction_cost
        asset_values = balance_after_cost * target_weights

        end_step = min(self.current_step + self.rebalance_period, len(self.features_df) - 1)

        if self.current_step >= len(self.features_df) - 1:
            return self._get_observation(), self.last_reward, True, False, self._get_info()

        period_returns = []

        for t in range(self.current_step, end_step):
            r_t = self.return_matrix[t]
            asset_values = asset_values * (1 + r_t)
            new_balance = np.sum(asset_values)
            prev_balance_t = self.portfolio_history[-1]
            step_return = (new_balance / prev_balance_t) - 1
            period_returns.append(step_return)
            self.episode_returns.append(step_return)
            self.returns_history.append(step_return)
            self.portfolio_history.append(new_balance)
            self.recent_returns.append(step_return)

        self.balance = np.sum(asset_values)
        if self.balance > 0:
            self.current_weights = asset_values / self.balance
        else:
            self.current_weights = np.zeros(self.n_assets)
        self.weights_history.append(self.current_weights.copy())

        if len(period_returns) > 0:
            self.recent_volatility = np.std(period_returns)

        self.current_step = end_step

        reward = self._calculate_reward(period_returns, turnover)
        self.last_reward = reward

        terminated = self.current_step >= len(self.features_df) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _get_info(self):
        returns_array = np.array(list(self.returns_history)) if self.returns_history else np.array([0])

        total_return = (self.balance / self.initial_balance - 1)

        if len(returns_array) > 1:
            sharpe = self._calculate_sharpe(returns_array)
            sortino = self._calculate_sortino(returns_array)
            max_dd = self._calculate_max_drawdown()
        else:
            sharpe = sortino = max_dd = 0

        concentration = np.sum(self.current_weights ** 2)
        effective_n = 1 / concentration if concentration != 0 else 0

        return {
            'balance': self.balance,
            'weights': self.current_weights.copy(),
            'total_return': total_return,
            'period_return': np.mean(list(self.recent_returns)) if self.recent_returns else 0,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'volatility': self.recent_volatility,
            'turnover': self._calculate_avg_turnover(),
            'effective_n_assets': effective_n
        }

    def _calculate_sharpe(self, returns, risk_free_rate=0.05/52):
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(52)

    def _calculate_sortino(self, returns, risk_free_rate=0.05/52):
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) < 2:
            return np.mean(excess_returns) / (np.std(returns) + 1e-8) * np.sqrt(52)

        downside_std = np.std(downside_returns)
        if downside_std < 1e-8:
            return 0

        return np.mean(excess_returns) / downside_std * np.sqrt(52)
    
    def _calculate_max_drawdown(self):
        if len(self.portfolio_history) < 2:
            return 0

        cumulative = np.array(self.portfolio_history) / self.initial_balance
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))

    def _calculate_avg_turnover(self):
        if len(self.weights_history) < 2:
            return 0

        turnovers = []
        for i in range(1, len(self.weights_history)):
            turnover = np.sum(np.abs(self.weights_history[i] - self.weights_history[i-1]))
            turnovers.append(turnover)

        return np.mean(turnovers) if turnovers else 0

# --- VecNormalize class for sharing/serialization of normalization stats ---

class VecNormalize(gym.ObservationWrapper):
    """
    Vectorized Observation Normalizer that allows saving/loading statistics to disk.

    Usage for training:
        env_train = VecNormalize(env_train_base)
        # ...do training to collect obs statistics ...
        env_train.save(stats_path)

    For validation (use trained stats, don't update them!):
        env_val = VecNormalize.load(stats_path, env_val_base)
        env_val.training = False  # No stat update
    """
    def __init__(self, env, epsilon=1e-8, training=True):
        super().__init__(env)
        shape = self.observation_space.shape
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self.training = training
        self.epsilon = epsilon

    def observation(self, obs):
        if self.training:
            self._update(obs)
        norm_obs = (obs - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return norm_obs.astype(np.float32)

    def _update(self, obs):
        obs = np.asarray(obs, dtype=np.float64)
        batch_mean = obs
        batch_var = np.square(obs - batch_mean)
        batch_count = 1.0

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def save(self, filename):
        stats = {
            'mean': self.mean,
            'var': self.var,
            'count': self.count
        }
        with open(filename, 'wb') as f:
            cloudpickle.dump(stats, f)

    @classmethod
    def load(cls, filename, env, training=False):
        obj = cls(env, training=training)
        with open(filename, 'rb') as f:
            stats = cloudpickle.load(f)
        obj.mean = stats['mean']
        obj.var = stats['var']
        obj.count = stats['count']
        obj.training = training
        return obj
