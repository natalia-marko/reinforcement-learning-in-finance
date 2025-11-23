import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from core.config import *

class PortfolioEnv(gym.Env):
    """
    Enhanced Portfolio Environment with Better Architecture
    
    Key Improvements:
    1. Multi-scale feature support
    2. Correlation-aware state space
    3. Regime-conditional rewards
    4. Better risk metrics
    """
    
    def __init__(self, features_df, prices_df=None, tickers=TICKERS, 
                 rebalance_period=REBALANCE_PERIOD, initial_balance=INITIAL_BALANCE,
                 transaction_cost=TOTAL_COST_BPS, use_correlation=True):
        super().__init__()
        
        self.features_df = features_df
        self.prices_df = prices_df
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.rebalance_period = rebalance_period
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.use_correlation = use_correlation
        
        # Extract returns and features
        self.ret_cols = [f'{t}_simple_ret' for t in tickers]
        
        # Check if we have simple returns
        if not all(col in features_df.columns for col in self.ret_cols):
            self.ret_cols = [f'{t}_log_ret' for t in tickers]
            self.using_log_returns = True
        else:
            self.using_log_returns = False
            
        self.feat_cols = [c for c in features_df.columns if c not in self.ret_cols]
        
        self.data_matrix = features_df[self.feat_cols].values
        self.return_matrix = features_df[self.ret_cols].values
        
        # Calculate correlation features if requested
        if use_correlation and prices_df is not None:
            self.correlation_features = self._calculate_correlation_features()
        else:
            self.correlation_features = None
        
        # Action Space: Portfolio weights (will be softmaxed)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation Space: Market features + Correlations + Current weights + Portfolio stats
        n_market_features = len(self.feat_cols)
        n_correlation_features = self.n_assets * 2 if use_correlation else 0  # Avg corr + beta
        n_portfolio_features = self.n_assets + 4  # Weights + recent returns/vol
        
        self.obs_shape = (n_market_features + n_correlation_features + n_portfolio_features,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32
        )
        
        # Enhanced tracking
        self.returns_history = deque(maxlen=52)
        self.weights_history = deque(maxlen=52)
        self.regime_detector = MarketRegimeDetector()
        
    def _calculate_correlation_features(self, window=52):
        """Calculate rolling correlation and beta features"""
        if self.prices_df is None:
            return None
            
        returns = self.prices_df[self.tickers].pct_change()
        features = []
        
        for i in range(len(returns)):
            if i < window:
                # Not enough history
                features.append(np.zeros(self.n_assets * 2))
            else:
                # Calculate correlation matrix
                window_returns = returns.iloc[i-window:i]
                corr_matrix = window_returns.corr()
                
                # Average correlation for each asset
                avg_correlations = []
                betas = []
                
                market_returns = window_returns.mean(axis=1)
                market_var = market_returns.var()
                
                for ticker in self.tickers:
                    # Average correlation with other assets
                    other_corrs = corr_matrix[ticker].drop(ticker)
                    avg_correlations.append(other_corrs.mean())
                    
                    # Beta to market
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
        self.recent_returns = deque(maxlen=4)  # Last 4 periods
        self.recent_volatility = 0.01  # Initialize
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Enhanced observation with portfolio stats and correlations"""
        idx = min(self.current_step, len(self.features_df) - 1)
        
        # Market features
        market_obs = self.data_matrix[idx]
        
        # Correlation features
        if self.correlation_features is not None:
            corr_obs = self.correlation_features[idx]
        else:
            corr_obs = np.array([])
        
        # Portfolio state
        portfolio_stats = np.array([
            np.mean(list(self.recent_returns)) if self.recent_returns else 0,
            np.std(list(self.recent_returns)) if len(self.recent_returns) > 1 else 0.01,
            len(self.returns_history) / 52,  # Time in market (normalized)
            self.balance / self.initial_balance - 1,  # Total return
        ])
        
        # Combine all features
        obs = np.concatenate([
            market_obs,
            corr_obs,
            self.current_weights,
            portfolio_stats
        ])
        
        return obs.astype(np.float32)

    def _softmax_with_constraints(self, x):
        """Apply softmax with concentration limits"""
        # Basic softmax
        e_x = np.exp(x - np.max(x))
        weights = e_x / e_x.sum()
        
        # Iterative projection to satisfy max_weight constraint
        # This is necessary because simple clipping + renormalization can violate constraints
        max_weight = 0.4
        weights = np.clip(weights, 0, 1)  # Ensure non-negative
        weights = weights / weights.sum() # Initial normalization
        
        for _ in range(10): # Max 10 iterations usually sufficient
            if np.all(weights <= max_weight + 1e-6):
                break
            
            # Clip elements exceeding max_weight
            weights = np.clip(weights, 0, max_weight)
            
            # Calculate residual to redistribute
            current_sum = weights.sum()
            if current_sum >= 1.0:
                break
                
            residual = 1.0 - current_sum
            
            # Distribute residual to elements that are NOT capped
            uncapped_mask = weights < max_weight - 1e-6
            if not np.any(uncapped_mask):
                break # Cannot satisfy constraint (shouldn't happen with reasonable max_weight)
                
            # Add proportional share of residual
            weights[uncapped_mask] += residual * (weights[uncapped_mask] / weights[uncapped_mask].sum())
        
        # Final safety normalization (might slightly violate max_weight if solver failed, but better than before)
        weights = weights / weights.sum()
        return weights

    def _calculate_robust_reward(self, period_returns, turnover):
        """
        Enhanced reward function with multiple components
        """
        if len(period_returns) == 0:
            return 0.0
        
        # 1. Basic statistics
        mean_return = np.mean(period_returns)
        volatility = np.std(period_returns) + 1e-8
        
        # 2. Sharpe component (40% weight)
        risk_free_rate = 0.02 / 52  # Weekly risk-free rate
        excess_returns = np.array(period_returns) - risk_free_rate
        sharpe = np.mean(excess_returns) / volatility * np.sqrt(52 / self.rebalance_period)
        sharpe_reward = np.tanh(sharpe * 0.5)  # Scale and bound
        
        # 3. Downside protection (30% weight)
        downside_returns = [r for r in period_returns if r < 0]
        if len(downside_returns) > 0:
            downside_vol = np.std(downside_returns)
            sortino = mean_return / (downside_vol + 1e-8) * np.sqrt(52 / self.rebalance_period)
            sortino_reward = np.tanh(sortino * 0.3)
        else:
            sortino_reward = 0.2  # Bonus for no losses
        
        # 4. Diversification reward (20% weight)
        concentration = np.sum(self.current_weights ** 2)  # HHI
        diversification_reward = 1 - concentration  # Higher is better
        
        # 5. Transaction cost penalty (10% weight)
        cost_penalty = -turnover * 2  # Magnify to encourage lower turnover
        
        # 6. Regime-conditional adjustment
        if hasattr(self, 'regime_detector'):
            regime = self.regime_detector.detect_regime(self.returns_history)
            if regime == 'bear':
                # In bear markets, prioritize capital preservation
                sharpe_weight = 0.2
                sortino_weight = 0.5  # More weight on downside protection
            elif regime == 'high_vol':
                # In high volatility, reduce turnover
                cost_penalty *= 2
                sharpe_weight = 0.3
                sortino_weight = 0.4
            else:  # Bull or normal
                sharpe_weight = 0.4
                sortino_weight = 0.3
        else:
            sharpe_weight = 0.4
            sortino_weight = 0.3
        
        # Combine components
        reward = (
            sharpe_reward * sharpe_weight +
            sortino_reward * sortino_weight +
            diversification_reward * 0.2 +
            np.tanh(cost_penalty) * 0.1
        )
        
        return float(reward)

    def step(self, action):
        """Execute one step with enhanced metrics"""
        # Convert action to portfolio weights
        target_weights = self._softmax_with_constraints(action)
        
        # Calculate turnover and costs
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        transaction_cost = self.balance * turnover * self.transaction_cost
        
        # Minimum trade filter (reduce tiny trades)
        min_trade = 0.02  # 2% minimum change
        weight_changes = np.abs(target_weights - self.current_weights)
        if np.max(weight_changes) < min_trade:
            # Skip rebalancing if all changes are tiny
            target_weights = self.current_weights
            turnover = 0
            transaction_cost = 0
        
        # Determine period end
        end_step = min(self.current_step + self.rebalance_period, len(self.features_df) - 1)
        
        # Check for termination
        if self.current_step >= len(self.features_df) - 1:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Initialize period
        period_balance = self.balance - transaction_cost
        period_returns = []
        
        # Simulate returns for the rebalancing period
        for t in range(self.current_step, end_step):
            if self.using_log_returns:
                log_returns = self.return_matrix[t]
                simple_returns = np.exp(log_returns) - 1
            else:
                simple_returns = self.return_matrix[t]
            
            # Portfolio return
            portfolio_return = np.dot(target_weights, simple_returns)
            period_balance *= (1 + portfolio_return)
            period_returns.append(portfolio_return)
            
            # Track for analysis
            self.episode_returns.append(portfolio_return)
            self.returns_history.append(portfolio_return)
        
        # Update state
        self.balance = period_balance
        self.portfolio_history.append(self.balance)
        self.current_step = end_step
        self.current_weights = target_weights
        self.weights_history.append(target_weights.copy())
        
        # Update recent tracking
        self.recent_returns.extend(period_returns)
        if len(period_returns) > 0:
            self.recent_volatility = np.std(period_returns)
        
        # Calculate reward
        reward = self._calculate_robust_reward(period_returns, turnover)
        
        # Check if episode is done
        terminated = self.current_step >= len(self.features_df) - 1
        truncated = False
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()
    
    def _get_info(self):
        """Enhanced info dictionary with more metrics"""
        returns_array = np.array(list(self.returns_history)) if self.returns_history else np.array([0])
        
        # Calculate comprehensive metrics
        total_return = (self.balance / self.initial_balance - 1)
        
        if len(returns_array) > 1:
            sharpe = self._calculate_sharpe(returns_array)
            sortino = self._calculate_sortino(returns_array)
            max_dd = self._calculate_max_drawdown()
        else:
            sharpe = sortino = max_dd = 0
        
        # Portfolio concentration
        concentration = np.sum(self.current_weights ** 2)
        effective_n = 1 / concentration  # Effective number of assets
        
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
            'effective_n_assets': effective_n,
            'regime': self.regime_detector.current_regime if hasattr(self, 'regime_detector') else 'unknown'
        }
    
    def _calculate_sharpe(self, returns, risk_free_rate=0.02/52):
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate
        if len(excess_returns) < 2:
            return 0
        return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8) * np.sqrt(52)
    
    def _calculate_sortino(self, returns, risk_free_rate=0.02/52):
        """Calculate Sortino ratio (downside risk-adjusted)"""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return np.mean(excess_returns) * 10  # Reward no downside
        
        downside_std = np.std(downside_returns)
        if downside_std < 1e-8:
            return 0
            
        return np.mean(excess_returns) / downside_std * np.sqrt(52)
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        if len(self.portfolio_history) < 2:
            return 0
        
        cumulative = np.array(self.portfolio_history) / self.initial_balance
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return float(np.min(drawdown))
    
    def _calculate_avg_turnover(self):
        """Calculate average portfolio turnover"""
        if len(self.weights_history) < 2:
            return 0
        
        turnovers = []
        for i in range(1, len(self.weights_history)):
            turnover = np.sum(np.abs(self.weights_history[i] - self.weights_history[i-1]))
            turnovers.append(turnover)
        
        return np.mean(turnovers) if turnovers else 0


class MarketRegimeDetector:
    """
    Simple market regime detection based on returns and volatility
    """
    def __init__(self, lookback=20):
        self.lookback = lookback
        self.current_regime = 'normal'
    
    def detect_regime(self, returns_history):
        """
        Detect current market regime
        Returns: 'bull', 'bear', 'high_vol', 'normal'
        """
        if len(returns_history) < self.lookback:
            return 'normal'
        
        recent_returns = list(returns_history)[-self.lookback:]
        
        # Calculate metrics
        mean_return = np.mean(recent_returns)
        volatility = np.std(recent_returns)
        
        # Annualized metrics
        annual_return = mean_return * 52
        annual_vol = volatility * np.sqrt(52)
        
        # Simple regime detection
        if annual_vol > 0.25:  # >25% annualized vol
            self.current_regime = 'high_vol'
        elif annual_return > 0.15:  # >15% annualized return
            self.current_regime = 'bull'
        elif annual_return < -0.10:  # <-10% annualized return
            self.current_regime = 'bear'
        else:
            self.current_regime = 'normal'
        
        return self.current_regime