"""
Portfolio Training System V3.0 - Weekly Trading with Complete Implementation
============================================================================
Fixed version with all configuration attributes and weekly trading adjustments.

Author: System Architecture
Date: November 2024
Version: 3.0
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import json
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os

# Import configuration and data pipeline
from portfolio_system import SystemConfig, DataPipeline


# ============================================================================
# SEED MANAGEMENT FOR REPRODUCIBILITY
# ============================================================================

def set_seed(seed: int, deterministic: bool = True):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic algorithms (may be slower)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: Some operations may still be non-deterministic on GPU
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    logging.info(f"Random seed set to {seed} (deterministic={deterministic})")


# ============================================================================
# PORTFOLIO ENVIRONMENT FOR WEEKLY TRADING
# ============================================================================

class PortfolioEnvironment(gym.Env):
    """Portfolio trading environment adjusted for weekly data."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: SystemConfig,
                 mode: str = 'train'):
        """
        Initialize environment for weekly trading.
        
        Args:
            data: DataFrame with weekly data
            config: System configuration
            mode: 'train', 'val', or 'test'
        """
        super().__init__()
        
        self.config = config
        self.mode = mode
        self.logger = self._setup_logger()
        
        # Ensure seed is set before creating RNG
        set_seed(config.random_seed, deterministic=True)
        
        # Initialize seeded random number generator for reproducibility
        # Use deterministic seed offset based on mode to ensure consistent states
        # Train mode: seed + 0, Val mode: seed + 1, Test mode: seed + 2
        mode_offset = {'train': 0, 'val': 1, 'test': 2}.get(mode, 0)
        env_seed = config.random_seed + mode_offset
        self.rng = np.random.RandomState(env_seed)
        
        # Prepare weekly data
        self.data = data.sort_values(['date', 'ticker'])
        self.dates = sorted(data['date'].unique())
        
        # Handle ticker setup
        self.tickers = sorted(data['ticker'].unique())
        self.n_assets = len(self.tickers)
        
        # Validate tickers
        data_tickers_upper = set([t.upper() for t in self.tickers])
        config_tickers_upper = set([t.upper() for t in config.tickers])
        assert data_tickers_upper == config_tickers_upper, f"Ticker mismatch! Data: {self.tickers}, Config: {config.tickers}"
        
        # Normalize column names
        data.columns = [col.lower() if col not in ['date', 'ticker'] else col for col in data.columns]
        
        # Get feature columns
        exclude_cols = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume',
                       'adj close', 'adj_close', 'dividends', 'stock splits', 'stock_splits',
                       'returns', 'log_returns', 'high_low_ratio', 'close_open_ratio']
        self.feature_cols = [col for col in data.columns if col not in exclude_cols]
        self.n_features = len(self.feature_cols)
        
        self.logger.info(f"Weekly environment initialized: {self.n_assets} assets, {self.n_features} features")
        
        # Prepare data tensors
        self._prepare_data_tensors()
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space dimensions
        obs_dim = (
            self.n_features * self.n_assets +  # Asset features
            self.n_assets +                     # Current weights
            6 +                                 # Portfolio stats (includes drawdown duration)
            4                                   # Time features
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Initialize tracking variables
        self.reset()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for environment."""
        logger = logging.getLogger(f"{self.__class__.__name__}_{self.mode}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.config.log_dir / f'env_{self.mode}.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger
    
    def _prepare_data_tensors(self):
        """Prepare data tensors for fast access during training."""
        # Create pivot tables for features and prices
        self.price_tensor = {}
        self.feature_tensor = {}
        
        for date in self.dates:
            date_data = self.data[self.data['date'] == date]
            
            # Price data
            prices = []
            features = []
            
            for ticker in self.tickers:
                ticker_data = date_data[date_data['ticker'] == ticker]
                
                if not ticker_data.empty:
                    prices.append(ticker_data['close'].values[0])
                    features.append(ticker_data[self.feature_cols].values[0])
                else:
                    # Handle missing data
                    prices.append(np.nan)
                    features.append(np.full(self.n_features, np.nan))
            
            self.price_tensor[date] = np.array(prices)
            self.feature_tensor[date] = np.array(features)
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment for new episode."""
        # Don't pass seed to super().reset() to avoid interfering with our seeded RNG
        # gymnasium's reset might use its own RNG which we don't control
        super().reset(seed=None)
        
        # Random start date (leave room for history)
        min_start = 20  # Need some history for features
        max_start = len(self.dates) - 52  # Leave at least 1 year for episode
        
        if self.mode == 'train':
            # Use our seeded RNG for deterministic episode start
            self.current_step = self.rng.randint(min_start, max_start)
        else:
            # Validation/test always start at the beginning
            self.current_step = min_start
        
        # Initialize portfolio
        self.current_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        self.portfolio_value = self.config.initial_capital
        self.initial_value = self.config.initial_capital
        self.peak_value = self.config.initial_capital
        
        # Tracking
        self.portfolio_history = [self.portfolio_value]
        self.weights_history = [self.current_weights.copy()]
        self.returns_history = []
        self.transaction_costs = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation including weekly features."""
        current_date = self.dates[self.current_step]
        
        # Get features for all assets
        features = self.feature_tensor[current_date].flatten()
        
        # Portfolio statistics
        returns_array = np.array(self.returns_history[-self.config.reward_lookback:]) if self.returns_history else np.array([0])
        
        # Calculate drawdown and related metrics
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Calculate drawdown duration (weeks since peak)
        if len(self.portfolio_history) > 1:
            peak_idx = np.argmax(self.portfolio_history)
            drawdown_duration = len(self.portfolio_history) - peak_idx - 1
            drawdown_duration_norm = min(drawdown_duration / 26.0, 1.0)  # Normalize to 26 weeks (6 months)
        else:
            drawdown_duration_norm = 0.0
        
        portfolio_stats = np.array([
            self.portfolio_value / self.initial_value - 1,  # Total return
            current_drawdown,  # Current drawdown
            drawdown_duration_norm,  # Drawdown duration (normalized)
            returns_array.mean() if len(returns_array) > 0 else 0,  # Mean return
            returns_array.std() if len(returns_array) > 1 else 0,  # Volatility
            self._calculate_sharpe(returns_array) if len(returns_array) > 1 else 0  # Sharpe
        ])
        
        # Time features (adjusted for weekly)
        progress = self.current_step / len(self.dates)
        week_of_month = (self.current_step % 4) / 4  # Approximate
        month_of_year = (self.current_step % 52) / 52  # Approximate
        quarter_of_year = (self.current_step % 13) / 13  # Approximate
        
        time_features = np.array([progress, week_of_month, month_of_year, quarter_of_year])
        
        # Combine all features
        observation = np.concatenate([
            features,
            self.current_weights,
            portfolio_stats,
            time_features
        ])
        
        # Handle NaN values
        observation = np.nan_to_num(observation, 0)
        
        return observation.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one week of trading."""
        
        # Calculate current drawdown to adjust constraints dynamically
        current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0
        
        # Dynamic position constraint: reduce max position size when drawdown is high
        # This forces diversification when portfolio is under stress
        if current_drawdown > 0.15:  # If drawdown > 15%
            # Reduce max position size proportionally to drawdown severity
            # At 15% drawdown: max_position = 30% (no change)
            # At 25% drawdown: max_position = 25% (reduced)
            # At 35% drawdown: max_position = 20% (more reduced)
            dynamic_max_position = self.config.max_position_size * (1 - (current_drawdown - 0.15) * 0.5)
            dynamic_max_position = max(0.15, min(dynamic_max_position, self.config.max_position_size))  # Clamp between 15% and 30%
        else:
            dynamic_max_position = self.config.max_position_size
        
        # Normalize action to valid portfolio weights
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Apply position constraints: ensure no single asset exceeds dynamic_max_position
        # Use iterative clipping and normalization to enforce constraint properly
        max_iterations = 10
        for _ in range(max_iterations):
            # Clip weights that exceed dynamic max
            excess_mask = action > dynamic_max_position
            if not excess_mask.any():
                break  # Constraint satisfied
            
            # Calculate excess to redistribute
            excess = (action[excess_mask] - dynamic_max_position).sum()
            action[excess_mask] = dynamic_max_position
            
            # Redistribute excess to weights below max (proportionally to their current values)
            below_max_mask = ~excess_mask & (action > 0)
            if below_max_mask.any():
                below_max_sum = action[below_max_mask].sum()
                if below_max_sum > 1e-8:
                    # Distribute excess proportionally
                    action[below_max_mask] += excess * (action[below_max_mask] / below_max_sum)
                else:
                    # If all below-max weights are zero, distribute equally
                    action[below_max_mask] += excess / below_max_mask.sum()
            else:
                # If no weights below max, distribute equally among all (shouldn't happen)
                action += excess / len(action)
            
            # Renormalize to ensure sum = 1
            action = action / (action.sum() + 1e-8)
        
        # Final validation: ensure constraint is satisfied using robust projection algorithm
        # This algorithm guarantees convergence by using iterative projection onto the feasible set
        # The feasible set is: {w: w_i >= 0, sum(w) = 1, w_i <= max_position_size}
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            # Step 1: Clip to [min, max] bounds (using dynamic max)
            action = np.clip(action, self.config.min_position_size, dynamic_max_position)
            
            # Step 2: Normalize to sum = 1
            action_sum = action.sum()
            if abs(action_sum - 1.0) < tolerance:
                # Sum is correct, check max constraint
                max_violation = action.max() - dynamic_max_position
                if max_violation <= tolerance:
                    # Both constraints satisfied - we're done!
                    break
                # Max constraint violated, need to redistribute
                # Find weights at max and redistribute excess
                excess_mask = action >= dynamic_max_position - tolerance
                excess = (action[excess_mask] - dynamic_max_position).sum()
                action[excess_mask] = dynamic_max_position
                
                # Redistribute excess to weights below max
                below_max_mask = ~excess_mask
                if below_max_mask.any() and excess > tolerance:
                    below_max_sum = action[below_max_mask].sum()
                    if below_max_sum > tolerance:
                        # Distribute proportionally
                        action[below_max_mask] += excess * (action[below_max_mask] / below_max_sum)
                    else:
                        # Distribute equally
                        action[below_max_mask] += excess / below_max_mask.sum()
                    # Renormalize
                    action = action / action.sum()
                else:
                    # All weights at max - this is impossible, fall back to equal weights
                    action = np.ones_like(action) / len(action)
                    action = np.clip(action, self.config.min_position_size, self.config.max_position_size)
                    action = action / action.sum()
            else:
                # Normalize
                action = action / action_sum
                
                # After normalization, check if max constraint is violated
                max_violation = action.max() - dynamic_max_position
                if max_violation > tolerance:
                    # Constraint violated after normalization, continue iterating
                    continue
            
            # Check convergence
            max_violation = action.max() - dynamic_max_position
            sum_error = abs(action.sum() - 1.0)
            if max_violation <= tolerance and sum_error < tolerance:
                break
        
        # Final safety: ensure constraints are satisfied (with small tolerance for floating point)
        action = np.clip(action, self.config.min_position_size, dynamic_max_position)
        action_sum = action.sum()
        if abs(action_sum - 1.0) > tolerance:
            action = action / action_sum
        
        # Final check: if normalization pushed values above max, use projection
        max_violation = action.max() - dynamic_max_position
        if max_violation > 1e-5:  # Only fix if violation is significant
            # Project onto feasible set: clip excess and redistribute
            excess_mask = action > dynamic_max_position
            excess = (action[excess_mask] - dynamic_max_position).sum()
            action[excess_mask] = dynamic_max_position
            
            below_max_mask = ~excess_mask
            if below_max_mask.any() and excess > 1e-8:
                below_max_sum = action[below_max_mask].sum()
                if below_max_sum > 1e-8:
                    action[below_max_mask] += excess * (action[below_max_mask] / below_max_sum)
                else:
                    action[below_max_mask] += excess / below_max_mask.sum()
            
            action = action / action.sum()
        
        # Final validation with reasonable tolerance for floating point errors
        max_weight = action.max()
        floating_point_tolerance = 1e-4  # Accept floating point errors up to 0.0001
        if max_weight > dynamic_max_position + floating_point_tolerance:
            # This should never happen, but if it does, use hard clip as last resort
            action = np.clip(action, self.config.min_position_size, dynamic_max_position)
            action = action / action.sum()
            max_weight = action.max()
            
            # If still violated, something is seriously wrong
            if max_weight > dynamic_max_position + floating_point_tolerance:
                raise AssertionError(
                    f"Constraint violation after all fixes: max weight {max_weight:.8f} > dynamic_max_position {dynamic_max_position:.8f} "
                    f"(difference: {max_weight - dynamic_max_position:.2e})"
                )
        
        # Check sum constraint
        action_sum = action.sum()
        if abs(action_sum - 1.0) > floating_point_tolerance:
            raise AssertionError(
                f"Weights don't sum to 1: sum = {action_sum:.10f} (difference: {abs(action_sum - 1.0):.2e})"
            )
        
        # Calculate transaction costs
        weight_change = np.abs(action - self.current_weights)
        transaction_cost = weight_change.sum() * self.config.transaction_cost_bps / 10000
        self.transaction_costs += transaction_cost
        
        # Get current and next prices
        current_date = self.dates[self.current_step]
        current_prices = self.price_tensor[current_date]
        
        # Move to next week
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1
        
        # Initialize net_return for info dict
        net_return = 0.0
        
        if not done:
            next_date = self.dates[self.current_step]
            next_prices = self.price_tensor[next_date]
            
            # Calculate returns (handle NaN prices)
            price_changes = np.nan_to_num(next_prices / current_prices - 1, 0)
            
            # Portfolio return
            portfolio_return = np.sum(action * price_changes)
            
            # Apply slippage
            slippage = weight_change.sum() * self.config.slippage_bps / 10000
            portfolio_return -= slippage
            
            # Calculate NET return (after transaction costs) - CRITICAL FIX
            net_return = portfolio_return - transaction_cost
            
            # Update portfolio value
            self.portfolio_value *= (1 + net_return)
            self.peak_value = max(self.peak_value, self.portfolio_value)
            
            # Update weights (adjust for price changes)
            self.current_weights = action * (1 + price_changes)
            self.current_weights = self.current_weights / (self.current_weights.sum() + 1e-8)
            
            # Re-enforce position constraints after price changes
            # (price changes can cause weights to drift outside constraints)
            # Use dynamic max position based on current drawdown
            current_drawdown_after_price = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0
            if current_drawdown_after_price > 0.15:
                dynamic_max_after_price = self.config.max_position_size * (1 - (current_drawdown_after_price - 0.15) * 0.5)
                dynamic_max_after_price = max(0.15, min(dynamic_max_after_price, self.config.max_position_size))
            else:
                dynamic_max_after_price = self.config.max_position_size
            
            max_iterations = 10
            for _ in range(max_iterations):
                excess_mask = self.current_weights > dynamic_max_after_price
                if not excess_mask.any():
                    break
                
                excess = (self.current_weights[excess_mask] - dynamic_max_after_price).sum()
                self.current_weights[excess_mask] = dynamic_max_after_price
                
                below_max_mask = ~excess_mask & (self.current_weights > 0)
                if below_max_mask.any():
                    below_max_sum = self.current_weights[below_max_mask].sum()
                    if below_max_sum > 1e-8:
                        self.current_weights[below_max_mask] += excess * (self.current_weights[below_max_mask] / below_max_sum)
                    else:
                        self.current_weights[below_max_mask] += excess / below_max_mask.sum()
                else:
                    self.current_weights += excess / len(self.current_weights)
                
                self.current_weights = self.current_weights / (self.current_weights.sum() + 1e-8)
            
            # Final clip and normalize with iterative approach to handle floating point precision
            max_iterations = 20
            for _ in range(max_iterations):
                self.current_weights = np.clip(self.current_weights, self.config.min_position_size, dynamic_max_after_price)
                weight_sum = self.current_weights.sum()
                if abs(weight_sum - 1.0) < 1e-8:
                    break
                self.current_weights = self.current_weights / weight_sum
                if np.all(self.current_weights <= dynamic_max_after_price + 1e-6):
                    break
            
            # Final safety clip
            self.current_weights = np.clip(self.current_weights, self.config.min_position_size, dynamic_max_after_price)
            weight_sum = self.current_weights.sum()
            if abs(weight_sum - 1.0) > 1e-8:
                self.current_weights = self.current_weights / weight_sum
            
            # Store history - FIX: Store NET return (after transaction costs)
            self.portfolio_history.append(self.portfolio_value)
            self.weights_history.append(self.current_weights.copy())
            self.returns_history.append(net_return)
            
            # IMMEDIATE VERIFICATION: Assert returns_history matches actual portfolio changes
            if len(self.portfolio_history) > 1:
                actual_pct_change = (self.portfolio_value / self.portfolio_history[-2]) - 1
                assert abs(self.returns_history[-1] - actual_pct_change) < 1e-6, \
                    f"Returns mismatch at step {self.current_step}: " \
                    f"stored={self.returns_history[-1]:.6f}, actual={actual_pct_change:.6f}, " \
                    f"diff={abs(self.returns_history[-1] - actual_pct_change):.6f}"
            
            # Calculate reward using SAME metric as validation (full episode Sharpe)
            # This ensures training optimizes exactly what we evaluate
            reward = self._calculate_reward()

            # REMOVED: Stop-loss check for consistency with validation
            # The agent should learn risk management through reward penalties
            # rather than artificial episode termination
        else:
            reward = 0.0
            portfolio_return = 0
            net_return = 0.0
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Info dictionary
        # Use net_return (after transaction costs) for consistency
        info = {
            'portfolio_value': self.portfolio_value,
            'return': net_return,  # Net return after transaction costs
            'cumulative_return': self.portfolio_value / self.initial_value - 1,
            'drawdown': (self.peak_value - self.portfolio_value) / self.peak_value,
            'transaction_costs': self.transaction_costs,
            'current_weights': self.current_weights.copy(),
            'week': self.current_step
        }
        
        return next_obs, reward, done, False, info
    
    def _calculate_reward(self) -> float:
        """
        Calculate reward using SAME method as validation (full episode Sharpe).
        This ensures training optimizes exactly what we evaluate on.

        CRITICAL FIX: Previously used rolling window, now uses cumulative returns.
        This aligns training signal with validation metric.
        """
        # Require minimum 10 weeks for stable calculation
        if len(self.returns_history) < 10:
            return 0.0

        # Use ALL returns accumulated so far (same as validation)
        # NOT just recent window - this was causing train-val mismatch
        all_returns = np.array(self.returns_history)

        # Calculate using SAME metric as validation
        if self.config.reward_type == 'sharpe':
            reward = self._calculate_sharpe(all_returns)
        elif self.config.reward_type == 'sortino':
            reward = self._calculate_sortino(all_returns)
        elif self.config.reward_type == 'calmar':
            reward = self._calculate_calmar(all_returns)
        else:  # log_returns
            reward = np.mean(all_returns)

        # Apply scaling
        reward *= self.config.reward_scaling

        # Drawdown penalty: Only for extreme drawdowns >40%
        if self.config.penalty_drawdown:
            drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if drawdown > 0.40:
                reward -= 0.05 * drawdown

        # Clip for training stability
        reward = np.clip(reward, -3, 3)

        return float(reward)
    
    def _calculate_episode_sharpe(self) -> float:
        """Calculate Sharpe ratio from actual portfolio value changes (STEP 2)."""
        if len(self.portfolio_history) < 3:
            return 0.0
        
        # Calculate returns from portfolio values (TRUE returns)
        portfolio_values = np.array(self.portfolio_history)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Annualized Sharpe ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return < 1e-8:
            return 0.0
        
        weekly_rf = self.config.risk_free_rate / 52
        sharpe = (mean_return - weekly_rf) / std_return * np.sqrt(52)
        
        return float(sharpe)
    
    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio (annualized for weekly)."""
        # Require minimum 10 weeks for stable calculation (as per EXECUTIVE_SUMMARY)
        if len(returns) < 10:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Weekly risk-free rate
        weekly_rf = self.config.risk_free_rate / 52
        
        # Annualized Sharpe for weekly returns
        sharpe = (mean_return - weekly_rf) / (std_return + 1e-8) * np.sqrt(52)
        
        return sharpe
    
    def _calculate_sortino(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (annualized for weekly)."""
        # Require minimum 10 weeks for stable calculation (as per EXECUTIVE_SUMMARY)
        if len(returns) < 10:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
        else:
            downside_std = 1e-8
        
        weekly_rf = self.config.risk_free_rate / 52
        sortino = (mean_return - weekly_rf) / (downside_std + 1e-8) * np.sqrt(52)
        
        return sortino
    
    def _calculate_calmar(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        # Require minimum 10 weeks for stable calculation (as per EXECUTIVE_SUMMARY)
        if len(returns) < 10:
            return 0.0
        
        # Annualized return (weekly)
        total_return = np.prod(1 + returns) - 1
        annual_return = (1 + total_return) ** (52 / len(returns)) - 1
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        calmar = annual_return / (max_drawdown + 1e-8)
        
        return calmar


# ============================================================================
# MULTI-HEAD ATTENTION MODULE
# ============================================================================

class MultiHeadAttentionModule(nn.Module):
    """Multi-head attention for asset relationships."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)


# ============================================================================
# PORTFOLIO ACTOR NETWORK
# ============================================================================

class PortfolioActorNetwork(nn.Module):
    """Actor network for portfolio allocation with complete configuration."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: SystemConfig):
        super().__init__()
        
        self.config = config
        self.action_dim = action_dim
        
        # Choose activation function
        if config.activation == 'gelu':
            self.activation = nn.GELU()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dims[0]),
            self.activation,
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(config.hidden_dims[0]),
            
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            self.activation,
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(config.hidden_dims[1])
        )
        
        # Attention module for asset relationships
        if config.use_attention:
            self.attention = MultiHeadAttentionModule(
                embed_dim=config.hidden_dims[1],
                num_heads=config.attention_heads,
                dropout=config.dropout_rate
            )
        
        # LSTM for temporal patterns
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_dims[1],
                hidden_size=config.hidden_dims[2],
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0
            )
        
        # Output layers
        final_dim = config.hidden_dims[2] if config.use_lstm else config.hidden_dims[1]
        
        self.output_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            self.activation,
            nn.Dropout(config.dropout_rate),
            nn.Linear(final_dim // 2, action_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.orthogonal_(param)
                    else:
                        nn.init.normal_(param)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to generate portfolio weights."""
        batch_size = state.shape[0]
        
        # Check for NaN or Inf in input and replace silently
        if torch.isnan(state).any() or torch.isinf(state).any():
            state = torch.where(torch.isnan(state) | torch.isinf(state), 
                               torch.zeros_like(state), state)
        
        # Extract features
        features = self.feature_extractor(state)
        
        # Check for NaN in features and replace silently
        if torch.isnan(features).any() or torch.isinf(features).any():
            features = torch.where(torch.isnan(features) | torch.isinf(features),
                                  torch.zeros_like(features), features)
        
        # Apply attention if configured
        if self.config.use_attention:
            features = features.unsqueeze(1)  # Add sequence dimension
            features = self.attention(features)
            features = features.squeeze(1)  # Remove sequence dimension
            
            # Check for NaN after attention
            if torch.isnan(features).any() or torch.isinf(features).any():
                features = torch.where(torch.isnan(features) | torch.isinf(features),
                                      torch.zeros_like(features), features)
        
        # Apply LSTM if configured
        if self.config.use_lstm:
            features = features.unsqueeze(1)  # Add sequence dimension
            features, _ = self.lstm(features)
            features = features.squeeze(1)  # Take last output
            
            # Check for NaN after LSTM
            if torch.isnan(features).any() or torch.isinf(features).any():
                features = torch.where(torch.isnan(features) | torch.isinf(features),
                                      torch.zeros_like(features), features)
        
        # Generate weights
        logits = self.output_head(features)
        
        # Check for NaN in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits),
                                torch.zeros_like(logits), logits)
        
        # Apply softmax to ensure valid portfolio weights
        weights = F.softmax(logits, dim=-1)
        
        # Final check: ensure weights are valid (no NaN, all positive, sum to ~1)
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            weights = torch.ones_like(weights) / weights.shape[-1]
        
        # Ensure weights are positive and sum to 1
        weights = torch.clamp(weights, min=1e-8, max=1.0)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        
        # Enforce max_position_size constraint using iterative clip and normalize
        # This ensures both constraints (sum=1 and max constraint) are satisfied
        # Iterate a few times to handle floating point precision issues
        for _ in range(5):
            weights = torch.clamp(weights, max=self.config.max_position_size)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            # Check if constraint is satisfied
            if (weights <= self.config.max_position_size + 1e-6).all():
                break
        
        # Final safety clip and normalize
        weights = torch.clamp(weights, max=self.config.max_position_size)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights
    
    def _constrained_softmax(self, logits: torch.Tensor, max_weight: float) -> torch.Tensor:
        """
        Apply softmax with max position constraint.
        
        Uses iterative clipping and normalization to ensure:
        1. Weights sum to 1.0
        2. No weight exceeds max_weight
        
        NOTE: Uses out-of-place operations to preserve gradients.
        """
        # Start with standard softmax
        weights = F.softmax(logits, dim=-1)
        
        # Iteratively enforce constraint using out-of-place operations
        max_iterations = 10
        for _ in range(max_iterations):
            # Check if constraint is satisfied
            if (weights <= max_weight + 1e-6).all():
                break
            
            # Clip weights that exceed max (completely out-of-place)
            excess_mask = weights > max_weight
            excess = (weights[excess_mask] - max_weight).sum()
            
            # Clip using torch.clamp (out-of-place)
            clipped_weights = torch.clamp(weights, max=max_weight)
            
            # Redistribute excess to weights below max (out-of-place)
            below_max_mask = ~excess_mask & (weights > 0)
            if below_max_mask.any():
                below_max_weights = clipped_weights[below_max_mask]
                below_max_sum = below_max_weights.sum()
                if below_max_sum > 1e-8:
                    # Distribute proportionally - use torch.where to create redistribution tensor
                    redistribution_factor = excess * (below_max_weights / below_max_sum)
                    # Create full redistribution tensor using torch.where (completely out-of-place)
                    redistribution = torch.where(
                        below_max_mask,
                        redistribution_factor.unsqueeze(0).expand_as(clipped_weights) if clipped_weights.dim() > 1 else redistribution_factor,
                        torch.zeros_like(clipped_weights)
                    )
                    # Handle batch dimension properly
                    if clipped_weights.dim() > 1:
                        # For batched inputs, we need to scatter properly
                        redistribution = torch.zeros_like(clipped_weights)
                        for i in range(clipped_weights.shape[0]):
                            batch_below_mask = below_max_mask[i] if below_max_mask.dim() > 1 else below_max_mask
                            if batch_below_mask.any():
                                batch_redist = excess[i] * (clipped_weights[i][batch_below_mask] / clipped_weights[i][batch_below_mask].sum())
                                redistribution[i][batch_below_mask] = batch_redist
                    else:
                        redistribution = torch.where(
                            below_max_mask,
                            redistribution_factor,
                            torch.zeros_like(clipped_weights)
                        )
                    new_weights = clipped_weights + redistribution
                else:
                    # Distribute equally if all below-max are zero
                    equal_redist = excess / below_max_mask.sum()
                    redistribution = torch.where(
                        below_max_mask,
                        equal_redist.unsqueeze(0).expand_as(clipped_weights) if clipped_weights.dim() > 1 else equal_redist,
                        torch.zeros_like(clipped_weights)
                    )
                    new_weights = clipped_weights + redistribution
            else:
                # Fallback: distribute equally (shouldn't happen)
                redistribution = excess / weights.shape[-1]
                new_weights = clipped_weights + redistribution
            
            # Renormalize (out-of-place)
            weights = new_weights / (new_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights
    
    def _enforce_max_position_constraint(self, weights: torch.Tensor, max_weight: float) -> torch.Tensor:
        """
        Enforce max position constraint with iterative clipping.
        
        This is a safety check to ensure the constraint is always satisfied.
        Uses out-of-place operations to preserve gradients.
        """
        max_iterations = 10
        current_weights = weights
        for _ in range(max_iterations):
            excess_mask = current_weights > max_weight + 1e-6
            if not excess_mask.any():
                break
            
            excess = (current_weights[excess_mask] - max_weight).sum()
            
            # Clip using torch.clamp (out-of-place)
            clipped_weights = torch.clamp(current_weights, max=max_weight)
            
            below_max_mask = ~excess_mask & (current_weights > 0)
            if below_max_mask.any():
                below_max_weights = clipped_weights[below_max_mask]
                below_max_sum = below_max_weights.sum()
                if below_max_sum > 1e-8:
                    redistribution_factor = excess * (below_max_weights / below_max_sum)
                    redistribution = torch.zeros_like(clipped_weights)
                    redistribution[below_max_mask] = redistribution_factor
                    new_weights = clipped_weights + redistribution
                else:
                    redistribution = torch.zeros_like(clipped_weights)
                    redistribution[below_max_mask] = excess / below_max_mask.sum()
                    new_weights = clipped_weights + redistribution
            else:
                redistribution = excess / current_weights.shape[-1]
                new_weights = clipped_weights + redistribution
            
            current_weights = new_weights / (new_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return current_weights


# ============================================================================
# PORTFOLIO CRITIC NETWORK
# ============================================================================

class PortfolioCriticNetwork(nn.Module):
    """Critic network for value estimation with complete configuration."""
    
    def __init__(self,
                 state_dim: int,
                 config: SystemConfig):
        super().__init__()
        
        # Choose activation
        if config.activation == 'gelu':
            activation = nn.GELU()
        elif config.activation == 'relu':
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()
        
        # Value network
        self.network = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dims[0]),
            activation,
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(config.hidden_dims[0]),
            
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            activation,
            nn.Dropout(config.dropout_rate),
            nn.LayerNorm(config.hidden_dims[1]),
            
            nn.Linear(config.hidden_dims[1], config.hidden_dims[2]),
            activation,
            nn.Dropout(config.dropout_rate),
            
            nn.Linear(config.hidden_dims[2], 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass to estimate state value."""
        return self.network(state)


# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """PPO agent with complete configuration support."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: SystemConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure seed is set before creating RNG
        set_seed(config.random_seed, deterministic=True)
        
        # Initialize seeded random number generator for reproducibility
        # Agent gets seed + 1000 to ensure different state from environment
        agent_seed = config.random_seed + 1000
        self.rng = np.random.RandomState(agent_seed)
        
        # Initialize running baseline tracker for advantage normalization
        self.returns_history = deque(maxlen=1000)  # Track recent returns for baseline
        self.baseline_value = 0.0  # Running baseline
        
        # Initialize networks
        self.actor = PortfolioActorNetwork(state_dim, action_dim, config).to(self.device)
        self.critic = PortfolioCriticNetwork(state_dim, config).to(self.device)
        
        # Optimizers with configured learning rates
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4
        )
        
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.learning_rate * config.critic_lr_multiplier,
            weight_decay=1e-4
        )
        
        # Learning rate schedulers
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer,
            mode='max',
            factor=config.lr_decay_factor,
            patience=config.lr_patience
        )
        
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer,
            mode='max',
            factor=config.lr_decay_factor,
            patience=config.lr_patience
        )
        
        # Trajectory storage for PPO (on-policy learning)
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_values = []
        self.trajectory_log_probs = []
        self.trajectory_dones = []

        # Exploration parameters
        self.exploration_rate = config.exploration_noise
        
        # Training metrics
        self.training_step = 0
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        
        # Diagnostic metrics
        self.gradient_norms_actor = []
        self.gradient_norms_critic = []
        self.actor_output_stats = []  # Track mean, std, max of actor outputs
        self.critic_value_stats = []  # Track critic predictions vs actual returns
        self.nan_sources = []  # Track sources of NaN/Inf with context
        
        # Initialize logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for agent."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.WARNING)  # Use WARNING to reduce noise
        
        if not logger.handlers:
            fh = logging.FileHandler(self.config.log_dir / 'agent.log')
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger
    
    def select_action(self, state: np.ndarray, training: bool = True, store_trajectory: bool = False) -> Tuple[np.ndarray, Optional[float], Optional[float]]:
        """
        Select action with optional exploration.

        Args:
            state: Current state
            training: Whether in training mode (enables exploration)
            store_trajectory: Whether to compute and return log_prob and value for trajectory storage

        Returns:
            action: Portfolio weights
            log_prob: Log probability of action (only if store_trajectory=True)
            value: State value estimate (only if store_trajectory=True)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            weights = self.actor(state_tensor)

            # Get log probability and value for trajectory storage
            if store_trajectory:
                # Calculate log probability
                log_probs = torch.log(weights + 1e-8)
                # For continuous action, we'll store the full log prob vector
                # and compute weighted sum during training
                log_prob = log_probs.cpu().numpy()[0]

                # Get value estimate
                value = self.critic(state_tensor).cpu().numpy()[0, 0]
            else:
                log_prob = None
                value = None

            weights = weights.cpu().numpy()[0]

        if training and self.rng.random() < self.exploration_rate:
            # Add exploration noise (ensure it respects max_position_size constraint)
            noise = self.rng.dirichlet(np.ones(len(weights)) * 10)
            weights = weights * (1 - self.exploration_rate) + noise * self.exploration_rate
            weights = weights / weights.sum()

            # Enforce constraint on exploration noise using robust projection
            # Use same algorithm as environment to ensure consistency
            max_iterations = 50
            tolerance = 1e-6

            for _ in range(max_iterations):
                # Clip to bounds
                weights = np.clip(weights, 0, self.config.max_position_size)

                # Normalize
                weight_sum = weights.sum()
                if abs(weight_sum - 1.0) < tolerance:
                    # Check max constraint
                    max_violation = weights.max() - self.config.max_position_size
                    if max_violation <= tolerance:
                        break
                    # Redistribute excess
                    excess_mask = weights >= self.config.max_position_size - tolerance
                    excess = (weights[excess_mask] - self.config.max_position_size).sum()
                    weights[excess_mask] = self.config.max_position_size

                    below_max_mask = ~excess_mask
                    if below_max_mask.any() and excess > tolerance:
                        below_max_sum = weights[below_max_mask].sum()
                        if below_max_sum > tolerance:
                            weights[below_max_mask] += excess * (weights[below_max_mask] / below_max_sum)
                        else:
                            weights[below_max_mask] += excess / below_max_mask.sum()
                        weights = weights / weights.sum()
                    else:
                        # Fallback to equal weights
                        weights = np.ones_like(weights) / len(weights)
                        weights = np.clip(weights, 0, self.config.max_position_size)
                        weights = weights / weights.sum()
                else:
                    weights = weights / weight_sum

                    # Check if max constraint violated after normalization
                    if weights.max() <= self.config.max_position_size + tolerance:
                        break

            # Final safety
            weights = np.clip(weights, 0, self.config.max_position_size)
            weights = weights / weights.sum()

            # If exploration was applied, need to recompute log_prob for the new action
            if store_trajectory:
                weights_tensor = torch.FloatTensor(weights).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    # Recompute log prob for the exploration-modified action
                    actor_probs = self.actor(state_tensor)
                    log_probs = torch.log(actor_probs + 1e-8)
                    log_prob = log_probs.cpu().numpy()[0]

        return weights.astype(np.float32), log_prob, value
    
    def store_step(self,
                   state: np.ndarray,
                   action: np.ndarray,
                   reward: float,
                   log_prob: np.ndarray,
                   value: float,
                   done: bool):
        """
        Store a step in the current trajectory for PPO.

        Args:
            state: Current state
            action: Action taken (portfolio weights)
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate from critic
            done: Whether episode ended
        """
        self.trajectory_states.append(state)
        self.trajectory_actions.append(action)
        self.trajectory_rewards.append(reward)
        self.trajectory_log_probs.append(log_prob)
        self.trajectory_values.append(value)
        self.trajectory_dones.append(done)

    def clear_trajectory(self):
        """Clear trajectory after training."""
        self.trajectory_states = []
        self.trajectory_actions = []
        self.trajectory_rewards = []
        self.trajectory_log_probs = []
        self.trajectory_values = []
        self.trajectory_dones = []
    
    def train(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        Train agent using proper PPO algorithm with trajectory-based learning.

        Args:
            next_value: Value estimate of next state after trajectory ends (0 if terminal)

        Returns:
            dict with training metrics
        """
        # Check if we have a trajectory to learn from
        if len(self.trajectory_states) == 0:
            return {}

        # Convert trajectory to tensors
        states = torch.FloatTensor(np.array(self.trajectory_states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.trajectory_actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.trajectory_rewards)).to(self.device)
        old_log_probs_np = np.array(self.trajectory_log_probs)  # Shape: (T, n_assets)
        values = torch.FloatTensor(np.array(self.trajectory_values)).to(self.device)
        dones = torch.FloatTensor(np.array(self.trajectory_dones)).to(self.device)
        
        # Check for NaN/Inf in inputs
        if torch.isnan(states).any() or torch.isinf(states).any():
            self.logger.warning("NaN/Inf in states, clearing trajectory")
            self.clear_trajectory()
            return {}
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            self.logger.warning("NaN/Inf in actions, clearing trajectory")
            self.clear_trajectory()
            return {}
        if torch.isnan(values).any() or torch.isinf(values).any():
            self.logger.warning("NaN/Inf in values, clearing trajectory")
            self.clear_trajectory()
            return {}

        # Compute advantages using GAE (Generalized Advantage Estimation)
        # This is the correct way to compute advantages for PPO
        with torch.no_grad():
            # Append next_value to values for GAE calculation
            values_with_next = torch.cat([values, torch.tensor([next_value]).to(self.device)])

            # Compute TD residuals: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            deltas = rewards + self.config.gamma * values_with_next[1:] * (1 - dones) - values

            # Compute GAE: A_t = Σ (γλ)^l * δ_{t+l}
            # Using λ = 0.95 (standard value for PPO)
            gae_lambda = 0.95
            advantages = torch.zeros_like(rewards)
            gae = 0

            # Compute GAE backwards through trajectory
            for t in reversed(range(len(rewards))):
                gae = deltas[t] + self.config.gamma * gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae

            # Compute returns: R_t = A_t + V(s_t)
            returns = advantages + values

            # Normalize advantages for stable training
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store old log probs as tensor for PPO ratio calculation
        # Compute the weighted sum: sum(action * log_prob) for each timestep
        old_log_probs_tensor = torch.FloatTensor(old_log_probs_np).to(self.device)
        old_log_probs = (actions * old_log_probs_tensor).sum(dim=-1)

        # Track metrics
        successful_epochs = 0
        actor_loss = None
        critic_loss = None
        entropy = None

        # PPO update: Train on full trajectory for multiple epochs
        for epoch in range(self.config.n_epochs):
            # Get current action probabilities
            action_probs = self.actor(states)

            # Validate action_probs
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                self.logger.warning(f"NaN/Inf in action_probs at epoch {epoch}")
                continue

            # Ensure valid probabilities
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            action_probs = action_probs / (action_probs.sum(dim=-1, keepdim=True) + 1e-8)

            # Calculate new log probabilities
            log_probs_dist = torch.log(action_probs + 1e-8)
            new_log_probs = (actions * log_probs_dist).sum(dim=-1)

            # Check for NaN
            if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                self.logger.warning(f"NaN/Inf in new_log_probs at epoch {epoch}")
                continue

            # Calculate importance sampling ratio: π_new / π_old
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO clipped objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - self.config.clip_epsilon,
                1.0 + self.config.clip_epsilon
            ) * advantages

            # Actor loss: -E[min(r(θ)A, clip(r(θ))A)]
            epoch_actor_loss = -torch.min(surr1, surr2).mean()

            # Entropy bonus for exploration: H = -Σ p * log(p)
            epoch_entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()

            # Check for NaN in losses
            if torch.isnan(epoch_actor_loss) or torch.isnan(epoch_entropy):
                self.logger.warning(f"NaN in actor losses at epoch {epoch}")
                continue

            # Total actor loss with entropy bonus
            total_actor_loss = epoch_actor_loss - self.config.entropy_coef * epoch_entropy

            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()

            # Check for NaN gradients
            has_nan_grad = False
            for param in self.actor.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                self.logger.warning(f"NaN in actor gradients at epoch {epoch}")
                self.actor_optimizer.zero_grad()
                continue

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()

            # Critic loss: MSE between predicted values and actual returns
            current_values = self.critic(states).squeeze()

            if torch.isnan(current_values).any() or torch.isinf(current_values).any():
                self.logger.warning(f"NaN in current_values at epoch {epoch}")
                continue

            # Use returns (not td_target) for critic loss
            epoch_critic_loss = F.mse_loss(current_values, returns)

            if torch.isnan(epoch_critic_loss) or torch.isinf(epoch_critic_loss):
                self.logger.warning(f"NaN in critic_loss at epoch {epoch}")
                continue

            # Update critic
            self.critic_optimizer.zero_grad()
            epoch_critic_loss.backward()

            # Check for NaN gradients
            has_nan_grad = False
            for param in self.critic.parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    has_nan_grad = True
                    break

            if has_nan_grad:
                self.logger.warning(f"NaN in critic gradients at epoch {epoch}")
                self.critic_optimizer.zero_grad()
                continue

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            # Store successful epoch metrics
            successful_epochs += 1
            actor_loss = total_actor_loss
            critic_loss = epoch_critic_loss
            entropy = epoch_entropy
        
        # Update exploration
        self.exploration_rate *= self.config.exploration_decay
        self.exploration_rate = max(self.exploration_rate, self.config.min_exploration)

        # Clear trajectory after training (PPO is on-policy)
        self.clear_trajectory()

        # Store metrics if any epoch succeeded
        if successful_epochs > 0 and actor_loss is not None:
            self.training_step += 1
            self.actor_losses.append(actor_loss.item())
            self.critic_losses.append(critic_loss.item())
            self.entropies.append(entropy.item())

            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'entropy': entropy.item(),
                'exploration_rate': self.exploration_rate,
                'successful_epochs': successful_epochs
            }
        else:
            self.logger.warning("No successful training epochs")
            return {}
    
    def update_learning_rate(self, metric: float):
        """Update learning rate based on performance."""
        self.actor_scheduler.step(metric)
        self.critic_scheduler.step(metric)
    
    def save(self, path: Path):
        """Save model checkpoint."""
        # Convert config dict, converting Path objects to strings for serialization
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'exploration_rate': self.exploration_rate,
            'training_step': self.training_step,
            'config': config_dict
        }
        torch.save(checkpoint, path)
        
    def load(self, path: Path):
        """Load model checkpoint."""
        # Use weights_only=False to allow loading Path objects (we trust our own checkpoints)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.exploration_rate = checkpoint['exploration_rate']
        self.training_step = checkpoint['training_step']


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self,
                 patience: int = 50,
                 min_delta: float = 0.01,
                 mode: str = 'max',
                 metric_name: str = 'val_sharpe',
                 restore_best_weights: bool = True,
                 baseline: Optional[float] = None,
                 verbose: bool = True):
        
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.metric_name = metric_name
        self.restore_best_weights = restore_best_weights
        self.baseline = baseline
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_checkpoint_path = None
        self.early_stop = False
        
    def __call__(self,
                 score: float,
                 episode: int,
                 model_save_path: Optional[Path] = None) -> bool:
        """Check if should stop training."""
        
        # Check baseline
        if self.baseline is not None and score < self.baseline:
            if self.verbose:
                print(f"Episode {episode}: {self.metric_name}={score:.3f} below baseline {self.baseline}")
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = episode
            self.best_checkpoint_path = model_save_path
            if self.verbose:
                print(f"Episode {episode}: Initial {self.metric_name}={score:.3f}")
        elif (self.mode == 'max' and score > self.best_score + self.min_delta) or \
             (self.mode == 'min' and score < self.best_score - self.min_delta):
            self.best_score = score
            self.best_epoch = episode
            self.best_checkpoint_path = model_save_path
            self.counter = 0
            if self.verbose:
                print(f"Episode {episode}: New best {self.metric_name}={score:.3f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Episode {episode}: {self.metric_name}={score:.3f}, patience {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered! Best {self.metric_name}={self.best_score:.3f} at episode {self.best_epoch}")
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping."""
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_checkpoint_path = None
        self.early_stop = False
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to best checkpoint."""
        return self.best_checkpoint_path


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

class TrainingPipeline:
    """Complete training pipeline with monitoring and early stopping."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = self._setup_logger()
        
        # Set random seed for reproducibility
        set_seed(config.random_seed, deterministic=True)
        self.logger.info(f"Random seed set to {config.random_seed} for reproducibility")
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.min_delta,
            mode='max',
            metric_name='val_sharpe',
            restore_best_weights=True,
            verbose=True
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            fh = logging.FileHandler(self.config.log_dir / 'training.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        return logger
    
    def train(self,
              train_env: PortfolioEnvironment,
              val_env: PortfolioEnvironment,
              agent: PPOAgent,
              episodes: int = 1000) -> Dict[str, Any]:
        """Train agent with validation and early stopping."""
        
        train_history = []
        val_history = []
        best_val_sharpe = -np.inf
        best_episode = 0
        val_sharpe = None  # Initialize to track if validation has been run
        
        self.logger.info("Starting training...")
        self.logger.info(f"Episodes: {episodes}")
        self.logger.info(f"Early stopping patience: {self.config.early_stopping_patience}")
        
        for episode in range(episodes):
            # Training episode
            train_reward = self._run_episode(train_env, agent, training=True)
            train_history.append(train_reward)
            
            # Skip verbose reward verification (rewards are cumulative, not comparable to Sharpe)
            
            # Validation check
            if episode % self.config.validate_every_n_episodes == 0:
                val_reward = self._run_episode(val_env, agent, training=False)
                val_history.append(val_reward)
                
                # Calculate validation Sharpe (TRUE Sharpe from portfolio values)
                val_sharpe = self._calculate_episode_sharpe(val_env)
                
                # Calculate training Sharpe for comparison
                train_sharpe = self._calculate_episode_sharpe(train_env)
                
                # Track divergence
                sharpe_gap = abs(train_sharpe - val_sharpe)
                # Warning removed - too verbose during training
                
                # Early stopping check
                checkpoint_path = self.config.model_dir / f'checkpoint_ep{episode}.pth'
                if self.early_stopping(val_sharpe, episode, checkpoint_path):
                    self.logger.info(f"Early stopping at episode {episode}")
                    break
                
                # Save best model
                if val_sharpe > best_val_sharpe:
                    best_val_sharpe = val_sharpe
                    best_episode = episode
                    agent.save(self.config.model_dir / 'best_model.pth')
                    self.logger.info(f"New best model saved: val_sharpe={val_sharpe:.3f}")
                
                # Update learning rate
                agent.update_learning_rate(val_sharpe)
            
            # Logging (reduced verbosity)
            if episode % self.config.log_every_n_episodes == 0:
                val_sharpe_str = f"{val_sharpe:.3f}" if val_sharpe is not None else "N/A"
                self.logger.info(f"Episode {episode}: train_reward={train_reward:.1f}, val_sharpe={val_sharpe_str}")
            
            # Save checkpoint
            if episode % self.config.save_every_n_episodes == 0:
                agent.save(self.config.model_dir / f'model_ep{episode}.pth')
        
        # Save final results
        results = {
            'train_history': train_history,
            'val_history': val_history,
            'best_val_sharpe': best_val_sharpe,
            'best_episode': best_episode,
            'final_episode': episode,
            'stopped_early': self.early_stopping.early_stop,
            'early_stopping': {
                'triggered': self.early_stopping.early_stop,
                'best_score': self.early_stopping.best_score,
                'best_epoch': self.early_stopping.best_epoch
            }
        }
        
        # Save history
        with open(self.config.results_dir / 'training_history.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best validation Sharpe: {best_val_sharpe:.3f} at episode {best_episode}")
        
        return results
    
    def _run_episode(self,
                     env: PortfolioEnvironment,
                     agent: PPOAgent,
                     training: bool = True) -> float:
        """
        Run single episode and train agent on collected trajectory.

        Args:
            env: Portfolio environment
            agent: PPO agent
            training: Whether to train (collect trajectory and update)

        Returns:
            episode_reward: Total reward for episode
        """
        obs = env.reset()
        episode_reward = 0

        # Collect trajectory for PPO
        while True:
            # Select action with trajectory storage if training
            if training:
                action, log_prob, value = agent.select_action(obs, training=True, store_trajectory=True)
            else:
                action, _, _ = agent.select_action(obs, training=False, store_trajectory=False)

            next_obs, reward, done, truncated, info = env.step(action)

            # Store step in trajectory if training
            if training:
                agent.store_step(obs, action, reward, log_prob, value, done or truncated)

            obs = next_obs
            episode_reward += reward

            if done or truncated:
                break

        # Train agent on collected trajectory (PPO uses on-policy learning)
        if training:
            # Get value of next state for GAE calculation
            if done or truncated:
                next_value = 0.0  # Terminal state has zero value
            else:
                with torch.no_grad():
                    next_state_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(agent.device)
                    next_value = agent.critic(next_state_tensor).cpu().numpy()[0, 0]

            # Train on full trajectory
            agent.train(next_value=next_value)

        return episode_reward
    
    def _calculate_episode_sharpe(self, env: PortfolioEnvironment) -> float:
        """Calculate Sharpe from actual portfolio values (not returns_history) - TRUE Sharpe."""
        return self._calculate_episode_sharpe_from_portfolio(env)
    
    def _calculate_episode_sharpe_from_portfolio(self, env: PortfolioEnvironment) -> float:
        """Calculate TRUE Sharpe from portfolio value changes (not returns_history)."""
        if len(env.portfolio_history) < 3:
            return 0.0
        
        # Calculate TRUE returns from portfolio value changes
        pv = np.array(env.portfolio_history)
        returns = (pv[1:] - pv[:-1]) / pv[:-1]  # TRUE returns from portfolio changes
        
        # Weekly Sharpe (annualized)
        weekly_rf = self.config.risk_free_rate / 52
        sharpe = (returns.mean() - weekly_rf) / (returns.std() + 1e-8) * np.sqrt(52)
        
        # Validate returns_history matches portfolio-derived returns
        if len(env.returns_history) >= 20 and len(env.portfolio_history) >= 21:
            stored_returns = np.array(env.returns_history[-20:])
            pv_returns = np.diff(env.portfolio_history[-21:]) / np.array(env.portfolio_history[-21:-1])
            
            # Compare last 20 returns
            returns_error = np.abs(stored_returns - pv_returns[-20:]).mean()
            
            if returns_error > 0.001:
                self.logger.error(
                    f"Returns storage ERROR: mean discrepancy = {returns_error:.6f}, "
                    f"max_discrepancy = {np.abs(stored_returns - pv_returns[-20:]).max():.6f}"
                )
        
        return sharpe
    
    def _calculate_sharpe_from_history(self, returns: list) -> float:
        """Helper to calculate Sharpe from returns_history for comparison."""
        if len(returns) < 10:
            return 0.0
        returns_array = np.array(returns)
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        weekly_rf = self.config.risk_free_rate / 52
        return (mean_return - weekly_rf) / (std_return + 1e-8) * np.sqrt(52)


if __name__ == '__main__':
    print("Portfolio Training System V3.0 - Ready for weekly trading!")
    print("Import this module to use the training components.")
