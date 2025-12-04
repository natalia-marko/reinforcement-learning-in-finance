"""
tech_env_module.py - REFACTORED (NO LEAKAGE)
Trading environment for RL - accepts PRE-NORMALIZED features only
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Dict, Any
import warnings


class TechnicalEnv(gym.Env):
    """
    Gym-compatible trading environment with advanced reward functions
    
    CRITICAL CHANGES TO PREVENT LEAKAGE:
    1. NO feature normalization in env - accepts pre-normalized features
    2. Assumes features are already properly normalized by train-only scaler
    3. Validates input data integrity
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        price_data: np.ndarray,
        technical_features: np.ndarray,  # PRE-NORMALIZED!
        max_position: float = 0.70,
        transaction_cost: float = 0.001,
        use_log_returns: bool = True,
        initial_balance: float = 100_000.0,
        reward_function: str = 'differential_sharpe',
        reward_scaling: float = 1.0,
        reward_lookback: int = 52,
        annualization_factor: int = 52,
        composite_return_weight: float = 1.0,
        composite_risk_weight: float = 0.3,
        composite_drawdown_weight: float = 0.5,
        composite_turnover_weight: float = 0.1,
        validate_inputs: bool = True
    ):
        """
        Initialize trading environment
        
        Args:
            price_data: Price series (NOT normalized)
            technical_features: PRE-NORMALIZED technical features
            validate_inputs: Whether to validate input data
            ... (other args same as before)
        
        IMPORTANT: technical_features must be pre-normalized using a scaler
        fitted ONLY on training data. Do NOT pass raw features.
        """
        super().__init__()
        
        # Convert prices to 2D numpy array.  Accept both Series (univariate)
        # and DataFrame (multi‑asset).  For a Series, we reshape to
        # (n_periods, 1) so that subsequent vectorised operations work
        if hasattr(price_data, 'values'):
            arr = price_data.values
        else:
            arr = np.asarray(price_data)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # arr.ndim should now be 2: (n_periods, n_assets)
        self.price_data = arr.astype(float)
        self.n_assets = self.price_data.shape[1]

        # Convert features to 2D numpy array
        if hasattr(technical_features, 'values'):
            feat_arr = technical_features.values
        else:
            feat_arr = np.asarray(technical_features)
        self.technical_features = np.asarray(feat_arr, dtype=float)
        
        # Validate inputs
        if validate_inputs:
            self._validate_inputs()
        
        # Clean any remaining issues (inf/nan)
        self.technical_features = np.nan_to_num(
            self.technical_features, nan=0.0, posinf=0.0, neginf=0.0
        )
        
        # ✅ NO NORMALIZATION HERE - features assumed pre-normalized!
        
        # Parameters
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.use_log_returns = use_log_returns
        self.initial_balance = initial_balance
        self.reward_function = reward_function.lower()
        self.reward_scaling = reward_scaling
        self.reward_lookback = reward_lookback
        self.ann_factor = annualization_factor
        self.returns_buffer = deque(maxlen=reward_lookback)
        
        # Composite weights
        self.composite_return_weight = composite_return_weight
        self.composite_risk_weight = composite_risk_weight
        self.composite_drawdown_weight = composite_drawdown_weight
        self.composite_turnover_weight = composite_turnover_weight
        
        # Episode tracking
        self.current_step = 0
        # Number of steps is rows minus one (can't compute return on last row)
        self.max_steps = self.price_data.shape[0] - 1

        # Portfolio state
        self.portfolio_value = initial_balance
        # Position vectors per asset
        self.position: np.ndarray = np.zeros(self.n_assets, dtype=float)
        self.previous_position: np.ndarray = np.zeros(self.n_assets, dtype=float)
        # Not used but kept for backwards compatibility
        self.cash = initial_balance
        self.shares = np.zeros(self.n_assets, dtype=float)
        
        # History
        self.portfolio_history: list[float] = [initial_balance]
        # Store a copy of positions at each step
        self.position_history: list[np.ndarray] = [self.position.copy()]
        self.log_returns_history: list[float] = []
        self.simple_returns_history: list[float] = []
        self.trade_history: list[dict] = []
        self.portfolio_peak: float = initial_balance
        
        # Define spaces
        n_features = self.technical_features.shape[1]
        # Observation includes all features, current positions per asset, and drawdown
        obs_low = np.array([-np.inf] * n_features + [-self.max_position] * self.n_assets + [-np.inf], dtype=np.float32)
        obs_high = np.array([np.inf] * n_features + [self.max_position] * self.n_assets + [np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action is target positions for each asset in [-1, 1] which will be scaled by max_position
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
    
    def _validate_inputs(self):
        """
        Validate input data to catch common errors
        """
        
        # Check number of rows match
        if self.price_data.shape[0] != self.technical_features.shape[0]:
            raise ValueError(
                f"Price data length ({self.price_data.shape[0]}) != "
                f"Features length ({self.technical_features.shape[0]})"
            )
        
        # Check for all-zero features (might indicate normalization issue)
        feature_stds = np.std(self.technical_features, axis=0)
        zero_std_features = np.sum(feature_stds < 1e-10)
        
        if zero_std_features > 0:
            warnings.warn(
                f"{zero_std_features} features have near-zero std. "
                f"This might indicate constant features or normalization issues."
            )
        
        # Check if features look normalized (mean ~0, std ~1)
        feature_means = np.mean(self.technical_features, axis=0)
        mean_of_means = float(np.mean(np.abs(feature_means)))
        mean_of_stds = float(np.mean(feature_stds))

        if mean_of_means > 0.5:
            warnings.warn(
                f"Features have high mean ({mean_of_means:.3f}). "
                f"Are they normalized? Expected mean ~0 for normalized features."
            )

        if mean_of_stds < 0.5 or mean_of_stds > 2.0:
            warnings.warn(
                f"Features have unusual std ({mean_of_stds:.3f}). "
                f"Expected std ~1 for normalized features."
            )

        # Check for NaN/inf in prices
        if not np.all(np.isfinite(self.price_data)):
            raise ValueError("Price data contains NaN or inf values!")

        # Check for non-positive prices
        if np.any(self.price_data <= 0):
            raise ValueError("Price data contains non-positive values!")

        print(f"  ✓ Input validation passed")
        print(f"    Periods: {self.price_data.shape[0]}")
        print(f"    Assets: {self.n_assets}")
        print(f"    Features: {self.technical_features.shape[1]}")
        print(f"    Feature mean: {mean_of_means:.3f} (expected ~0)")
        print(f"    Feature std:  {mean_of_stds:.3f} (expected ~1)")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_balance
        # Reset positions per asset
        self.position = np.zeros(self.n_assets, dtype=float)
        self.previous_position = np.zeros(self.n_assets, dtype=float)
        self.cash = self.initial_balance
        self.shares = np.zeros(self.n_assets, dtype=float)

        self.portfolio_history = [self.initial_balance]
        self.position_history = [self.position.copy()]
        self.log_returns_history = []
        self.simple_returns_history = []
        self.trade_history = []
        self.portfolio_peak = self.initial_balance
        self.returns_buffer.clear()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step"""
        
        # Ensure action is 1D array of target weights for each asset
        action = np.asarray(action, dtype=float).flatten()
        # Map action range [-1,1] to positions within [-max_position, max_position]
        target_position = np.clip(action, -1.0, 1.0) * self.max_position

        # Compute change from current position
        self.previous_position = self.position.copy()
        position_change = target_position - self.position

        # Move to next step
        self.current_step += 1

        # Compute asset returns for this step
        price_prev = self.price_data[self.current_step - 1, :]
        price_curr = self.price_data[self.current_step, :]
        if self.use_log_returns:
            asset_returns = np.log(price_curr / price_prev)
        else:
            asset_returns = (price_curr - price_prev) / price_prev

        # Portfolio return is dot of previous positions and asset returns
        portfolio_return = float(np.dot(self.previous_position, asset_returns))

        # Transaction cost proportional to turnover across all assets
        cost = float(np.sum(np.abs(position_change)) * self.transaction_cost)

        # Update portfolio value
        if self.use_log_returns:
            self.portfolio_value *= np.exp(portfolio_return - cost)
        else:
            self.portfolio_value *= (1 + portfolio_return - cost)

        # Update positions after accounting for return
        self.position = target_position

        # Track history
        self.portfolio_history.append(self.portfolio_value)
        self.position_history.append(self.position.copy())

        if self.use_log_returns:
            self.log_returns_history.append(portfolio_return - cost)
            self.simple_returns_history.append(np.exp(portfolio_return - cost) - 1)
        else:
            self.simple_returns_history.append(portfolio_return - cost)
            self.log_returns_history.append(np.log(1 + portfolio_return - cost))

        if np.any(np.abs(position_change) > 1e-6):
            self.trade_history.append({
                'step': self.current_step,
                'position_change': position_change,
                'cost': cost
            })

        # Update portfolio peak
        self.portfolio_peak = max(self.portfolio_peak, self.portfolio_value)

        # Calculate reward
        reward = self._calculate_reward(portfolio_return, cost, position_change)

        # Termination and truncation flags
        terminated = (self.current_step >= self.max_steps)
        truncated = False

        # Bankruptcy check: if portfolio drops below 30% of initial balance
        if self.portfolio_value < self.initial_balance * 0.3:
            terminated = True
            reward -= 10.0

        # Build observation and info
        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(
        self,
        portfolio_return: float,
        cost: float,
        position_change: np.ndarray
    ) -> float:
        """Calculate reward using selected reward function.

        The reward functions operate on the net return (after transaction costs)
        and optional additional risk/turnover penalties.  For multi‑asset
        portfolios the turnover component sums absolute position changes
        across all assets.
        """

        net_return = portfolio_return - cost

        # Simple net return scaled
        if self.reward_function == 'simple':
            return float(net_return * self.reward_scaling)

        # Differential Sharpe: reward is the difference between current return and
        # the mean of the buffer, divided by the buffer's std
        elif self.reward_function == 'differential_sharpe':
            self.returns_buffer.append(net_return)

            # Require at least two observations
            if len(self.returns_buffer) < 2:
                return 0.0

            returns_array = np.array(self.returns_buffer)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)

            if std_return < 1e-8:
                return 0.0

            differential_sharpe = (net_return - mean_return) / std_return
            return float(differential_sharpe * self.reward_scaling)

        # Sortino ratio based reward
        elif self.reward_function == 'sortino':
            self.returns_buffer.append(net_return)

            if len(self.returns_buffer) < 2:
                return float(net_return * 100)

            returns_array = np.array(self.returns_buffer)
            target = 0.0
            downside_returns = returns_array[returns_array < target] - target

            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            if downside_std < 1e-8:
                downside_std = 1e-8

            sortino = (net_return - target) / downside_std
            return float(sortino * self.reward_scaling)

        # Composite reward: combination of return, risk, drawdown and turnover
        elif self.reward_function == 'composite':
            self.returns_buffer.append(net_return)

            # Return component
            return_component = net_return * self.composite_return_weight

            # Risk component: penalise volatility of returns buffer
            if len(self.returns_buffer) >= 2:
                volatility = np.std(np.array(self.returns_buffer))
                risk_component = -volatility * self.composite_risk_weight
            else:
                risk_component = 0.0

            # Drawdown component: penalise current drawdown
            current_drawdown = (self.portfolio_peak - self.portfolio_value) / self.portfolio_peak
            drawdown_component = -current_drawdown * self.composite_drawdown_weight

            # Turnover component: penalise sum of absolute position changes across assets
            turnover_component = -np.sum(np.abs(position_change)) * self.composite_turnover_weight

            total_reward = (
                return_component +
                risk_component +
                drawdown_component +
                turnover_component
            )

            return float(total_reward * self.reward_scaling)

        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}")
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (features already normalized)"""
        # Features for the current step (already normalised)
        tech_features = self.technical_features[self.current_step]

        # Portfolio state: positions per asset and current drawdown
        current_drawdown = (self.portfolio_peak - self.portfolio_value) / self.portfolio_peak

        # Flatten positions and concatenate with features and drawdown
        obs_array = np.concatenate([
            tech_features,
            self.position.astype(float),
            np.array([current_drawdown], dtype=float)
        ]).astype(np.float32)

        return obs_array
    
    def _get_info(self) -> Dict:
        """Get step info"""
        return {
            'step': self.current_step,
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'n_trades': len(self.trade_history)
        }
    
    def get_portfolio_metrics(self) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        
        portfolio_values = np.array(self.portfolio_history)
        log_returns = np.array(self.log_returns_history)
        
        if len(log_returns) == 0:
            return self._empty_metrics()
        
        ann_factor = self.ann_factor
        
        # Total return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        
        # Annualized metrics
        mean_log_return = np.mean(log_returns) * ann_factor
        volatility = np.std(log_returns) * np.sqrt(ann_factor)
        
        # Sharpe
        sharpe_ratio = mean_log_return / volatility if volatility > 0 else 0
        
        # Sortino
        downside_returns = log_returns[log_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(ann_factor) if len(downside_returns) > 0 else 1e-6
        sortino_ratio = mean_log_return / downside_std
        
        # Max Drawdown
        cumulative = np.exp(np.cumsum(log_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Calmar
        calmar_ratio = abs(mean_log_return / max_drawdown) if max_drawdown != 0 else 0
        
        # Win Rate
        winning_returns = log_returns[log_returns > 0]
        win_rate = len(winning_returns) / len(log_returns)
        
        # Profit Factor
        gross_profit = winning_returns.sum() if len(winning_returns) > 0 else 0
        losing_returns = log_returns[log_returns < 0]
        gross_loss = abs(losing_returns.sum()) if len(losing_returns) > 0 else 1e-6
        profit_factor = gross_profit / gross_loss
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'n_trades': len(self.trade_history),
            'final_value': portfolio_values[-1]
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics"""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'volatility': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'n_trades': 0,
            'final_value': self.initial_balance
        }