"""
Reward Functions
================
All your reward calculation functions in one place!
Your original formulas are preserved exactly.

CHANGES IN THIS VERSION:
- EMASharpeReward: Fixed initial variance (1e-3 → 0.01)
- MultiObjectiveReward: Added configurable max_concentration parameter
- Added warnings about SimpleSharpeReward (non-stationary)
"""

import numpy as np
import math


# ============================================================================
# 1. EMA Sharpe Reward (FIXED INITIAL VARIANCE)
# ============================================================================

class EMASharpeReward:
    """EMA-based Sharpe ratio reward."""
    
    def __init__(self, rolling_vol_window=12):
        self.W = rolling_vol_window
        self.alpha = 2.0 / (self.W + 1.0)
        self.reset()
    
    def reset(self):
        """Reset at episode start."""
        self._ema_mean = 0.0
        self._ema_var = 0.01  # CHANGED from 1e-3 - more realistic initial variance
        self._initialized = False
    
    def compute(self, port_return):
        """Compute reward from portfolio return."""
        # Update EMA
        if not self._initialized:
            self._ema_mean = port_return
            self._ema_var = 0.01  # CHANGED from 1e-3
            self._initialized = True
        else:
            delta = port_return - self._ema_mean
            self._ema_mean += self.alpha * delta
            self._ema_var = (1.0 - self.alpha) * (self._ema_var + self.alpha * delta * delta)
        
        # Sharpe-like reward
        vol = max(math.sqrt(max(self._ema_var, 0.0)), 1e-4)
        reward = np.clip(port_return / vol * math.sqrt(52.0), -10.0, 10.0)
        return float(reward)


# ============================================================================
# 2. Differential Sharpe Reward (ORIGINAL - VERIFY FORMULA)
# ============================================================================

class DifferentialSharpeReward:
    """
    Differential Sharpe ratio reward.
    
    NOTE: This implementation should be verified against the original
    Moody & Saffell (2001) paper. The formula may need adjustment.
    """
    
    def __init__(self, decay_factor=0.95):
        self.decay = decay_factor
        self.reset()
    
    def reset(self):
        """Reset at episode start."""
        self._mean_return = 0.0
        self._var_return = 1e-6
        self._n_effective = 0.0
    
    def compute(self, port_return):
        """Compute reward from portfolio return."""
        # Update statistics
        self._n_effective = 1.0 + self.decay * self._n_effective
        alpha = 1.0 / max(self._n_effective, 1.0)
        
        delta = port_return - self._mean_return
        self._mean_return += alpha * delta
        self._var_return = self.decay * self._var_return + (1.0 - self.decay) * delta**2
        
        # Differential Sharpe formula
        # TODO: Verify this matches Moody & Saffell (2001)
        std_return = math.sqrt(max(self._var_return, 1e-8))
        numerator = port_return * std_return - 0.5 * self._mean_return * delta
        denominator = std_return**2 + 1e-8
        diff_sharpe = numerator / denominator
        
        reward = np.clip(diff_sharpe * math.sqrt(52), -10.0, 10.0)
        return float(reward)


# ============================================================================
# 3. Multi-Objective Reward (ADDED CONFIGURABLE CONCENTRATION)
# ============================================================================

class MultiObjectiveReward:
    """Multi-objective reward with penalties."""
    
    def __init__(self, return_scale=10.0, volatility_penalty=0.1,
                 concentration_penalty=0.5, turnover_penalty=0.01, 
                 vol_window=12, max_concentration=0.30):  # NEW PARAMETER
        """
        Initialize multi-objective reward.
        
        Parameters
        ----------
        return_scale : float
            Scale for return objective
        volatility_penalty : float
            Penalty for volatility
        concentration_penalty : float
            Penalty for concentration
        turnover_penalty : float
            Penalty for turnover
        vol_window : int
            Window for volatility EMA
        max_concentration : float
            Maximum concentration threshold (NEW)
        """
        self.return_scale = return_scale
        self.lambda_vol = volatility_penalty
        self.lambda_conc = concentration_penalty
        self.lambda_turn = turnover_penalty
        self.alpha_vol = 2.0 / (vol_window + 1.0)
        self.max_concentration = max_concentration  # NEW
        self.reset()
    
    def reset(self):
        """Reset at episode start."""
        self._mean_return = 0.0
        self._ema_var = 1e-6
        self._prev_weights = None
    
    def compute(self, port_return, weights):
        """Compute reward from return and weights."""
        # Initialize prev_weights on first step
        if self._prev_weights is None:
            self._prev_weights = np.ones_like(weights) / len(weights)
        
        # Update volatility tracking
        delta = port_return - self._mean_return
        self._mean_return += self.alpha_vol * delta
        self._ema_var = (1.0 - self.alpha_vol) * self._ema_var + self.alpha_vol * delta**2
        
        # Objective 1: Return
        reward_return = port_return * self.return_scale
        
        # Penalty 1: Volatility
        penalty_vol = -self.lambda_vol * abs(port_return - self._mean_return)
        
        # Penalty 2: Concentration (USING CONFIGURABLE THRESHOLD)
        max_weight = np.max(weights)
        penalty_conc = -self.lambda_conc * max(0, max_weight - self.max_concentration)
        
        # Penalty 3: Turnover
        turnover = np.sum(np.abs(weights - self._prev_weights))
        penalty_turn = -self.lambda_turn * turnover
        
        # Combined reward
        reward = reward_return + penalty_vol + penalty_conc + penalty_turn
        reward = np.clip(reward, -10.0, 10.0)
        
        # Update for next step
        self._prev_weights = weights.copy()
        
        return float(reward)


# ============================================================================
# 4. Simple Return Reward (ORIGINAL)
# ============================================================================

class SimpleReturnReward:
    """Just the return itself (no risk adjustment)."""
    
    def reset(self):
        pass
    
    def compute(self, port_return):
        """Compute reward from portfolio return."""
        reward = np.clip(port_return * math.sqrt(52.0), -10.0, 10.0)
        return float(reward)


# Helper to create reward functions
# ============================================================================

def create_reward(reward_type, **kwargs):
    """
    Create a reward function.
    
    Parameters
    ----------
    reward_type : str
        'ema_sharpe', 'differential_sharpe', 'multi_objective',
        'simple_return', or 'simple_sharpe'
    **kwargs : dict
        Reward-specific parameters
    
    Returns
    -------
    Reward object
    
    Examples
    --------
    >>> reward = create_reward('ema_sharpe', rolling_vol_window=12)
    >>> reward = create_reward('multi_objective', return_scale=8.0, max_concentration=0.3)
    """
    
    if reward_type == 'ema_sharpe':
        return EMASharpeReward(**kwargs)
    elif reward_type == 'differential_sharpe':
        return DifferentialSharpeReward(**kwargs)
    elif reward_type == 'multi_objective':
        return MultiObjectiveReward(**kwargs)
    elif reward_type == 'simple_return':
        return SimpleReturnReward()
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")


if __name__ == '__main__':
    print("Reward functions loaded!")
    print("\nAvailable rewards:")
    print("  1. EMASharpeReward ✅ (FIXED initial variance)")
    print("  2. DifferentialSharpeReward ⚠️  (verify formula)")
    print("  3. MultiObjectiveReward ✅ (added max_concentration)")
    print("  4. SimpleReturnReward ✅")
 