import numpy as np
import math

class EMASharpeReward:
    """EMA-based Sharpe ratio reward."""

    def __init__(self, rolling_vol_window=12):
        self.W = rolling_vol_window
        self.alpha = 2.0 / (self.W + 1.0)
        self.reset()

    def reset(self):
        self._ema_mean = 0.0
        self._ema_var = 0.01
        self._initialized = False

    def compute(self, port_return):
        if not self._initialized:
            self._ema_mean = port_return
            self._ema_var = 0.01
            self._initialized = True
        else:
            delta = port_return - self._ema_mean
            self._ema_mean += self.alpha * delta
            self._ema_var = (1.0 - self.alpha) * (self._ema_var + self.alpha * delta * delta)
        vol = max(math.sqrt(max(self._ema_var, 0.0)), 1e-4)
        reward = np.clip(port_return / vol * math.sqrt(52.0), -10.0, 10.0)
        return float(reward)

class MultiObjectiveReward:
    """Multi-objective reward with penalties."""

    def __init__(
        self, 
        return_scale=8.0, 
        volatility_penalty=0.050, 
        concentration_penalty=0.250,
        turnover_penalty=0.005, 
        vol_window=12, 
        max_concentration=0.35
    ):
        self.return_scale = return_scale
        self.lambda_vol = volatility_penalty
        self.lambda_conc = concentration_penalty
        self.lambda_turn = turnover_penalty
        self.alpha_vol = 2.0 / (vol_window + 1.0)
        self.max_concentration = max_concentration
        self.reset()

    def reset(self):
        self._mean_return = 0.0
        self._ema_var = 1e-6
        self._prev_weights = None

    def compute(self, port_return, weights):
        if self._prev_weights is None:
            self._prev_weights = np.ones_like(weights) / len(weights)
        delta = port_return - self._mean_return
        self._mean_return += self.alpha_vol * delta
        self._ema_var = (1.0 - self.alpha_vol) * self._ema_var + self.alpha_vol * delta**2

        reward_return = port_return * self.return_scale
        penalty_vol = -self.lambda_vol * abs(port_return - self._mean_return)
        max_weight = np.max(weights)
        penalty_conc = -self.lambda_conc * max(0, max_weight - self.max_concentration)
        turnover = np.sum(np.abs(weights - self._prev_weights))
        penalty_turn = -self.lambda_turn * turnover

        reward = reward_return + penalty_vol + penalty_conc + penalty_turn
        reward = np.clip(reward, -10.0, 10.0)
        self._prev_weights = weights.copy()
        return float(reward)

def create_reward(reward_type, **kwargs):
    if reward_type == 'ema_sharpe':
        return EMASharpeReward(**kwargs)
    elif reward_type == 'multi_objective':
        return MultiObjectiveReward(**kwargs)
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

if __name__ == '__main__':
    print("Simplified reward functions loaded!")
    print("\nAvailable rewards:")
    print("  1. EMASharpeReward")
    print("  2. MultiObjectiveReward")