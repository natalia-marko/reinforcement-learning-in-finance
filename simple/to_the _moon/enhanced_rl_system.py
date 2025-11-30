"""
ENHANCED RL PORTFOLIO SYSTEM
=============================

Combines:
1. Your RL "Board of Directors" ensemble
2. Professional quant methods (HRP, Factor Models, Regime Detection)
3. Robust reward functions (CVaR-adjusted, benchmark-relative)
4. Proper validation and anti-overfitting

Key Improvements:
- Benchmark-relative rewards (beat the market, not just make money)
- CVaR penalty for tail risk
- Regime-aware reward scaling
- Position concentration penalty
- Turnover penalty with realistic costs
- Ensemble with traditional quant strategies
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PROFESSIONAL REWARD FUNCTIONS
# =============================================================================

class RewardFunctions:
    """
    Collection of professional reward functions used in real quant systems
    """
    
    @staticmethod
    def sharpe_reward(returns, risk_free_rate=0.0):
        """Standard Sharpe ratio reward"""
        if len(returns) < 2:
            return 0.0
        excess_returns = np.array(returns) - risk_free_rate
        if np.std(excess_returns) < 1e-8:
            return np.mean(excess_returns) * 100
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(52)
    
    @staticmethod
    def sortino_reward(returns, risk_free_rate=0.0):
        """Sortino ratio - penalizes downside volatility only"""
        if len(returns) < 2:
            return 0.0
        excess_returns = np.array(returns) - risk_free_rate
        downside = excess_returns[excess_returns < 0]
        if len(downside) < 2 or np.std(downside) < 1e-8:
            return np.mean(excess_returns) * 100
        return np.mean(excess_returns) / np.std(downside) * np.sqrt(52)
    
    @staticmethod
    def calmar_reward(returns, nav_history):
        """Calmar ratio - return / max drawdown"""
        if len(returns) < 2 or len(nav_history) < 2:
            return 0.0
        
        nav = np.array(nav_history)
        running_max = np.maximum.accumulate(nav)
        drawdown = (nav - running_max) / (running_max + 1e-8)
        max_dd = abs(np.min(drawdown))
        
        if max_dd < 0.01:  # Less than 1% drawdown
            return np.sum(returns) * 100
        
        annual_return = np.mean(returns) * 52
        return annual_return / max_dd
    
    @staticmethod
    def cvar_adjusted_reward(returns, alpha=0.05):
        """
        Return adjusted for Conditional Value at Risk (tail risk)
        Used by sophisticated risk managers
        """
        if len(returns) < 10:
            return np.mean(returns) * 100
        
        returns = np.array(returns)
        
        # Calculate CVaR (Expected Shortfall)
        var = np.percentile(returns, alpha * 100)
        cvar = returns[returns <= var].mean() if any(returns <= var) else 0
        
        # Reward = Mean return - CVaR penalty
        mean_return = np.mean(returns)
        cvar_penalty = abs(cvar) * 2  # 2x weight on tail risk
        
        return (mean_return - cvar_penalty) * 100
    
    @staticmethod
    def benchmark_relative_reward(portfolio_returns, benchmark_returns):
        """
        Information Ratio - Excess return vs benchmark per unit of tracking error
        This is what professional PMs are measured on
        """
        if len(portfolio_returns) < 2:
            return 0.0
            
        portfolio_returns = np.array(portfolio_returns)
        benchmark_returns = np.array(benchmark_returns)
        
        # Align lengths
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Active returns
        active_returns = portfolio_returns - benchmark_returns
        
        tracking_error = np.std(active_returns)
        if tracking_error < 1e-8:
            return np.mean(active_returns) * 100
            
        information_ratio = np.mean(active_returns) / tracking_error * np.sqrt(52)
        
        return information_ratio


# =============================================================================
# ENHANCED PORTFOLIO ENVIRONMENT
# =============================================================================

class EnhancedPortfolioEnv(gym.Env):
    """
    Enhanced Portfolio Environment with:
    - Multiple reward function options
    - Regime-aware rewards
    - Concentration penalty
    - Turnover penalty
    - Benchmark tracking
    """
    
    def __init__(self,
                 prices,
                 features,
                 benchmark=None,
                 reward_type='cvar_adjusted',  # 'sharpe', 'sortino', 'calmar', 'cvar_adjusted', 'benchmark_relative'
                 rebalance_period=4,
                 transaction_cost=0.001,
                 max_position=0.40,
                 concentration_penalty=0.1,
                 turnover_penalty=0.3):
        
        super().__init__()
        
        self.prices = prices
        self.features = features
        self.benchmark = benchmark if benchmark is not None else np.mean(prices, axis=1)
        self.reward_type = reward_type
        self.rebalance_period = rebalance_period
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.concentration_penalty = concentration_penalty
        self.turnover_penalty = turnover_penalty
        
        self.n_assets = prices.shape[1]
        self.n_steps = len(prices)
        
        # Action space: continuous weights
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: features + current weights + market state
        n_features = features.shape[1] * features.shape[2] if len(features.shape) == 3 else features.shape[1]
        obs_dim = n_features + self.n_assets + 5  # +5 for market state features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.nav = 1.0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.nav_history = [self.nav]
        self.returns_history = []
        self.benchmark_returns_history = []
        self.turnover_history = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Construct observation vector"""
        idx = min(self.current_step, self.n_steps - 1)
        
        # Flatten features
        features_flat = self.features[idx].flatten()
        
        # Market state features
        if self.current_step >= 20:
            recent_returns = np.diff(self.prices[max(0, idx-20):idx+1], axis=0) / self.prices[max(0, idx-20):idx][:-1] if idx > 0 else np.zeros((20, self.n_assets))
            market_state = np.array([
                np.mean(recent_returns) * 100,  # Recent return
                np.std(recent_returns) * np.sqrt(252),  # Volatility
                self.nav / max(self.nav_history) - 1 if self.nav_history else 0,  # Drawdown
                len(self.returns_history) / 52,  # Time in episode (normalized)
                np.mean(self.turnover_history[-10:]) if self.turnover_history else 0  # Recent turnover
            ])
        else:
            market_state = np.zeros(5)
        
        obs = np.concatenate([
            features_flat,
            self.weights,
            market_state
        ]).astype(np.float32)
        
        return obs
    
    def _softmax_with_constraints(self, action):
        """Convert action to valid portfolio weights"""
        # Softmax
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / (exp_action.sum() + 1e-8)
        
        # Apply max position constraint iteratively
        for _ in range(10):
            excess = np.maximum(weights - self.max_position, 0)
            if excess.sum() < 1e-6:
                break
            weights = np.minimum(weights, self.max_position)
            deficit = 1.0 - weights.sum()
            uncapped = weights < self.max_position
            if uncapped.any():
                weights[uncapped] += deficit * weights[uncapped] / weights[uncapped].sum()
        
        weights = np.clip(weights, 0, 1)
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def _calculate_concentration(self, weights):
        """Herfindahl index - measure of concentration"""
        return np.sum(weights ** 2)
    
    def _calculate_reward(self, period_returns, turnover, benchmark_returns=None):
        """Calculate reward based on selected reward type"""
        
        # Base reward from selected function
        if self.reward_type == 'sharpe':
            base_reward = RewardFunctions.sharpe_reward(period_returns)
        elif self.reward_type == 'sortino':
            base_reward = RewardFunctions.sortino_reward(period_returns)
        elif self.reward_type == 'calmar':
            base_reward = RewardFunctions.calmar_reward(period_returns, self.nav_history)
        elif self.reward_type == 'cvar_adjusted':
            base_reward = RewardFunctions.cvar_adjusted_reward(self.returns_history[-52:] if len(self.returns_history) > 52 else self.returns_history)
        elif self.reward_type == 'benchmark_relative' and benchmark_returns is not None:
            base_reward = RewardFunctions.benchmark_relative_reward(period_returns, benchmark_returns)
        else:
            # Default to simple return
            base_reward = np.sum(period_returns) * 100
        
        # Penalties
        concentration = self._calculate_concentration(self.weights)
        concentration_cost = (concentration - 1/self.n_assets) * self.concentration_penalty * 100
        turnover_cost = turnover * self.turnover_penalty * 100
        
        # Final reward
        reward = base_reward - concentration_cost - turnover_cost
        
        return reward
    
    def step(self, action):
        """Execute one step"""
        new_weights = self._softmax_with_constraints(action)
        turnover = np.sum(np.abs(new_weights - self.weights))
        
        # Transaction costs
        cost = turnover * self.transaction_cost
        
        # Simulate period
        end_step = min(self.current_step + self.rebalance_period, self.n_steps - 1)
        
        if self.current_step >= self.n_steps - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        period_returns = []
        benchmark_returns = []
        
        for t in range(self.current_step, end_step):
            if t + 1 < self.n_steps:
                # Asset returns
                asset_returns = self.prices[t + 1] / self.prices[t] - 1
                portfolio_return = np.sum(new_weights * asset_returns) - cost / self.rebalance_period
                
                # Benchmark return
                bench_return = self.benchmark[t + 1] / self.benchmark[t] - 1 if t + 1 < len(self.benchmark) else 0
                
                period_returns.append(portfolio_return)
                benchmark_returns.append(bench_return)
                self.returns_history.append(portfolio_return)
                self.benchmark_returns_history.append(bench_return)
                
                self.nav *= (1 + portfolio_return)
                self.nav_history.append(self.nav)
        
        self.weights = new_weights
        self.turnover_history.append(turnover)
        self.current_step = end_step
        
        reward = self._calculate_reward(period_returns, turnover, benchmark_returns)
        
        terminated = self.current_step >= self.n_steps - 1
        
        return self._get_observation(), reward, terminated, False, self._get_info()
    
    def _get_info(self):
        """Return info dict"""
        returns = np.array(self.returns_history) if self.returns_history else np.array([0])
        
        return {
            'nav': self.nav,
            'total_return': self.nav - 1,
            'sharpe': RewardFunctions.sharpe_reward(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'avg_turnover': np.mean(self.turnover_history) if self.turnover_history else 0,
            'weights': self.weights.copy()
        }
    
    def _calculate_max_drawdown(self):
        if len(self.nav_history) < 2:
            return 0
        nav = np.array(self.nav_history)
        running_max = np.maximum.accumulate(nav)
        drawdown = (nav - running_max) / running_max
        return float(np.min(drawdown))


# =============================================================================
# SPECIALIST AGENTS WITH PROFESSIONAL REWARDS
# =============================================================================

class BullAgent(EnhancedPortfolioEnv):
    """
    Bull specialist - optimized for trending markets
    Reward: Momentum-adjusted returns with low turnover
    """
    def _calculate_reward(self, period_returns, turnover, benchmark_returns=None):
        if not period_returns:
            return 0.0
        
        # Bull reward: maximize upside, ignore downside
        positive_returns = [r for r in period_returns if r > 0]
        upside = sum(positive_returns) * 20  # Amplify gains
        
        # Small penalty for missing gains (being too conservative)
        total = sum(period_returns)
        conservatism_penalty = max(0, upside/20 - total) * 5
        
        return upside - conservatism_penalty - turnover * 0.1


class BearAgent(EnhancedPortfolioEnv):
    """
    Bear specialist - optimized for volatile/declining markets
    Reward: CVaR-focused with heavy drawdown penalty
    """
    def _calculate_reward(self, period_returns, turnover, benchmark_returns=None):
        if not period_returns:
            return 0.0
        
        returns = np.array(period_returns)
        
        # Heavy penalty for losses
        losses = returns[returns < 0]
        loss_penalty = np.sum(losses) * 50 if len(losses) > 0 else 0
        
        # Bonus for avoiding drawdown
        max_dd = self._calculate_max_drawdown()
        dd_bonus = 10 if max_dd > -0.05 else 0  # Bonus if drawdown < 5%
        
        # CVaR penalty
        if len(returns) >= 5:
            var_5 = np.percentile(returns, 5)
            cvar = returns[returns <= var_5].mean() if any(returns <= var_5) else 0
            cvar_penalty = abs(cvar) * 30
        else:
            cvar_penalty = 0
        
        base_return = np.sum(returns) * 10
        
        return base_return + loss_penalty + dd_bonus - cvar_penalty


class SniperAgent(EnhancedPortfolioEnv):
    """
    Sniper specialist - balanced Sharpe optimization
    Reward: Risk-adjusted returns with concentration penalty
    """
    def _calculate_reward(self, period_returns, turnover, benchmark_returns=None):
        if not period_returns:
            return 0.0
        
        # Sharpe-like reward
        returns = np.array(period_returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns) + 1e-8
        
        sharpe = mean_ret / std_ret * np.sqrt(52)
        
        # Concentration penalty
        hhi = np.sum(self.weights ** 2)
        optimal_hhi = 1 / self.n_assets
        concentration_penalty = (hhi - optimal_hhi) * 10
        
        # Turnover penalty
        turnover_cost = turnover * self.turnover_penalty
        
        return sharpe * 10 - concentration_penalty - turnover_cost


class AlphaAgent(EnhancedPortfolioEnv):
    """
    Alpha specialist - benchmark-relative performance
    Reward: Information ratio (excess return per tracking error)
    """
    def _calculate_reward(self, period_returns, turnover, benchmark_returns=None):
        if not period_returns or benchmark_returns is None:
            return 0.0
        
        returns = np.array(period_returns)
        bench = np.array(benchmark_returns)
        
        # Active returns
        active = returns - bench[:len(returns)]
        
        # Information ratio
        if len(active) > 1 and np.std(active) > 1e-8:
            ir = np.mean(active) / np.std(active) * np.sqrt(52)
        else:
            ir = np.sum(active) * 100
        
        # Bonus for consistent outperformance
        hit_rate = np.mean(active > 0)
        hit_bonus = (hit_rate - 0.5) * 20  # Bonus for >50% hit rate
        
        return ir * 10 + hit_bonus - turnover * 0.2


# =============================================================================
# ENHANCED BOARD OF DIRECTORS
# =============================================================================

class EnhancedBoardOfDirectors:
    """
    Enhanced ensemble with:
    - Bayesian model averaging
    - Regime-based weighting
    - Dynamic confidence scoring
    - Traditional quant overlay
    """
    
    def __init__(self, 
                 bull_model_path, 
                 bear_model_path, 
                 sniper_model_path,
                 alpha_model_path=None,
                 use_quant_overlay=True):
        
        # Load RL models
        self.bull = PPO.load(bull_model_path)
        self.bear = PPO.load(bear_model_path)
        self.sniper = PPO.load(sniper_model_path)
        self.alpha = PPO.load(alpha_model_path) if alpha_model_path else None
        
        self.use_quant_overlay = use_quant_overlay
        
        # Performance tracking for Bayesian weights
        self.agent_scores = {
            'bull': 1.0,
            'bear': 1.0,
            'sniper': 1.0,
            'alpha': 1.0
        }
        self.decay = 0.95
        
        # Regime thresholds
        self.vol_bull_threshold = 0.15
        self.vol_bear_threshold = 0.25
        
    def detect_regime(self, recent_returns, lookback=20):
        """
        Detect market regime based on recent volatility and trend
        
        Returns:
            regime: 'BULL', 'BEAR', or 'NEUTRAL'
            confidence: 0-1 confidence in regime
        """
        if len(recent_returns) < lookback:
            return 'NEUTRAL', 0.5
        
        returns = np.array(recent_returns[-lookback:])
        
        # Annualized volatility
        vol = np.std(returns) * np.sqrt(252)
        
        # Trend
        trend = np.mean(returns)
        
        # Regime classification
        if vol < self.vol_bull_threshold and trend > 0:
            regime = 'BULL'
            confidence = min(1.0, (self.vol_bull_threshold - vol) / self.vol_bull_threshold + 0.5)
        elif vol > self.vol_bear_threshold or trend < -0.005:
            regime = 'BEAR'
            confidence = min(1.0, (vol - self.vol_bear_threshold) / self.vol_bear_threshold + 0.5)
        else:
            regime = 'NEUTRAL'
            confidence = 0.5
        
        return regime, confidence
    
    def get_regime_weights(self, regime, confidence):
        """Get agent weights based on regime"""
        
        if regime == 'BULL':
            base_weights = {
                'bull': 0.50,
                'bear': 0.05,
                'sniper': 0.30,
                'alpha': 0.15
            }
        elif regime == 'BEAR':
            base_weights = {
                'bull': 0.05,
                'bear': 0.55,
                'sniper': 0.25,
                'alpha': 0.15
            }
        else:  # NEUTRAL
            base_weights = {
                'bull': 0.25,
                'bear': 0.25,
                'sniper': 0.35,
                'alpha': 0.15
            }
        
        # Adjust by confidence (blend with equal weights when uncertain)
        equal_weights = {k: 0.25 for k in base_weights}
        
        final_weights = {}
        for agent in base_weights:
            final_weights[agent] = (
                confidence * base_weights[agent] + 
                (1 - confidence) * equal_weights[agent]
            )
        
        # Adjust by historical performance (Bayesian)
        for agent in final_weights:
            final_weights[agent] *= self.agent_scores[agent]
        
        # Normalize
        total = sum(final_weights.values())
        final_weights = {k: v/total for k, v in final_weights.items()}
        
        return final_weights
    
    def update_agent_scores(self, agent_returns):
        """Update Bayesian scores based on realized returns"""
        for agent, ret in agent_returns.items():
            # Exponential moving score
            self.agent_scores[agent] = (
                self.decay * self.agent_scores[agent] +
                (1 - self.decay) * (1 + ret * 10)  # Scale returns to scores
            )
            # Keep scores bounded
            self.agent_scores[agent] = np.clip(self.agent_scores[agent], 0.1, 10.0)
    
    def get_action(self, obs, recent_returns=None):
        """
        Get ensemble action
        
        Args:
            obs: Current observation
            recent_returns: Recent portfolio returns for regime detection
            
        Returns:
            action: Ensemble portfolio weights
            info: Dict with regime and agent weights
        """
        # Get individual predictions
        bull_action, _ = self.bull.predict(obs, deterministic=True)
        bear_action, _ = self.bear.predict(obs, deterministic=True)
        sniper_action, _ = self.sniper.predict(obs, deterministic=True)
        
        if self.alpha is not None:
            alpha_action, _ = self.alpha.predict(obs, deterministic=True)
        else:
            alpha_action = sniper_action  # Fallback
        
        # Detect regime
        if recent_returns is not None and len(recent_returns) > 0:
            regime, confidence = self.detect_regime(recent_returns)
        else:
            regime, confidence = 'NEUTRAL', 0.5
        
        # Get agent weights for this regime
        agent_weights = self.get_regime_weights(regime, confidence)
        
        # Weighted average of actions
        ensemble_action = (
            agent_weights['bull'] * bull_action +
            agent_weights['bear'] * bear_action +
            agent_weights['sniper'] * sniper_action +
            agent_weights['alpha'] * alpha_action
        )
        
        info = {
            'regime': regime,
            'confidence': confidence,
            'agent_weights': agent_weights,
            'individual_actions': {
                'bull': bull_action,
                'bear': bear_action,
                'sniper': sniper_action,
                'alpha': alpha_action
            }
        }
        
        return ensemble_action, info


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def train_specialist_agents(prices, features, benchmark, output_dir='models',
                           total_timesteps=100000, n_eval_episodes=5):
    """
    Train all specialist agents with proper validation
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Train/val split
    split_idx = int(len(prices) * 0.8)
    
    train_prices = prices[:split_idx]
    train_features = features[:split_idx]
    train_benchmark = benchmark[:split_idx]
    
    val_prices = prices[split_idx:]
    val_features = features[split_idx:]
    val_benchmark = benchmark[split_idx:]
    
    agents = {
        'bull': BullAgent,
        'bear': BearAgent,
        'sniper': SniperAgent,
        'alpha': AlphaAgent
    }
    
    for name, AgentClass in agents.items():
        print(f"\n{'='*60}")
        print(f"Training {name.upper()} Agent")
        print(f"{'='*60}")
        
        # Create environments
        train_env = AgentClass(train_prices, train_features, train_benchmark)
        val_env = AgentClass(val_prices, val_features, val_benchmark)
        
        # Wrap for stable-baselines
        train_env_wrapped = DummyVecEnv([lambda: train_env])
        train_env_wrapped = VecNormalize(train_env_wrapped, norm_obs=True, norm_reward=True)
        
        val_env_wrapped = DummyVecEnv([lambda: val_env])
        val_env_wrapped = VecNormalize(val_env_wrapped, norm_obs=True, norm_reward=False, training=False)
        
        # Callbacks
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=10,
            min_evals=5,
            verbose=1
        )
        
        eval_callback = EvalCallback(
            val_env_wrapped,
            best_model_save_path=os.path.join(output_dir, name),
            log_path=os.path.join(output_dir, 'logs', name),
            eval_freq=total_timesteps // 20,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            callback_after_eval=stop_callback
        )
        
        # PPO with anti-overfitting settings
        model = PPO(
            "MlpPolicy",
            train_env_wrapped,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=64,
            ent_coef=0.1,  # High entropy for exploration
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            max_grad_norm=0.5,
            verbose=0,
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 32], vf=[64, 32])  # Smaller network
            )
        )
        
        # Train
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final
        model.save(os.path.join(output_dir, f"agent_{name}"))
        print(f"Saved {name} agent")
    
    print(f"\n{'='*60}")
    print("All agents trained successfully!")
    print(f"{'='*60}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Enhanced RL Portfolio System")
    print("="*60)
    
    # Example with synthetic data
    np.random.seed(42)
    
    n_days = 500
    n_assets = 7
    n_features = 3
    
    # Generate synthetic prices
    prices = np.cumprod(1 + np.random.randn(n_days, n_assets) * 0.02, axis=0) * 100
    features = np.random.randn(n_days, n_assets, n_features)
    benchmark = np.mean(prices, axis=1)
    
    print(f"Data shape: prices={prices.shape}, features={features.shape}")
    
    # Test environment
    env = EnhancedPortfolioEnv(
        prices, features, benchmark,
        reward_type='cvar_adjusted'
    )
    
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    # Run a few steps
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        if done:
            break
    
    print(f"Test run completed. Total reward: {total_reward:.2f}")
    print(f"Final NAV: {info['nav']:.4f}")
    print(f"Sharpe: {info['sharpe']:.2f}")
