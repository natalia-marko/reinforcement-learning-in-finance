"""
Configuration module for Multi-Hierarchical RL Portfolio System.

This module contains all configuration parameters for:
- Portfolio setup
- Environment parameters
- Training hyperparameters
- Reward function settings
- Agent configurations
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data_hierarchical'
MODELS_DIR = PROJECT_ROOT / 'models'
AGENT_MODELS_DIR = MODELS_DIR / 'agent_models'

# Create directories if they don't exist
MODELS_DIR.mkdir(exist_ok=True)
AGENT_MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# PORTFOLIO CONFIGURATION
# ============================================================================

TICKERS = ['NVDA', 'MU', 'AAPL', 'AMD', 'ASML', 'MSFT', 'GOOG']
BENCHMARK = 'QQQ'
N_ASSETS = len(TICKERS)

# Date ranges
START_DATE = '2020-01-01'
END_DATE = '2025-11-04'

# Rebalancing
REBALANCING_FREQUENCY = 'weekly'  # Weekly rebalancing on Friday close

# ============================================================================
# ENVIRONMENT PARAMETERS
# ============================================================================

class EnvironmentConfig:
    """Configuration for portfolio environment."""

    # Transaction costs
    TRANSACTION_COST = 0.001  # 0.1% per trade (10 bps)

    # Position constraints
    MIN_POSITION = 0.0        # No short selling
    MAX_POSITION = 1.0        # Maximum 100% in single asset
    MIN_CASH = 0.0            # Minimum cash position (0% = fully invested allowed)

    # Initial capital
    INITIAL_CAPITAL = 100000  # $100k starting capital

    # Risk constraints
    MAX_LEVERAGE = 1.0        # No leverage (1.0 = 100% invested max)
    MAX_DRAWDOWN = 0.3        # 30% max drawdown before forced liquidation

    # Episode parameters
    LOOKBACK_WINDOW = 52      # Number of past weeks to include in state (1 year)

    # Normalization
    NORMALIZE_OBSERVATIONS = True
    NORMALIZE_RETURNS = True

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            'transaction_cost': cls.TRANSACTION_COST,
            'min_position': cls.MIN_POSITION,
            'max_position': cls.MAX_POSITION,
            'min_cash': cls.MIN_CASH,
            'initial_capital': cls.INITIAL_CAPITAL,
            'max_leverage': cls.MAX_LEVERAGE,
            'max_drawdown': cls.MAX_DRAWDOWN,
            'lookback_window': cls.LOOKBACK_WINDOW,
            'normalize_observations': cls.NORMALIZE_OBSERVATIONS,
            'normalize_returns': cls.NORMALIZE_RETURNS
        }


# ============================================================================
# REWARD FUNCTION PARAMETERS
# ============================================================================

class RewardConfig:
    """Configuration for reward functions."""

    # EMA Sharpe Reward
    EMA_ALPHA = 0.1           # EMA smoothing factor (0.1 = ~20 period)
    TARGET_SHARPE = 2.0       # Target annual Sharpe ratio
    RISK_FREE_RATE = 0.04     # 4% annual risk-free rate

    # Multi-Objective Reward Weights
    RETURN_WEIGHT = 0.5       # Weight for returns
    RISK_WEIGHT = 0.3         # Weight for risk (Sharpe)
    DRAWDOWN_WEIGHT = 0.2     # Weight for drawdown penalty

    # Penalties
    VIOLATION_PENALTY = -10.0  # Penalty for constraint violations
    BANKRUPTCY_PENALTY = -100.0  # Penalty for going broke

    # Reward scaling
    REWARD_SCALE = 100.0      # Scale rewards to ~[-10, 10] range

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            'ema_alpha': cls.EMA_ALPHA,
            'target_sharpe': cls.TARGET_SHARPE,
            'risk_free_rate': cls.RISK_FREE_RATE,
            'return_weight': cls.RETURN_WEIGHT,
            'risk_weight': cls.RISK_WEIGHT,
            'drawdown_weight': cls.DRAWDOWN_WEIGHT,
            'violation_penalty': cls.VIOLATION_PENALTY,
            'bankruptcy_penalty': cls.BANKRUPTCY_PENALTY,
            'reward_scale': cls.REWARD_SCALE
        }


# ============================================================================
# BASE AGENT TRAINING PARAMETERS
# ============================================================================

class BaseAgentConfig:
    """Configuration for base agent training (Technical & Sentiment)."""

    # PPO Hyperparameters
    LEARNING_RATE = 3e-4
    N_STEPS = 2048            # Steps to collect before update
    BATCH_SIZE = 64
    N_EPOCHS = 10             # Optimization epochs per update
    GAMMA = 0.99              # Discount factor
    GAE_LAMBDA = 0.95         # GAE lambda
    CLIP_RANGE = 0.2          # PPO clip range
    ENT_COEF = 0.01           # Entropy coefficient
    VF_COEF = 0.5             # Value function coefficient
    MAX_GRAD_NORM = 0.5       # Gradient clipping

    # Training duration
    TOTAL_TIMESTEPS = 200000  # Total training steps

    # Network architecture
    NET_ARCH = [256, 256]     # Hidden layer sizes
    ACTIVATION = 'tanh'       # Activation function

    # Evaluation
    EVAL_FREQ = 10000         # Evaluate every N steps
    N_EVAL_EPISODES = 10      # Episodes per evaluation

    # Early stopping
    PATIENCE = 5              # Evaluations without improvement before stopping
    MIN_DELTA = 0.01          # Minimum improvement threshold

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'n_steps': cls.N_STEPS,
            'batch_size': cls.BATCH_SIZE,
            'n_epochs': cls.N_EPOCHS,
            'gamma': cls.GAMMA,
            'gae_lambda': cls.GAE_LAMBDA,
            'clip_range': cls.CLIP_RANGE,
            'ent_coef': cls.ENT_COEF,
            'vf_coef': cls.VF_COEF,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'total_timesteps': cls.TOTAL_TIMESTEPS,
            'net_arch': cls.NET_ARCH,
            'activation': cls.ACTIVATION,
            'eval_freq': cls.EVAL_FREQ,
            'n_eval_episodes': cls.N_EVAL_EPISODES,
            'patience': cls.PATIENCE,
            'min_delta': cls.MIN_DELTA
        }


# ============================================================================
# SUPER AGENT TRAINING PARAMETERS
# ============================================================================

class SuperAgentConfig:
    """Configuration for super agent training (blends base agents)."""

    # PPO Hyperparameters (similar to base but potentially different)
    LEARNING_RATE = 1e-4      # Lower LR for meta-learning
    N_STEPS = 1024
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.005          # Lower entropy for more stable meta-policy
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    # Training duration
    TOTAL_TIMESTEPS = 100000  # Fewer steps (learns faster)

    # Network architecture
    NET_ARCH = [128, 128]     # Smaller network
    ACTIVATION = 'tanh'

    # Evaluation
    EVAL_FREQ = 5000
    N_EVAL_EPISODES = 10

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.01

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'n_steps': cls.N_STEPS,
            'batch_size': cls.BATCH_SIZE,
            'n_epochs': cls.N_EPOCHS,
            'gamma': cls.GAMMA,
            'gae_lambda': cls.GAE_LAMBDA,
            'clip_range': cls.CLIP_RANGE,
            'ent_coef': cls.ENT_COEF,
            'vf_coef': cls.VF_COEF,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'total_timesteps': cls.TOTAL_TIMESTEPS,
            'net_arch': cls.NET_ARCH,
            'activation': cls.ACTIVATION,
            'eval_freq': cls.EVAL_FREQ,
            'n_eval_episodes': cls.N_EVAL_EPISODES,
            'patience': cls.PATIENCE,
            'min_delta': cls.MIN_DELTA
        }


# ============================================================================
# META AGENT TRAINING PARAMETERS
# ============================================================================

class MetaAgentConfig:
    """Configuration for meta agent training (adjusts super agent)."""

    # PPO Hyperparameters
    LEARNING_RATE = 5e-5      # Even lower LR for high-level policy
    N_STEPS = 512
    BATCH_SIZE = 32
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.15         # Tighter clip for stability
    ENT_COEF = 0.001          # Minimal entropy
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5

    # Training duration
    TOTAL_TIMESTEPS = 50000   # Even fewer steps

    # Network architecture
    NET_ARCH = [64, 64]       # Smallest network
    ACTIVATION = 'tanh'

    # Evaluation
    EVAL_FREQ = 2500
    N_EVAL_EPISODES = 10

    # Early stopping
    PATIENCE = 5
    MIN_DELTA = 0.01

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary."""
        return {
            'learning_rate': cls.LEARNING_RATE,
            'n_steps': cls.N_STEPS,
            'batch_size': cls.BATCH_SIZE,
            'n_epochs': cls.N_EPOCHS,
            'gamma': cls.GAMMA,
            'gae_lambda': cls.GAE_LAMBDA,
            'clip_range': cls.CLIP_RANGE,
            'ent_coef': cls.ENT_COEF,
            'vf_coef': cls.VF_COEF,
            'max_grad_norm': cls.MAX_GRAD_NORM,
            'total_timesteps': cls.TOTAL_TIMESTEPS,
            'net_arch': cls.NET_ARCH,
            'activation': cls.ACTIVATION,
            'eval_freq': cls.EVAL_FREQ,
            'n_eval_episodes': cls.N_EVAL_EPISODES,
            'patience': cls.PATIENCE,
            'min_delta': cls.MIN_DELTA
        }


# ============================================================================
# AGENT FEATURE SPECIFICATIONS
# ============================================================================

class AgentFeatures:
    """Feature specifications for each agent type."""

    TECHNICAL = [
        # Trend Following (5)
        'price_to_sma_4w', 'price_to_sma_8w', 'price_to_sma_12w',
        'price_to_ema_8w', 'price_to_ema_12w',
        # Momentum (7)
        'macd_histogram', 'return_lag_1w', 'return_lag_2w', 'return_lag_3w',
        'rsi_14w', 'stochastic_14w', 'roc_4w',
        # Volatility (3)
        'volatility_12w', 'atr_pct_14w', 'bb_position_20w',
        # Volume (3)
        'volume_ratio_20w', 'mfi_14w', 'obv_roc_4w',
        # Benchmark (2)
        'bench_corr_12w', 'bench_beta_12w'
    ]

    SENTIMENT = [
        # Momentum Signals (3)
        'momentum_2w', 'momentum_4w', 'momentum_6w',
        # Trend Quality (2)
        'positive_weeks_pct', 'price_dispersion',
        # Volume Sentiment (2)
        'volume_sentiment', 'volume_surge',
        # Market Regime (2)
        'vol_regime', 'drawdown',
        # Fear/Greed (3)
        'market_fear', 'credit_sentiment', 'news_sentiment'
    ]

    MACRO = [
        # Interest Rate Environment (4)
        'treasury_10y', 'fed_funds_rate', 'yield_curve_2_10', 'real_yield_10y',
        # Market Volatility & Risk (2)
        'vix', 'vix_regime',
        # Economic Health (2)
        'credit_spread', 'unemployment_rate',
        # Calendar Effects (4)
        'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos'
    ]

    @classmethod
    def get_feature_count(cls, agent_type):
        """Get number of features for agent type."""
        if agent_type == 'technical':
            return len(cls.TECHNICAL)
        elif agent_type == 'sentiment':
            return len(cls.SENTIMENT)
        elif agent_type == 'macro':
            return len(cls.MACRO)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    @classmethod
    def get_features(cls, agent_type):
        """Get feature list for agent type."""
        if agent_type == 'technical':
            return cls.TECHNICAL
        elif agent_type == 'sentiment':
            return cls.SENTIMENT
        elif agent_type == 'macro':
            return cls.MACRO
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_all_configs():
    """Get all configuration dictionaries."""
    return {
        'environment': EnvironmentConfig.to_dict(),
        'reward': RewardConfig.to_dict(),
        'base_agent': BaseAgentConfig.to_dict(),
        'super_agent': SuperAgentConfig.to_dict(),
        'meta_agent': MetaAgentConfig.to_dict(),
        'portfolio': {
            'tickers': TICKERS,
            'benchmark': BENCHMARK,
            'n_assets': N_ASSETS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'rebalancing_frequency': REBALANCING_FREQUENCY
        },
        'features': {
            'technical': AgentFeatures.TECHNICAL,
            'sentiment': AgentFeatures.SENTIMENT,
            'macro': AgentFeatures.MACRO
        }
    }


def print_config_summary():
    """Print configuration summary."""
    print("="*70)
    print("MULTI-HIERARCHICAL RL PORTFOLIO SYSTEM - CONFIGURATION")
    print("="*70)

    print(f"\n📊 Portfolio:")
    print(f"   Stocks: {', '.join(TICKERS)}")
    print(f"   Benchmark: {BENCHMARK}")
    print(f"   Rebalancing: {REBALANCING_FREQUENCY}")

    print(f"\n🏗️  Environment:")
    print(f"   Transaction Cost: {EnvironmentConfig.TRANSACTION_COST*100:.2f}%")
    print(f"   Max Leverage: {EnvironmentConfig.MAX_LEVERAGE:.1f}x")
    print(f"   Max Drawdown: {EnvironmentConfig.MAX_DRAWDOWN*100:.0f}%")

    print(f"\n🎯 Rewards:")
    print(f"   EMA Alpha: {RewardConfig.EMA_ALPHA}")
    print(f"   Target Sharpe: {RewardConfig.TARGET_SHARPE}")
    print(f"   Risk-Free Rate: {RewardConfig.RISK_FREE_RATE*100:.1f}%")

    print(f"\n🤖 Base Agents:")
    print(f"   Learning Rate: {BaseAgentConfig.LEARNING_RATE}")
    print(f"   Total Timesteps: {BaseAgentConfig.TOTAL_TIMESTEPS:,}")
    print(f"   Network: {BaseAgentConfig.NET_ARCH}")

    print(f"\n🦸 Super Agent:")
    print(f"   Learning Rate: {SuperAgentConfig.LEARNING_RATE}")
    print(f"   Total Timesteps: {SuperAgentConfig.TOTAL_TIMESTEPS:,}")
    print(f"   Network: {SuperAgentConfig.NET_ARCH}")

    print(f"\n🧠 Meta Agent:")
    print(f"   Learning Rate: {MetaAgentConfig.LEARNING_RATE}")
    print(f"   Total Timesteps: {MetaAgentConfig.TOTAL_TIMESTEPS:,}")
    print(f"   Network: {MetaAgentConfig.NET_ARCH}")

    print(f"\n📈 Features:")
    print(f"   Technical: {len(AgentFeatures.TECHNICAL)} features")
    print(f"   Sentiment: {len(AgentFeatures.SENTIMENT)} features")
    print(f"   Macro: {len(AgentFeatures.MACRO)} features")
    print(f"   Total: {len(AgentFeatures.TECHNICAL) + len(AgentFeatures.SENTIMENT) + len(AgentFeatures.MACRO)} features")

    print("="*70)


if __name__ == '__main__':
    print_config_summary()
