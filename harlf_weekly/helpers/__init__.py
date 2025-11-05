"""
Multi-Hierarchical RL Portfolio System - Helpers Package

This package provides the core components for training and evaluating
a hierarchical reinforcement learning system for portfolio optimization.

Modules:
    - config: Configuration parameters for all components
    - rewards: Reward functions (EMA Sharpe, Multi-Objective, etc.)
    - environments: Portfolio environment compatible with Gym/SB3
    - utils: Data loading, metrics, visualization utilities
    - train: Training functions for base agents
    - super_agent: Hierarchical agent architecture (Super & Meta agents)

Quick Start:
    >>> from helpers import config, create_env, train_base_agent
    >>> env = create_env('technical', split='train')
    >>> model, history = train_base_agent('technical', 'my_agent')
"""

# Import main components
from .config import (
    EnvironmentConfig,
    RewardConfig,
    BaseAgentConfig,
    SuperAgentConfig,
    MetaAgentConfig,
    AgentFeatures,
    TICKERS,
    N_ASSETS,
    DATA_DIR,
    MODELS_DIR,
    AGENT_MODELS_DIR
)

from .rewards import (
    EMASharpeReward,
    MultiObjectiveReward,
    SimpleReturnReward,
    RiskAdjustedReturnReward,
    create_reward_function
)

from .environments import (
    PortfolioEnv,
    create_env,
    create_vectorized_env
)

from .utils import (
    load_features,
    load_returns,
    load_metadata,
    prepare_env_data,
    load_features_and_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_all_metrics,
    save_agent_results,
    load_agent_results,
    print_metrics,
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_allocation_heatmap
)

from .train import (
    train_base_agent,
    evaluate_agent,
    plot_training_history,
    plot_agent_performance,
    train_super_agent,
    evaluate_super_agent,
    train_meta_agent,
    evaluate_meta_agent
)

from .super_agent import (
    BaseAgentWrapper,
    SuperAgentEnv,
    MetaAgentEnv,
    create_super_agent_env,
    create_meta_agent_env
)

__version__ = '1.0.0'
__author__ = 'Multi-Hierarchical RL Portfolio System'

__all__ = [
    # Config
    'EnvironmentConfig',
    'RewardConfig',
    'BaseAgentConfig',
    'SuperAgentConfig',
    'MetaAgentConfig',
    'AgentFeatures',
    'TICKERS',
    'N_ASSETS',
    'DATA_DIR',
    'MODELS_DIR',
    'AGENT_MODELS_DIR',
    # Rewards
    'EMASharpeReward',
    'MultiObjectiveReward',
    'SimpleReturnReward',
    'RiskAdjustedReturnReward',
    'create_reward_function',
    # Environments
    'PortfolioEnv',
    'create_env',
    'create_vectorized_env',
    # Utils
    'load_features',
    'load_returns',
    'load_metadata',
    'prepare_env_data',
    'load_features_and_returns',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_max_drawdown',
    'calculate_all_metrics',
    'save_agent_results',
    'load_agent_results',
    'print_metrics',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_returns_distribution',
    'plot_allocation_heatmap',
    # Training
    'train_base_agent',
    'evaluate_agent',
    'plot_training_history',
    'plot_agent_performance',
    'train_super_agent',
    'evaluate_super_agent',
    'train_meta_agent',
    'evaluate_meta_agent',
    # Hierarchical
    'BaseAgentWrapper',
    'SuperAgentEnv',
    'MetaAgentEnv',
    'create_super_agent_env',
    'create_meta_agent_env'
]
