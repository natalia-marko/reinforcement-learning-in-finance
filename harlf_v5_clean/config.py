"""
Centralized configuration management for training parameters

This module provides a dataclass-based configuration system for all training
hyperparameters, making it easy to manage and version control configurations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import json
from pathlib import Path


@dataclass
class TrainingConfig:
    # Environment parameters
    initial_capital: int = 100_000
    transaction_cost: float = 0.002
    max_position: float = 0.30
    max_turnover: float = 0.25
    constraint_penalty: float = 50.0
    
    # Reward function weights
    alpha_returns: float = 3.0
    alpha_mdd: float = 1.0
    alpha_vol: float = 0.5
    alpha_concentration: float = 0.3
    
    # General training parameters
    seed: int = 42
    
    # Super Agent (SAC) parameters
    super_learning_rate: float = 3e-4
    super_timesteps: int = 100_000
    super_buffer_size: int = 100_000
    super_batch_size: int = 256
    super_tau: float = 0.005
    super_gamma: float = 0.99
    super_ent_coef: float = 0.1
    super_eval_freq: int = 2000
    super_patience: int = 10
    super_min_delta: float = 0.01
    super_network: List[int] = field(default_factory=lambda: [256, 256, 128])
    
    # Meta Agent (PPO) parameters
    meta_learning_rate: float = 3e-4
    meta_timesteps: int = 100_000
    meta_n_steps: int = 2048
    meta_batch_size: int = 64
    meta_n_epochs: int = 10
    meta_gamma: float = 0.99
    meta_ent_coef: float = 0.01
    meta_max_grad_norm: float = 0.5
    meta_gae_lambda: float = 0.95
    meta_eval_freq: int = 2000
    meta_patience: int = 10
    meta_min_delta: float = 0.01
    meta_network: List[int] = field(default_factory=lambda: [128, 128, 64])
    
    # Base Agent parameters (for reference)
    base_timesteps: int = 100_000
    base_eval_freq: int = 2000
    base_patience: int = 10
    base_min_delta: float = 0.01
    
    @classmethod
    def from_json(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: str):
        """Save config to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'initial_capital': self.initial_capital,
            'transaction_cost': self.transaction_cost,
            'max_position': self.max_position,
            'max_turnover': self.max_turnover,
            'constraint_penalty': self.constraint_penalty,
            'alpha_returns': self.alpha_returns,
            'alpha_mdd': self.alpha_mdd,
            'alpha_vol': self.alpha_vol,
            'alpha_concentration': self.alpha_concentration,
            'seed': self.seed,
            'super_learning_rate': self.super_learning_rate,
            'super_timesteps': self.super_timesteps,
            'super_buffer_size': self.super_buffer_size,
            'super_batch_size': self.super_batch_size,
            'super_tau': self.super_tau,
            'super_gamma': self.super_gamma,
            'super_ent_coef': self.super_ent_coef,
            'super_eval_freq': self.super_eval_freq,
            'super_patience': self.super_patience,
            'super_min_delta': self.super_min_delta,
            'super_network': self.super_network,
            'meta_learning_rate': self.meta_learning_rate,
            'meta_timesteps': self.meta_timesteps,
            'meta_n_steps': self.meta_n_steps,
            'meta_batch_size': self.meta_batch_size,
            'meta_n_epochs': self.meta_n_epochs,
            'meta_gamma': self.meta_gamma,
            'meta_ent_coef': self.meta_ent_coef,
            'meta_max_grad_norm': self.meta_max_grad_norm,
            'meta_gae_lambda': self.meta_gae_lambda,
            'meta_eval_freq': self.meta_eval_freq,
            'meta_patience': self.meta_patience,
            'meta_min_delta': self.meta_min_delta,
            'meta_network': self.meta_network,
            'base_timesteps': self.base_timesteps,
            'base_eval_freq': self.base_eval_freq,
            'base_patience': self.base_patience,
            'base_min_delta': self.base_min_delta,
        }
    
    def to_training_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format expected by training functions"""
        return {
            'transaction_cost': self.transaction_cost,
            'max_position': self.max_position,
            'max_turnover': self.max_turnover,
            'constraint_penalty': self.constraint_penalty,
            'alpha_returns': self.alpha_returns,
            'alpha_mdd': self.alpha_mdd,
            'alpha_vol': self.alpha_vol,
            'alpha_concentration': self.alpha_concentration,
            'seed': self.seed,
            'super_learning_rate': self.super_learning_rate,
            'super_timesteps': self.super_timesteps,
            'super_buffer_size': self.super_buffer_size,
            'super_batch_size': self.super_batch_size,
            'super_tau': self.super_tau,
            'super_gamma': self.super_gamma,
            'super_ent_coef': self.super_ent_coef,
            'super_eval_freq': self.super_eval_freq,
            'super_patience': self.super_patience,
            'super_min_delta': self.super_min_delta,
            'super_network': self.super_network,
            'meta_learning_rate': self.meta_learning_rate,
            'meta_timesteps': self.meta_timesteps,
            'meta_n_steps': self.meta_n_steps,
            'meta_batch_size': self.meta_batch_size,
            'meta_n_epochs': self.meta_n_epochs,
            'meta_gamma': self.meta_gamma,
            'meta_ent_coef': self.meta_ent_coef,
            'meta_max_grad_norm': self.meta_max_grad_norm,
            'meta_gae_lambda': self.meta_gae_lambda,
            'meta_eval_freq': self.meta_eval_freq,
            'meta_patience': self.meta_patience,
            'meta_min_delta': self.meta_min_delta,
            'meta_network': self.meta_network,
        }


def get_default_config() -> TrainingConfig:
    """Get default configuration"""
    return TrainingConfig()


def load_config(path: Optional[str] = None) -> TrainingConfig:
    """Load configuration from file or return default"""
    if path and Path(path).exists():
        return TrainingConfig.from_json(path)
    return get_default_config()

