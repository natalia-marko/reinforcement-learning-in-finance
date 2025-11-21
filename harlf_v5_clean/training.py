"""
Training functions for Super and Meta agents

This module contains all training functions for the hierarchical RL system.
"""

import numpy as np
import pandas as pd
from utile import EarlyStoppingCallback
from stable_baselines3 import PPO, SAC
from production_agent_wrapper_frozen import ProductionAgentWrapper
from model_loader import load_base_models
from data_validation import validate_training_data
from environments import SuperAgentEnv, MetaAgentEnv


# ============================================================================
# SUPER AGENT TRAINING
# ============================================================================

def train_super_agent_sac(train_data, val_data, config):
    """
    Train Super agent with SAC + Enhanced observations
    SIMPLIFIED: Single training run (no windows, no ensemble)
    
    Args:
        train_data: Tuple of (prices, technical_features, sentiment_features, regime)
        val_data: Tuple of validation data
        config: Dictionary containing training hyperparameters
        
    Returns:
        tuple: (trained_model, validation_sharpe_ratio)
        
    Raises:
        FileNotFoundError: If required base model files are missing
        ValueError: If data validation fails
        RuntimeError: If training fails
    """
    try:
        # Validate inputs
        validate_training_data(train_data, val_data)
        
        print("\n" + "="*70)
        print("TRAINING ENHANCED SUPER AGENT (SAC + FULL FEATURES)")
        print("="*70)
        
        train_prices, train_tech, train_sent, train_regime = train_data
        val_prices, val_tech, val_sent, val_regime = val_data
        
        n_assets = len(train_prices.columns)
        
        # Load base agents
        print("\n   Loading base agents...")
        base_models = load_base_models('models')
        
        base_agents_train = {
            'tech_PPO': ProductionAgentWrapper(base_models['tech_PPO'], train_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(base_models['tech_SAC'], train_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(base_models['sent_PPO'], train_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(base_models['sent_SAC'], train_sent, n_assets, 'sent_SAC'),
        }
        
        base_agents_val = {
            'tech_PPO': ProductionAgentWrapper(base_models['tech_PPO'], val_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(base_models['tech_SAC'], val_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(base_models['sent_PPO'], val_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(base_models['sent_SAC'], val_sent, n_assets, 'sent_SAC'),
        }
        
        # Create ENHANCED environments (with full features)
        print("\n   Creating environments...")
        super_train_env = SuperAgentEnv(
            train_prices, base_agents_train, train_tech, train_sent, **config
        )
        super_val_env = SuperAgentEnv(
            val_prices, base_agents_val, val_tech, val_sent, **config
        )
        
        # Train with SAC
        print(f"\n   Training with SAC algorithm...")
        model = SAC(
            'MlpPolicy',
            super_train_env,
            learning_rate=config['super_learning_rate'],
            buffer_size=config['super_buffer_size'],
            batch_size=config['super_batch_size'],
            tau=config['super_tau'],
            gamma=config['super_gamma'],
            ent_coef=config['super_ent_coef'],
            verbose=0,
            seed=config['seed'],
            policy_kwargs=dict(net_arch=config['super_network'])
        )
        
        callback = EarlyStoppingCallback(
            val_env=super_val_env,
            eval_freq=config['super_eval_freq'],
            patience=config['super_patience'],
            min_delta=config['super_min_delta'],
            min_steps=10000,
            save_path='models/super_agent_sac',
            verbose=1
        )
        
        print(f"\n   Training for up to {config['super_timesteps']} steps...")
        model.learn(total_timesteps=config['super_timesteps'], callback=callback)
        
        if callback.best_model_path:
            model = SAC.load(callback.best_model_path)
            print(f"\n   ✓ Loaded best model (Val Sharpe: {callback.best_val_sharpe:.3f})")
        
        return model, callback.best_val_sharpe
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}") from e
    except ValueError as e:
        raise ValueError(f"Data validation error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e


# ============================================================================
# META AGENT TRAINING
# ============================================================================

def train_meta_agent(train_data, val_data, super_model, config):
    """
    Train Meta agent
    SIMPLIFIED: Single training run (no windows, no ensemble)
    
    Args:
        train_data: Tuple of (prices, technical_features, sentiment_features, regime)
        val_data: Tuple of validation data
        super_model: Pre-trained Super agent model
        config: Dictionary containing training hyperparameters
        
    Returns:
        tuple: (trained_model, validation_sharpe_ratio)
        
    Raises:
        FileNotFoundError: If required base model files are missing
        ValueError: If data validation fails
        RuntimeError: If training fails
    """
    try:
        # Validate inputs
        validate_training_data(train_data, val_data)
        
        print("\n" + "="*70)
        print("TRAINING META AGENT")
        print("="*70)
        
        train_prices, train_tech, train_sent, train_regime = train_data
        val_prices, val_tech, val_sent, val_regime = val_data
        
        n_assets = len(train_prices.columns)
        
        # Load base agents
        print("\n   Loading base agents...")
        base_models = load_base_models('models')
        
        base_agents_train = {
            'tech_PPO': ProductionAgentWrapper(base_models['tech_PPO'], train_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(base_models['tech_SAC'], train_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(base_models['sent_PPO'], train_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(base_models['sent_SAC'], train_sent, n_assets, 'sent_SAC'),
        }
        
        base_agents_val = {
            'tech_PPO': ProductionAgentWrapper(base_models['tech_PPO'], val_tech, n_assets, 'tech_PPO'),
            'tech_SAC': ProductionAgentWrapper(base_models['tech_SAC'], val_tech, n_assets, 'tech_SAC'),
            'sent_PPO': ProductionAgentWrapper(base_models['sent_PPO'], val_sent, n_assets, 'sent_PPO'),
            'sent_SAC': ProductionAgentWrapper(base_models['sent_SAC'], val_sent, n_assets, 'sent_SAC'),
        }
        
        # Create Super environments
        print("\n   Creating Super agent environments...")
        super_train_env = SuperAgentEnv(
            train_prices, base_agents_train, train_tech, train_sent, **config
        )
        super_val_env = SuperAgentEnv(
            val_prices, base_agents_val, val_tech, val_sent, **config
        )
        
        # Wrap Super model
        super_train_wrap = ProductionAgentWrapper(
            super_model, super_train_env, n_assets, 'super', freeze_weights=True
        )
        super_val_wrap = ProductionAgentWrapper(
            super_model, super_val_env, n_assets, 'super', freeze_weights=True
        )
        
        # Create Meta environments
        print("   Creating Meta agent environments...")
        meta_train_env = MetaAgentEnv(train_prices, super_train_wrap, train_regime, **config)
        meta_val_env = MetaAgentEnv(val_prices, super_val_wrap, val_regime, **config)
        
        # Train Meta with PPO
        print(f"\n   Training with PPO algorithm...")
        model = PPO(
            'MlpPolicy',
            meta_train_env,
            learning_rate=config['meta_learning_rate'],
            n_steps=config['meta_n_steps'],
            batch_size=config['meta_batch_size'],
            n_epochs=config['meta_n_epochs'],
            gamma=config['meta_gamma'],
            ent_coef=config['meta_ent_coef'],
            max_grad_norm=config['meta_max_grad_norm'],
            gae_lambda=config['meta_gae_lambda'],
            verbose=0,
            seed=config['seed'],
            policy_kwargs=dict(net_arch=[dict(pi=config['meta_network'], 
                                              vf=config['meta_network'])])
        )
        
        callback = EarlyStoppingCallback(
            val_env=meta_val_env,
            eval_freq=config['meta_eval_freq'],
            patience=config['meta_patience'],
            min_delta=config['meta_min_delta'],
            min_steps=10000,
            save_path='models/meta_agent',
            verbose=1
        )
        
        print(f"\n   Training for up to {config['meta_timesteps']} steps...")
        model.learn(total_timesteps=config['meta_timesteps'], callback=callback)
        
        if callback.best_model_path:
            model = PPO.load(callback.best_model_path, env=meta_train_env)
            print(f"\n   ✓ Loaded best model (Val Sharpe: {callback.best_val_sharpe:.3f})")
        
        return model, callback.best_val_sharpe
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found: {e}") from e
    except ValueError as e:
        raise ValueError(f"Data validation error: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}") from e

