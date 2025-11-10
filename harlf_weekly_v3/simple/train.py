"""
Training Functions
==================
Simplified: Removed SAC/A2C support, hyperparameter sweep, quick test.
"""

from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th

from config import get_config
from environments import create_env, ValidationCallback
from utils import print_results, compare_results, save_results


def train(agent_type, algorithm, reward_type, config=None, verbose=True):
    if algorithm != 'PPO':
        raise ValueError("Only PPO supported in simplified version.")

    if config is None:
        config = get_config(reward_type, agent_type)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {agent_type.upper()} - {reward_type.upper()} - {algorithm}")
        print(f"{'='*70}")
        print(f"Gamma: {config['gamma']}")
        print(f"Softmax temp: {config['softmax_temperature']}")
        print(f"Patience: {config['patience']}")

    reward_kwargs = {}
    if reward_type == 'ema_sharpe':
        reward_kwargs = {'rolling_vol_window': config.get('rolling_vol_window', 12)}
    elif reward_type == 'multi_objective':
        reward_kwargs = {
            'return_scale': config.get('return_scale', 8.0),
            'volatility_penalty': config.get('volatility_penalty', 0.05),
            'concentration_penalty': config.get('concentration_penalty', 0.25),
            'turnover_penalty': config.get('turnover_penalty', 0.005),
            'vol_window': config.get('vol_window', 12),
            'max_concentration': config.get('max_concentration', 0.35)
        }

    train_env = create_env(
        config['data_dir'],
        agent_type,
        'train',
        reward_type,
        reward_kwargs,
        config['softmax_temperature'],
        random_start=True,
        transaction_cost=config.get('transaction_cost', 0.0)
    )
    val_env = create_env(
        config['data_dir'],
        agent_type,
        'val',
        reward_type,
        reward_kwargs,
        config['softmax_temperature'],
        random_start=False,
        transaction_cost=config.get('transaction_cost', 0.0)
    )
    test_env = create_env(
        config['data_dir'],
        agent_type,
        'val',
        reward_type,
        reward_kwargs,
        config['softmax_temperature'],
        random_start=False,
        transaction_cost=config.get('transaction_cost', 0.0)
    )

    train_env = DummyVecEnv([lambda: train_env])
    model = PPO(
        'MlpPolicy',
        train_env,
        learning_rate=config['learning_rate'],
        gamma=config['gamma'],
        seed=config['seed'],
        device='cuda' if th.cuda.is_available() else 'cpu'
    )

    save_path = Path(f"{config['models_dir']}/{agent_type}{reward_type}{algorithm}.zip")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    callback = ValidationCallback(
        val_env=val_env,
        eval_freq=config['eval_freq'],
        patience=config['patience'],
        save_path=str(save_path),
        verbose=verbose
    )

    model.learn(
        total_timesteps=config['total_steps'],
        callback=callback
    )

    train_sharpe = train_env.envs[0].run_full_pass(model)
    val_sharpe = val_env.run_full_pass(model)
    test_sharpe = test_env.run_full_pass(model)

    if verbose:
        print_results({
            'train_sharpe': train_sharpe,
            'val_sharpe': val_sharpe,
            'test_sharpe': test_sharpe
        })

    return {
        'model_path': str(save_path),
        'train_sharpe': train_sharpe,
        'val_sharpe': val_sharpe,
        'test_sharpe': test_sharpe,
        'gamma': config['gamma'],
        'softmax_temperature': config['softmax_temperature'],
        'transaction_cost': config['transaction_cost']
    }

if __name__ == '__main__':
    # Example usage
    result = train('technical', 'PPO', 'ema_sharpe')

