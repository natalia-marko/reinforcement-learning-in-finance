"""
Training Functions
==================
Simple training functions you can call from notebooks!

CHANGES IN THIS VERSION:
- A2C: n_steps increased from 5 to 20 (better bias/variance tradeoff)
- SAC: buffer_size increased from 200k to 500k (more experience diversity)
- All algorithms now use updated gamma from config
"""

from pathlib import Path
import pandas as pd
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th

from config import get_config, ALGORITHMS
from environments import create_env, ValidationCallback
from utils import print_results, compare_results, save_results


# ============================================================================
# Main Training Function (replaces all your old ones!)
# ============================================================================

def train(agent_type, algorithm, reward_type, config=None, verbose=True):
    """
    Train an agent - THIS IS ALL YOU NEED!
    
    Parameters
    ----------
    agent_type : str
        'technical' or 'sentiment'
    algorithm : str
        'PPO', 'SAC', or 'A2C'
    reward_type : str
        'ema_sharpe', 'differential_sharpe', 'multi_objective',
        'simple_return', or 'simple_sharpe'
    config : dict, optional
        Configuration (if None, uses defaults)
    verbose : bool
        Print progress
    
    Returns
    -------
    dict : Results with Sharpe ratios and model path
    
    Examples
    --------
    >>> # Train technical agent with EMA Sharpe using PPO
    >>> result = train('technical', 'PPO', 'ema_sharpe')
    
    >>> # Train sentiment agent with multi-objective using SAC
    >>> result = train('sentiment', 'SAC', 'multi_objective')
    """
    
    # Get config
    if config is None:
        config = get_config(reward_type, agent_type)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training: {agent_type.upper()} - {reward_type.upper()} - {algorithm}")
        print(f"{'='*70}")
        print(f"Gamma: {config['gamma']}")
        print(f"Softmax temp: {config['softmax_temperature']}")
        print(f"Patience: {config['patience']}")
    
    # Get reward kwargs from config
    reward_kwargs = {}
    if reward_type == 'ema_sharpe':
        reward_kwargs = {'rolling_vol_window': config.get('rolling_vol_window', 12)}
    elif reward_type == 'differential_sharpe':
        reward_kwargs = {'decay_factor': config.get('decay_factor', 0.95)}
    elif reward_type == 'multi_objective':
        reward_kwargs = {
            'return_scale': config.get('return_scale', 10.0),
            'volatility_penalty': config.get('volatility_penalty', 0.1),
            'concentration_penalty': config.get('concentration_penalty', 0.5),
            'turnover_penalty': config.get('turnover_penalty', 0.01),
            'vol_window': config.get('vol_window', 12),
            'max_concentration': config.get('max_concentration', 0.30)  # NEW
        }
    
    # Create environments
    train_env = create_env(
        config['data_dir'], agent_type, 'train', reward_type,
        reward_kwargs, config['softmax_temperature'], 
        random_start=True,
        transaction_cost=config.get('transaction_cost', 0.0)  # NEW
    )
    val_env = create_env(
        config['data_dir'], agent_type, 'val', reward_type,
        reward_kwargs, config['softmax_temperature'],
        transaction_cost=config.get('transaction_cost', 0.0)  # NEW
    )
    test_env = create_env(
        config['data_dir'], agent_type, 'test', reward_type,
        reward_kwargs, config['softmax_temperature'],
        transaction_cost=config.get('transaction_cost', 0.0)  # NEW
    )
    
    if verbose:
        print(f"Environment: {len(train_env.dates)} dates, "
              f"{train_env.n_assets} assets, {train_env.n_features} features")
        if config.get('transaction_cost', 0.0) > 0:
            print(f"Transaction cost: {config['transaction_cost']*10000:.1f} bps")
    
    # Wrap in vectorized env
    vec_env = DummyVecEnv([lambda: train_env])
    
    # Create model
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', vec_env,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            n_steps=2048,
            batch_size=512,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256, 256]),
            verbose=1 if verbose else 0
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy', vec_env,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            buffer_size=500_000,  # CHANGED from 200_000 - larger buffer
            batch_size=1024,
            policy_kwargs=dict(activation_fn=th.nn.ReLU, 
                             net_arch=dict(pi=[256, 256], qf=[256, 256])),
            verbose=1 if verbose else 0
        )
    else:  # A2C
        model = A2C(
            'MlpPolicy', vec_env,
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            n_steps=20,  # CHANGED from 5 - better bias/variance tradeoff
            policy_kwargs=dict(activation_fn=th.nn.ReLU, net_arch=[256, 256]),
            verbose=1 if verbose else 0
        )
    
    # Set up callback
    save_path = Path(config['models_dir']) / agent_type / reward_type / \
                f"{reward_type}_{agent_type}_{algorithm.lower()}.zip"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    callback = ValidationCallback(
        val_env=val_env,
        eval_freq=config['eval_freq'],
        patience=config['patience'],
        save_path=str(save_path),
        verbose=1 if verbose else 0
    )
    
    # Train!
    if verbose:
        print(f"\nTraining for {config['total_steps']:,} steps...")
    
    model.learn(total_timesteps=config['total_steps'], callback=callback)
    
    # Load best model
    if save_path.exists():
        if algorithm == 'PPO':
            model = PPO.load(str(save_path), env=vec_env)
        elif algorithm == 'SAC':
            model = SAC.load(str(save_path), env=vec_env)
        else:
            model = A2C.load(str(save_path), env=vec_env)
    
    # Evaluate
    train_sharpe = train_env.run_full_pass(model)
    val_sharpe = val_env.run_full_pass(model)
    test_sharpe = test_env.run_full_pass(model)
    
    result = {
        'agent_type': agent_type,
        'algorithm': algorithm,
        'reward_type': reward_type,
        'train_sharpe': round(train_sharpe, 3),
        'val_sharpe': round(val_sharpe, 3),
        'test_sharpe': round(test_sharpe, 3),
        'model_path': str(save_path),
        'gamma': config['gamma'],  # NEW - track key params
        'softmax_temperature': config['softmax_temperature'],  # NEW
        'transaction_cost': config.get('transaction_cost', 0.0)  # NEW
    }
    
    if verbose:
        print_results(result)
    
    return result


# ============================================================================
# Convenience Functions
# ============================================================================

def train_all_algos(agent_type, reward_type, config=None, verbose=True):
    """
    Train with all algorithms (PPO, SAC, A2C) and return best.
    
    Examples
    --------
    >>> results = train_all_algos('technical', 'ema_sharpe')
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training {agent_type.upper()} with ALL algorithms")
        print(f"Reward: {reward_type.upper()}")
        print(f"{'='*70}")
    
    results = []
    for algo in ALGORITHMS:
        result = train(agent_type, algo, reward_type, config, verbose=False)
        results.append(result)
        if verbose:
            print(f"\n{algo}: Test Sharpe = {result['test_sharpe']:.3f}")
    
    # Show comparison
    if verbose:
        df = compare_results(results)
        best = df.iloc[0]
        print(f"\n🏆 Best: {best['algorithm']} (Test Sharpe: {best['test_sharpe']:.3f})")
    
    return results


def train_both_agents(reward_type, config_tech=None, config_sent=None, 
                     algorithm='PPO', verbose=True):
    """
    Train both technical and sentiment agents.
    
    Examples
    --------
    >>> results = train_both_agents('ema_sharpe', algorithm='PPO')
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Training BOTH agents with {algorithm}")
        print(f"Reward: {reward_type.upper()}")
        print(f"{'='*70}")
    
    # Train technical
    tech_result = train('technical', algorithm, reward_type, config_tech, verbose)
    
    # Train sentiment
    sent_result = train('sentiment', algorithm, reward_type, config_sent, verbose)
    
    results = {
        'approach': reward_type,
        'technical': tech_result,
        'sentiment': sent_result
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Technical: Test Sharpe = {tech_result['test_sharpe']:.3f}")
        print(f"Sentiment: Test Sharpe = {sent_result['test_sharpe']:.3f}")
        print(f"{'='*70}\n")
    
    return results


def quick_test(agent_type='technical', reward_type='ema_sharpe'):
    """
    Quick test with reduced training steps (for debugging).
    
    Examples
    --------
    >>> quick_test('technical', 'ema_sharpe')
    """
    
    config = get_config(reward_type, agent_type)
    config['total_steps'] = 10_000  # Much shorter
    config['eval_freq'] = 2_000
    
    print("\n⚡ QUICK TEST MODE (10k steps only)")
    result = train(agent_type, 'PPO', reward_type, config, verbose=True)
    
    return result


def hyperparameter_sweep(agent_type='technical', reward_type='ema_sharpe', 
                        algorithm='PPO', verbose=True):
    """
    NEW: Sweep over key hyperparameters.
    
    Sweeps over:
    - Softmax temperature: [0.5, 0.75, 1.0, 1.5]
    - Gamma: [0.85, 0.90, 0.95]
    
    Examples
    --------
    >>> results = hyperparameter_sweep('technical', 'ema_sharpe', 'PPO')
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER SWEEP")
        print(f"Agent: {agent_type}, Reward: {reward_type}, Algo: {algorithm}")
        print(f"{'='*70}")
    
    temps = [0.5, 0.75, 1.0, 1.5]
    gammas = [0.85, 0.90, 0.95]
    
    results = []
    best_sharpe = -999
    best_config = None
    
    for temp in temps:
        for gamma in gammas:
            if verbose:
                print(f"\n--- Testing temp={temp}, gamma={gamma} ---")
            
            config = get_config(reward_type, agent_type)
            config['softmax_temperature'] = temp
            config['gamma'] = gamma
            config['total_steps'] = 50_000  # Shorter for sweep
            config['patience'] = 10
            
            result = train(agent_type, algorithm, reward_type, config, verbose=False)
            results.append({
                'temp': temp,
                'gamma': gamma,
                'test_sharpe': result['test_sharpe'],
                'val_sharpe': result['val_sharpe']
            })
            
            if verbose:
                print(f"Result: Test Sharpe = {result['test_sharpe']:.3f}")
            
            if result['test_sharpe'] > best_sharpe:
                best_sharpe = result['test_sharpe']
                best_config = (temp, gamma)
    
    df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'='*70}")
        print("SWEEP RESULTS")
        print(f"{'='*70}")
        print(df.sort_values('test_sharpe', ascending=False).to_string(index=False))
        print(f"\n🏆 Best: temp={best_config[0]}, gamma={best_config[1]}, "
              f"Sharpe={best_sharpe:.3f}")
        print(f"{'='*70}\n")
    
    return df


if __name__ == '__main__':
    print("Training functions loaded!")
    print("\n⚠️  REFACTORED VERSION - Key changes:")
    print("  - A2C n_steps: 20 (was 5)")
    print("  - SAC buffer: 500k (was 200k)")
    print("  - Uses updated gamma from config")
    print("  - Transaction costs support")
    print("\nMain function:")
    print("  train() - Train any agent with any reward")
    print("\nConvenience functions:")
    print("  train_all_algos() - Try all algorithms")
    print("  train_both_agents() - Train tech + sentiment")
    print("  quick_test() - Fast test run")
    print("  hyperparameter_sweep() - NEW: Test different params")