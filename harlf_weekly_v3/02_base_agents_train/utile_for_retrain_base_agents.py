"""
Train Base Agents for Super Agent
==================================
Trains 3 diverse base agents as ensembles for super agent architecture

Integrates with your existing train.py and config.py
"""

import numpy as np
import json
from pathlib import Path
from train import train
from config import get_config

# Ensure correct data directory path
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data_hierarchical'

# ============================================================================
# BASE AGENT CONFIGURATIONS
# ============================================================================

def get_base_agent_configs():
    """
    Get configurations for the 3 base agents
    
    Returns:
        dict: Dictionary with 3 base agent configs
    """
    
    configs = {
        # ====================================================================
        # BASE AGENT 1: RETURN MAXIMIZER (EMA Sharpe)
        # ====================================================================
        'agent_1_return_max': {
            'name': 'EMA_Sharpe_Technical_ReturnMax',
            'agent_type': 'technical',
            'algorithm': 'PPO',
            'reward_type': 'ema_sharpe',
            'purpose': 'Maximize returns - compromise params from best_choice',
            'config_overrides': {
                'gamma': 0.85,  # COMPROMISE: from best_choice (shorter horizon)
                'softmax_temperature': 1.0,  # COMPROMISE: down from 1.8, up from 0.9
                'rolling_vol_window': 12,  # COMPROMISE: from best_choice
                'transaction_cost': 0.001,  # COMPROMISE: from best_choice (lower costs)
                'total_steps': 500_000,
                'patience': 15,
                'learning_rate': 0.00026,  # COMPROMISE: from best_choice
            }
        },
        
        # ====================================================================
        # BASE AGENT 2: LOSS MINIMIZER (Multi-Objective)
        # ====================================================================
        'agent_2_loss_min': {
            'name': 'MultiObj_Technical_LossMin',
            'agent_type': 'technical',
            'algorithm': 'PPO',
            'reward_type': 'multi_objective',
            'purpose': 'Min losses - downside protection, still chases returns',
            'config_overrides': {
                'gamma': 0.87,  # ADJUSTED: slightly shorter horizon for responsiveness
                'softmax_temperature': 1.0,  # Balanced decisions
                'return_scale': 9.0,  # STILL AGGRESSIVE on returns
                'volatility_penalty': 0.08,  # Moderate risk control
                'concentration_penalty': 0.30,  # Allow concentrated winners
                'turnover_penalty': 0.006,  # Some trading flexibility
                'vol_window': 12,
                'max_concentration': 0.35,  # Can go 35% in best stock
                'transaction_cost': 0.001,  # ADJUSTED: lower costs like Agent 1
                'total_steps': 500_000,
                'patience': 15,
                'learning_rate': 0.00026,  # ADJUSTED: match Agent 1
            }
        },
        
        # ====================================================================
        # BASE AGENT 3: ALTERNATIVE SIGNAL (Sentiment EMA Sharpe)
        # ====================================================================
        'agent_3_alternative': {
            'name': 'EMA_Sharpe_Sentiment_Alternative',
            'agent_type': 'sentiment',
            'algorithm': 'PPO',
            'reward_type': 'ema_sharpe',
            'purpose': 'Alternative signal source, news-driven',
            'config_overrides': {
                'gamma': 0.88,  # ADJUSTED: slightly shorter for news responsiveness
                'softmax_temperature': 1.0,
                'rolling_vol_window': 12,
                'transaction_cost': 0.001,  # ADJUSTED: match other agents
                'total_steps': 500_000,
                'patience': 15,
                'learning_rate': 0.00026,  # ADJUSTED: match other agents
            }
        }
    }
    
    return configs


# ============================================================================
# TRAIN INDIVIDUAL BASE AGENT (WITH ENSEMBLE)
# ============================================================================

def train_base_agent(agent_config, n_seeds=10, verbose=True):
    """
    Train a single base agent as an ensemble
    
    Args:
        agent_config: Configuration dictionary for the agent
        n_seeds: Number of seeds to train (ensemble size)
        verbose: Print progress
    
    Returns:
        dict: Results including ensemble statistics
    """
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎯 Training Base Agent: {agent_config['name']}")
        print(f"{'='*70}")
        print(f"Agent Type:  {agent_config['agent_type']}")
        print(f"Algorithm:   {agent_config['algorithm']}")
        print(f"Reward:      {agent_config['reward_type']}")
        print(f"Purpose:     {agent_config['purpose']}")
        print(f"Ensemble:    {n_seeds} models")
        print(f"{'='*70}\n")
    
    # Get base config
    config = get_config(
        agent_config['reward_type'],
        agent_config['agent_type']
    )
    
    # Apply overrides
    config.update(agent_config['config_overrides'])
    
    # Use absolute path for data directory
    config['data_dir'] = str(DATA_DIR)
    
    # Display key configuration parameters
    if verbose:
        print(f"📋 Configuration Parameters:")
        print(f"  gamma:                {config.get('gamma', 'N/A')}")
        print(f"  softmax_temperature:  {config.get('softmax_temperature', 'N/A')}")
        print(f"  learning_rate:        {config.get('learning_rate', 'N/A')}")
        print(f"  total_steps:          {config.get('total_steps', 'N/A'):,}")
        print(f"  patience:             {config.get('patience', 'N/A')}")
        print(f"  transaction_cost:     {config.get('transaction_cost', 'N/A')}")
        
        # Reward-specific parameters
        if agent_config['reward_type'] == 'ema_sharpe':
            print(f"  rolling_vol_window:   {config.get('rolling_vol_window', 'N/A')}")
        elif agent_config['reward_type'] == 'multi_objective':
            print(f"  return_scale:         {config.get('return_scale', 'N/A')}")
            print(f"  volatility_penalty:   {config.get('volatility_penalty', 'N/A')}")
            print(f"  concentration_penalty: {config.get('concentration_penalty', 'N/A')}")
            print(f"  turnover_penalty:     {config.get('turnover_penalty', 'N/A')}")
            print(f"  max_concentration:    {config.get('max_concentration', 'N/A')}")
        
        print()
    
    # Train ensemble
    ensemble_results = []
    seeds = list(range(42, 42 + n_seeds))
    
    for i, seed in enumerate(seeds, 1):
        if verbose:
            print(f"\n[{i}/{n_seeds}] Training with seed {seed}...")
        
        # Update seed
        config['seed'] = seed
        
        # Train
        result = train(
            agent_type=agent_config['agent_type'],
            algorithm=agent_config['algorithm'],
            reward_type=agent_config['reward_type'],
            config=config,
            verbose=False
        )
        
        ensemble_results.append(result)
        
        if verbose:
            print(f"  Train: {result['train_sharpe']:.3f} | "
                  f"Val: {result['val_sharpe']:.3f} | "
                  f"Test: {result['test_sharpe']:.3f}")
    
    # Calculate ensemble statistics
    train_sharpes = [r['train_sharpe'] for r in ensemble_results]
    val_sharpes = [r['val_sharpe'] for r in ensemble_results]
    test_sharpes = [r['test_sharpe'] for r in ensemble_results]
    
    ensemble_stats = {
        'name': agent_config['name'],
        'agent_type': agent_config['agent_type'],
        'algorithm': agent_config['algorithm'],
        'reward_type': agent_config['reward_type'],
        'n_models': n_seeds,
        'seeds': seeds,
        
        # Individual results
        'individual_results': ensemble_results,
        
        # Ensemble statistics
        'ensemble_train_mean': float(np.mean(train_sharpes)),
        'ensemble_train_std': float(np.std(train_sharpes)),
        'ensemble_val_mean': float(np.mean(val_sharpes)),
        'ensemble_val_std': float(np.std(val_sharpes)),
        'ensemble_test_mean': float(np.mean(test_sharpes)),
        'ensemble_test_std': float(np.std(test_sharpes)),
        
        # Best/worst
        'best_val_sharpe': float(np.max(val_sharpes)),
        'worst_val_sharpe': float(np.min(val_sharpes)),
        'best_test_sharpe': float(np.max(test_sharpes)),
        'worst_test_sharpe': float(np.min(test_sharpes)),
        
        # Coefficient of variation
        'test_cv_percent': float(np.std(test_sharpes) / np.mean(test_sharpes) * 100),
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🏆 ENSEMBLE RESULTS: {agent_config['name']}")
        print(f"{'='*70}")
        print(f"\n📊 Test Set (Most Important):")
        print(f"  Mean:  {ensemble_stats['ensemble_test_mean']:.4f} ± {ensemble_stats['ensemble_test_std']:.4f}")
        print(f"  Range: [{ensemble_stats['worst_test_sharpe']:.3f}, {ensemble_stats['best_test_sharpe']:.3f}]")
        print(f"  CV:    {ensemble_stats['test_cv_percent']:.1f}%")
        
        # Stability assessment
        cv = ensemble_stats['test_cv_percent']
        if cv < 5:
            print(f"  ✅ EXCELLENT stability")
        elif cv < 10:
            print(f"  ✅ GOOD stability")
        elif cv < 15:
            print(f"  ⚠️  MODERATE stability")
        else:
            print(f"  ❌ POOR stability - consider more training")
        
        print(f"\n📊 Validation Set:")
        print(f"  Mean:  {ensemble_stats['ensemble_val_mean']:.4f} ± {ensemble_stats['ensemble_val_std']:.4f}")
        
        print(f"\n📊 Training Set:")
        print(f"  Mean:  {ensemble_stats['ensemble_train_mean']:.4f} ± {ensemble_stats['ensemble_train_std']:.4f}")
        
        print(f"{'='*70}\n")
    
    return ensemble_stats


# ============================================================================
# TRAIN ALL BASE AGENTS
# ============================================================================

def train_all_base_agents(n_seeds=10, save_path='results/base_agents.json', verbose=True):
    """
    Train all 3 base agents as ensembles
    
    Args:
        n_seeds: Number of seeds per agent (ensemble size)
        save_path: Where to save results
        verbose: Print progress
    
    Returns:
        dict: Results for all base agents
    """
    
    configs = get_base_agent_configs()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING ALL BASE AGENTS FOR SUPER AGENT")
        print(f"{'='*70}")
        print(f"Number of base agents: 3")
        print(f"Ensemble size per agent: {n_seeds}")
        print(f"Total models to train: {3 * n_seeds}")
        print(f"Estimated time: {3 * n_seeds * 4}-{3 * n_seeds * 6} minutes")
        print(f"{'='*70}\n")
    
    all_results = {}
    
    # Train each base agent
    for agent_key, agent_config in configs.items():
        result = train_base_agent(agent_config, n_seeds=n_seeds, verbose=verbose)
        all_results[agent_key] = result
    
    # Save results
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        if verbose:
            print(f"\n✅ All results saved to: {save_path}")
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 FINAL SUMMARY: BASE AGENTS FOR SUPER AGENT")
        print(f"{'='*70}\n")
        
        for agent_key, result in all_results.items():
            print(f"{result['name']}:")
            print(f"  Test Sharpe: {result['ensemble_test_mean']:.4f} ± {result['ensemble_test_std']:.4f}")
            print(f"  Stability:   CV = {result['test_cv_percent']:.1f}%")
            print(f"  Range:       [{result['worst_test_sharpe']:.3f}, {result['best_test_sharpe']:.3f}]")
            print()
        
        # Calculate diversity
        test_means = [r['ensemble_test_mean'] for r in all_results.values()]
        avg_sharpe = np.mean(test_means)
        spread = np.max(test_means) - np.min(test_means)
        
        print(f"Average Test Sharpe: {avg_sharpe:.4f}")
        print(f"Spread:              {spread:.4f} ({spread/avg_sharpe*100:.1f}% of mean)")
        print(f"\nℹ️  Good spread indicates diversity (want >10%)")
        
        print(f"\n{'='*70}")
        print(f"✅ BASE AGENTS READY FOR SUPER AGENT TRAINING!")
        print(f"{'='*70}\n")
    
    return all_results


# ============================================================================
# ANALYZE DIVERSITY
# ============================================================================

def analyze_diversity(base_agents_results, verbose=True):
    """
    Analyze diversity between base agents
    
    Args:
        base_agents_results: Results from train_all_base_agents()
        verbose: Print analysis
    
    Returns:
        dict: Diversity metrics
    """
    
    # Extract test sharpes
    agent_names = []
    test_sharpes = []
    
    for agent_key, result in base_agents_results.items():
        agent_names.append(result['name'])
        test_sharpes.append(result['ensemble_test_mean'])
    
    # Calculate diversity metrics
    diversity_metrics = {
        'agent_names': agent_names,
        'test_sharpes': test_sharpes,
        'mean_sharpe': float(np.mean(test_sharpes)),
        'std_sharpe': float(np.std(test_sharpes)),
        'min_sharpe': float(np.min(test_sharpes)),
        'max_sharpe': float(np.max(test_sharpes)),
        'spread': float(np.max(test_sharpes) - np.min(test_sharpes)),
        'spread_percent': float((np.max(test_sharpes) - np.min(test_sharpes)) / np.mean(test_sharpes) * 100)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔬 DIVERSITY ANALYSIS")
        print(f"{'='*70}\n")
        
        print(f"Base Agent Performance:")
        for name, sharpe in zip(agent_names, test_sharpes):
            print(f"  {name}: {sharpe:.4f}")
        
        print(f"\nDiversity Metrics:")
        print(f"  Mean:        {diversity_metrics['mean_sharpe']:.4f}")
        print(f"  Std:         {diversity_metrics['std_sharpe']:.4f}")
        print(f"  Range:       [{diversity_metrics['min_sharpe']:.3f}, {diversity_metrics['max_sharpe']:.3f}]")
        print(f"  Spread:      {diversity_metrics['spread']:.4f} ({diversity_metrics['spread_percent']:.1f}% of mean)")
        
        print(f"\n💡 Interpretation:")
        spread_pct = diversity_metrics['spread_percent']
        if spread_pct > 20:
            print(f"  ✅ EXCELLENT diversity (spread = {spread_pct:.1f}%)")
            print(f"     → Super agent has strong potential to improve")
        elif spread_pct > 10:
            print(f"  ✅ GOOD diversity (spread = {spread_pct:.1f}%)")
            print(f"     → Super agent should provide meaningful gains")
        elif spread_pct > 5:
            print(f"  ⚠️  MODERATE diversity (spread = {spread_pct:.1f}%)")
            print(f"     → Super agent may provide modest gains")
        else:
            print(f"  ❌ LOW diversity (spread = {spread_pct:.1f}%)")
            print(f"     → Consider using more different reward functions")
        
        print(f"\n{'='*70}\n")
    
    return diversity_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train base agents for super agent')
    parser.add_argument('--n-seeds', type=int, default=5,
                       help='Number of seeds per agent (default: 5)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 2 seeds and 100k steps')
    parser.add_argument('--save-path', type=str, default='results/base_agents.json',
                       help='Where to save results')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        print("\n⚡ QUICK TEST MODE")
        print("  - 2 seeds per agent (instead of 5)")
        print("  - 100k steps (instead of 500k)")
        print("  - For testing only!\n")
        
        # Modify configs for quick test
        configs = get_base_agent_configs()
        for config in configs.values():
            config['config_overrides']['total_steps'] = 100_000
        
        # Train with 2 seeds
        results = train_all_base_agents(n_seeds=2, save_path=args.save_path, verbose=True)
    else:
        # Full training
        results = train_all_base_agents(n_seeds=args.n_seeds, save_path=args.save_path, verbose=True)
    
    # Analyze diversity
    diversity = analyze_diversity(results, verbose=True)
    
    print("\n🎯 Next Steps:")
    print("  1. Check results in:", args.save_path)
    print("  2. Train super agent: python train_super_agent.py")
    print("  3. Evaluate: python evaluate_super_agent.py")
