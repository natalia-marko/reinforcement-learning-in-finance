#!/usr/bin/env python3
"""
Train 3 Base Agents - PRODUCTION VERSION
=========================================
Full training with 300k steps per agent (~1-2 hours total)
"""

from pathlib import Path
import json
from train import train
from config import get_config

# HARDCODED PATHS
BASE_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3'
MODELS_DIR = f'{BASE_DIR}/models'

print("="*70)
print("TRAINING 3 BASE AGENTS - PRODUCTION")
print("="*70)
print("⏱️  Estimated time: 1-2 hours")
print("="*70)

# Define 3 base agents
agents = [
    {
        'name': 'agent_1_return_max',
        'agent_type': 'technical',
        'algorithm': 'PPO',
        'reward_type': 'ema_sharpe',
        'description': 'Technical agent - EMA Sharpe (return maximizer)'
    },
    {
        'name': 'agent_2_loss_min',
        'agent_type': 'technical',
        'algorithm': 'PPO',
        'reward_type': 'multi_objective',
        'description': 'Technical agent - Multi-objective (loss minimizer)'
    },
    {
        'name': 'agent_3_alternative',
        'agent_type': 'sentiment',
        'algorithm': 'PPO',
        'reward_type': 'ema_sharpe',
        'description': 'Sentiment agent - EMA Sharpe (alternative)'
    }
]

results = {}

# Train each agent
for i, agent in enumerate(agents, 1):
    print(f"\n{'='*70}")
    print(f"AGENT {i}/3: {agent['name']}")
    print(f"Type: {agent['agent_type']} | Reward: {agent['reward_type']}")
    print(f"{agent['description']}")
    print(f"{'='*70}\n")
    
    # Get config with FULL training settings
    config = get_config(agent['reward_type'], agent['agent_type'])
    # These are already 300k in config.py, but we'll be explicit
    config['total_steps'] = 300_000  # Full training!
    config['eval_freq'] = 5_000
    config['patience'] = 15
    
    result = train(
        agent_type=agent['agent_type'],
        algorithm=agent['algorithm'],
        reward_type=agent['reward_type'],
        config=config,
        verbose=True
    )
    
    results[agent['name']] = {
        'name': agent['name'],
        'agent_type': agent['agent_type'],
        'algorithm': agent['algorithm'],
        'reward_type': agent['reward_type'],
        'model_path': result['model_path'],
        'train_sharpe': result['train_sharpe'],
        'val_sharpe': result['val_sharpe'],
        'test_sharpe': result['test_sharpe'],
        'gamma': result.get('gamma', 0.90),
        'softmax_temperature': result.get('softmax_temperature', 1.0),
        'transaction_cost': result.get('transaction_cost', 0.0)
    }
    
    print(f"\n✓ Agent {i} complete!")
    print(f"  Train Sharpe: {result['train_sharpe']:.3f}")
    print(f"  Val Sharpe:   {result['val_sharpe']:.3f}")
    print(f"  Test Sharpe:  {result['test_sharpe']:.3f}")

# Save results
print(f"\n{'='*70}")
print("ALL AGENTS TRAINED!")
print(f"{'='*70}\n")

summary_file = Path(f'{MODELS_DIR}/base_agents_production.json')
summary_file.parent.mkdir(parents=True, exist_ok=True)

with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {summary_file}")

# Print summary
print("\nSUMMARY:")
print("-" * 70)
for agent_name, data in results.items():
    print(f"\n{data['name']}:")
    print(f"  Type: {data['agent_type']:10s} | Reward: {data['reward_type']:15s}")
    print(f"  Test Sharpe: {data['test_sharpe']:.3f}")
    print(f"  Model: {data['model_path']}")

print("\n" + "="*70)
print("Next: Run train_super_agent_PRODUCTION.py")
print("="*70)
