#!/usr/bin/env python3
"""
Test Fixed Weight Combinations
================================
Test different combinations of the 3 base agents without meta-learning.
This is simpler and often performs just as well.
"""

from pathlib import Path
import json
import numpy as np
from super_agent import BaseAgentWrapper
from environments import create_env
from config import get_config
from utils import compute_sharpe

# HARDCODED PATHS
BASE_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3'
DATA_DIR = f'{BASE_DIR}/data_hierarchical'
MODELS_DIR = f'{BASE_DIR}/models'

print("="*70)
print("TESTING FIXED WEIGHT COMBINATIONS")
print("="*70)

# Load base agents
base_agents_file = Path(f'{MODELS_DIR}/base_agents_production.json')
if not base_agents_file.exists():
    print("\n❌ ERROR: Base agents not found!")
    print("Please run train_3_base_agents_PRODUCTION.py first.")
    exit(1)

print("\n1. Loading base agents...")
with open(base_agents_file, 'r') as f:
    base_agents_info = json.load(f)

# Create base agent wrappers for TEST set
print("\n2. Creating base agent wrappers...")
base_agents = []

for agent_key, agent_data in base_agents_info.items():
    config = {
        'gamma': agent_data.get('gamma', 0.90),
        'softmax_temperature': agent_data.get('softmax_temperature', 1.0),
        'transaction_cost': agent_data.get('transaction_cost', 0.0),
    }
    
    if agent_data['reward_type'] == 'ema_sharpe':
        config['rolling_vol_window'] = 12
    elif agent_data['reward_type'] == 'multi_objective':
        config.update({
            'return_scale': 9.0,
            'volatility_penalty': 0.08,
            'concentration_penalty': 0.30,
            'turnover_penalty': 0.006,
            'vol_window': 12,
            'max_concentration': 0.35
        })
    
    agent = BaseAgentWrapper(
        model_paths=[agent_data['model_path']],
        agent_name=agent_data['name'],
        agent_type=agent_data['agent_type'],
        reward_type=agent_data['reward_type'],
        data_dir=DATA_DIR,
        split='test',  # Use TEST split
        algorithm=agent_data['algorithm'],
        config=config
    )
    
    base_agents.append(agent)

print(f"   ✓ Loaded {len(base_agents)} agents\n")

# Align date counts
date_counts = [len(agent.env.dates) for agent in base_agents]
min_dates = min(date_counts)

if len(set(date_counts)) > 1:
    print(f"   ⚠️  Aligning to {min_dates} dates")
    for agent in base_agents:
        if len(agent.env.dates) > min_dates:
            agent.env.dates = agent.env.dates[:min_dates]

# Create test environment
print("3. Creating test environment...")
main_config = get_config('ema_sharpe', 'technical')

test_env = create_env(
    data_dir=DATA_DIR,
    agent_type='technical',
    split='test',
    reward_type='ema_sharpe',
    reward_kwargs={'rolling_vol_window': 12},
    softmax_temperature=main_config['softmax_temperature'],
    transaction_cost=main_config.get('transaction_cost', 0.0)
)

if len(test_env.dates) > min_dates:
    test_env.dates = test_env.dates[:min_dates]

print(f"   ✓ Test environment: {len(test_env.dates)} dates\n")

# Test individual agents first (baselines)
print("="*70)
print("BASELINE: INDIVIDUAL AGENT PERFORMANCE")
print("="*70)

agent_names = [agent.agent_name for agent in base_agents]
baselines = {}

for i, agent in enumerate(base_agents):
    state, _ = test_env.reset()
    returns = []
    done = False
    
    agent.env._t = test_env._t
    
    while not done:
        current_t = test_env._t
        
        if current_t >= len(agent.env.dates):
            break
        
        try:
            agent_state = agent.get_state_for_timestep(current_t)
            action = agent.predict(agent_state, deterministic=True)
            
            state, _, done, truncated, info = test_env.step(action)
            done = done or truncated
            
            if 'port_log_r' in info:
                returns.append(info['port_log_r'])
        except (ValueError, IndexError):
            break
    
    sharpe = compute_sharpe(returns) if len(returns) >= 2 else 0.0
    baselines[agent_names[i]] = sharpe
    print(f"{agent_names[i]:45s}: {sharpe:.3f}")

avg_baseline = np.mean(list(baselines.values()))
best_baseline = max(baselines.values())
best_agent = max(baselines, key=baselines.get)

print(f"\nAverage: {avg_baseline:.3f}")
print(f"Best:    {best_baseline:.3f} ({best_agent})")

# Test weight combinations
print("\n" + "="*70)
print("TESTING WEIGHT COMBINATIONS")
print("="*70)

weights_to_test = [
    ([1.0, 0.0, 0.0], "Agent 1 only"),
    ([0.0, 1.0, 0.0], "Agent 2 only"),
    ([0.0, 0.0, 1.0], "Agent 3 only"),
    ([0.333, 0.333, 0.333], "Equal weights"),
    ([0.5, 0.3, 0.2], "Favor best"),
    ([0.6, 0.3, 0.1], "Strongly favor best"),
    ([0.4, 0.4, 0.2], "Top 2 equal"),
    ([0.7, 0.2, 0.1], "Very aggressive"),
]

results = []

for weights, description in weights_to_test:
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    # Reset
    state, _ = test_env.reset()
    returns = []
    done = False
    
    # Sync agents
    for agent in base_agents:
        agent.env._t = test_env._t
    
    while not done:
        current_t = test_env._t
        
        # Check bounds
        valid = True
        for agent in base_agents:
            if current_t >= len(agent.env.dates):
                valid = False
                done = True
                break
        
        if not valid:
            break
        
        # Get predictions
        base_actions = []
        for agent in base_agents:
            try:
                agent_state = agent.get_state_for_timestep(current_t)
                action = agent.predict(agent_state, deterministic=True)
                base_actions.append(action)
            except (ValueError, IndexError):
                done = True
                break
        
        if len(base_actions) != len(base_agents):
            break
        
        base_actions = np.array(base_actions)
        
        # Combine using weights
        combined_action = np.sum(weights[:, None] * base_actions, axis=0)
        combined_action = combined_action / (combined_action.sum() + 1e-12)
        
        # Execute
        state, _, done, truncated, info = test_env.step(combined_action)
        done = done or truncated
        
        if 'port_log_r' in info:
            returns.append(info['port_log_r'])
    
    test_sharpe = compute_sharpe(returns) if len(returns) >= 2 else 0.0
    improvement = (test_sharpe - best_baseline) / best_baseline * 100
    
    results.append({
        'weights': weights.tolist(),
        'description': description,
        'test_sharpe': float(test_sharpe),
        'improvement': float(improvement)
    })
    
    weights_str = np.array2string(weights, precision=2, suppress_small=True)
    
    print(f"\n{description}")
    print(f"  Weights: {weights_str}")
    print(f"  Test Sharpe: {test_sharpe:.3f}")
    print(f"  vs Best:     {improvement:+.1f}%")
    
    if test_sharpe > best_baseline:
        print(f"  ✓ BEATS BASELINE!")
    elif test_sharpe > best_baseline * 0.98:
        print(f"  ~ Close to baseline")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

best_result = max(results, key=lambda x: x['test_sharpe'])

print(f"\nBest baseline (single agent): {best_baseline:.3f} ({best_agent})")
print(f"\nBest combination:")
print(f"  Weights:     {np.array(best_result['weights'])}")
print(f"  Description: {best_result['description']}")
print(f"  Test Sharpe: {best_result['test_sharpe']:.3f}")
print(f"  Improvement: {best_result['improvement']:+.1f}%")

if best_result['test_sharpe'] > best_baseline * 1.02:
    print(f"\n✅ COMBINATION SIGNIFICANTLY BETTER!")
    print(f"   Use these weights: {np.array(best_result['weights'])}")
elif best_result['test_sharpe'] > best_baseline:
    print(f"\n✓ Combination slightly better")
    print(f"   Marginal improvement over best single agent")
else:
    print(f"\n❌ COMBINATION DOESN'T HELP")
    print(f"   Just use best single agent: {best_agent}")

print("="*70 + "\n")

# Save results
output_file = Path(f'{MODELS_DIR}/fixed_weights_results.json')
output_data = {
    'baselines': baselines,
    'results': results,
    'best_result': best_result,
    'recommendation': 'combination' if best_result['test_sharpe'] > best_baseline else 'single_agent'
}

with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"✓ Results saved to: {output_file}")
