#!/usr/bin/env python3
"""Quick Test - 10 minute training"""

from pathlib import Path
import json
import torch
from train import train
from super_agent import SuperAgent, BaseAgentWrapper
from environments import create_env
from config import get_config

# YOUR PATHS
BASE_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3'
DATA_DIR = f'{BASE_DIR}/data_hierarchical'
MODELS_DIR = f'{BASE_DIR}/models'

print("="*70)
print("QUICK TEST - Training 3 agents")
print("="*70)

agents = [
    {'name': 'agent_1_return_max', 'agent_type': 'technical', 'reward_type': 'ema_sharpe'},
    {'name': 'agent_2_loss_min', 'agent_type': 'technical', 'reward_type': 'multi_objective'},
    {'name': 'agent_3_alternative', 'agent_type': 'sentiment', 'reward_type': 'ema_sharpe'}
]

results = {}

# Train 3 base agents
for i, agent in enumerate(agents, 1):
    print(f"\nAgent {i}/3: {agent['name']}...")
    
    config = get_config(agent['reward_type'], agent['agent_type'])
    config['total_steps'] = 10_000
    config['eval_freq'] = 2_000
    config['patience'] = 3
    
    result = train(
        agent_type=agent['agent_type'],
        algorithm='PPO',
        reward_type=agent['reward_type'],
        config=config,
        verbose=False
    )
    
    results[agent['name']] = {
        'name': agent['name'],
        'agent_type': agent['agent_type'],
        'algorithm': 'PPO',
        'reward_type': agent['reward_type'],
        'model_path': result['model_path'],
        'train_sharpe': result['train_sharpe'],
        'val_sharpe': result['val_sharpe'],
        'test_sharpe': result['test_sharpe'],
        'gamma': result.get('gamma', 0.90),
        'softmax_temperature': result.get('softmax_temperature', 1.0),
        'transaction_cost': result.get('transaction_cost', 0.0)
    }
    
    print(f"✓ Test Sharpe: {result['test_sharpe']:.3f}")

# Save base agents
summary_file = Path(f'{MODELS_DIR}/base_agents_simple.json')
summary_file.parent.mkdir(parents=True, exist_ok=True)
with open(summary_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Base agents saved")

# Load base agents for super agent
print("\nTraining super agent...")
base_agents = []

for agent_key, agent_data in results.items():
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
        split='train',
        algorithm=agent_data['algorithm'],
        config=config
    )
    base_agents.append(agent)

# Create environments
main_config = get_config('ema_sharpe', 'technical')

train_env = create_env(
    DATA_DIR, 'technical', 'train', 'ema_sharpe',
    {'rolling_vol_window': 12}, 
    main_config['softmax_temperature'], True,
    main_config.get('transaction_cost', 0.0)
)

val_env = create_env(
    DATA_DIR, 'technical', 'val', 'ema_sharpe',
    {'rolling_vol_window': 12},
    main_config['softmax_temperature'], False,
    main_config.get('transaction_cost', 0.0)
)

test_env = create_env(
    DATA_DIR, 'technical', 'test', 'ema_sharpe',
    {'rolling_vol_window': 12},
    main_config['softmax_temperature'], False,
    main_config.get('transaction_cost', 0.0)
)

# Create super agent
super_agent = SuperAgent(
    base_agents=base_agents,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# FIXED: Correct parameter names!
# train_env (not env!)
# n_episodes (not n_epochs!)
# eval_every (not eval_freq!)
super_agent.train(
    train_env=train_env,
    val_env=val_env,
    n_episodes=30,
    eval_every=10,
    patience=5,
    verbose=True
)

# Evaluate
test_sharpe = super_agent.evaluate(test_env, deterministic=True, n_runs=3)

# Save
save_path = Path(f'{MODELS_DIR}/super_agent_simple.pt')
super_agent.save(str(save_path))

print(f"\n✓ Super agent saved")
print(f"✓ Test Sharpe: {test_sharpe:.3f}")
print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nResults:")
print(f"  {MODELS_DIR}/base_agents_simple.json")
print(f"  {MODELS_DIR}/super_agent_simple.pt")
