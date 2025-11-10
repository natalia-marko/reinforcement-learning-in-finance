
"""
Train Super Agent - PRODUCTION VERSION
=======================================
"""

from pathlib import Path
import json
import torch
from super_agent import SuperAgent, BaseAgentWrapper
from environments import create_env
from config import get_config

# HARDCODED PATHS
BASE_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3'
DATA_DIR = f'{BASE_DIR}/data_hierarchical'
MODELS_DIR = f'{BASE_DIR}/models'

print("="*70)
print("TRAINING SUPER AGENT - PRODUCTION")
print("="*70)

# Check if base agents exist
base_agents_file = Path(f'{MODELS_DIR}/base_agents_production.json')

# Load base agent info
print("\n1. Loading base agents...")
with open(base_agents_file, 'r') as f:
    base_agents_info = json.load(f)

print(f"   ✓ Found {len(base_agents_info)} base agents")

# Create base agent wrappers
print("\n2. Creating base agent wrappers...")
base_agents = []

for agent_key, agent_data in base_agents_info.items():
    print(f"   Loading {agent_data['name']}...")
    
    model_path = agent_data['model_path']
    
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
        model_paths=[model_path],
        agent_name=agent_data['name'],
        agent_type=agent_data['agent_type'],
        reward_type=agent_data['reward_type'],
        data_dir=DATA_DIR,
        split='train',
        algorithm=agent_data['algorithm'],
        config=config
    )
    
    base_agents.append(agent)
    print(f"   ✓ Loaded")

if len(base_agents) != 3:
    print(f"\n❌ ERROR: Expected 3 base agents, found {len(base_agents)}")
    exit(1)

print(f"\n✓ All 3 base agents loaded!")

# Create environments
print("\n3. Creating environments...")
main_config = get_config('ema_sharpe', 'technical')

train_env = create_env(
    data_dir=DATA_DIR,
    agent_type='technical',
    split='train',
    reward_type='ema_sharpe',
    reward_kwargs={'rolling_vol_window': 12},
    softmax_temperature=main_config['softmax_temperature'],
    random_start=True,
    transaction_cost=main_config.get('transaction_cost', 0.0)
)

val_env = create_env(
    data_dir=DATA_DIR,
    agent_type='technical',
    split='val',
    reward_type='ema_sharpe',
    reward_kwargs={'rolling_vol_window': 12},
    softmax_temperature=main_config['softmax_temperature'],
    random_start=False,
    transaction_cost=main_config.get('transaction_cost', 0.0)
)

test_env = create_env(
    data_dir=DATA_DIR,
    agent_type='technical',
    split='test',
    reward_type='ema_sharpe',
    reward_kwargs={'rolling_vol_window': 12},
    softmax_temperature=main_config['softmax_temperature'],
    random_start=False,
    transaction_cost=main_config.get('transaction_cost', 0.0)
)

print(f"   ✓ Environments created")

# Create super agent
print("\n4. Creating super agent...")
super_agent = SuperAgent(
    base_agents=base_agents,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"   ✓ Super agent created")

# Train super agent
print("\n5. Training super agent...")
print("   (This will take 20-30 minutes...)\n")

save_path = Path(f'{MODELS_DIR}/super_agent_production.pt')

super_agent.train(
    train_env=train_env,       
    val_env=val_env,
    n_episodes=100,            
    eval_every=10,            
    patience=20,                
    save_path=str(save_path),
    verbose=True
)

print("\n   ✓ Training complete!")

# Evaluate on test set
print("\n6. Evaluating on test set...")
test_sharpe = super_agent.evaluate(test_env, deterministic=True, n_runs=5)
print(f"   Test Sharpe: {test_sharpe:.3f}")

# Analyze agent usage
print("\n7. Analyzing agent usage...")
analysis = super_agent.analyze_agent_usage(test_env, deterministic=True)

print("\n" + "="*70)
print("AGENT USAGE ON TEST SET")
print("="*70)
agent_names = [a.agent_name for a in base_agents]
for i, name in enumerate(agent_names):
    mean_w = analysis['mean_weights'][i]
    std_w = analysis['std_weights'][i]
    print(f"\n{name}:")
    print(f"  Mean weight: {mean_w:.3f} ± {std_w:.3f}")
    print(f"  Range: [{analysis['min_weights'][i]:.3f}, {analysis['max_weights'][i]:.3f}]")

# Save analysis
print("\n8. Saving results...")
analysis_file = Path(f'{MODELS_DIR}/super_agent_production_analysis.json')
analysis_data = {
    'test_sharpe': float(test_sharpe),
    'agent_weights': {
        agent_names[i]: {
            'mean': float(analysis['mean_weights'][i]),
            'std': float(analysis['std_weights'][i]),
            'min': float(analysis['min_weights'][i]),
            'max': float(analysis['max_weights'][i])
        }
        for i in range(len(agent_names))
    }
}

with open(analysis_file, 'w') as f:
    json.dump(analysis_data, f, indent=2)

print(f"   ✓ Analysis saved")

print("\n" + "="*70)
print("SUPER AGENT TRAINING COMPLETE!")
print("="*70)
print(f"\nTest Sharpe Ratio: {test_sharpe:.3f}")
print(f"\nFiles created:")
print(f"  - {save_path}")
print(f"  - {analysis_file}")
print("\n" + "="*70)
