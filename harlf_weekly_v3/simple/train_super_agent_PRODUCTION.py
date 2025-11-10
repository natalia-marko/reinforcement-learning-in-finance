"""
Train Super Agent - PRODUCTION VERSION (FIXED & SIMPLIFIED)
=======================================
Fixed: Use split-specific base agents for val/test.
Use get_config for consistency.
Set n_runs=1.
"""

from pathlib import Path
import json
import torch

from super_agent import SuperAgent, load_base_agents_from_results
from environments import create_env
from config import get_config

BASE_DIR = '/Users/nataliamarko/Documents/GitHub_Projects/reinforcement_learning_in_finance/harlf_weekly_v3'
DATA_DIR = f"{BASE_DIR}/data_hierarchical"
MODELS_DIR = f"{BASE_DIR}/models"

print("=" * 70)
print("TRAINING SUPER AGENT - PRODUCTION")
print("=" * 70)

base_agents_file = Path(f"{MODELS_DIR}/base_agents_production.json")

print("\n1. Loading base agents for splits...")
with open(base_agents_file, 'r') as f:
    base_agents_info = json.load(f)

bases_train = load_base_agents_from_results(base_agents_file, DATA_DIR, split='train')
bases_val   = load_base_agents_from_results(base_agents_file, DATA_DIR, split='val')
bases_test  = load_base_agents_from_results(base_agents_file, DATA_DIR, split='test')

print("   ✓ Loaded bases for train/val/test")

print("\n2. Creating environments...")
main_config = get_config("ema_sharpe", "technical")

train_env = create_env(
    data_dir=DATA_DIR,
    agent_type="technical",
    split="train",
    reward_type="ema_sharpe",
    reward_kwargs={"rolling_vol_window": 12},
    softmax_temperature=main_config["softmax_temperature"],
    random_start=True,
    transaction_cost=main_config.get("transaction_cost", 0.0)
)

val_env = create_env(
    data_dir=DATA_DIR,
    agent_type="technical",
    split="val",
    reward_type="ema_sharpe",
    reward_kwargs={"rolling_vol_window": 12},
    softmax_temperature=main_config["softmax_temperature"],
    random_start=False,
    transaction_cost=main_config.get("transaction_cost", 0.0)
)

test_env = create_env(
    data_dir=DATA_DIR,
    agent_type="technical",
    split="test",
    reward_type="ema_sharpe",
    reward_kwargs={"rolling_vol_window": 12},
    softmax_temperature=main_config["softmax_temperature"],
    random_start=False,
    transaction_cost=main_config.get("transaction_cost", 0.0)
)

print("   ✓ Environments created")

print("\n3. Creating super agent...")
super_agent = SuperAgent(
    base_agents=bases_train,
    learning_rate=3e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("   ✓ Super agent created")

print("\n4. Training super agent...")
save_path = Path(f"{MODELS_DIR}/super_agent_production.pt")
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

print("\n5. Evaluating on test set...")
test_sharpe = super_agent.evaluate(
    test_env,
    deterministic=True,
    n_runs=1,
    base_agents=bases_test
)
print(f"   Test Sharpe: {test_sharpe:.3f}")

print("\n6. Analyzing agent usage...")
analysis = super_agent.analyze_agent_usage(
    test_env,
    deterministic=True,
    base_agents=bases_test
)

print("\n" + "=" * 70)
print("AGENT USAGE ON TEST SET")
print("=" * 70)
agent_names = [a.agent_name for a in bases_test]

for i, name in enumerate(agent_names):
    mean_w = analysis['mean_weights'][i]
    std_w  = analysis['std_weights'][i]
    print(f"\n{name}:")
    print(f"  Mean weight: {mean_w:.3f} ± {std_w:.3f}")
    print(f"  Range: [{analysis['min_weights'][i]:.3f}, {analysis['max_weights'][i]:.3f}]")

print("\n7. Saving results...")
analysis_file = Path(f"{MODELS_DIR}/super_agent_production_analysis.json")

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

print("   ✓ Analysis saved")
print("\n" + "=" * 70)
print("SUPER AGENT TRAINING COMPLETE!")
print("=" * 70)
print(f"\nTest Sharpe Ratio: {test_sharpe:.3f}")
print(f"\nFiles created:")
print(f"  - {save_path}")
print(f"  - {analysis_file}")
print("\n" + "=" * 70)