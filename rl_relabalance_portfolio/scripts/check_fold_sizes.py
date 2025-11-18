"""
Check all fold sizes to identify the discrepancy
"""
import pandas as pd
import json
from pathlib import Path
from rl_system import create_walk_forward_folds

# Load data
DATA_DIR = Path('data/processed')
train_df = pd.read_parquet(DATA_DIR / 'train.parquet')

with open(DATA_DIR / 'metadata.json', 'r') as f:
    metadata = json.load(f)

# Create folds
folds = create_walk_forward_folds(train_df, n_folds=5)

print("=" * 80)
print("FOLD SIZES")
print("=" * 80)

for fold in folds:
    fold_idx = fold['fold_idx']
    train_dates = len(fold['train']['date'].unique())
    val_dates = len(fold['val']['date'].unique())

    print(f"\nFold {fold_idx}:")
    print(f"  Train unique dates: {train_dates}")
    print(f"  Val unique dates: {val_dates}")
    print(f"  Expected episode length: {val_dates - 1}")
    print(f"  Date range: {fold['val_start'].date()} to {fold['val_end'].date()}")

# Compare with results
results_path = Path('models/baseline/fold_results.csv')
if results_path.exists():
    results_df = pd.read_csv(results_path)
    print("\n" + "=" * 80)
    print("ACTUAL RESULTS (from fold_results.csv)")
    print("=" * 80)
    for _, row in results_df.iterrows():
        print(f"\nFold {int(row['fold_idx'])}:")
        print(f"  Episode length: {int(row['val_episode_length'])}")
        print(f"  Portfolio value: {row['val_portfolio_value']}")
        print(f"  Sharpe: {row['val_sharpe_ratio']}")
