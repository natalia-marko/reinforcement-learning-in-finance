#!/usr/bin/env python3
"""Diagnose LSTM vs MLP PPO performance issues."""

import pandas as pd
import numpy as np
from pathlib import Path

# Load results
lstm_results = pd.read_csv('models/lstm_ppo_results.csv')
mlp_results = pd.read_csv('models/ppo_log_return_results.csv')

print('=' * 80)
print('PORTFOLIO VALUE ANALYSIS')
print('=' * 80)

print('\nMLP PPO (ppo_log_return):')
for _, row in mlp_results.iterrows():
    fold = int(row['fold'])
    pv = row['portfolio_value']
    sharpe = row['sharpe_ratio']
    ret = row['total_return']

    status = "✓ VALID" if pv > 1000 else "✗ COLLAPSED"
    print(f"  Fold {fold}: ${pv:>15.2f}  |  Return: {ret:>8.2%}  |  Sharpe: {sharpe:>6.3f}  {status}")

print('\nLSTM PPO:')
for _, row in lstm_results.iterrows():
    fold = int(row['fold'])
    pv = row['portfolio_value']
    sharpe = row['sharpe_ratio']
    ret = row['total_return']

    status = "✓ VALID"
    print(f"  Fold {fold}: ${pv:>15.2f}  |  Return: {ret:>8.2%}  |  Sharpe: {sharpe:>6.3f}  {status}")

print('\n' + '=' * 80)
print('COMPARISON')
print('=' * 80)

# Valid folds only
mlp_valid = mlp_results[mlp_results['portfolio_value'] > 1000]
lstm_valid = lstm_results[lstm_results['portfolio_value'] > 1000]

print(f"\nMLP PPO:")
print(f"  Valid folds: {len(mlp_valid)}/3")
if len(mlp_valid) > 0:
    print(f"  Valid mean Sharpe: {mlp_valid['sharpe_ratio'].mean():.3f}")
    print(f"  Valid mean Return: {mlp_valid['total_return'].mean():.2%}")

print(f"\nLSTM PPO:")
print(f"  Valid folds: {len(lstm_valid)}/3")
if len(lstm_valid) > 0:
    print(f"  Mean Sharpe: {lstm_valid['sharpe_ratio'].mean():.3f}")
    print(f"  Mean Return: {lstm_valid['total_return'].mean():.2%}")

print('\n' + '=' * 80)
print('DIAGNOSIS')
print('=' * 80)

collapsed_folds = mlp_results[mlp_results['portfolio_value'] < 1000]
if len(collapsed_folds) > 0:
    print(f"\n⚠️  WARNING: {len(collapsed_folds)} MLP PPO folds have portfolio value collapse!")
    print("   This indicates numerical instability or a bug in the environment.")
    print("   These results are INVALID and should be discarded.")
    print("\n   Possible causes:")
    print("   - Old training used PortfolioEnv with incorrect return formula")
    print("   - Transaction costs applied incorrectly")
    print("   - Numerical overflow/underflow in log calculations")

print("\n✓ RECOMMENDATION:")
print("  Retrain MLP PPO with PortfolioEnv2 for valid comparison")

print('\n' + '=' * 80)
print('LSTM PERFORMANCE ASSESSMENT')
print('=' * 80)

# Analyze LSTM variance
lstm_sharpes = lstm_results['sharpe_ratio'].values
print(f"\nLSTM Sharpe ratios: {lstm_sharpes}")
print(f"  Best: {lstm_sharpes.max():.3f} (Fold {lstm_sharpes.argmax()})")
print(f"  Worst: {lstm_sharpes.min():.3f} (Fold {lstm_sharpes.argmin()})")
print(f"  Mean: {lstm_sharpes.mean():.3f}")
print(f"  Std: {lstm_sharpes.std():.3f}")

if lstm_sharpes.std() > 0.5:
    print("\n⚠️  High variance across folds suggests:")
    print("   - Unstable training")
    print("   - Insufficient training timesteps")
    print("   - Learning rate too high")
    print("   - LSTM hidden size too small")

# Check training convergence
print("\n" + "=" * 80)
print("CHECKING TRAINING CONVERGENCE")
print("=" * 80)

for fold_idx in range(3):
    val_log = Path(f'models/lstm_ppo/fold_{fold_idx}/logs/val_evaluations.csv')
    if val_log.exists():
        df = pd.read_csv(val_log)
        best_step = df.loc[df['sharpe_ratio'].idxmax(), 'step']
        best_sharpe = df['sharpe_ratio'].max()
        final_sharpe = df['sharpe_ratio'].iloc[-1]

        print(f"\nFold {fold_idx}:")
        print(f"  Best Sharpe: {best_sharpe:.3f} at step {int(best_step)}")
        print(f"  Final Sharpe: {final_sharpe:.3f}")

        if best_step < 50000:
            print(f"  ⚠️  Peaked early at {int(best_step)}/100000 steps")
        if best_sharpe > final_sharpe + 0.1:
            print(f"  ⚠️  Degraded after peak (overfitting?)")
