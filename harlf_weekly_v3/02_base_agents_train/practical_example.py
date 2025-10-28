"""
PRACTICAL EXAMPLE: Integrating Optuna with Your Training Code
===============================================================

This shows exactly how to plug Optuna into your existing training setup.
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import json
from pathlib import Path

# Assuming your project structure:
from train import train
from config import get_config


# ============================================================================
# MINIMAL WORKING EXAMPLE: EMA Sharpe
# ============================================================================

def objective_ema_minimal(trial):
    """
    Minimal objective function - tune only gamma and softmax_temperature.
    This is the fastest and most practical approach.
    """
    # Sample just 2 key parameters
    gamma = trial.suggest_float('gamma', 0.85, 0.95, step=0.01)
    softmax_temp = trial.suggest_float('softmax_temperature', 0.75, 1.5, step=0.05)
    
    # Build config
    config = {
        'gamma': gamma,
        'softmax_temperature': softmax_temp,
        'rolling_vol_window': 12,  # Fixed
        'learning_rate': 3e-4,
        'total_steps': 300_000,
        'patience': 15,
        'transaction_cost': 0.0025,
        'seed': 42,
        'random_start': True,
    }
    
    # Train model (replace with your actual train function)
    result = train(
        agent_type='technical',
        algorithm='PPO',
        reward_type='ema_sharpe',
        config=config,
        verbose=False
    )
    
    # Return validation Sharpe (what Optuna optimizes)
    return result['val_sharpe']


# ============================================================================
# RUN OPTIMIZATION
# ============================================================================

def run_light_tune():
    """Run a light tuning session - perfect for overnight run."""
    
    print("🎯 Starting Light Hyperparameter Tuning")
    print("=" * 60)
    print("Model: EMA Sharpe + Technical + PPO")
    print("Trials: 20")
    print("Time: ~5 hours")
    print("Tuning: gamma, softmax_temperature")
    print("=" * 60)
    print()
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
    )
    
    # Run optimization
    study.optimize(objective_ema_minimal, n_trials=20)
    
    # Print results
    print("\n" + "=" * 60)
    print("✅ TUNING COMPLETE!")
    print("=" * 60)
    print(f"\n🏆 Best Validation Sharpe: {study.best_value:.4f}")
    print(f"\n📊 Best Parameters:")
    print(f"   gamma: {study.best_params['gamma']:.3f}")
    print(f"   softmax_temperature: {study.best_params['softmax_temperature']:.2f}")
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
    }
    
    with open('optuna_best_params.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: optuna_best_params.json")
    
    return study


# ============================================================================
# APPLY BEST PARAMETERS
# ============================================================================

def retrain_with_best_params(params_file='optuna_best_params.json'):
    """
    Load best parameters from Optuna and retrain for final evaluation.
    """
    print("🔄 Retraining with best parameters...")
    
    # Load best params
    with open(params_file, 'r') as f:
        results = json.load(f)
    
    best_params = results['best_params']
    print(f"\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Build config with best params
    config = {
        'gamma': best_params['gamma'],
        'softmax_temperature': best_params['softmax_temperature'],
        'rolling_vol_window': 12,
        'learning_rate': 3e-4,
        'total_steps': 300_000,
        'patience': 15,
        'transaction_cost': 0.0025,
        'seed': 42,
        'random_start': True,
    }
    
    # Final training run
    result = train(
        agent_type='technical',
        algorithm='PPO',
        reward_type='ema_sharpe',
        config=config,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Train Sharpe:  {result['train_sharpe']:.4f}")
    print(f"Val Sharpe:    {result['val_sharpe']:.4f}")
    print(f"Test Sharpe:   {result['test_sharpe']:.4f}")
    print("=" * 60)
    
    return result


# ============================================================================
# COMPARISON: BEFORE vs AFTER TUNING
# ============================================================================

def compare_before_after():
    """
    Compare baseline performance vs tuned performance.
    """
    print("\n" + "=" * 60)
    print("📊 BEFORE vs AFTER COMPARISON")
    print("=" * 60)
    
    # Baseline (your current params)
    baseline_config = {
        'gamma': 0.90,
        'softmax_temperature': 1.0,
        'rolling_vol_window': 12,
        'learning_rate': 3e-4,
        'total_steps': 300_000,
        'patience': 15,
        'transaction_cost': 0.0025,
        'seed': 42,
    }
    
    print("\n🔵 BASELINE (current params):")
    baseline_result = train('technical', 'PPO', 'ema_sharpe', baseline_config, verbose=False)
    print(f"   Val Sharpe:  {baseline_result['val_sharpe']:.4f}")
    print(f"   Test Sharpe: {baseline_result['test_sharpe']:.4f}")
    
    # Load tuned params
    with open('optuna_best_params.json', 'r') as f:
        best_params = json.load(f)['best_params']
    
    tuned_config = baseline_config.copy()
    tuned_config.update(best_params)
    
    print("\n🟢 TUNED (Optuna params):")
    tuned_result = train('technical', 'PPO', 'ema_sharpe', tuned_config, verbose=False)
    print(f"   Val Sharpe:  {tuned_result['val_sharpe']:.4f}")
    print(f"   Test Sharpe: {tuned_result['test_sharpe']:.4f}")
    
    # Calculate improvement
    val_improvement = (tuned_result['val_sharpe'] - baseline_result['val_sharpe']) / baseline_result['val_sharpe'] * 100
    test_improvement = (tuned_result['test_sharpe'] - baseline_result['test_sharpe']) / baseline_result['test_sharpe'] * 100
    
    print("\n📈 IMPROVEMENT:")
    print(f"   Val Sharpe:  {val_improvement:+.1f}%")
    print(f"   Test Sharpe: {test_improvement:+.1f}%")
    print("=" * 60)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    
    # Example 1: Run light tuning (20 trials, ~5 hours)
    print("\n" + "=" * 60)
    print("OPTION 1: Run Light Tuning")
    print("=" * 60)
    study = run_light_tune()
    
    # Example 2: Retrain with best params
    print("\n" + "=" * 60)
    print("OPTION 2: Retrain with Best Parameters")
    print("=" * 60)
    result = retrain_with_best_params()
    
    # Example 3: Compare before/after
    print("\n" + "=" * 60)
    print("OPTION 3: Compare Before vs After")
    print("=" * 60)
    compare_before_after()


# ============================================================================
# STEP-BY-STEP WORKFLOW
# ============================================================================

"""
RECOMMENDED WORKFLOW:
═════════════════════

Step 1: Initial Tuning (Run tonight)
────────────────────────────────────
$ python practical_example.py

This will:
- Run 20 trials (5-6 hours)
- Save best params to optuna_best_params.json
- You wake up to optimized parameters!


Step 2: Verify Results (Next morning)
─────────────────────────────────────
>>> from practical_example import retrain_with_best_params
>>> result = retrain_with_best_params()

This will:
- Load best params from JSON
- Retrain one final time
- Show final test set performance


Step 3: Deploy (If results are good)
────────────────────────────────────
If test Sharpe > 1.55:
✅ Deploy with tuned parameters

If test Sharpe < 1.55:
⚠️  Keep baseline params (gamma=0.9, temp=1.0)
   Or try tuning multi-objective instead


EXPECTED TIMELINE:
──────────────────
Day 1 Evening: Start tuning (5-6 hours)
Day 2 Morning: Check results (15 min)
Day 2 Morning: Retrain with best (15 min)
Day 2 Noon:    Deploy or iterate

Total time: ~6 hours compute, ~30 min human time


EXPECTED IMPROVEMENT:
─────────────────────
Conservative estimate: +3-5% test Sharpe
Optimistic estimate:   +8-12% test Sharpe

Your current:  1.513
After tuning:  1.56-1.65 (realistic)
               1.65-1.70 (if lucky!)
"""
