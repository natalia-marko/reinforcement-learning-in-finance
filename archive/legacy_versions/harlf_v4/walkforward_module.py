"""
walkforward_module.py - REFACTORED (NO LEAKAGE)
Walk-forward validation with proper train-only normalization
"""

import numpy as np
import pandas as pd
from stable_baselines3 import SAC, TD3, PPO
import os
from typing import Dict, Tuple

from tech_env_module import TechnicalEnv
from utils import prepare_walkforward_window, clip_to_valid_range
from config import MODEL_CONFIG, WALKFORWARD_CONFIG, DATA_CONFIG


def train_and_evaluate(
    train_env: TechnicalEnv,
    val_env: TechnicalEnv,
    test_env: TechnicalEnv,
    algorithm: str,
    model_config: Dict,
    window_num: int
) -> Tuple[Dict, list]:
    """
    Train model with early stopping and evaluate
    
    Returns:
        test_metrics: Dict of test metrics
        test_history: List of portfolio values
    """
    
    # Create model
    policy_kwargs = dict(net_arch=model_config['network_arch'])
    
    if algorithm == 'SAC':
        model = SAC(
            'MlpPolicy', train_env,
            learning_rate=model_config['learning_rate'],
            buffer_size=model_config['buffer_size'],
            batch_size=model_config['batch_size'],
            gamma=model_config['gamma'],
            tau=model_config['tau'],
            policy_kwargs=policy_kwargs,
            verbose=0
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy', train_env,
            learning_rate=model_config['learning_rate'],
            buffer_size=model_config['buffer_size'],
            batch_size=model_config['batch_size'],
            gamma=model_config['gamma'],
            tau=model_config['tau'],
            policy_kwargs=policy_kwargs,
            verbose=0
        )
    elif algorithm == 'PPO':
        model = PPO(
            'MlpPolicy', train_env,
            learning_rate=model_config['learning_rate'],
            n_steps=min(2048, len(train_env.price_data) * 10),
            batch_size=model_config['batch_size'],
            gamma=model_config['gamma'],
            policy_kwargs=policy_kwargs,
            verbose=0
        )
    
    # Training with early stopping
    best_val_sortino = -np.inf
    patience = model_config['early_stopping_patience']
    wait = 0
    max_epochs = int(np.ceil(model_config['total_training_steps'] / model_config['steps_per_epoch']))
    
    os.makedirs('models/temp', exist_ok=True)
    
    for epoch in range(max_epochs):
        model.learn(
            total_timesteps=model_config['steps_per_epoch'],
            reset_num_timesteps=(epoch == 0)
        )
        
        # Validate (on VALIDATION set, NOT training!)
        val_metrics = evaluate_model(model, val_env, verbose=False)
        
        improvement_threshold = max(best_val_sortino * 1.02, best_val_sortino + 0.01)
        
        if val_metrics['sortino_ratio'] > improvement_threshold:
            best_val_sortino = val_metrics['sortino_ratio']
            model.save(f'models/temp/best_model_w{window_num}')
            wait = 0
        else:
            wait += 1
        
        if wait >= patience:
            break
    
    # Load best model
    if algorithm == 'SAC':
        model = SAC.load(f'models/temp/best_model_w{window_num}')
    elif algorithm == 'TD3':
        model = TD3.load(f'models/temp/best_model_w{window_num}')
    elif algorithm == 'PPO':
        model = PPO.load(f'models/temp/best_model_w{window_num}')
    
    # Evaluate on TEST set (out-of-sample!)
    test_metrics = evaluate_model(model, test_env, verbose=False)
    test_history = test_env.portfolio_history
    
    return test_metrics, test_history


def evaluate_model(model, env: TechnicalEnv, verbose: bool = False) -> Dict:
    """Evaluate model on environment"""
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    
    metrics = env.get_portfolio_metrics()
    
    if verbose:
        print(f"  Sharpe:  {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino: {metrics['sortino_ratio']:.3f}")
        print(f"  Return:  {metrics['total_return']*100:.1f}%")
    
    return metrics


def walk_forward_validation(
    price_data: pd.Series | pd.DataFrame,
    technical_features: pd.DataFrame,
    asset_name: str | None,
    env_config: Dict,
    model_config: Dict,
    wf_config: Dict,
    algorithm: str = 'SAC',
    clip_pre_ipo: bool = True
) -> Dict:
    """
    Run walk-forward validation with NO LEAKAGE
    
    CRITICAL: Each window normalizes ONLY on its own training data
    
    Args:
        price_data: Price series.  May be a pandas Series (single asset) or
            DataFrame (multi‑asset), indexed by date.
        technical_features: RAW (unnormalized) features as a DataFrame aligned
            with ``price_data``.  Features should be pre‑computed for all
            assets if using a DataFrame of prices.
        asset_name: Optional asset ticker for logging and pre‑IPO clipping;
            ignored when ``price_data`` is a multi‑asset DataFrame.
        env_config: Environment configuration
        model_config: Model configuration
        wf_config: Walk-forward configuration
        algorithm: Algorithm to use
        clip_pre_ipo: Whether to clip pre-IPO backfill
    
    Returns:
        results: Dict with results DataFrame and summary statistics
    """
    
    print("=" * 80)
    # Use a label for multi‑asset runs where asset_name may be None
    label = asset_name if asset_name is not None else 'MULTI-ASSET'
    print(f"WALK-FORWARD VALIDATION: {label} - {algorithm}")
    print("=" * 80)
    
    # Clip pre-IPO data if requested
    if clip_pre_ipo:
        print(f"\nClipping pre-IPO backfill...")
        price_data, technical_features = clip_to_valid_range(
            price_data,
            technical_features,
            asset_name,
            lookback_long=DATA_CONFIG['lookback_long']
        )
    
    print(f"\nWindow Configuration:")
    print(f"  Train: {wf_config['train_weeks']} weeks (~{wf_config['train_weeks']/52:.1f} years)")
    print(f"  Val:   {wf_config['val_weeks']} weeks (~{wf_config['val_weeks']/52:.1f} years)")
    print(f"  Test:  {wf_config['test_weeks']} weeks (~{wf_config['test_weeks']/52:.1f} years)")
    print(f"  Gap:   {wf_config['test_gap_weeks']} weeks")
    print(f"  Step:  {wf_config['step_size']} weeks")
    
    results = []
    total_len = len(price_data)
    window_size = (wf_config['train_weeks'] + wf_config['val_weeks'] + 
                   wf_config['test_gap_weeks'] + wf_config['test_weeks'])
    
    if total_len < window_size:
        raise ValueError(
            f"Not enough data! Have {total_len} weeks, need {window_size} weeks. "
            f"Consider reducing window sizes or using asset with more history."
        )
    
    window_num = 0
    
    for start_idx in range(0, total_len - window_size + 1, wf_config['step_size']):
        window_num += 1
        
        # Define boundaries
        train_end = start_idx + wf_config['train_weeks']
        val_end = train_end + wf_config['val_weeks']
        test_start = val_end + wf_config['test_gap_weeks']
        test_end = test_start + wf_config['test_weeks']
        
        print(f"\n{'─'*80}")
        print(f"Window {window_num}:")
        # Access the datetime index from either Series or DataFrame
        idx = price_data.index
        print(f"  Train: {idx[start_idx].date()} to {idx[train_end - 1].date()}")
        print(f"  Val:   {idx[train_end].date()} to {idx[val_end - 1].date()}")
        print(f"  Gap:   {idx[val_end].date()} to {idx[test_start - 1].date()}")
        print(f"  Test:  {idx[test_start].date()} to {idx[test_end - 1].date()}")
        
        # Prepare window with proper normalization (NO LEAKAGE!)
        print(f"  Preparing data (normalizing on train only)...")
        (train_prices, train_features_norm), \
        (val_prices, val_features_norm), \
        (test_prices, test_features_norm), \
        scaler = prepare_walkforward_window(
            price_data,
            technical_features,
            asset_name,
            start_idx,
            train_end,
            val_end,
            test_start,
            test_end,
            lookback_long=DATA_CONFIG['lookback_long'],
            clip_pre_ipo=False  # Already clipped above
        )
        
        # Create environments (features pre-normalized, no leakage)
        train_env = TechnicalEnv(
            train_prices,
            train_features_norm,
            validate_inputs=True,  # Validate first window
            **env_config
        )
        val_env = TechnicalEnv(val_prices, val_features_norm, validate_inputs=False, **env_config)
        test_env = TechnicalEnv(test_prices, test_features_norm, validate_inputs=False, **env_config)
        
        # Train and evaluate
        print(f"  Training (validation-based early stopping)...")
        test_metrics, test_history = train_and_evaluate(
            train_env, val_env, test_env,
            algorithm, model_config, window_num
        )
        
        print(f"  ✓ Test Results (OUT-OF-SAMPLE):")
        print(f"    Sharpe:  {test_metrics['sharpe_ratio']:>7.3f}")
        print(f"    Sortino: {test_metrics['sortino_ratio']:>7.3f}")
        print(f"    Return:  {test_metrics['total_return']*100:>6.1f}%")
        print(f"    Max DD:  {test_metrics['max_drawdown']*100:>6.1f}%")
        
        results.append({
            'window': window_num,
            'test_start': test_prices.index[0],
            'test_end': test_prices.index[-1],
            'sharpe': test_metrics['sharpe_ratio'],
            'sortino': test_metrics['sortino_ratio'],
            'calmar': test_metrics['calmar_ratio'],
            'return': test_metrics['total_return'],
            'mdd': test_metrics['max_drawdown'],
            'volatility': test_metrics['volatility'],
            'win_rate': test_metrics['win_rate'],
            'profit_factor': test_metrics['profit_factor'],
            'n_trades': test_metrics['n_trades']
        })
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATE RESULTS (ALL OUT-OF-SAMPLE)")
    print(f"{'='*80}")
    
    df_results = pd.DataFrame(results)
    
    print(f"\nNumber of windows: {len(results)}")
    print(f"\nMean Metrics:")
    print(f"  Sharpe:      {df_results['sharpe'].mean():.3f} ± {df_results['sharpe'].std():.3f}")
    print(f"  Sortino:     {df_results['sortino'].mean():.3f} ± {df_results['sortino'].std():.3f}")
    print(f"  Return:      {df_results['return'].mean()*100:.1f}% ± {df_results['return'].std()*100:.1f}%")
    print(f"  Max DD:      {df_results['mdd'].mean()*100:.1f}% ± {df_results['mdd'].std()*100:.1f}%")
    
    print(f"\nConsistency:")
    print(f"  % Positive Sharpe:  {(df_results['sharpe'] > 0).mean()*100:.1f}%")
    print(f"  % Positive Return:  {(df_results['return'] > 0).mean()*100:.1f}%")
    
    # Statistical significance
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(df_results['sharpe'], 0)
    print(f"\nStatistical Significance:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significant (p<0.05)? {'✓ YES' if p_value < 0.05 else '✗ NO'}")
    
    return {
        'results': df_results,
        'mean_sharpe': df_results['sharpe'].mean(),
        'std_sharpe': df_results['sharpe'].std(),
        'mean_sortino': df_results['sortino'].mean(),
        'mean_return': df_results['return'].mean(),
        'consistency': (df_results['sharpe'] > 0).mean(),
        'p_value': p_value
    }


def compare_algorithms(
    price_data: pd.Series | pd.DataFrame,
    technical_features: pd.DataFrame,
    asset_name: str | None,
    env_config: Dict,
    model_config: Dict,
    wf_config: Dict,
    algorithms: list = ['SAC', 'TD3', 'PPO']
) -> pd.DataFrame:
    """
    Compare multiple algorithms with proper validation
    """
    
    print("=" * 80)
    label = asset_name if asset_name is not None else 'MULTI-ASSET'
    print(f"COMPARING ALGORITHMS: {label}")
    print("=" * 80)
    
    comparison = []
    
    for algo in algorithms:
        print(f"\n{'#'*80}")
        print(f"# ALGORITHM: {algo}")
        print(f"{'#'*80}")
        
        results = walk_forward_validation(
            price_data, technical_features, asset_name,
            env_config, model_config, wf_config,
            algorithm=algo
        )
        
        comparison.append({
            'Algorithm': algo,
            'Mean Sharpe': results['mean_sharpe'],
            'Std Sharpe': results['std_sharpe'],
            'Mean Sortino': results['mean_sortino'],
            'Mean Return': results['mean_return'],
            'Consistency': results['consistency'],
            'p-value': results['p_value']
        })
    
    df_comparison = pd.DataFrame(comparison)
    
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON")
    print(f"{'='*80}")
    print(df_comparison.to_string(index=False))
    
    return df_comparison