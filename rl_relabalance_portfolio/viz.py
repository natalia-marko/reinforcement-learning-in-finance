"""
Visualization functions for RL portfolio rebalancing analysis.
Contains all plotting and visualization utilities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.ensemble import RandomForestRegressor


# ============================================================================
# DATA VISUALIZATION
# ============================================================================

def plot_price_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tickers: List[str],
    results_dir: Path
) -> None:
    """
    Visualize price data over time.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    tickers : List[str]
        List of ticker symbols
    results_dir : Path
        Directory to save plots
    """
    train_end = pd.to_datetime(train_df['date'].max())
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot closing prices
    for ticker in tickers:
        train_ticker = train_df[train_df['ticker'] == ticker]
        test_ticker = test_df[test_df['ticker'] == ticker]
        
        axes[0].plot(train_ticker['date'], train_ticker['close'], label=ticker, alpha=0.7)
        axes[0].plot(test_ticker['date'], test_ticker['close'], alpha=0.7)
    
    axes[0].axvline(x=train_end, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    axes[0].set_title('Close Prices (Train | Test)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot returns
    for ticker in tickers:
        train_ticker = train_df[train_df['ticker'] == ticker]
        test_ticker = test_df[test_df['ticker'] == ticker]
        
        train_returns = train_ticker['close'].pct_change()
        test_returns = test_ticker['close'].pct_change()
        
        axes[1].plot(train_ticker['date'], train_returns, label=ticker, alpha=0.7)
        axes[1].plot(test_ticker['date'], test_returns, alpha=0.7)
    
    axes[1].axvline(x=train_end, color='red', linestyle='--', linewidth=2, label='Train/Test Split')
    axes[1].set_title('Weekly Returns (Train | Test)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'price_data_overview.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_regime_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tickers: List[str],
    results_dir: Path
) -> None:
    """
    Plot regime and concentration features.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    tickers : List[str]
        List of ticker symbols
    results_dir : Path
        Directory to save plots
    """
    regime_features = ['bull_market', 'bear_market', 'relative_momentum', 'market_concentration', 'returns_52w']
    available_regime_features = [f for f in regime_features if f in train_df.columns]
    
    if not available_regime_features:
        print("No regime features found to plot")
        return
    
    # Count how many features we'll actually plot
    features_to_plot = []
    if 'bull_market' in train_df.columns:
        features_to_plot.append('bull_market')
    if 'relative_momentum' in train_df.columns:
        features_to_plot.append('relative_momentum')
    if 'market_concentration' in train_df.columns:
        features_to_plot.append('market_concentration')
    if 'returns_52w' in train_df.columns:
        features_to_plot.append('returns_52w')
    # Add VIX as 4th indicator if available
    if 'vix' in train_df.columns:
        features_to_plot.append('vix')
    
    # Create grid based on number of features to plot
    n_features = len(features_to_plot)
    if n_features == 0:
        print("No features available to plot")
        return
    
    # Determine grid size (use 2x2 for 3+ features, adjust for fewer)
    if n_features == 1:
        nrows, ncols = 1, 1
    elif n_features == 2:
        nrows, ncols = 1, 2
    else:
        nrows, ncols = 2, 2
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 10))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot bull_market and bear_market over time
    if 'bull_market' in train_df.columns:
        train_dates = pd.to_datetime(train_df['date'].unique())
        bull_by_date = train_df.groupby('date')['bull_market'].mean()
        bear_by_date = train_df.groupby('date')['bear_market'].mean()
        
        axes[plot_idx].plot(train_dates, bull_by_date, label='Bull Market', color='green', alpha=0.7)
        axes[plot_idx].plot(train_dates, bear_by_date, label='Bear Market', color='red', alpha=0.7)
        axes[plot_idx].set_title('Market Regime Indicators Over Time', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Date')
        axes[plot_idx].set_ylabel('Proportion of Assets')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot relative momentum distribution
    if 'relative_momentum' in train_df.columns:
        train_rm = train_df['relative_momentum'].dropna()
        axes[plot_idx].hist(train_rm, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[plot_idx].set_title('Relative Momentum Distribution', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Relative Momentum (Asset Return - Benchmark Return)')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot market concentration over time
    if 'market_concentration' in train_df.columns:
        conc_by_date = train_df.groupby('date')['market_concentration'].first()
        train_dates = pd.to_datetime(conc_by_date.index)
        axes[plot_idx].plot(train_dates, conc_by_date.values, color='purple', alpha=0.7)
        axes[plot_idx].set_title('Market Concentration (Herfindahl Index)', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Date')
        axes[plot_idx].set_ylabel('Herfindahl Index')
        axes[plot_idx].axhline(y=1/len(tickers), color='red', linestyle='--', 
                       label=f'Equal Weight (1/{len(tickers)})', linewidth=2)
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot 52-week returns distribution
    if 'returns_52w' in train_df.columns:
        train_52w = train_df['returns_52w'].dropna()
        axes[plot_idx].hist(train_52w, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[plot_idx].set_title('52-Week Returns Distribution', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('52-Week Log Return')
        axes[plot_idx].set_ylabel('Frequency')
        axes[plot_idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot VIX over time (market volatility/fear gauge)
    if 'vix' in train_df.columns:
        vix_by_date = train_df.groupby('date')['vix'].first()
        train_dates = pd.to_datetime(vix_by_date.index)
        axes[plot_idx].plot(train_dates, vix_by_date.values, color='darkred', alpha=0.7, linewidth=2)
        axes[plot_idx].set_title('VIX (Market Volatility Index)', fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Date')
        axes[plot_idx].set_ylabel('VIX Level')
        # Add reference lines for common VIX thresholds
        axes[plot_idx].axhline(y=20, color='green', linestyle='--', alpha=0.5, label='Low Volatility (<20)')
        axes[plot_idx].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='High Volatility (>30)')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'regime_features.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_distributions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_categories: Dict[str, List[str]],
    results_dir: Path
) -> None:
    """
    Plot normalized feature distributions.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    test_df : pd.DataFrame
        Test dataframe
    feature_categories : Dict[str, List[str]]
        Dictionary mapping category names to feature lists
    results_dir : Path
        Directory to save plots
    """
    # Sample 9 features from different categories
    sample_features = []
    for category, features in sorted(feature_categories.items(), key=lambda x: -len(x[1])):
        if features and category != 'Other':
            sample_features.extend(features[:2])
        if len(sample_features) >= 9:
            break
    sample_features = sample_features[:9]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feat in enumerate(sample_features):
        if feat in train_df.columns:
            train_vals = train_df[feat].dropna()
            test_vals = test_df[feat].dropna()
            
            axes[idx].hist(train_vals, bins=50, alpha=0.6, label='Train', density=True, color='blue')
            axes[idx].hist(test_vals, bins=50, alpha=0.6, label='Test', density=True, color='orange')
            axes[idx].set_title(feat, fontsize=10)
            axes[idx].set_xlabel('Normalized Value')
            axes[idx].set_ylabel('Density')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Normalized Feature Distributions (Train vs Test)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    results_dir: Path,
    top_n: int = 30
) -> None:
    """
    Plot feature importance visualization.
    
    Parameters:
    -----------
    feature_importance : pd.DataFrame
        DataFrame with columns 'feature' and 'importance'
    results_dir : Path
        Directory to save plots
    top_n : int
        Number of top features to visualize (default: 30)
    """
    top_features = feature_importance.head(top_n)
    
    # Create color mapping by category
    colors = []
    for feat in top_features['feature']:
        if any(x in feat.lower() for x in ['macro', 'treasury', 'fed', 'vix', 'cpi', 'unemployment', 
                                             'yield_curve', 'dxy', 'oil', 'spread', 'baa', 'pce']):
            colors.append('#e74c3c')  # Red for macro
        elif any(x in feat.lower() for x in ['return', 'roc', 'momentum', 'rsi', 'macd', 'stochastic']):
            colors.append('#3498db')  # Blue for momentum
        elif any(x in feat.lower() for x in ['volatility', 'atr', 'parkinson']):
            colors.append('#f39c12')  # Orange for volatility
        elif any(x in feat.lower() for x in ['volume', 'obv', 'mfi', 'chaikin']):
            colors.append('#9b59b6')  # Purple for volume
        elif any(x in feat.lower() for x in ['sharpe', 'sortino', 'calmar']):
            colors.append('#1abc9c')  # Teal for risk-adjusted
        elif any(x in feat.lower() for x in ['drawdown', 'recovery', 'mae', 'mfe']):
            colors.append('#e67e22')  # Dark orange for drawdown
        else:
            colors.append('#95a5a6')  # Gray for technical
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_features['importance'], color=colors)
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features for Predicting Future Returns', fontsize=14, pad=20)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(results_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_feature_importance(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    results_dir: Path,
    n_estimators: int = 100,
    max_depth: int = 10,
    top_n: int = 50
) -> pd.DataFrame:
    """
    Analyze feature importance using Random Forest and create visualization.
    
    Parameters:
    -----------
    train_df : pd.DataFrame
        Training dataframe
    feature_cols : List[str]
        List of feature column names
    results_dir : Path
        Directory to save plots
    n_estimators : int
        Number of trees in Random Forest
    max_depth : int
        Maximum depth of trees
    top_n : int
        Number of top features to display
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    # Prepare target variable: future 1-week returns
    train_sorted = train_df.sort_values(['ticker', 'date']).copy()
    
    # Calculate forward returns (next week's log return)
    train_sorted['next_close'] = train_sorted.groupby('ticker')['close'].shift(-1)
    train_sorted['future_return'] = np.log(train_sorted['next_close'] / train_sorted['close'])
    train_sorted = train_sorted.drop(columns=['next_close'])
    
    # Remove rows with missing future returns
    train_clean = train_sorted.dropna(subset=['future_return']).copy()
    
    # Prepare features and target
    X = train_clean[feature_cols].fillna(0)
    y = train_clean['future_return']
    
    print(f"Training Random Forest for feature importance...")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    
    # Train Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=20,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X, y)
    r2_score = rf_model.score(X, y)
    
    print(f"✓ Model trained. R² score: {r2_score:.4f}")
    
    # Extract feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Display top features
    print("\n" + "=" * 80)
    print(f"TOP {top_n} MOST IMPORTANT FEATURES")
    print("=" * 80)
    print("\nThese features have the highest predictive power for future returns:\n")
    
    for i, row in feature_importance.head(top_n).iterrows():
        # Categorize feature
        feat = row['feature']
        if any(x in feat.lower() for x in ['macro', 'treasury', 'fed', 'vix', 'cpi', 'unemployment', 
                                             'yield_curve', 'dxy', 'oil', 'spread', 'baa', 'pce']):
            category = 'MACRO'
        elif any(x in feat.lower() for x in ['return', 'roc', 'momentum', 'rsi', 'macd', 'stochastic']):
            category = 'MOMENTUM'
        elif any(x in feat.lower() for x in ['volatility', 'atr', 'parkinson']):
            category = 'VOLATILITY'
        elif any(x in feat.lower() for x in ['volume', 'obv', 'mfi', 'chaikin']):
            category = 'VOLUME'
        elif any(x in feat.lower() for x in ['sharpe', 'sortino', 'calmar']):
            category = 'RISK-ADJ'
        elif any(x in feat.lower() for x in ['drawdown', 'recovery', 'mae', 'mfe']):
            category = 'DRAWDOWN'
        else:
            category = 'TECHNICAL'
        
        bar_length = int(row['importance'] * 500)
        bar = '█' * bar_length
        print(f"{i+1:2d}. {feat:35s} [{category:10s}] {bar} {row['importance']:.6f}")
    
    # Visualize top 30 features
    plot_feature_importance(feature_importance, results_dir, top_n=30)
    
    # Category breakdown
    print("\n" + "=" * 80)
    print("IMPORTANCE BY CATEGORY (Top 50 Features)")
    print("=" * 80)
    
    category_importance = {}
    for _, row in feature_importance.head(50).iterrows():
        feat = row['feature']
        if any(x in feat.lower() for x in ['macro', 'treasury', 'fed', 'vix', 'cpi', 'unemployment', 
                                             'yield_curve', 'dxy', 'oil', 'spread', 'baa', 'pce']):
            category = 'Macro Economic'
        elif any(x in feat.lower() for x in ['return', 'roc', 'momentum', 'rsi', 'macd', 'stochastic']):
            category = 'Momentum/Trend'
        elif any(x in feat.lower() for x in ['volatility', 'atr', 'parkinson']):
            category = 'Volatility/Risk'
        elif any(x in feat.lower() for x in ['volume', 'obv', 'mfi', 'chaikin']):
            category = 'Volume/Liquidity'
        elif any(x in feat.lower() for x in ['sharpe', 'sortino', 'calmar']):
            category = 'Risk-Adjusted'
        elif any(x in feat.lower() for x in ['drawdown', 'recovery', 'mae', 'mfe']):
            category = 'Drawdown/Path'
        else:
            category = 'Technical'
        
        if category not in category_importance:
            category_importance[category] = {'count': 0, 'total_importance': 0}
        category_importance[category]['count'] += 1
        category_importance[category]['total_importance'] += row['importance']
    
    for category, stats in sorted(category_importance.items(), key=lambda x: -x[1]['total_importance']):
        avg_importance = stats['total_importance'] / stats['count']
        print(f"{category:20s}: {stats['count']:2d} features, Avg importance: {avg_importance:.6f}")
    
    return feature_importance


# ============================================================================
# OVERFITTING INVESTIGATION VISUALIZATION
# ============================================================================

def plot_learning_curves(
    train_logs: Dict[int, pd.DataFrame],
    val_logs: Dict[int, pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Plot learning curves showing train vs validation Sharpe ratio over training steps.
    
    Parameters:
    -----------
    train_logs : Dict[int, pd.DataFrame]
        Dictionary mapping fold indices to training log DataFrames
        Each DataFrame should have columns: 'step', 'sharpe_ratio'
    val_logs : Dict[int, pd.DataFrame]
        Dictionary mapping fold indices to validation log DataFrames
        Each DataFrame should have columns: 'step', 'sharpe_ratio'
    output_dir : Path
        Directory to save the plot
    """
    n_folds = len(val_logs)
    fig, axes = plt.subplots(n_folds, 1, figsize=(14, 5 * n_folds))
    
    if n_folds == 1:
        axes = [axes]
    
    for idx, fold_idx in enumerate(sorted(val_logs.keys())):
        ax = axes[idx]
        
        # Training data: use rolling average to smooth
        if fold_idx in train_logs:
            train_df = train_logs[fold_idx]
            train_steps = train_df['step'].values
            train_sharpe = train_df['sharpe_ratio'].values
            
            # Rolling average for smoother visualization
            window = max(10, len(train_sharpe) // 20)
            train_sharpe_smooth = pd.Series(train_sharpe).rolling(window=window, center=True).mean()
            
            ax.plot(train_steps, train_sharpe_smooth, 
                    label='Train Sharpe (smoothed)', color='blue', alpha=0.6, linewidth=2)
            ax.scatter(train_steps[::max(1, len(train_steps)//50)], 
                      train_sharpe[::max(1, len(train_steps)//50)],
                      alpha=0.2, s=10, color='blue')
        
        # Validation data
        val_df = val_logs[fold_idx]
        val_steps = val_df['step'].values
        val_sharpe = val_df['sharpe_ratio'].values
        
        ax.plot(val_steps, val_sharpe, 
                label='Validation Sharpe', color='red', marker='o', linewidth=2, markersize=6)
        
        # Mark best validation Sharpe
        best_idx = val_sharpe.argmax()
        best_step = val_steps[best_idx]
        best_sharpe = val_sharpe[best_idx]
        ax.axvline(x=best_step, color='green', linestyle='--', alpha=0.5, 
                   label=f'Best Val Sharpe: {best_sharpe:.3f} at step {best_step:,}')
        ax.scatter([best_step], [best_sharpe], color='green', s=100, zorder=5)
        
        # Check if Sharpe peaked early
        peak_position = best_idx / len(val_sharpe)
        if peak_position < 0.5:
            ax.text(0.02, 0.98, f'WARNING: Early peak at {peak_position*100:.1f}% of training',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Fold {fold_idx}: Learning Curves')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Learning curves saved to: {output_path}")


def analyze_peak_positions(
    val_logs: Dict[int, pd.DataFrame],
    output_dir: Path
) -> pd.DataFrame:
    """
    Analyze when validation Sharpe ratio peaked for each fold and create visualization.
    
    Parameters:
    -----------
    val_logs : Dict[int, pd.DataFrame]
        Dictionary mapping fold indices to validation log DataFrames
        Each DataFrame should have columns: 'step', 'sharpe_ratio'
    output_dir : Path
        Directory to save the plot
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with peak analysis results for each fold
    """
    peak_analysis = []
    
    for fold_idx in sorted(val_logs.keys()):
        val_df = val_logs[fold_idx]
        val_sharpe = val_df['sharpe_ratio'].values
        val_steps = val_df['step'].values
        
        best_idx = val_sharpe.argmax()
        peak_step = val_steps[best_idx]
        peak_sharpe = val_sharpe[best_idx]
        final_sharpe = val_sharpe[-1]
        
        # Calculate peak position as fraction of training
        peak_position = best_idx / len(val_sharpe)
        
        # Calculate decline from peak
        decline_from_peak = peak_sharpe - final_sharpe
        decline_pct = (decline_from_peak / abs(peak_sharpe)) * 100 if peak_sharpe != 0 else 0
        
        peak_analysis.append({
            'fold': fold_idx,
            'peak_step': peak_step,
            'peak_sharpe': peak_sharpe,
            'final_sharpe': final_sharpe,
            'peak_position': peak_position,
            'decline_from_peak': decline_from_peak,
            'decline_pct': decline_pct,
            'n_evaluations': len(val_sharpe)
        })
    
    peak_df = pd.DataFrame(peak_analysis)
    
    # Display results
    print("Validation Sharpe Peak Analysis:")
    print("=" * 80)
    for _, row in peak_df.iterrows():
        print(f"\nFold {row['fold']}:")
        print(f"  Peak Sharpe: {row['peak_sharpe']:.3f} at step {row['peak_step']:,}")
        print(f"  Final Sharpe: {row['final_sharpe']:.3f}")
        print(f"  Peak position: {row['peak_position']*100:.1f}% of training")
        print(f"  Decline from peak: {row['decline_from_peak']:.3f} ({row['decline_pct']:.1f}%)")
        
        if row['peak_position'] < 0.5:
            print(f"  WARNING: Peak occurred in first half of training (potential overfitting)")
        if row['decline_pct'] > 20:
            print(f"  WARNING: Significant decline from peak ({row['decline_pct']:.1f}%)")
    
    # Visualize peak positions
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['red' if p < 0.5 else 'green' for p in peak_df['peak_position']]
    bars = ax.bar(peak_df['fold'], peak_df['peak_position'] * 100, color=colors, alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', label='50% threshold (overfitting risk)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Peak Position (% of training)')
    ax.set_title('When Did Validation Sharpe Peak?')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, pos in zip(bars, peak_df['peak_position']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{pos*100:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / 'peak_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nPeak analysis plot saved to: {output_path}")
    
    return peak_df


def plot_train_val_comparison(
    train_logs: Dict[int, pd.DataFrame],
    val_logs: Dict[int, pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Plot train vs validation returns and Sharpe ratio comparisons for all folds.
    
    Creates side-by-side plots for each fold showing:
    1. Returns comparison (train scatter vs validation line)
    2. Sharpe ratio comparison (train scatter vs validation line)
    
    Also prints statistical summary of train vs validation performance gaps.
    
    Parameters:
    -----------
    train_logs : Dict[int, pd.DataFrame]
        Dictionary mapping fold indices to training log DataFrames
        Each DataFrame should have columns: 'step', 'total_return', 'sharpe_ratio'
    val_logs : Dict[int, pd.DataFrame]
        Dictionary mapping fold indices to validation log DataFrames
        Each DataFrame should have columns: 'step', 'total_return', 'sharpe_ratio'
    output_dir : Path
        Directory to save the plot
    """
    n_folds = len(val_logs)
    fig, axes = plt.subplots(n_folds, 2, figsize=(16, 5 * n_folds))
    
    if n_folds == 1:
        axes = axes.reshape(1, -1)
    
    for idx, fold_idx in enumerate(sorted(val_logs.keys())):
        # Returns comparison
        ax_ret = axes[idx, 0]
        
        if fold_idx in train_logs:
            train_df = train_logs[fold_idx]
            # Sample training episodes for visualization (too many points)
            train_sample = train_df.sample(min(200, len(train_df)))
            ax_ret.scatter(train_sample['step'], train_sample['total_return'] * 100,
                          alpha=0.3, s=20, label='Train Returns', color='blue')
        
        val_df = val_logs[fold_idx]
        ax_ret.plot(val_df['step'], val_df['total_return'] * 100,
                   marker='o', label='Validation Returns', color='red', linewidth=2, markersize=6)
        
        ax_ret.set_xlabel('Training Step')
        ax_ret.set_ylabel('Total Return (%)')
        ax_ret.set_title(f'Fold {fold_idx}: Returns Comparison')
        ax_ret.legend()
        ax_ret.grid(True, alpha=0.3)
        
        # Sharpe comparison
        ax_sharpe = axes[idx, 1]
        
        if fold_idx in train_logs:
            train_df = train_logs[fold_idx]
            train_sample = train_df.sample(min(200, len(train_df)))
            ax_sharpe.scatter(train_sample['step'], train_sample['sharpe_ratio'],
                             alpha=0.3, s=20, label='Train Sharpe', color='blue')
        
        ax_sharpe.plot(val_df['step'], val_df['sharpe_ratio'],
                      marker='o', label='Validation Sharpe', color='red', linewidth=2, markersize=6)
        
        ax_sharpe.set_xlabel('Training Step')
        ax_sharpe.set_ylabel('Sharpe Ratio')
        ax_sharpe.set_title(f'Fold {fold_idx}: Sharpe Comparison')
        ax_sharpe.legend()
        ax_sharpe.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'train_val_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("\nTrain vs Validation Returns Summary:")
    print("=" * 80)
    for fold_idx in sorted(val_logs.keys()):
        val_df = val_logs[fold_idx]
        val_mean_return = val_df['total_return'].mean() * 100
        val_std_return = val_df['total_return'].std() * 100
        
        if fold_idx in train_logs:
            train_df = train_logs[fold_idx]
            train_mean_return = train_df['total_return'].mean() * 100
            train_std_return = train_df['total_return'].std() * 100
            
            gap = train_mean_return - val_mean_return
            print(f"\nFold {fold_idx}:")
            print(f"  Train Return: {train_mean_return:.2f}% ± {train_std_return:.2f}%")
            print(f"  Val Return:   {val_mean_return:.2f}% ± {val_std_return:.2f}%")
            print(f"  Gap: {gap:.2f}%")
            
            if gap > 20:
                print(f"  WARNING: Large gap suggests overfitting")
        else:
            print(f"\nFold {fold_idx}:")
            print(f"  Val Return: {val_mean_return:.2f}% ± {val_std_return:.2f}%")
    
    print(f"\nTrain vs validation comparison plot saved to: {output_path}")


def plot_validation_test_gap(
    fold_results: pd.DataFrame,
    test_sharpe: float,
    tested_fold_idx: int,
    output_dir: Path
) -> None:
    """
    Plot validation vs test performance gap to visualize overfitting.
    
    Creates two plots:
    1. Sharpe ratio comparison across all folds (validation vs test)
    2. Performance degradation waterfall for the tested fold
    
    Parameters:
    -----------
    fold_results : pd.DataFrame
        DataFrame with columns: 'fold_idx', 'val_sharpe_ratio', 'best_val_sharpe'
    test_sharpe : float
        Test Sharpe ratio for the tested fold
    tested_fold_idx : int
        Index of the fold that was tested (typically 2)
    output_dir : Path
        Directory to save the plot
    """
    # Extract validation metrics for tested fold
    fold_val_sharpe = fold_results[fold_results['fold_idx'] == tested_fold_idx]['val_sharpe_ratio'].values[0]
    fold_best_sharpe = fold_results[fold_results['fold_idx'] == tested_fold_idx]['best_val_sharpe'].values[0]
    
    # Calculate gap
    gap_from_best = ((test_sharpe - fold_best_sharpe) / fold_best_sharpe) * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ============================================================================
    # Plot 1: Sharpe Ratio Comparison
    # ============================================================================
    ax1 = axes[0]
    
    # Data for all folds
    folds = fold_results['fold_idx'].values
    val_sharpes = fold_results['val_sharpe_ratio'].values
    best_sharpes = fold_results['best_val_sharpe'].values
    
    x_pos = np.arange(len(folds))
    width = 0.25
    
    # Validation bars
    bars1 = ax1.bar(x_pos - width, val_sharpes, width, 
                    label='Final Validation Sharpe', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x_pos, best_sharpes, width,
                    label='Best Validation Sharpe', color='green', alpha=0.8)
    
    # Test performance (only for tested fold)
    test_values = [np.nan if i != tested_fold_idx else test_sharpe for i in range(len(folds))]
    bars3 = ax1.bar(x_pos + width, test_values, width,
                    label='Test Sharpe (2024)', color='red', alpha=0.8)
    
    # Styling
    ax1.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Validation vs Test Performance\n(Overfitting Evidence)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Fold {int(i)}' for i in folds])
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=9)
    
    # Highlight the overfitting gap for tested fold
    tested_fold_pos = np.where(folds == tested_fold_idx)[0][0]
    ax1.annotate('', xy=(tested_fold_pos + width, test_sharpe), 
                 xytext=(tested_fold_pos, fold_best_sharpe),
                 arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    gap_text = f'{gap_from_best:.0f}% drop'
    ax1.text(tested_fold_pos + width/2, (test_sharpe + fold_best_sharpe)/2, gap_text,
             ha='center', va='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # ============================================================================
    # Plot 2: Performance Gap Waterfall
    # ============================================================================
    ax2 = axes[1]
    
    # Create waterfall data for tested fold
    stages = ['Best\nValidation\nSharpe', 'Final\nValidation\nSharpe', 'Test\nSharpe\n(2024)']
    values = [fold_best_sharpe, fold_val_sharpe, test_sharpe]
    colors = ['green', 'orange', 'red']
    
    bars = ax2.bar(stages, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add change arrows
    # From best to final val
    ax2.annotate('', xy=(1, fold_val_sharpe), xytext=(0, fold_val_sharpe),
                 arrowprops=dict(arrowstyle='->', color='orange', lw=2, linestyle='--'))
    decline1 = fold_best_sharpe - fold_val_sharpe
    ax2.text(0.5, fold_val_sharpe + 0.1, f'-{decline1:.2f}',
             ha='center', fontsize=10, color='orange', fontweight='bold')
    
    # From final val to test
    ax2.annotate('', xy=(2, test_sharpe), xytext=(1, test_sharpe),
                 arrowprops=dict(arrowstyle='->', color='red', lw=2, linestyle='--'))
    decline2 = fold_val_sharpe - test_sharpe
    ax2.text(1.5, test_sharpe + 0.1, f'-{decline2:.2f}',
             ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax2.set_title(f'Fold {tested_fold_idx}: Performance Degradation\n(Best Val → Final Val → Test)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, fold_best_sharpe * 1.2])
    
    # Add warning box
    warning_text = f"WARNING: OVERFITTING DETECTED\n{gap_from_best:.0f}% performance drop\nfrom best validation to test"
    ax2.text(0.5, 0.95, warning_text,
             transform=ax2.transAxes,
             fontsize=11, fontweight='bold',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    plt.tight_layout()
    output_path = output_dir / 'validation_test_gap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nPlot saved to: {output_path}")


# ============================================================================
# MULTI-MODEL COMPARISON VISUALIZATION
# ============================================================================

def plot_multi_model_learning_curves(
    models_data: Dict[str, Dict[str, Dict[int, pd.DataFrame]]],
    output_dir: Path
) -> None:
    """
    Plot learning curves for multiple models across all folds.

    Parameters:
    -----------
    models_data : Dict[str, Dict[str, Dict[int, pd.DataFrame]]]
        Nested dictionary: {model_name: {'train': {fold: df}, 'val': {fold: df}}}
    output_dir : Path
        Directory to save the plot
    """
    n_models = len(models_data)
    n_folds = max(len(data['val']) for data in models_data.values())

    fig, axes = plt.subplots(n_models, n_folds, figsize=(6 * n_folds, 5 * n_models))

    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_folds == 1:
        axes = axes.reshape(-1, 1)

    for model_idx, (model_name, data) in enumerate(sorted(models_data.items())):
        train_logs = data.get('train', {})
        val_logs = data.get('val', {})

        for fold_idx in range(n_folds):
            ax = axes[model_idx, fold_idx]

            # Training data
            if fold_idx in train_logs:
                train_df = train_logs[fold_idx]
                train_steps = train_df['step'].values
                train_sharpe = train_df['sharpe_ratio'].values

                window = max(10, len(train_sharpe) // 20)
                train_sharpe_smooth = pd.Series(train_sharpe).rolling(window=window, center=True).mean()

                ax.plot(train_steps, train_sharpe_smooth,
                        label='Train', color='blue', alpha=0.6, linewidth=2)

            # Validation data
            if fold_idx in val_logs:
                val_df = val_logs[fold_idx]
                val_steps = val_df['step'].values
                val_sharpe = val_df['sharpe_ratio'].values

                ax.plot(val_steps, val_sharpe,
                        label='Validation', color='red', marker='o', linewidth=2, markersize=4)

                # Mark best
                best_idx = val_sharpe.argmax()
                best_step = val_steps[best_idx]
                best_sharpe = val_sharpe[best_idx]
                ax.axvline(x=best_step, color='green', linestyle='--', alpha=0.5)
                ax.scatter([best_step], [best_sharpe], color='green', s=100, zorder=5)

            ax.set_xlabel('Training Step', fontsize=9)
            ax.set_ylabel('Sharpe Ratio', fontsize=9)
            ax.set_title(f'{model_name.upper()} | Fold {fold_idx}', fontsize=10, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.suptitle('Learning Curves: All Models & Folds', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / 'multi_model_learning_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Multi-model learning curves saved to: {output_path}")


def compare_overfitting_metrics(
    models_peak_analysis: Dict[str, pd.DataFrame],
    output_dir: Path
) -> None:
    """
    Compare overfitting metrics across multiple models.

    Parameters:
    -----------
    models_peak_analysis : Dict[str, pd.DataFrame]
        Dictionary mapping model names to their peak analysis DataFrames
    output_dir : Path
        Directory to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data
    model_names = list(models_peak_analysis.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    # Plot 1: Average Peak Position
    ax1 = axes[0, 0]
    avg_peaks = [df['peak_position'].mean() for df in models_peak_analysis.values()]
    bars = ax1.bar(model_names, np.array(avg_peaks) * 100, color=colors, alpha=0.8, edgecolor='black')
    ax1.axhline(y=50, color='red', linestyle='--', linewidth=2, label='50% threshold')
    ax1.set_ylabel('Average Peak Position (%)', fontsize=12, fontweight='bold')
    ax1.set_title('When Does Validation Sharpe Peak?\n(Earlier = More Overfitting)',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_peaks):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{val*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 2: Average Decline from Peak
    ax2 = axes[0, 1]
    avg_declines = [df['decline_pct'].mean() for df in models_peak_analysis.values()]
    bars = ax2.bar(model_names, avg_declines, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=20, color='red', linestyle='--', linewidth=2, label='20% threshold')
    ax2.set_ylabel('Average Decline from Peak (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Sharpe Decline\n(Higher = More Unstable)',
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, avg_declines):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Peak Position Distribution
    ax3 = axes[1, 0]
    for model_name, color in zip(model_names, colors):
        df = models_peak_analysis[model_name]
        ax3.scatter(df['fold'], df['peak_position'] * 100,
                   label=model_name.upper(), s=150, alpha=0.7, color=color, edgecolors='black', linewidths=2)
    ax3.axhline(y=50, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Peak Position (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Peak Position by Fold', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Best vs Final Sharpe
    ax4 = axes[1, 1]
    x = np.arange(len(model_names))
    width = 0.35

    best_sharpes = [df['peak_sharpe'].mean() for df in models_peak_analysis.values()]
    final_sharpes = [df['final_sharpe'].mean() for df in models_peak_analysis.values()]

    bars1 = ax4.bar(x - width/2, best_sharpes, width, label='Best Val Sharpe',
                    color='green', alpha=0.7, edgecolor='black')
    bars2 = ax4.bar(x + width/2, final_sharpes, width, label='Final Val Sharpe',
                    color='orange', alpha=0.7, edgecolor='black')

    ax4.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax4.set_title('Best vs Final Validation Sharpe', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / 'overfitting_metrics_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Overfitting metrics comparison saved to: {output_path}")


def plot_multi_model_val_test_gap(
    models_results: Dict[str, Dict],
    output_dir: Path
) -> None:
    """
    Plot validation vs test performance gap for multiple models.

    Parameters:
    -----------
    models_results : Dict[str, Dict]
        Dictionary mapping model names to their results:
        {model_name: {'fold_results': df, 'test_sharpe': float, 'tested_fold': int}}
    output_dir : Path
        Directory to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Prepare data
    model_names = list(models_results.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

    # Plot 1: Validation vs Test Sharpe
    ax1 = axes[0]

    x = np.arange(len(model_names))
    width = 0.25

    best_val_sharpes = []
    final_val_sharpes = []
    test_sharpes = []

    for model_name in model_names:
        data = models_results[model_name]
        fold_results = data['fold_results']
        tested_fold = data['tested_fold']

        best_val = fold_results[fold_results['fold_idx'] == tested_fold]['best_val_sharpe'].values[0]
        final_val = fold_results[fold_results['fold_idx'] == tested_fold]['val_sharpe_ratio'].values[0]
        test = data['test_sharpe']

        best_val_sharpes.append(best_val)
        final_val_sharpes.append(final_val)
        test_sharpes.append(test)

    bars1 = ax1.bar(x - width, best_val_sharpes, width, label='Best Val Sharpe',
                    color='green', alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x, final_val_sharpes, width, label='Final Val Sharpe',
                    color='orange', alpha=0.7, edgecolor='black')
    bars3 = ax1.bar(x + width, test_sharpes, width, label='Test Sharpe',
                    color='red', alpha=0.7, edgecolor='black')

    ax1.set_ylabel('Sharpe Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Validation vs Test Performance\nAcross All Models',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Performance Degradation %
    ax2 = axes[1]

    gaps_from_best = []
    gaps_from_final = []

    for i, model_name in enumerate(model_names):
        best_val = best_val_sharpes[i]
        final_val = final_val_sharpes[i]
        test = test_sharpes[i]

        gap_best = ((test - best_val) / best_val) * 100 if best_val != 0 else 0
        gap_final = ((test - final_val) / final_val) * 100 if final_val != 0 else 0

        gaps_from_best.append(gap_best)
        gaps_from_final.append(gap_final)

    bars1 = ax2.bar(x - width/2, gaps_from_best, width, label='From Best Val',
                    color='red', alpha=0.7, edgecolor='black')
    bars2 = ax2.bar(x + width/2, gaps_from_final, width, label='From Final Val',
                    color='orange', alpha=0.7, edgecolor='black')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=-30, color='red', linestyle='--', linewidth=2, alpha=0.5, label='-30% threshold')

    ax2.set_ylabel('Performance Degradation (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Val-to-Test Performance Drop\n(Lower = More Overfitting)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([name.upper() for name in model_names], rotation=15, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height - 2 if height < 0 else height + 1,
                    f'{height:.0f}%', ha='center', va='top' if height < 0 else 'bottom',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = output_dir / 'multi_model_val_test_gap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Multi-model val-test gap comparison saved to: {output_path}")


def create_overfitting_summary_table(
    models_peak_analysis: Dict[str, pd.DataFrame],
    models_test_results: Dict[str, Dict],
    output_dir: Path
) -> pd.DataFrame:
    """
    Create comprehensive overfitting summary table for all models.

    Parameters:
    -----------
    models_peak_analysis : Dict[str, pd.DataFrame]
        Dictionary mapping model names to peak analysis DataFrames
    models_test_results : Dict[str, Dict]
        Dictionary with test results for each model
    output_dir : Path
        Directory to save the table

    Returns:
    --------
    pd.DataFrame
        Summary table with overfitting metrics
    """
    summary_data = []

    for model_name in sorted(models_peak_analysis.keys()):
        peak_df = models_peak_analysis[model_name]
        test_data = models_test_results.get(model_name, {})

        # Average metrics across folds
        avg_peak_pos = peak_df['peak_position'].mean()
        avg_decline = peak_df['decline_pct'].mean()
        avg_best_sharpe = peak_df['peak_sharpe'].mean()
        avg_final_sharpe = peak_df['final_sharpe'].mean()

        # Test performance
        test_sharpe = test_data.get('test_sharpe', np.nan)
        tested_fold = test_data.get('tested_fold', np.nan)

        # Calculate gaps
        if not np.isnan(test_sharpe) and not np.isnan(tested_fold):
            fold_best = peak_df[peak_df['fold'] == tested_fold]['peak_sharpe'].values[0]
            gap_pct = ((test_sharpe - fold_best) / fold_best) * 100 if fold_best != 0 else np.nan
        else:
            gap_pct = np.nan

        # Overfitting risk assessment
        risk_signals = []
        if avg_peak_pos < 0.5:
            risk_signals.append('Early Peak')
        if avg_decline > 20:
            risk_signals.append('Large Decline')
        if not np.isnan(gap_pct) and gap_pct < -30:
            risk_signals.append('Val-Test Gap')

        risk_level = 'HIGH' if len(risk_signals) >= 2 else 'MEDIUM' if len(risk_signals) == 1 else 'LOW'

        summary_data.append({
            'Model': model_name.upper(),
            'Avg Peak Position': f'{avg_peak_pos*100:.1f}%',
            'Avg Decline from Peak': f'{avg_decline:.1f}%',
            'Avg Best Val Sharpe': f'{avg_best_sharpe:.3f}',
            'Avg Final Val Sharpe': f'{avg_final_sharpe:.3f}',
            'Test Sharpe': f'{test_sharpe:.3f}' if not np.isnan(test_sharpe) else 'N/A',
            'Val-Test Gap': f'{gap_pct:.0f}%' if not np.isnan(gap_pct) else 'N/A',
            'Risk Signals': ', '.join(risk_signals) if risk_signals else 'None',
            'Overfitting Risk': risk_level
        })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_path = output_dir / 'overfitting_summary_table.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"\nOverfitting summary table saved to: {output_path}")

    return summary_df

