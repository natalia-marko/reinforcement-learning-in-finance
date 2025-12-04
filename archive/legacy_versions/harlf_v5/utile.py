import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================================
# EARLY STOPPING CALLBACK
# ============================================================================

class EarlyStoppingCallback(BaseCallback):
    """Early stopping with better diagnostics"""
    
    def __init__(self, val_env, eval_freq=2000, patience=10,
                 min_delta=0.01, min_steps=10000, save_path=None, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.min_steps = min_steps
        self.save_path = save_path
        self.best_val_sharpe = -np.inf
        self.wait = 0
        self.val_sharpes = []
        self.eval_steps = []
        self.best_model_path = None
    
    def _on_step(self):
        if self.n_calls % self.eval_freq != 0:
            return True
        
        if self.n_calls < self.min_steps:
            return True
        
        obs, _ = self.val_env.reset()
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _, _ = self.val_env.step(action)
        
        val_metrics = self.val_env.get_portfolio_metrics()
        val_sharpe = val_metrics['sharpe_ratio']
        
        self.val_sharpes.append(val_sharpe)
        self.eval_steps.append(self.n_calls)
        
        if self.verbose:
            print(f"  Step {self.n_calls}: Val Sharpe = {val_sharpe:.3f}")
        
        if val_sharpe > self.best_val_sharpe + self.min_delta:
            self.best_val_sharpe = val_sharpe
            self.wait = 0
            
            if self.save_path:
                self.best_model_path = f"{self.save_path}_best"
                self.model.save(self.best_model_path)
                if self.verbose:
                    print(f"New best! (Sharpe: {val_sharpe:.3f})")
        else:
            self.wait += 1
            if self.verbose:
                print(f"No improvement ({self.wait}/{self.patience})")
        
        if self.wait >= self.patience:
            if self.verbose:
                print(f"\n  Early stopping at step {self.n_calls}")
            return False
        
        return True




# ============================================================================
# NEW: SIMPLE 70/20/10 DATA SPLIT (replaces WalkForwardValidator)
# ============================================================================

def create_train_val_test_split(price_data, technical_features, sentiment_features, regime_indicators):
    """Create standard 70/20/10 train/validation/test split"""
    
    print("\n" + "="*70)
    print("DATA SPLIT: 70% TRAIN / 20% VALIDATION / 10% TEST")
    print("="*70)
    
    data_length = len(price_data)
    dates = price_data.index
    
    # Calculate split indices
    train_end = int(data_length * 0.70)
    val_end = int(data_length * 0.90)
    
    print(f"\nTotal data: {data_length} months")
    print(f"Date range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    
    print(f"\nTrain Set (70%):")
    print(f"  Dates: {dates[0].strftime('%Y-%m')} to {dates[train_end-1].strftime('%Y-%m')}")
    print(f"  Size:  {train_end} months")
    
    print(f"\nValidation Set (20%):")
    print(f"  Dates: {dates[train_end].strftime('%Y-%m')} to {dates[val_end-1].strftime('%Y-%m')}")
    print(f"  Size:  {val_end - train_end} months")
    
    print(f"\nTest Set (10%):")
    print(f"  Dates: {dates[val_end].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
    print(f"  Size:  {data_length - val_end} months")
    
    # Create splits
    train_data = (
        price_data.iloc[:train_end],
        technical_features.iloc[:train_end],
        sentiment_features.iloc[:train_end],
        regime_indicators.iloc[:train_end] if regime_indicators is not None else None
    )
    
    val_data = (
        price_data.iloc[train_end:val_end],
        technical_features.iloc[train_end:val_end],
        sentiment_features.iloc[train_end:val_end],
        regime_indicators.iloc[train_end:val_end] if regime_indicators is not None else None
    )
    
    test_data = (
        price_data.iloc[val_end:],
        technical_features.iloc[val_end:],
        sentiment_features.iloc[val_end:],
        regime_indicators.iloc[val_end:] if regime_indicators is not None else None
    )
    
    print("\n" + "="*70)
    print("Data split created successfully!")
    print("="*70)
    
    return train_data, val_data, test_data



def plot_performance_comparison(test_prices, all_portfolio_histories, qqq_benchmark=None, save_path='results/performance_comparison.png'):
    """
    Plot cumulative returns comparison for all strategies
    
    Args:
        test_prices: Test price data for date alignment
        all_portfolio_histories: Dict of {strategy_name: portfolio_history_list}
        qqq_benchmark: Optional QQQ benchmark data
        save_path: Where to save the plot
    """
    
    print("\n   Creating performance comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Convert to cumulative returns
    colors = {
        'QQQ Buy & Hold': '#FFA500',
        'Best Base (sent_PPO)': '#4169E1', 
        'Enhanced Super (SAC+Full)': '#32CD32',
        'Meta Agent': '#DC143C'
    }
    
    # Plot 1: Cumulative Returns
    for strategy_name, portfolio_history in all_portfolio_histories.items():
        portfolio_series = pd.Series(portfolio_history, index=test_prices.index[:len(portfolio_history)])
        cumulative_returns = (portfolio_series / portfolio_series.iloc[0] - 1) * 100
        
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                label=strategy_name, linewidth=2, color=colors.get(strategy_name, None))
    
    ax1.set_title('Cumulative Returns Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    # Plot 2: Drawdown
    for strategy_name, portfolio_history in all_portfolio_histories.items():
        portfolio_series = pd.Series(portfolio_history, index=test_prices.index[:len(portfolio_history)])
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max * 100
        
        ax2.plot(drawdown.index, drawdown.values, 
                label=strategy_name, linewidth=2, color=colors.get(strategy_name, None))
    
    ax2.set_title('Drawdown Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.fill_between(ax2.get_xlim(), 0, -100, alpha=0.1, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()


def plot_portfolio_allocations(all_final_weights, asset_names, save_path='results/portfolio_allocations.png'):
    """
    Plot final portfolio allocations for all strategies
    
    Args:
        all_final_weights: Dict of {strategy_name: weights_array}
        asset_names: List of asset names
        save_path: Where to save the plot
    """
    
    print("\n   Creating portfolio allocation plot...")
    
    n_strategies = len(all_final_weights)
    fig, axes = plt.subplots(1, n_strategies, figsize=(5*n_strategies, 6))
    
    if n_strategies == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(range(len(asset_names)))
    
    for idx, (strategy_name, weights) in enumerate(all_final_weights.items()):
        ax = axes[idx]
        
        # Create bar chart
        bars = ax.bar(range(len(weights)), weights * 100, color=colors)
        ax.set_title(f'{strategy_name}\nAllocation', fontsize=12, fontweight='bold')
        ax.set_xlabel('Assets', fontsize=10)
        ax.set_ylabel('Allocation (%)', fontsize=10)
        ax.set_xticks(range(len(asset_names)))
        ax.set_xticklabels(asset_names, rotation=45, ha='right')
        ax.set_ylim(0, max(weights) * 110)  # 10% padding
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:  # Only show if > 0.5%
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
        
        # Add HHI (concentration) metric
        hhi = np.sum(weights ** 2)
        ax.text(0.02, 0.98, f'HHI: {hhi:.3f}', 
               transform=ax.transAxes, fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
    plt.close()

()