import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import BaseCallback
from sentiment_enviroment import SentimentEnv
from technical_enviroment import TechnicalEnv
from custom_function import split_data_chronologically


class EarlyStoppingCallback(BaseCallback):
    """
    Enhanced early stopping callback with best model saving.
    
    Args:
        val_env: Validation environment for evaluation
        eval_freq: Frequency (in steps) to evaluate on validation set
        patience: Number of evaluations without improvement before stopping
        min_delta: Minimum change in Sharpe to be considered improvement
        save_path: Directory to save the best model
        verbose: Verbosity level (0: silent, 1: print updates)
    """
    def __init__(self, val_env, eval_freq=1000, patience=5, 
                 min_delta=0.0, save_path='./models/', verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_sharpe = -np.inf
        self.wait = 0
        self.val_sharpes = []
        self.eval_steps = []
        self.stopped_step = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        
        # Evaluate on validation set
        metrics = evaluate_agent(self.model, self.val_env, phase="Validation", verbose=False)
        val_sharpe = metrics['sharpe_ratio']
        self.val_sharpes.append(val_sharpe)
        self.eval_steps.append(self.n_calls)
        
        if self.verbose > 0:
            print(f"Step {self.n_calls:>6d}: Val Sharpe = {val_sharpe:>6.3f} | "
                  f"Best = {self.best_sharpe:>6.3f} | Wait = {self.wait}/{self.patience}")

        # Check for improvement
        if val_sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = val_sharpe
            self.wait = 0
            
            # Save best model
            model_path = os.path.join(self.save_path, 'best_model')
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"  ✓ New best model saved! Sharpe = {val_sharpe:.3f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"\n⚠ Early stopping triggered at step {self.n_calls}")
                    print(f"  Best validation Sharpe: {self.best_sharpe:.3f}")
                    print(f"  No improvement for {self.patience} evaluations")
                self.stopped_step = self.n_calls
                return False  # Stop training
        
        return True

    def plot_curves(self, agent_name):
        """Plot validation Sharpe ratio over training steps"""
        if len(self.val_sharpes) == 0:
            print(f"No validation data to plot for {agent_name}")
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(self.eval_steps, self.val_sharpes, label='Validation Sharpe', 
                marker='o', linewidth=2, markersize=6)
        
        # Mark best point
        best_idx = np.argmax(self.val_sharpes)
        plt.scatter(self.eval_steps[best_idx], self.val_sharpes[best_idx], 
                   color='red', s=200, marker='*', zorder=5, 
                   label=f'Best: {self.val_sharpes[best_idx]:.3f}')
        
        # Mark early stopping point if triggered
        if self.stopped_step > 0:
            plt.axvline(x=self.stopped_step, color='red', linestyle='--', 
                       alpha=0.5, label=f'Early Stop: {self.stopped_step}')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Validation Performance: {agent_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = os.path.join(self.save_path, f'{agent_name}_validation_curve.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Plot saved: {plot_path}")


class AgentWrapper:
    """Wrapper to add weights attribute and algorithm info to agents"""
    def __init__(self, model, env, algorithm_name, agent_type):
        self.model = model
        self.env = env
        self.algorithm = algorithm_name  # 'PPO' or 'SAC'
        self.agent_type = agent_type  # 'sentiment' or 'technical'
        n_assets = len(env.price_data.columns)
        self.weights = np.ones(n_assets) / n_assets
    
    def predict(self, obs, deterministic=True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        action = np.clip(action, 0, 1)
        total = action.sum()
        if total > 1e-6:
            self.weights = action / total
        return action, state
    
    def reset(self, seed=None):
        return self.env.reset(seed=seed)
    
    def save(self, path):
        self.model.save(path)
    
    def __str__(self):
        return f"{self.agent_type.title()} {self.algorithm} Agent"


def get_policy_kwargs(small_network=True):
    """
    Get policy network configuration for regularization.
    
    Args:
        small_network: If True, use smaller network (less overfitting)
    """
    if small_network:
        # Smaller network = less capacity to overfit
        net_arch = [dict(pi=[128, 128], vf=[128, 128])]
    else:
        # Default larger network
        net_arch = [dict(pi=[256, 256], vf=[256, 256])]
    
    policy_kwargs = dict(
        net_arch=net_arch,
        activation_fn=torch.nn.ReLU,
        ortho_init=True,  # Better weight initialization
    )
    
    return policy_kwargs


def train_agent(env, val_env, agent_type, algorithm, timesteps=30000, 
                ppo_params=None, sac_params=None, save_path='./models/'): 
    """
    Train a single agent with anti-overfitting measures.
    
    Returns:
        best_model: The model with best validation performance
        callback: The callback containing training history
    """
    
    print(f"\n{'='*70}")
    print(f"Training {agent_type} with {algorithm}")
    print(f"{'='*70}")
    
    # Get regularized network architecture
    policy_kwargs = get_policy_kwargs(small_network=True)
    
    # Merge policy_kwargs with user parameters
    if algorithm == 'PPO':
        params = ppo_params or {}
        params['policy_kwargs'] = policy_kwargs
        model = PPO("MlpPolicy", env, **params)
    else:  # SAC
        params = sac_params or {}
        params['policy_kwargs'] = policy_kwargs
        model = SAC("MlpPolicy", env, **params)
    
    # Create callback with best model saving
    agent_save_path = os.path.join(save_path, f'{agent_type}_{algorithm}')
    os.makedirs(agent_save_path, exist_ok=True)
    
    callback = EarlyStoppingCallback(
        val_env, 
        eval_freq=1000,  # Evaluate every 1k steps
        patience=5,  # Stop if no improvement for 5 evaluations
        min_delta=0.0,  # Any improvement counts
        save_path=agent_save_path,
        verbose=1
    )
    
    # Train with callback
    print(f"\nStarting training for up to {timesteps:,} steps...")
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    
    # Plot validation curves
    callback.plot_curves(f"{agent_type}_{algorithm}")
    
    # Load best model (from validation, not final checkpoint!)
    best_model_path = os.path.join(agent_save_path, 'best_model')
    if os.path.exists(best_model_path + '.zip'):
        print(f"\n✓ Loading best model from validation...")
        if algorithm == 'PPO':
            best_model = PPO.load(best_model_path, env=env)
        else:
            best_model = SAC.load(best_model_path, env=env)
        print(f"  Best validation Sharpe: {callback.best_sharpe:.3f}")
    else:
        print(f"\n⚠ Best model not found, using final model")
        best_model = model
    
    return best_model, callback


def evaluate_agent(model, env, phase="Test", verbose=True):
    """Evaluate agent on given environment"""
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    metrics = env.get_portfolio_metrics()
    
    if verbose:
        print(f"\n{phase} Results:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:>8.3f}")
        print(f"  Total Return: {metrics['total_return']:>8.2%}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:>8.2%}")
        print(f"  Win Rate:     {metrics['win_rate']:>8.2%}")
    
    return metrics


def train_and_evaluate_with_split(price_data, technical_features, sentiment_features,
                                  train_ratio=0.60, val_ratio=0.20,
                                  timesteps=30000, algorithm='ppo',
                                  sentiment_env_params=None, technical_env_params=None,
                                  ppo_params=None, sac_params=None,
                                  save_path='./models/'):
    """
    Complete workflow with proper train/val/test split and anti-overfitting measures.
    
    Args:
        algorithm: 'ppo', 'sac', or 'both'
        timesteps: Max training steps (will stop early if overfitting detected)
        sentiment_env_params: dict of parameters for SentimentEnv
        technical_env_params: dict of parameters for TechnicalEnv
        ppo_params: dict of PPO hyperparameters (with anti-overfitting defaults)
        sac_params: dict of SAC hyperparameters (with anti-overfitting defaults)
        save_path: Directory to save models and plots
    """
    
    # Default environment parameters with higher transaction costs
    sentiment_env_params = sentiment_env_params or {'transaction_cost': 0.002}
    technical_env_params = technical_env_params or {'transaction_cost': 0.002}
    
    # Create save directory
    os.makedirs(save_path, exist_ok=True)
    
    # Step 1: Split data
    splits = split_data_chronologically(
        price_data, technical_features, sentiment_features,
        train_ratio, val_ratio
    )
    
    train_prices, train_technical, train_sentiment = splits['train']
    val_prices, val_technical, val_sentiment = splits['val']
    test_prices, test_technical, test_sentiment = splits['test']
    
    print(f"\nData Split:")
    print(f"  Training:   {len(train_prices)} months")
    print(f"  Validation: {len(val_prices)} months")
    print(f"  Test:       {len(test_prices)} months")
    
    results = {}
    training_histories = {}
    
    # Algorithms to train
    algorithms = ['PPO', 'SAC'] if algorithm == 'both' else [algorithm.upper()]
    
    # Step 2: Train Sentiment Agent
    print("\n" + "="*70)
    print("TRAINING SENTIMENT AGENTS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n[Sentiment {algo}]")
        
        # Create environments
        sent_train_env = SentimentEnv(train_prices, train_sentiment, **sentiment_env_params)
        sent_val_env = SentimentEnv(val_prices, val_sentiment, **sentiment_env_params)
        
        # Train with best model selection
        sent_model, callback = train_agent(
            sent_train_env, sent_val_env, "Sentiment", algo, 
            timesteps, ppo_params, sac_params, save_path
        )
        
        # Store training history
        training_histories[f'sentiment_{algo.lower()}'] = callback
        
        # Evaluate on all sets
        print(f"\n{'='*70}")
        print(f"Evaluating Sentiment {algo} on all datasets...")
        print(f"{'='*70}")
        
        train_metrics = evaluate_agent(sent_model, sent_train_env, "Training")
        val_metrics = evaluate_agent(sent_model, sent_val_env, "Validation")
        
        sent_test_env = SentimentEnv(test_prices, test_sentiment, **sentiment_env_params)
        test_metrics = evaluate_agent(sent_model, sent_test_env, "Test")
        
        results[f'sentiment_{algo.lower()}'] = {
            'model': sent_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_env': sent_train_env,
            'val_env': sent_val_env,
            'test_env': sent_test_env
        }
    
    # Step 3: Train Technical Agent
    print("\n" + "="*70)
    print("TRAINING TECHNICAL AGENTS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n[Technical {algo}]")
        
        # Create environments
        tech_train_env = TechnicalEnv(train_prices, train_technical, **technical_env_params)
        tech_val_env = TechnicalEnv(val_prices, val_technical, **technical_env_params)
        
        # Train with best model selection
        tech_model, callback = train_agent(
            tech_train_env, tech_val_env, "Technical", algo, 
            timesteps, ppo_params, sac_params, save_path
        )
        
        # Store training history
        training_histories[f'technical_{algo.lower()}'] = callback
        
        # Evaluate on all sets
        print(f"\n{'='*70}")
        print(f"Evaluating Technical {algo} on all datasets...")
        print(f"{'='*70}")
        
        train_metrics = evaluate_agent(tech_model, tech_train_env, "Training")
        val_metrics = evaluate_agent(tech_model, tech_val_env, "Validation")
        
        tech_test_env = TechnicalEnv(test_prices, test_technical, **technical_env_params)
        test_metrics = evaluate_agent(tech_model, tech_test_env, "Test")
        
        results[f'technical_{algo.lower()}'] = {
            'model': tech_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'train_env': tech_train_env,
            'val_env': tech_val_env,
            'test_env': tech_test_env
        }
    
    # Step 4: Summary with Generalization Analysis
    print("\n" + "="*70)
    print("FINAL COMPARISON - TRAIN vs VALIDATION vs TEST")
    print("="*70)
    
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Agent': name.replace('_', ' ').title(),
            'Train Sharpe': res['train_metrics']['sharpe_ratio'],
            'Val Sharpe': res['val_metrics']['sharpe_ratio'],
            'Test Sharpe': res['test_metrics']['sharpe_ratio'],
            'Train Return': res['train_metrics']['total_return'],
            'Test Return': res['test_metrics']['total_return'],
            'Best Val Step': training_histories[name].stopped_step or timesteps
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Generalization Analysis
    print("\n" + "="*70)
    print("GENERALIZATION ANALYSIS (Lower is Better):")
    print("="*70)
    
    for name, res in results.items():
        train_sharpe = res['train_metrics']['sharpe_ratio']
        val_sharpe = res['val_metrics']['sharpe_ratio']
        test_sharpe = res['test_metrics']['sharpe_ratio']
        
        # Calculate degradation
        train_val_deg = (train_sharpe - val_sharpe) / abs(train_sharpe) * 100 if train_sharpe != 0 else 0
        val_test_deg = (val_sharpe - test_sharpe) / abs(val_sharpe) * 100 if val_sharpe != 0 else 0
        train_test_deg = (train_sharpe - test_sharpe) / abs(train_sharpe) * 100 if train_sharpe != 0 else 0
        
        agent_name = name.replace('_', ' ').title()
        
        # Status based on train->test degradation
        if train_test_deg < 20:
            status = "✓ Excellent"
            color = "green"
        elif train_test_deg < 40:
            status = "⚠ Acceptable"
            color = "orange"
        else:
            status = "✗ Overfitting"
            color = "red"
        
        print(f"\n{agent_name:25s}")
        print(f"  Train → Val:  {train_val_deg:>6.1f}%")
        print(f"  Val → Test:   {val_test_deg:>6.1f}%")
        print(f"  Train → Test: {train_test_deg:>6.1f}% [{status}]")
    
    return results, summary_df, training_histories


def plot_train_val_test_comparison(results):
    """Create visualization comparing train/val/test performance"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Train vs Validation vs Test Performance', 
                 fontsize=16, fontweight='bold')
    
    # Extract data
    agents = list(results.keys())
    train_sharpes = [results[a]['train_metrics']['sharpe_ratio'] for a in agents]
    val_sharpes = [results[a]['val_metrics']['sharpe_ratio'] for a in agents]
    test_sharpes = [results[a]['test_metrics']['sharpe_ratio'] for a in agents]
    
    train_returns = [results[a]['train_metrics']['total_return']*100 for a in agents]
    val_returns = [results[a]['val_metrics']['total_return']*100 for a in agents]
    test_returns = [results[a]['test_metrics']['total_return']*100 for a in agents]
    
    # Clean names
    agent_names = [a.replace('_', '\n').title() for a in agents]
    
    x = np.arange(len(agents))
    width = 0.25
    
    # Sharpe Ratio comparison
    axes[0, 0].bar(x - width, train_sharpes, width, label='Train', alpha=0.8, color='#2E86AB')
    axes[0, 0].bar(x, val_sharpes, width, label='Validation', alpha=0.8, color='#A23B72')
    axes[0, 0].bar(x + width, test_sharpes, width, label='Test', alpha=0.8, color='#F18F01')
    axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=11)
    axes[0, 0].set_title('Sharpe Ratio: Train vs Val vs Test', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(agent_names, fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # Total Return comparison
    axes[0, 1].bar(x - width, train_returns, width, label='Train', alpha=0.8, color='#2E86AB')
    axes[0, 1].bar(x, val_returns, width, label='Validation', alpha=0.8, color='#A23B72')
    axes[0, 1].bar(x + width, test_returns, width, label='Test', alpha=0.8, color='#F18F01')
    axes[0, 1].set_ylabel('Total Return (%)', fontsize=11)
    axes[0, 1].set_title('Total Return: Train vs Val vs Test', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agent_names, fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Performance degradation (train → test)
    degradation = [(t - s) / abs(t) * 100 if t != 0 else 0 
                   for t, s in zip(train_sharpes, test_sharpes)]
    colors = ['#06A77D' if d < 20 else '#F77F00' if d < 40 else '#D62828' 
              for d in degradation]
    
    bars = axes[1, 0].bar(agent_names, degradation, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Degradation (%)', fontsize=11)
    axes[1, 0].set_title('Generalization Gap (Train → Test)', fontsize=12, fontweight='bold')
    axes[1, 0].axhline(y=20, color='orange', linestyle='--', alpha=0.5, 
                      label='Acceptable (<20%)')
    axes[1, 0].axhline(y=40, color='red', linestyle='--', alpha=0.5,
                      label='Severe (>40%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)
    
    # Portfolio value - Test set only
    for agent in agents:
        test_env = results[agent]['test_env']
        label = agent.replace('_', ' ').title()
        axes[1, 1].plot(test_env.portfolio_history, label=label, linewidth=2.5)
    
    axes[1, 1].set_title('Portfolio Value on Test Set', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time Step', fontsize=11)
    axes[1, 1].set_ylabel('Portfolio Value ($)', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig