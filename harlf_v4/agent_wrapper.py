

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from sentiment_enviroment import SentimentEnv
from technical_enviroment import TechnicalEnv
from custom_function import split_data_chronologically
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt



class EarlyStoppingCallback(BaseCallback):
    def __init__(self, val_env, eval_freq=5000, patience=5, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_sharpe = -np.inf
        self.wait = 0
        self.train_sharpes = []  # For plotting
        self.val_sharpes = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True
        
        # Evaluate on val
        metrics = evaluate_agent(self.model, self.val_env, phase="Validation (Callback)")
        val_sharpe = metrics['sharpe_ratio']
        self.val_sharpes.append(val_sharpe)
        
        # Approximate train Sharpe (from recent rewards; rough estimate)
        train_sharpe = np.mean(self.locals['rewards']) / (np.std(self.locals['rewards']) + 1e-8) if len(self.locals['rewards']) > 0 else 0
        self.train_sharpes.append(train_sharpe)
        
        if self.verbose > 0:
            print(f"Step {self.n_calls}: Val Sharpe = {val_sharpe:.3f}")
        
        if val_sharpe > self.best_sharpe:
            self.best_sharpe = val_sharpe
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print(f"Early stopping: No val Sharpe improvement for {self.patience} evals.")
                return False  # Stop training
        return True

    def plot_curves(self, agent_name):
        plt.figure(figsize=(10, 5))
        steps = np.arange(len(self.train_sharpes)) * self.eval_freq
        plt.plot(steps, self.train_sharpes, label='Train Sharpe (approx)')
        plt.plot(steps, self.val_sharpes, label='Val Sharpe')
        plt.xlabel('Training Steps')
        plt.ylabel('Sharpe Ratio')
        plt.title(f'Sharpe Curves for {agent_name}')
        plt.legend()
        plt.grid(True)
        #plt.savefig(f'./models/{agent_name}_sharpe_curves.png')
        plt.close()
        print(f"✓ Plot saved: ./models/{agent_name}_sharpe_curves.png")


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



def train_agent(env, val_env, agent_type, algorithm, timesteps=100000, 
                ppo_params=None, sac_params=None): 
    """Train a single agent with configurable hyperparameters"""
    
    print(f"\nTraining {agent_type} with {algorithm}...")
    
    if algorithm == 'PPO':
        default_ppo = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'ent_coef': 0.01,
            'verbose': 1
        }
        params = {**default_ppo, **(ppo_params or {})}
        model = PPO("MlpPolicy", env, **params)
    else:  # SAC
        default_sac = {
            'learning_rate': 3e-4,
            'buffer_size': 20000,
            'learning_starts': 100,
            'batch_size': 128,
            'ent_coef': 'auto',
            'verbose': 1
        }
        params = {**default_sac, **(sac_params or {})}
        model = SAC("MlpPolicy", env, **params)

    callback = EarlyStoppingCallback(val_env, eval_freq=5000, patience=5)
    model.learn(total_timesteps=timesteps, callback=callback)
    callback.plot_curves(f"{agent_type}_{algorithm}")
    
    return model


def evaluate_agent(model, env, phase="Test"):
    """Evaluate agent on given environment"""
    
    obs, _ = env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
    
    metrics = env.get_portfolio_metrics()
    
    print(f"\n{phase} Results:")
    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:>8.3f}")
    print(f"  Total Return: {metrics['total_return']:>8.2%}")
    print(f"  Max Drawdown: {metrics['max_drawdown']:>8.2%}")
    print(f"  Win Rate:     {metrics['win_rate']:>8.2%}")
    
    return metrics


def train_and_evaluate_with_split(price_data, technical_features, sentiment_features,
                                  train_ratio=0.60, val_ratio=0.20,
                                  timesteps=100000, algorithm='both',
                                  sentiment_env_params=None, technical_env_params=None,
                                  ppo_params=None, sac_params=None):
    """
    Complete workflow with proper train/val/test split.
    
    Args:
        algorithm: 'ppo', 'sac', or 'both'
        sentiment_env_params: dict of parameters for SentimentEnv
        technical_env_params: dict of parameters for TechnicalEnv
        ppo_params: dict of PPO hyperparameters
        sac_params: dict of SAC hyperparameters
    """
    
    # Default environment parameters
    sentiment_env_params = sentiment_env_params or {}
    technical_env_params = technical_env_params or {}
    
    # Step 1: Split data
    splits = split_data_chronologically(
        price_data, technical_features, sentiment_features,
        train_ratio, val_ratio
    )
    
    train_prices, train_technical, train_sentiment = splits['train']
    val_prices, val_technical, val_sentiment = splits['val']
    test_prices, test_technical, test_sentiment = splits['test']
    
    results = {}
    
    # Algorithms to train
    algorithms = ['PPO', 'SAC'] if algorithm == 'both' else [algorithm.upper()]
    
    # Step 2: Train Sentiment Agent
    print("\n" + "="*70)
    print("TRAINING SENTIMENT AGENTS")
    print("="*70)
    
    for algo in algorithms:
        print(f"\n[Sentiment {algo}]")
        
        # Create training environment with custom parameters
        sent_train_env = SentimentEnv(train_prices, train_sentiment, **sentiment_env_params)
        sent_val_env = SentimentEnv(val_prices, val_sentiment, **sentiment_env_params)
        
        # Train with custom RL parameters
        sent_model = train_agent(sent_train_env, sent_val_env, "Sentiment", algo, 
                                timesteps, ppo_params, sac_params)
        
        # Evaluate on all sets
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
        
        # Create training environment with custom parameters
        tech_train_env = TechnicalEnv(train_prices, train_technical, **technical_env_params)
        tech_val_env = TechnicalEnv(val_prices, val_technical, **technical_env_params)
        
        # Train with custom RL parameters
        tech_model = train_agent(tech_train_env, tech_val_env, "Technical", algo, 
                                timesteps, ppo_params, sac_params)
        
        # Evaluate on all sets
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
    
    # Step 4: Summary
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
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Check for overfitting
    print("\n" + "="*70)
    print("OVERFITTING ANALYSIS:")
    print("="*70)
    
    for name, res in results.items():
        train_sharpe = res['train_metrics']['sharpe_ratio']
        test_sharpe = res['test_metrics']['sharpe_ratio']
        degradation = (train_sharpe - test_sharpe) / train_sharpe * 100
        
        agent_name = name.replace('_', ' ').title()
        
        if degradation < 20:
            status = "✓ Good generalization"
        elif degradation < 40:
            status = "⚠️  Moderate overfitting"
        else:
            status = "✗ Severe overfitting"
        
        print(f"{agent_name:20s}: {degradation:>5.1f}% degradation - {status}")
    
    return results, summary_df


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
    axes[0, 0].bar(x - width, train_sharpes, width, label='Train', alpha=0.8)
    axes[0, 0].bar(x, val_sharpes, width, label='Validation', alpha=0.8)
    axes[0, 0].bar(x + width, test_sharpes, width, label='Test', alpha=0.8)
    axes[0, 0].set_ylabel('Sharpe Ratio')
    axes[0, 0].set_title('Sharpe Ratio: Train vs Val vs Test')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(agent_names, fontsize=9)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
    
    # Total Return comparison
    axes[0, 1].bar(x - width, train_returns, width, label='Train', alpha=0.8)
    axes[0, 1].bar(x, val_returns, width, label='Validation', alpha=0.8)
    axes[0, 1].bar(x + width, test_returns, width, label='Test', alpha=0.8)
    axes[0, 1].set_ylabel('Total Return (%)')
    axes[0, 1].set_title('Total Return: Train vs Val vs Test')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(agent_names, fontsize=9)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Performance degradation (train - test)
    degradation = [(t - s) / t * 100 if t > 0 else 0 
                   for t, s in zip(train_sharpes, test_sharpes)]
    colors = ['green' if d < 20 else 'orange' if d < 40 else 'red' 
              for d in degradation]
    
    axes[1, 0].bar(agent_names, degradation, color=colors, alpha=0.8)
    axes[1, 0].set_ylabel('Degradation (%)')
    axes[1, 0].set_title('Overfitting Analysis (Train → Test)')
    axes[1, 0].axhline(y=20, color='orange', linestyle='--', alpha=0.5, 
                      label='Acceptable (<20%)')
    axes[1, 0].axhline(y=40, color='red', linestyle='--', alpha=0.5,
                      label='Severe (>40%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Portfolio value - Test set only
    for agent in agents:
        test_env = results[agent]['test_env']
        label = agent.replace('_', ' ').title()
        axes[1, 1].plot(test_env.portfolio_history, label=label, linewidth=2)
    
    axes[1, 1].set_title('Portfolio Value on Test Set')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Portfolio Value ($)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=100000, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return fig
