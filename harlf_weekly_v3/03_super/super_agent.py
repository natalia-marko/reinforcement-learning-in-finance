<DOCUMENT filename="super_agent.py">
"""
Super Agent Implementation - WITH PPO & VISUALIZATION
=====================================================
Updated: Use PPO from Stable Baselines3 for proper gradient-based training.
Added post-training visualization of val_sharpe vs. episodes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from environments import create_env
from utils import compute_sharpe


class BaseAgentWrapper:
    """Wrapper for base agents."""
    
    def __init__(self, model_paths: List[str], agent_name: str, 
                 agent_type: str, reward_type: str,
                 data_dir: str, split: str = 'train',
                 algorithm: str = 'PPO',
                 config: dict = None):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.reward_type = reward_type
        self.algorithm = algorithm
        self.model_paths = model_paths
        self.models = []
        
        if config is None:
            config = {}
        
        reward_kwargs = {}
        if reward_type == 'ema_sharpe':
            reward_kwargs = {'rolling_vol_window': config.get('rolling_vol_window', 12)}
        elif reward_type == 'multi_objective':
            reward_kwargs = {
                'return_scale': config.get('return_scale', 8.0),
                'volatility_penalty': config.get('volatility_penalty', 0.05),
                'concentration_penalty': config.get('concentration_penalty', 0.25),
                'turnover_penalty': config.get('turnover_penalty', 0.005),
                'vol_window': config.get('vol_window', 12),
                'max_concentration': config.get('max_concentration', 0.35)
            }
        
        self.env = create_env(
            data_dir=data_dir,
            agent_type=agent_type,
            split=split,
            reward_type=reward_type,
            reward_kwargs=reward_kwargs,
            softmax_temperature=config.get('softmax_temperature', 1.0),
            random_start=False,
            transaction_cost=config.get('transaction_cost', 0.0)
        )
        
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.n_assets
        
        for path in model_paths:
            model = PPO.load(path)
            self.models.append(model)
    
    def get_state(self, timestep):
        self.env._t = timestep
        return self.env._get_obs()
    
    def predict(self, obs, deterministic=True):
        actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action)
        return np.mean(actions, axis=0)


class SuperAgentWrapperEnv(gym.Env):
    """Wrapper env for super agent: obs=concat states, action=agent_weights."""
    
    def __init__(self, base_agents, main_env):
        super().__init__()
        self.base_agents = base_agents
        self.main_env = main_env
        self.n_agents = len(base_agents)
        self.total_state_dim = sum(agent.state_dim for agent in base_agents)
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.total_state_dim,))
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_agents,))
    
    def reset(self, seed=None, options=None):
        self.main_env.reset()
        return self._get_obs(), {}
    
    def step(self, action):
        agent_weights = np.softmax(action)  # Softmax actions to weights
        states = [agent.get_state(self.main_env._t) for agent in self.base_agents]
        
        base_actions = []
        for i, agent in enumerate(self.base_agents):
            base_actions.append(agent.predict(states[i]))
        
        blended_action = np.sum(agent_weights[:, None] * np.array(base_actions), axis=0)
        
        obs, reward, done, info = self.main_env.step(blended_action)
        return self._get_obs(), reward, done, info
    
    def _get_obs(self):
        states = [agent.get_state(self.main_env._t) for agent in self.base_agents]
        return np.concatenate(states)


class SuperAgentPolicy(ActorCriticPolicy):
    """Custom policy for super agent (outputs n_agents weights)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )


class SuperAgent:
    def __init__(self, base_agents: List[BaseAgentWrapper], learning_rate=3e-4, device='cpu'):
        self.base_agents = base_agents
        self.n_agents = len(base_agents)
        self.action_dim = base_agents[0].action_dim
        self.total_state_dim = sum(agent.state_dim for agent in base_agents)
        self.device = device
        self.training_history = []
    
    def train(self, train_env, val_env, n_episodes=100, eval_every=10, patience=20, save_path=None, verbose=True):
        # Wrap envs for PPO
        train_wrapper = SuperAgentWrapperEnv(self.base_agents, train_env)
        train_wrapper = DummyVecEnv([lambda: train_wrapper])
        
        # PPO model with custom policy
        self.model = PPO(
            SuperAgentPolicy,
            train_wrapper,
            learning_rate=3e-4,
            gamma=0.90,
            device=self.device,
            verbose=1 if verbose else 0
        )
        
        best_val_sharpe = -np.inf
        no_improve = 0
        
        for episode in range(0, n_episodes, eval_every):  # Train in batches
            self.model.learn(total_timesteps=eval_every * train_env.n_steps)  # Adjust steps
            
            val_sharpe = self.evaluate(val_env)
            
            self.training_history.append({
                'episode': episode + eval_every,
                'val_sharpe': val_sharpe
            })
            
            if verbose:
                print(f"Episode {episode + eval_every}: Val Sharpe {val_sharpe:.3f}")
            
            if val_sharpe > best_val_sharpe:
                best_val_sharpe = val_sharpe
                no_improve = 0
                if save_path:
                    self.model.save(save_path.replace('.pt', '.zip'))  # Save PPO model
            else:
                no_improve += 1
                if no_improve >= patience:
                    if verbose:
                        print("Early stopping")
                    break
        
        # Visualize training history
        self.visualize_training_history()
    
    def visualize_training_history(self, save_path=None):
        if not self.training_history:
            print("No training history to visualize.")
            return
        
        episodes = [h['episode'] for h in self.training_history]
        sharpes = [h['val_sharpe'] for h in self.training_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, sharpes, marker='o', label='Val Sharpe')
        plt.xlabel('Episodes')
        plt.ylabel('Validation Sharpe Ratio')
        plt.title('Super Agent Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Training history plot saved to: {save_path}")
        else:
            plt.show()
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
    def evaluate(self, env, deterministic=True, n_runs=1, base_agents=None):
        wrapper = SuperAgentWrapperEnv(base_agents or self.base_agents, env)
        wrapper = DummyVecEnv([lambda: wrapper])
        
        sharpes = []
        for _ in range(n_runs):
            obs = wrapper.reset()
            done = False
            returns = []
            
            while not done:
                action = self.predict(obs)
                obs, reward, done, info = wrapper.step(action)
                if info[0].get('port_return') is not None:
                    returns.append(info[0]['port_return'])
            
            sharpe = compute_sharpe(returns)
            sharpes.append(sharpe)
        
        return np.mean(sharpes)
    
    def analyze_agent_usage(self, env, deterministic=True, base_agents=None):
        wrapper = SuperAgentWrapperEnv(base_agents or self.base_agents, env)
        
        obs = wrapper.reset()
        done = False
        weights_history = []
        returns = []
        
        while not done:
            action = self.predict(obs)
            agent_weights = np.softmax(action)
            weights_history.append(agent_weights)
            obs, _, done, info = wrapper.step(action)
            if info.get('port_return') is not None:
                returns.append(info['port_return'])
        
        weights_history = np.array(weights_history)
        analysis = {
            'mean_weights': np.mean(weights_history, axis=0),
            'std_weights': np.std(weights_history, axis=0),
            'min_weights': np.min(weights_history, axis=0),
            'max_weights': np.max(weights_history, axis=0),
            'sharpe': compute_sharpe(returns)
        }
        
        return analysis


def load_base_agents_from_results(
    results_path: str,
    data_dir: str,
    split: str = 'train'
) -> List[BaseAgentWrapper]:
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    base_agents = []
    
    for agent_key, agent_data in results.items():
        model_path = agent_data['model_path']  # Single model per agent
        
        config = get_config(agent_data['reward_type'], agent_data['agent_type'])
        
        agent = BaseAgentWrapper(
            model_paths=[model_path],
            agent_name=agent_data['name'],
            agent_type=agent_data['agent_type'],
            reward_type=agent_data['reward_type'],
            data_dir=data_dir,
            split=split,
            algorithm=agent_data['algorithm'],
            config=config
        )
        
        base_agents.append(agent)
    
    return base_agents
</DOCUMENT>