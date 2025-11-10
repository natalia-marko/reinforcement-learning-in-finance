# Wrapper for pre-trained frozen agents
import numpy as np

class ProductionAgentWrapper:
    """Wrapper for pre-trained frozen agents"""
    
    def __init__(self, model, features_or_env, n_assets, agent_type: str, 
                 freeze_weights: bool = True):
        self.model = model
        self.agent_type = agent_type
        self.n_assets = n_assets
        self.weights = np.ones(n_assets) / n_assets
        
        if hasattr(features_or_env, 'index'):
            self.features = features_or_env
            self.env = None
            self.common_dates = features_or_env.index
        else:
            self.features = None
            self.env = features_or_env
            self.common_dates = features_or_env.common_dates
        
        if freeze_weights:
            self.freeze()
    
    def freeze(self):
        if hasattr(self.model, 'policy'):
            for param in self.model.policy.parameters():
                param.requires_grad = False
    
    def predict(self, obs, deterministic=True):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        action = (action + 1.0) / 2.0
        action = np.clip(action, 0, 1)
        
        total = action.sum()
        if total > 1e-6:
            self.weights = action / total
        else:
            self.weights = np.ones(self.n_assets) / self.n_assets
        
        return self.weights, _
    
    def reset(self, seed=None):
        self.weights = np.ones(self.n_assets) / self.n_assets
        return self.weights
    
    def get_observation(self, date):
        if self.features is not None:
            if date in self.features.index:
                obs = self.features.loc[date].values
                obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                return obs.astype(np.float32)
            return np.zeros(self.features.shape[1], dtype=np.float32)
        elif self.env is not None:
            return self.env._get_observation()
        else:
            raise ValueError("No features or environment available")

