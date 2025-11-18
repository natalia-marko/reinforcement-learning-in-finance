"""
Neural network models for RL portfolio rebalancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleActor(nn.Module):
    """
    Baseline simple actor network for portfolio allocation.
    
    Architecture:
    - Linear(state_dim, 256) -> ReLU -> Dropout(0.2)
    - Linear(256, 128) -> ReLU
    - Linear(128, n_assets) -> Softmax
    """
    
    def __init__(self, state_dim: int, n_assets: int, dropout: float = 0.2):
        """
        Initialize SimpleActor.
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state space
        n_assets : int
            Number of assets in portfolio
        dropout : float
            Dropout probability (default: 0.2)
        """
        super(SimpleActor, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_assets)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters:
        -----------
        x : torch.Tensor
            State tensor of shape (batch_size, state_dim)
        
        Returns:
        --------
        torch.Tensor
            Portfolio weights of shape (batch_size, n_assets)
        """
        logits = self.net(x)
        weights = F.softmax(logits, dim=-1)
        return weights

