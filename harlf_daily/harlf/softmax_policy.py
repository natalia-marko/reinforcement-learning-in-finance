"""
Custom PPO Policy with Softmax Activation for Portfolio Weights.

This policy ensures that portfolio weights always sum to 1.0 by using
Softmax activation instead of relying on environment normalization.

⚠️  EXPERIMENTAL - Test thoroughly before production use
Created: 2025-01-08
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.type_aliases import Schedule
from gymnasium import spaces


class DirichletDistribution(Distribution):
    """
    Dirichlet distribution for portfolio weights.

    Ensures actions sum to 1.0 and are in [0, 1].
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.concentration = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create network that outputs concentration parameters.

        Returns a linear layer that outputs action_dim values (logits).
        """
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, concentration_logits: torch.Tensor) -> "DirichletDistribution":
        """
        Set distribution parameters from network output.

        Args:
            concentration_logits: Raw network outputs (will be converted to positive concentrations)
        """
        # Convert logits to positive concentrations
        # Use softmax + scale to get reasonable concentration values
        self.concentration = torch.softmax(concentration_logits, dim=-1) * 10.0 + 0.1
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Compute log probability of actions under Dirichlet."""
        dist = torch.distributions.Dirichlet(self.concentration)
        return dist.log_prob(actions)

    def entropy(self) -> torch.Tensor:
        """Compute entropy of the distribution."""
        dist = torch.distributions.Dirichlet(self.concentration)
        return dist.entropy()

    def sample(self) -> torch.Tensor:
        """Sample from Dirichlet distribution."""
        dist = torch.distributions.Dirichlet(self.concentration)
        return dist.sample()

    def mode(self) -> torch.Tensor:
        """
        Return mode (deterministic action).

        For Dirichlet, we use normalized concentration as a deterministic action.
        """
        # Normalize concentration to sum to 1
        return self.concentration / self.concentration.sum(dim=-1, keepdim=True)

    def actions_from_params(self, concentration_logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get actions from parameters."""
        self.proba_distribution(concentration_logits)
        if deterministic:
            return self.mode()
        return self.sample()

    def log_prob_from_params(self, concentration_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and compute log prob."""
        actions = self.actions_from_params(concentration_logits, deterministic=False)
        log_prob = self.log_prob(actions)
        return actions, log_prob


class SoftmaxActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy using Dirichlet distribution for portfolio weights.

    Key features:
    - Actions always sum to 1.0 (valid portfolio weights)
    - Actions always in [0, 1] (long-only)
    - Natural exploration via Dirichlet sampling

    Usage:
        from helpers.softmax_policy import SoftmaxActorCriticPolicy

        model = PPO(
            SoftmaxActorCriticPolicy,  # Instead of 'MlpPolicy'
            env,
            learning_rate=3e-4,
            ...
        )
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs
    ):
        # Store for later
        self.action_dim = action_space.shape[0]

        # Call parent init
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build networks. Called by parent __init__."""
        # Call parent to build feature extractor and mlp_extractor
        super()._build(lr_schedule)

        # Replace action distribution with Dirichlet
        self.action_dist = DirichletDistribution(self.action_dim)

        # Build action network (outputs concentration logits)
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)

        # Value network already built by parent

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            actions: Portfolio weights (sum to 1.0)
            values: Value estimates
            log_probs: Log probabilities
        """
        # Extract features
        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Value
        values = self.value_net(latent_vf)

        # Policy
        concentration_logits = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(concentration_logits)

        actions = distribution.mode() if deterministic else distribution.sample()
        log_probs = distribution.log_prob(actions)

        # Safety: ensure sum to 1.0 (should already be true, but for numerical stability)
        actions = actions / (actions.sum(dim=-1, keepdim=True) + 1e-10)

        return actions, values, log_probs

    def _predict(self, observation: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict action from observation."""
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.

        Returns:
            values: Value estimates
            log_probs: Log probabilities of actions
            entropy: Entropy of policy
        """
        features = self.extract_features(obs)

        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Value
        values = self.value_net(latent_vf)

        # Policy
        concentration_logits = self.action_net(latent_pi)
        distribution = self.action_dist.proba_distribution(concentration_logits)

        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_probs, entropy


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

def test_softmax_policy(verbose: bool = True) -> bool:
    """
    Test the SoftmaxActorCriticPolicy.

    Validates:
    1. Policy instantiation
    2. Actions sum to 1.0
    3. Actions in [0, 1]
    4. Forward pass works
    5. Evaluation works
    6. No NaN values

    Returns:
        True if all tests pass
    """
    if verbose:
        print("="*80)
        print("TESTING SOFTMAX POLICY")
        print("="*80)

    try:
        # Setup
        obs_dim = 147  # Technical features
        action_dim = 7  # 7 assets

        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        action_space = spaces.Box(low=0.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        if verbose:
            print(f"\nSetup:")
            print(f"  Observation dim: {obs_dim}")
            print(f"  Action dim: {action_dim}")

        # Test 1: Instantiation
        if verbose:
            print("\n1. Testing instantiation...")

        def lr_schedule(x):
            return 3e-4

        policy = SoftmaxActorCriticPolicy(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=[64, 64]
        )

        if verbose:
            print("   ✓ Policy created")

        # Test 2-4: Action generation
        if verbose:
            print("\n2. Testing action generation (100 samples)...")

        policy.eval()

        for i in range(100):
            # Random observation
            obs = torch.FloatTensor(obs_space.sample()).unsqueeze(0)

            # Deterministic action
            with torch.no_grad():
                action, value, log_prob = policy.forward(obs, deterministic=True)

            action_np = action.cpu().numpy()[0]

            # Check sum
            action_sum = action_np.sum()
            if not np.isclose(action_sum, 1.0, atol=1e-5):
                if verbose:
                    print(f"   ✗ FAIL: Action sum = {action_sum:.8f} (expected 1.0)")
                return False

            # Check range
            if action_np.min() < -1e-6 or action_np.max() > 1.0 + 1e-6:
                if verbose:
                    print(f"   ✗ FAIL: Action out of range: min={action_np.min():.6f}, max={action_np.max():.6f}")
                return False

            # Check for NaN
            if np.isnan(action_np).any():
                if verbose:
                    print(f"   ✗ FAIL: NaN in actions")
                return False

        if verbose:
            print("   ✓ All 100 actions valid (sum=1.0, range=[0,1])")

        # Test 5: Evaluate actions
        if verbose:
            print("\n3. Testing evaluate_actions...")

        obs = torch.FloatTensor(obs_space.sample()).unsqueeze(0)
        action, _, _ = policy.forward(obs, deterministic=True)

        values, log_probs, entropy = policy.evaluate_actions(obs, action)

        if torch.isnan(values).any() or torch.isnan(log_probs).any() or torch.isnan(entropy).any():
            if verbose:
                print("   ✗ FAIL: NaN in evaluate_actions")
            return False

        if verbose:
            print("   ✓ evaluate_actions works")

        # Test 6: Stochastic vs deterministic
        if verbose:
            print("\n4. Testing stochastic sampling...")

        actions_det = []
        actions_stoch = []

        for _ in range(10):
            with torch.no_grad():
                a_det, _, _ = policy.forward(obs, deterministic=True)
                a_stoch, _, _ = policy.forward(obs, deterministic=False)

            actions_det.append(a_det.cpu().numpy()[0])
            actions_stoch.append(a_stoch.cpu().numpy()[0])

        # Deterministic should be same
        det_std = np.std(actions_det, axis=0).sum()
        if det_std > 1e-6:
            if verbose:
                print(f"   ✗ FAIL: Deterministic actions vary (std={det_std:.8f})")
            return False

        # Stochastic should vary
        stoch_std = np.std(actions_stoch, axis=0).sum()
        if stoch_std < 1e-6:
            if verbose:
                print(f"   ⚠ WARNING: Stochastic actions don't vary")

        if verbose:
            print(f"   ✓ Deterministic std: {det_std:.8f}")
            print(f"   ✓ Stochastic std: {stoch_std:.6f}")

        if verbose:
            print("\n" + "="*80)
            print("✅ ALL TESTS PASSED")
            print("="*80)

        return True

    except Exception as e:
        if verbose:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    # Run tests
    success = test_softmax_policy(verbose=True)

    if success:
        print("\n🎉 Policy is ready to use!")
        print("\nUsage:")
        print("  from helpers.softmax_policy import SoftmaxActorCriticPolicy")
        print("  model = PPO(SoftmaxActorCriticPolicy, env, ...)")
    else:
        print("\n❌ Tests failed - DO NOT USE")
