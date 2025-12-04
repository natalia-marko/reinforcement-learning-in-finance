import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
from utile import EarlyStoppingCallback
from stable_baselines3 import PPO, SAC
from production_agent_wrapper_frozen import ProductionAgentWrapper

class SuperAgentEnv(gymnasium.Env):
    """
    Super Agent: Sees BOTH base agent weights AND raw features
    
    Fixes the information bottleneck problem!
    """
    
    def __init__(self, price_data, base_agents, technical_features, 
                 sentiment_features, **config):
        super().__init__()
        
        self.price_data = price_data
        self.base_agents = base_agents
        self.technical_features = technical_features
        self.sentiment_features = sentiment_features
        
        self.n_assets = len(price_data.columns)
        self.n_base_agents = len(base_agents)
        
        # Config
        self.initial_capital = 100_000
        self.transaction_cost = config.get('transaction_cost', 0.002)
        self.max_position = config.get('max_position', 0.30)
        self.max_turnover = config.get('max_turnover', 0.25)
        self.constraint_penalty = config.get('constraint_penalty', 50.0)
        
        # Reward parameters
        self.alpha_returns = config.get('alpha_returns', 1.0)
        self.alpha_mdd = config.get('alpha_mdd', 1.0)
        self.alpha_vol = config.get('alpha_vol', 0.5)
        self.alpha_concentration = config.get('alpha_concentration', 0.3)
        
        self.obs_scaler = None
        self.normalize = True
        self.common_dates = price_data.index
        
        # Action space
        self.action_space = spaces.Box(
            low=0.0, high=self.max_position,
            shape=(self.n_assets,), dtype=np.float32
        )
        
        # OBSERVATION SPACE (base weights + raw features)
        n_tech_features = technical_features.shape[1]
        n_sent_features = sentiment_features.shape[1]
        obs_dim = (self.n_base_agents * self.n_assets +  # Base agent weights
                   n_tech_features +                      # Technical features
                   n_sent_features)                       # Sentiment features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        
        print(f"   Super Agent observation size: {obs_dim}")
        print(f"      - Base weights: {self.n_base_agents * self.n_assets}")
        print(f"      - Technical:    {n_tech_features}")
        print(f"      - Sentiment:    {n_sent_features}")
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_history = [self.initial_capital]
        self.returns_history = []
        
        for agent in self.base_agents.values():
            agent.reset(seed)
        
        if self.normalize and self.obs_scaler is None:
            self._fit_scaler()
        
        self._update_base_agents()
        
        return self._get_observation(), {}
    
    def _fit_scaler(self):
        obs_samples = []
        n_samples = min(len(self.common_dates), 200)
        
        for step in range(n_samples):
            self.current_step = step
            self._update_base_agents()
            obs = self._get_observation_raw()
            obs_samples.append(obs)
        
        self.obs_scaler = StandardScaler()
        self.obs_scaler.fit(obs_samples)
        self.current_step = 0
    
    def _get_observation_raw(self):
        """Include base weights + raw features"""
        
        # Get base agent weights
        agent_weights = [agent.weights for agent in self.base_agents.values()]
        obs_weights = np.concatenate(agent_weights)
        
        # Get raw features for current date
        current_date = self.common_dates[self.current_step]
        
        tech_obs = np.zeros(self.technical_features.shape[1])
        sent_obs = np.zeros(self.sentiment_features.shape[1])
        
        if current_date in self.technical_features.index:
            tech_obs = self.technical_features.loc[current_date].values
            tech_obs = np.nan_to_num(tech_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        if current_date in self.sentiment_features.index:
            sent_obs = self.sentiment_features.loc[current_date].values
            sent_obs = np.nan_to_num(sent_obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine: weights + technical + sentiment
        obs = np.concatenate([obs_weights, tech_obs, sent_obs])
        
        return obs
    
    def _get_observation(self):
        obs = self._get_observation_raw()
        
        if self.normalize and self.obs_scaler is not None:
            obs = self.obs_scaler.transform(obs.reshape(1, -1))[0]
            obs = np.clip(obs, -10, 10)
        
        return obs.astype(np.float32)
    
    def _apply_constraints(self, weights):
        violation_penalty = 0.0
        
        max_weight = np.max(weights)
        if max_weight > self.max_position:
            violation_penalty += self.constraint_penalty * (max_weight - self.max_position)
        
        weights = np.clip(weights, 0, self.max_position)
        weights = weights / weights.sum()
        
        turnover = np.sum(np.abs(weights - self.weights))
        if turnover > self.max_turnover:
            violation_penalty += self.constraint_penalty * (turnover - self.max_turnover)
            scale = self.max_turnover / turnover
            weights = self.weights + scale * (weights - self.weights)
            weights = np.clip(weights, 0, 1)
            weights = weights / weights.sum()
        
        return weights, violation_penalty
    
    def step(self, action):
        action = np.clip(action, 0, 1)
        total = action.sum()
        weights = action / total if total > 1e-6 else np.ones(self.n_assets) / self.n_assets
        
        weights, constraint_penalty = self._apply_constraints(weights)
        
        current_date = self.common_dates[self.current_step]
        next_date = self.common_dates[min(self.current_step + 1, len(self.common_dates) - 1)]
        
        log_returns = np.log(
            self.price_data.loc[next_date] / self.price_data.loc[current_date]
        )
        log_returns = np.nan_to_num(log_returns.values, nan=0.0, posinf=0.0, neginf=0.0)
        arithmetic_returns = np.exp(log_returns) - 1
        
        portfolio_return = np.sum(weights * arithmetic_returns)
        turnover = np.sum(np.abs(weights - self.weights))
        portfolio_return -= self.transaction_cost * turnover
        
        self.portfolio_value *= (1 + portfolio_return)
        self.portfolio_history.append(self.portfolio_value)
        self.returns_history.append(portfolio_return)
        
        portfolio_series = pd.Series(self.portfolio_history)
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        current_mdd = abs(drawdown.iloc[-1])
        
        volatility = np.std(self.returns_history) if len(self.returns_history) > 1 else 0.0
        hhi = np.sum(weights ** 2)
        
        reward = (
            self.alpha_returns * np.log(1 + portfolio_return + 1e-8) -
            self.alpha_mdd * current_mdd -
            self.alpha_vol * volatility -
            self.alpha_concentration * hhi -
            constraint_penalty
        )
        
        self.weights = weights
        self.current_step += 1
        done = self.current_step >= len(self.common_dates) - 1
        
        if not done:
            self._update_base_agents()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "weights": weights,
            "turnover": turnover,
            "hhi": hhi
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _update_base_agents(self):
        if self.current_step >= len(self.common_dates):
            return
        
        current_date = self.common_dates[self.current_step]
        
        for agent_name, agent in self.base_agents.items():
            obs = agent.get_observation(current_date)
            agent.predict(obs, deterministic=True)
    
    def get_portfolio_metrics(self):
        portfolio_series = pd.Series(self.portfolio_history)
        returns = portfolio_series.pct_change().dropna()
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        sharpe_ratio = 0.0
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12)
        
        max_drawdown = 0.0
        if len(portfolio_series) > 0:
            peak = portfolio_series.expanding().max()
            drawdown = (portfolio_series - peak) / peak
            max_drawdown = abs(drawdown.min())
        
        volatility = returns.std() * np.sqrt(12) if len(returns) > 0 else 0.0
        win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.0
        
        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "final_value": self.portfolio_value,
            "win_rate": win_rate
        }



# ============================================================================
# MODIFIED: SIMPLIFIED TRAINING FUNCTIONS (no window loops, no ensemble)
# ============================================================================

def train_super_agent_sac(train_data, val_data, config):
    """
    Train Super agent with SAC + Enhanced observations
    SIMPLIFIED: Single training run (no windows, no ensemble)
    """
    
    print("\n" + "="*70)
    print("TRAINING ENHANCED SUPER AGENT (SAC + FULL FEATURES)")
    print("="*70)
    
    train_prices, train_tech, train_sent, train_regime = train_data
    val_prices, val_tech, val_sent, val_regime = val_data
    
    n_assets = len(train_prices.columns)
    
    # Load base agents
    print("\n   Loading base agents...")
    tech_ppo_model = PPO.load('models/best_technical_PPO.zip')
    tech_sac_model = SAC.load('models/best_technical_SAC.zip')
    sent_ppo_model = PPO.load('models/best_sentiment_PPO.zip')
    sent_sac_model = SAC.load('models/best_sentiment_SAC.zip')
    
    base_agents_train = {
        'tech_PPO': ProductionAgentWrapper(tech_ppo_model, train_tech, n_assets, 'tech_PPO'),
        'tech_SAC': ProductionAgentWrapper(tech_sac_model, train_tech, n_assets, 'tech_SAC'),
        'sent_PPO': ProductionAgentWrapper(sent_ppo_model, train_sent, n_assets, 'sent_PPO'),
        'sent_SAC': ProductionAgentWrapper(sent_sac_model, train_sent, n_assets, 'sent_SAC'),
    }
    
    base_agents_val = {
        'tech_PPO': ProductionAgentWrapper(tech_ppo_model, val_tech, n_assets, 'tech_PPO'),
        'tech_SAC': ProductionAgentWrapper(tech_sac_model, val_tech, n_assets, 'tech_SAC'),
        'sent_PPO': ProductionAgentWrapper(sent_ppo_model, val_sent, n_assets, 'sent_PPO'),
        'sent_SAC': ProductionAgentWrapper(sent_sac_model, val_sent, n_assets, 'sent_SAC'),
    }
    
    # Create ENHANCED environments (with full features)
    print("\n   Creating environments...")
    super_train_env = SuperAgentEnv(
        train_prices, base_agents_train, train_tech, train_sent, **config
    )
    super_val_env = SuperAgentEnv(
        val_prices, base_agents_val, val_tech, val_sent, **config
    )
    
    # Train with SAC
    print(f"\n   Training with SAC algorithm...")
    model = SAC(
        'MlpPolicy',
        super_train_env,
        learning_rate=config['super_learning_rate'],
        buffer_size=config['super_buffer_size'],
        batch_size=config['super_batch_size'],
        tau=config['super_tau'],
        gamma=config['super_gamma'],
        ent_coef=config['super_ent_coef'],
        verbose=0,
        seed=config['seed'],
        policy_kwargs=dict(net_arch=config['super_network'])
    )
    
    callback = EarlyStoppingCallback(
        val_env=super_val_env,
        eval_freq=config['super_eval_freq'],
        patience=config['super_patience'],
        min_delta=config['super_min_delta'],
        min_steps=10000,
        save_path='models/super_agent_sac',
        verbose=1
    )
    
    print(f"\n   Training for up to {config['super_timesteps']} steps...")
    model.learn(total_timesteps=config['super_timesteps'], callback=callback)
    
    if callback.best_model_path:
        model = SAC.load(callback.best_model_path)
        print(f"\n   ✓ Loaded best model (Val Sharpe: {callback.best_val_sharpe:.3f})")
    
    return model, callback.best_val_sharpe
