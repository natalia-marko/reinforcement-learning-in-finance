"""
Quick Reference: Reward Function Configuration

Use these parameters when initializing environments.
"""

# Base Agents (Sentiment & Technical)
base_agent_config = {
    'vol_window': 10,           # Rolling window for Sharpe-like calculation
    'transaction_cost': 0.001,  # 0.1% transaction cost per trade
    'risk_lambda': 0.5,         # Risk penalty weight (legacy parameter)
}

# Super Agent & Meta Agent
hierarchical_agent_config = {
    'alpha1': 1.0,              # Weight for log returns
    'alpha2': 0.5,              # Weight for MDD penalty
    'alpha3': 0.5,              # Weight for volatility penalty
    'exploration_bias': 0.001,  # Small positive constant for exploration
}

# Pre-configured Risk Profiles
risk_profiles = {
    'balanced': {
        'alpha1': 1.0,
        'alpha2': 0.5,
        'alpha3': 0.5,
        'exploration_bias': 0.001
    },
    'conservative': {
        'alpha1': 1.0,
        'alpha2': 1.0,     # Higher penalty for drawdowns
        'alpha3': 1.0,     # Higher penalty for volatility
        'exploration_bias': 0.0005
    },
    'aggressive': {
        'alpha1': 1.0,
        'alpha2': 0.2,     # Lower penalty for drawdowns
        'alpha3': 0.2,     # Lower penalty for volatility
        'exploration_bias': 0.002
    },
    'growth': {
        'alpha1': 1.5,     # Emphasize returns
        'alpha2': 0.3,
        'alpha3': 0.3,
        'exploration_bias': 0.001
    }
}

# Example usage:
"""
from sentiment_enviroment import SentimentEnv
from super_agent_envoriment import SuperAgentEnv

# Base agent with custom window
sentiment_env = SentimentEnv(
    price_data=prices,
    sentiment_features=features,
    vol_window=15,           # Longer window for smoother signals
    transaction_cost=0.001
)

# Super agent with conservative profile
super_env = SuperAgentEnv(
    price_data=prices,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    **risk_profiles['conservative']
)

# Or with custom parameters
super_env = SuperAgentEnv(
    price_data=prices,
    sentiment_agent=sent_agent,
    technical_agent=tech_agent,
    alpha1=1.2,              # Slightly emphasize returns
    alpha2=0.7,              # More conservative on drawdowns
    alpha3=0.4,              # Less concerned about volatility
    exploration_bias=0.0015
)
"""

