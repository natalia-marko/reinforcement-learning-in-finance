"""
Configuration settings for HARLF sentiment analysis.

This module contains all configurable parameters for the sentiment
analysis pipeline, following the KISS principle.

Created: 2025-01-27 15:15 UTC
Version: 1.0.0
Status: Complete - Ready for production use
"""

# Data Collection Settings
SENTIMENT_CONFIG = {
    'sources': {
        'yahoo': {
            'enabled': True,
            'rate_limit': 0.5,  # seconds between requests
            'weight': 0.4
        },
        'google': {
            'enabled': True, 
            'rate_limit': 1.0,
            'weight': 0.4
        },
        'reddit': {
            'enabled': True,
            'rate_limit': 1.0,
            'weight': 0.2
        }
    },
    
    'finbert': {
        'model': 'ProsusAI/finbert',
        'fallback_model': 'yiyanghkust/finbert-tone',
        'batch_size': 20,
        'max_length': 512,
        'confidence_threshold': 0.5
    },
    
    'aggregation': {
        'formula': 'weighted_mean',
        'monthly_frequency': True,
        'source_weights': {
            'yahoo': 0.4,
            'google': 0.4, 
            'reddit': 0.2
        }
    },
    
    'harlf': {
        'alpha1': 1.0,  # ROI weight
        'alpha2': 2.0,  # Max Drawdown penalty
        'alpha3': 0.5   # Volatility penalty
    },
    
    'cache': {
        'enabled': True,
        'validity_hours': 24,
        'compression': True
    },
    
    'validation': {
        'correlation_threshold': 0.1,
        'significance_level': 0.05,
        'min_sample_size': 10
    }
}

# Portfolio Settings
PORTFOLIO_CONFIG = {
    'tickers': [
        'RDDT', 'NVDA', 'SMR', 'MU', 'MRVL', 'MSFT', 'ASML', 'AEM',
        'AMD', 'VERU', 'AI', 'GOOGL', 'INGM', 'PLUG', 'IONQ', 'CHYM',
        'RGTI', 'ARBE'
    ],
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'rebalance_frequency': 'monthly'
}

# RL Environment Settings
RL_CONFIG = {
    'observation_space': {
        'sentiment_features': 1,
        'technical_features': 2,  # returns, volatility
        'total_features': 3
    },
    'action_space': {
        'min_weight': 0.0,
        'max_weight': 1.0,
        'constraint': 'sum_to_one'
    },
    'reward_function': {
        'roi_weight': 1.0,
        'drawdown_penalty': 2.0,
        'volatility_penalty': 0.5
    }
}

# File Paths
PATHS = {
    'cache_dir': '../cache',
    'news_cache': '../cache/news_cache',
    'sentiment_cache': '../cache/sentiment_cache',
    'outputs_dir': '../outputs',
    'models_dir': '../models',
    'notebooks_dir': '../notebooks'
}

# API Settings (for future implementation)
API_CONFIG = {
    'yahoo_finance': {
        'base_url': 'https://finance.yahoo.com',
        'timeout': 30,
        'max_retries': 3
    },
    'google_news': {
        'api_key': None,  # Set your API key here
        'base_url': 'https://newsapi.org/v2',
        'timeout': 30
    },
    'reddit': {
        'client_id': None,  # Set your Reddit API credentials
        'client_secret': None,
        'user_agent': 'HARLF-Sentiment-Analysis/1.0'
    }
}

# Logging Settings
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': '../outputs/sentiment_analysis.log'
} 