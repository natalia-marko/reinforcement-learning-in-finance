"""
Centralized path management

This module provides a single source of truth for all file paths used
throughout the project, making it easy to change directory structures.
"""

from pathlib import Path


class Paths:
    """Centralized path management"""
    
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    PLOTS_DIR = PROJECT_ROOT / 'plots'
    CONFIGS_DIR = PROJECT_ROOT / 'configs'
    ARCHIVE_DIR = PROJECT_ROOT / 'archive'
    
    # Data files
    PRICE_DATA = DATA_DIR / 'price_data.csv'
    TECHNICAL_FEATURES = DATA_DIR / 'technical_features.csv'
    SENTIMENT_FEATURES = DATA_DIR / 'sentiment_features.csv'
    REGIME_INDICATORS = DATA_DIR / 'regime_indicators.csv'
    QQQ_BENCHMARK = DATA_DIR / 'qqq_benchmark.csv'
    
    # Model files
    BASE_MODELS_DIR = MODELS_DIR
    SUPER_MODEL_DIR = MODELS_DIR / 'super_agent_sac'
    META_MODEL_DIR = MODELS_DIR / 'meta_agent'
    
    # Required base model files
    TECH_PPO_MODEL = BASE_MODELS_DIR / 'best_technical_PPO.zip'
    TECH_SAC_MODEL = BASE_MODELS_DIR / 'best_technical_SAC.zip'
    SENT_PPO_MODEL = BASE_MODELS_DIR / 'best_sentiment_PPO.zip'
    SENT_SAC_MODEL = BASE_MODELS_DIR / 'best_sentiment_SAC.zip'
    
    # Results files
    WALK_FORWARD_RESULTS = RESULTS_DIR / 'walk_forward_results.csv'
    PORTFOLIO_ALLOCATIONS = RESULTS_DIR / 'portfolio_allocations.png'
    
    # Config files
    DEFAULT_CONFIG = CONFIGS_DIR / 'default_config.json'
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories if they don't exist"""
        dirs = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR,
            cls.CONFIGS_DIR,
            cls.ARCHIVE_DIR,
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get path for a specific model file"""
        return cls.MODELS_DIR / model_name
    
    @classmethod
    def get_result_path(cls, result_name: str) -> Path:
        """Get path for a specific result file"""
        return cls.RESULTS_DIR / result_name

