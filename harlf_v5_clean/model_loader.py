"""
Model loading utilities with error handling and validation
"""
import os
from stable_baselines3 import PPO, SAC


def load_base_models(models_dir='models'):
    """
    Load all required base models with error handling
    
    Args:
        models_dir: Directory containing model files (default: 'models')
        
    Returns:
        dict: Dictionary of loaded models with keys:
            - 'tech_PPO': PPO model for technical features
            - 'tech_SAC': SAC model for technical features
            - 'sent_PPO': PPO model for sentiment features
            - 'sent_SAC': SAC model for sentiment features
            
    Raises:
        FileNotFoundError: If any required model file is missing
    """
    required_models = {
        'tech_PPO': 'best_technical_PPO.zip',
        'tech_SAC': 'best_technical_SAC.zip',
        'sent_PPO': 'best_sentiment_PPO.zip',
        'sent_SAC': 'best_sentiment_SAC.zip'
    }
    
    models = {}
    missing_models = []
    
    for name, filename in required_models.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            missing_models.append(path)
            continue
        
        try:
            if 'SAC' in name:
                models[name] = SAC.load(path)
            else:
                models[name] = PPO.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {path}: {e}") from e
    
    if missing_models:
        error_msg = "Required model files not found:\n"
        error_msg += "\n".join(f"  - {path}" for path in missing_models)
        error_msg += "\n\nPlease train base agents first using 01_base_agents.ipynb"
        raise FileNotFoundError(error_msg)
    
    return models


def check_required_models(models_dir='models'):
    """
    Check if all required model files exist
    
    Args:
        models_dir: Directory containing model files (default: 'models')
        
    Returns:
        bool: True if all models exist, False otherwise
        
    Raises:
        FileNotFoundError: If any required model file is missing
    """
    required_models = [
        'best_technical_PPO.zip',
        'best_technical_SAC.zip',
        'best_sentiment_PPO.zip',
        'best_sentiment_SAC.zip'
    ]
    
    missing = []
    existing = []
    
    for filename in required_models:
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            existing.append(f"  ✓ {path}")
        else:
            missing.append(f"  ✗ {path}")
    
    print("Model File Status:")
    print("=" * 70)
    
    if existing:
        print("\nExisting models:")
        print("\n".join(existing))
    
    if missing:
        print("\nMissing models:")
        print("\n".join(missing))
        print("\n" + "=" * 70)
        raise FileNotFoundError(
            f"Missing {len(missing)} required model file(s). "
            "Please train base agents first using 01_base_agents.ipynb"
        )
    
    print("\n" + "=" * 70)
    print(f"✓ All {len(required_models)} required models found!")
    print("=" * 70)
    
    return True

