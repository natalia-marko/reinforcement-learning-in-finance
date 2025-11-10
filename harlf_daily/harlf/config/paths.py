"""
Path configurations for HARLF project
Auto-generated during migration
"""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
WALK_FORWARD_DIR = DATA_DIR / 'walk_forward'
METADATA_DIR = DATA_DIR / 'metadata'

# Model directories
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
TENSORBOARD_DIR = PROJECT_ROOT / 'tensorboard_logs'
