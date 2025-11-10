#!/usr/bin/env python3
"""
Run data pipeline end-to-end
Usage: python scripts/run_data_pipeline.py [--max-folds N]
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from harlf.data_pipeline import pipeline


def main():
    parser = argparse.ArgumentParser(description='Run data pipeline')
    parser.add_argument('--max-folds', type=int, default=None,
                       help='Maximum number of folds (default: all)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip data download if raw data exists')
    args = parser.parse_args()
    
    result = pipeline.run_full_pipeline(
        max_folds=args.max_folds,
        skip_download=args.skip_download
    )
    
    print("\n✅ Pipeline complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
