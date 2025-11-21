"""
Verification script to test data pipeline correctness
Run this AFTER implementing fixes to ensure everything is correct
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path


def test_global_normalization(data_dir='data/walk_forward'):
    """Test that all folds use the same global scaler"""
    print("\n" + "="*60)
    print("TEST 1: Global Normalization")
    print("="*60)
    
    try:
        # Load global scaler
        global_scaler_path = Path(data_dir) / 'global_scaler.pkl'
        global_scaler = joblib.load(global_scaler_path)
        
        print(f"✅ Global scaler found: {global_scaler_path}")
        print(f"   Features: {len(global_scaler.mean_)}")
        print(f"   Mean range: [{global_scaler.mean_.min():.3f}, {global_scaler.mean_.max():.3f}]")
        print(f"   Std range: [{global_scaler.scale_.min():.3f}, {global_scaler.scale_.max():.3f}]")
        
        # Verify all folds reference same scaler
        with open('data/metadata/preprocessing_params.json', 'r') as f:
            params = json.load(f)
        
        scaler_paths = set()
        for fold_key in params.keys():
            scaler_path = params[fold_key].get('global_scaler_path')
            if scaler_path:
                scaler_paths.add(scaler_path)
        
        if len(scaler_paths) == 1:
            print(f"✅ All {len(params)} folds use the same global scaler")
            return True
        else:
            print(f"❌ FAILED: Folds use {len(scaler_paths)} different scalers!")
            print(f"   Scaler paths: {scaler_paths}")
            return False
            
    except FileNotFoundError as e:
        print(f"❌ FAILED: {e}")
        print("   Global scaler not found - per-fold normalization detected!")
        return False
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_feature_target_alignment(fold_id=0, data_dir='data/walk_forward'):
    """Test that features and targets are properly aligned"""
    print("\n" + "="*60)
    print("TEST 2: Feature-Target Alignment")
    print("="*60)
    
    try:
        fold_dir = Path(data_dir) / f'fold_{fold_id}'
        
        # Load features and targets
        train_features = pd.read_csv(fold_dir / 'train_features.csv', index_col=0)
        train_targets = pd.read_csv(fold_dir / 'train_targets.csv', index_col=0)
        
        print(f"Fold {fold_id}:")
        print(f"  Features: {train_features.shape}")
        print(f"  Targets:  {train_targets.shape}")
        print(f"  Feature date range: {train_features.index[0]} to {train_features.index[-1]}")
        print(f"  Target date range:  {train_targets.index[0]} to {train_targets.index[-1]}")
        
        # Check alignment
        if train_features.index[0] != train_targets.index[0]:
            print(f"❌ FAILED: Start dates don't match!")
            return False
        
        if train_features.index[-1] != train_targets.index[-1]:
            print(f"❌ FAILED: End dates don't match!")
            return False
        
        if len(train_features) != len(train_targets):
            print(f"❌ FAILED: Different number of samples!")
            return False
        
        print("✅ Features and targets are properly aligned")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_feature_normalization(fold_id=0, data_dir='data/walk_forward'):
    """Test that features are normalized correctly"""
    print("\n" + "="*60)
    print("TEST 3: Feature Normalization")
    print("="*60)

    try:
        fold_dir = Path(data_dir) / f'fold_{fold_id}'
        train_features = pd.read_csv(fold_dir / 'train_features.csv', index_col=0)

        # Remove non-normalized columns (ticker and regime one-hot features)
        exclude_cols = ['ticker'] + [f'regime_{i}' for i in range(3)]
        normalized_features = train_features.drop(
            columns=[c for c in exclude_cols if c in train_features.columns]
        )

        # Calculate statistics on ONLY normalized features
        feature_means = normalized_features.mean()
        feature_stds = normalized_features.std()

        mean_of_means = feature_means.mean()
        mean_of_stds = feature_stds.mean()

        print(f"Feature statistics (fold {fold_id} training data):")
        print(f"  Normalized features only: {len(normalized_features.columns)}")
        print(f"  Mean of means: {mean_of_means:.6f} (should be ~0)")
        print(f"  Mean of stds:  {mean_of_stds:.6f} (should be ~1)")
        print(f"  Mean range: [{feature_means.min():.3f}, {feature_means.max():.3f}]")
        print(f"  Std range:  [{feature_stds.min():.3f}, {feature_stds.max():.3f}]")

        # Check if normalized (means near 0, stds near 1)
        if abs(mean_of_means) < 0.1 and 0.8 < mean_of_stds < 1.2:
            print("✅ Features are properly normalized")
            return True
        else:
            print("❌ FAILED: Features are not normalized correctly!")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_target_not_normalized(fold_id=0, data_dir='data/walk_forward'):
    """Test that targets are NOT normalized"""
    print("\n" + "="*60)
    print("TEST 4: Target NOT Normalized")
    print("="*60)
    
    try:
        fold_dir = Path(data_dir) / f'fold_{fold_id}'
        train_targets = pd.read_csv(fold_dir / 'train_targets.csv', index_col=0)
        
        # Check forward_log_return
        if 'forward_log_return' not in train_targets.columns:
            print("❌ FAILED: forward_log_return not found in targets!")
            return False
        
        log_returns = train_targets['forward_log_return'].dropna()
        
        mean = log_returns.mean()
        std = log_returns.std()
        min_val = log_returns.min()
        max_val = log_returns.max()
        
        print(f"Target statistics (forward_log_return):")
        print(f"  Mean: {mean:.6f}")
        print(f"  Std:  {std:.6f}")
        print(f"  Min:  {min_val:.6f} ({min_val*100:.2f}%)")
        print(f"  Max:  {max_val:.6f} ({max_val*100:.2f}%)")
        
        # Check if raw (NOT normalized)
        # Daily log returns should have std between 0.005 and 0.1
        if 0.005 < std < 0.1:
            print("✅ Targets are raw (not normalized)")
            return True
        else:
            print("❌ FAILED: Targets appear to be normalized!")
            print("   Expected std between 0.005 and 0.1 for daily log returns")
            return False
            
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_close_price_removed(fold_id=0, data_dir='data/walk_forward'):
    """Test that close_price is not in targets"""
    print("\n" + "="*60)
    print("TEST 5: close_price Removed from Targets")
    print("="*60)
    
    try:
        fold_dir = Path(data_dir) / f'fold_{fold_id}'
        train_targets = pd.read_csv(fold_dir / 'train_targets.csv', index_col=0)
        
        target_cols = [c for c in train_targets.columns if c != 'ticker']
        print(f"Target columns: {target_cols}")
        
        if 'close_price' in target_cols:
            print("❌ FAILED: close_price found in targets!")
            print("   close_price should NOT be a target column")
            return False
        
        # Check that we have the expected targets
        expected_targets = ['forward_log_return', 'forward_return']
        missing = set(expected_targets) - set(target_cols)
        
        if missing:
            print(f"⚠️  WARNING: Missing expected targets: {missing}")
        
        print("✅ close_price removed from targets")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_extreme_returns(fold_id=0, data_dir='data/walk_forward', threshold=0.30):
    """Test for extreme returns exceeding the cap (should be ±30%)"""
    print("\n" + "="*60)
    print("TEST 6: Check for Extreme Returns (Cap = ±30%)")
    print("="*60)

    try:
        fold_dir = Path(data_dir) / f'fold_{fold_id}'
        train_targets = pd.read_csv(fold_dir / 'train_targets.csv', index_col=0)

        log_returns = train_targets['forward_log_return'].dropna()

        # Find extreme returns that exceed the cap
        extreme = train_targets[train_targets['forward_log_return'].abs() > threshold]

        print(f"Return statistics:")
        print(f"  Mean: {log_returns.mean():.6f}")
        print(f"  Std:  {log_returns.std():.6f}")
        print(f"  Min:  {log_returns.min():.6f} ({log_returns.min()*100:.2f}%)")
        print(f"  Max:  {log_returns.max():.6f} ({log_returns.max()*100:.2f}%)")
        print(f"  Returns exceeding ±{threshold*100:.0f}% cap: {len(extreme)}")

        if len(extreme) > 0:
            print(f"\n❌ FAILED: Found returns exceeding ±{threshold*100:.0f}% cap!")
            print(extreme[['ticker', 'forward_log_return']].head(10))
            print("\nThis indicates the return capping is not working correctly.")
            return False
        else:
            print(f"✅ All returns within ±{threshold*100:.0f}% cap")
            # Show warning if there are returns between 10-30% (high but acceptable)
            high_returns = train_targets[
                (train_targets['forward_log_return'].abs() > 0.10) &
                (train_targets['forward_log_return'].abs() <= threshold)
            ]
            if len(high_returns) > 0:
                print(f"\nℹ️  Note: Found {len(high_returns)} returns between 10-30% (high but within cap)")
            return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_scaler_consistency(data_dir='data/walk_forward', n_folds=5):
    """Test that early and late folds use same normalization"""
    print("\n" + "="*60)
    print("TEST 7: Scaler Consistency Across Folds")
    print("="*60)

    try:
        # Load features from fold 0 and last fold
        fold_0_features = pd.read_csv(Path(data_dir) / 'fold_0' / 'train_features.csv', index_col=0)

        # Find last fold
        fold_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir() and d.name.startswith('fold_')])
        last_fold_id = len(fold_dirs) - 1
        last_fold_features = pd.read_csv(Path(data_dir) / f'fold_{last_fold_id}' / 'train_features.csv', index_col=0)

        # Remove non-normalized columns (ticker and regime one-hot features)
        exclude_cols = ['ticker'] + [f'regime_{i}' for i in range(3)]
        fold_0_features = fold_0_features.drop(
            columns=[c for c in exclude_cols if c in fold_0_features.columns]
        )
        last_fold_features = last_fold_features.drop(
            columns=[c for c in exclude_cols if c in last_fold_features.columns]
        )

        # Compare feature statistics
        fold_0_mean = fold_0_features.mean().mean()
        fold_0_std = fold_0_features.std().mean()

        last_fold_mean = last_fold_features.mean().mean()
        last_fold_std = last_fold_features.std().mean()

        print(f"Fold 0 feature statistics (normalized features only):")
        print(f"  Mean: {fold_0_mean:.6f}")
        print(f"  Std:  {fold_0_std:.6f}")

        print(f"\nFold {last_fold_id} feature statistics (normalized features only):")
        print(f"  Mean: {last_fold_mean:.6f}")
        print(f"  Std:  {last_fold_std:.6f}")

        # Both should be close to 0 and 1 (normalized)
        if abs(fold_0_mean) < 0.1 and abs(last_fold_mean) < 0.1 and \
           0.8 < fold_0_std < 1.2 and 0.8 < last_fold_std < 1.2:
            print("\n✅ Both early and late folds are properly normalized")
            return True
        else:
            print("\n❌ FAILED: Inconsistent normalization across folds!")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def run_all_tests():
    """Run all verification tests"""
    print("="*60)
    print("DATA PIPELINE VERIFICATION")
    print("="*60)
    print("\nRunning comprehensive tests to verify:")
    print("1. Global normalization (no per-fold fitting)")
    print("2. Feature-target alignment (no misalignment from lagging)")
    print("3. Features normalized (mean~0, std~1)")
    print("4. Targets NOT normalized (realistic returns)")
    print("5. close_price removed from targets")
    print("6. No extreme returns (potential data issues)")
    print("7. Scaler consistency across folds")
    
    results = {}
    
    # Run tests
    results['global_normalization'] = test_global_normalization()
    results['alignment'] = test_feature_target_alignment()
    results['feature_normalized'] = test_feature_normalization()
    results['target_raw'] = test_target_not_normalized()
    results['close_price_removed'] = test_close_price_removed()
    results['extreme_returns'] = test_extreme_returns()
    results['scaler_consistency'] = test_scaler_consistency()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*60)
        print("🟢 ALL TESTS PASSED - READY TO TRAIN!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("🔴 SOME TESTS FAILED - DO NOT TRAIN YET!")
        print("="*60)
        print("\nPlease fix the failing tests before training.")
        print("See pipeline_analysis_and_fixes.md for detailed instructions.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
