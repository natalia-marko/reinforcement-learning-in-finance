#!/usr/bin/env python3
"""
Quick verification that system is ready to train with fixed early stopping
"""
import sys
from pathlib import Path

def verify_system():
    """Verify all components are ready"""
    print("="*80)
    print("TRAINING READINESS CHECK")
    print("="*80)

    checks = []

    # 1. Check custom callback exists
    print("\n1. Checking custom_callbacks.py...")
    try:
        from custom_callbacks import StopTrainingOnNoModelImprovementFixed
        print("   ✓ Custom callback imported successfully")
        checks.append(True)
    except ImportError as e:
        print(f"   ✗ Failed to import callback: {e}")
        checks.append(False)

    # 2. Check config
    print("\n2. Checking config.py...")
    try:
        import config
        print(f"   ✓ Early stopping patience: {config.EARLY_STOP_PATIENCE}")
        print(f"   ✓ Eval frequency: {config.EVAL_FREQ}")
        print(f"   ✓ LSTM hidden size: {config.LSTM_HIDDEN_SIZE}")
        print(f"   ✓ Features: {len(config.FEATURE_COLS)}")
        checks.append(True)
    except Exception as e:
        print(f"   ✗ Config error: {e}")
        checks.append(False)

    # 3. Check old models deleted
    print("\n3. Checking old models deleted...")
    models_dir = Path('models_minimal_features')
    if models_dir.exists():
        print(f"   ⚠️  WARNING: {models_dir} still exists!")
        print(f"   Run: rm -rf {models_dir}")
        checks.append(False)
    else:
        print(f"   ✓ Old models directory deleted")
        checks.append(True)

    # 4. Check training notebook exists
    print("\n4. Checking training notebook...")
    notebook = Path('07_train_minimal_features.ipynb')
    if notebook.exists():
        print(f"   ✓ Notebook found: {notebook}")

        # Check for fixed callback in notebook
        content = notebook.read_text()
        if 'StopTrainingOnNoModelImprovementFixed' in content:
            print("   ✓ Notebook uses fixed callback")
            checks.append(True)
        else:
            print("   ✗ Notebook doesn't use fixed callback!")
            checks.append(False)
    else:
        print(f"   ✗ Notebook not found: {notebook}")
        checks.append(False)

    # 5. Check data files exist
    print("\n5. Checking data files...")
    try:
        import config
        if config.DATA_PATH.exists():
            print(f"   ✓ Training data exists: {config.DATA_PATH.name}")
            checks.append(True)
        else:
            print(f"   ✗ Training data not found: {config.DATA_PATH}")
            checks.append(False)
    except Exception as e:
        print(f"   ✗ Error checking data: {e}")
        checks.append(False)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(checks)
    total = len(checks)
    print(f"Checks passed: {passed}/{total}")

    if all(checks):
        print("\n✓ SYSTEM READY TO TRAIN!")
        print("\nNext steps:")
        print("  1. Open: jupyter notebook 07_train_minimal_features.ipynb")
        print("  2. Run: Kernel → Restart & Run All")
        print("  3. Wait: 4-6 hours")
        print("  4. Compare: python analyze_overfitting.py")
        return True
    else:
        print("\n✗ SYSTEM NOT READY - Fix issues above")
        return False

if __name__ == '__main__':
    success = verify_system()
    sys.exit(0 if success else 1)
