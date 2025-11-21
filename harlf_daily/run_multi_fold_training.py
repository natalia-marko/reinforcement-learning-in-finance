"""
Automated Multi-Fold Training Script

Trains PPO agent on multiple folds sequentially or in parallel.
Tracks progress, saves checkpoints, and generates comparison reports.

Usage:
    python run_multi_fold_training.py --folds 0 1 2 3 4 --timesteps 50000
    python run_multi_fold_training.py --all-folds --reward ema_sharpe
    python run_multi_fold_training.py --folds 0 --quick  # 10k steps for testing

Created: 2025-01-11
"""

import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class MultiFoldTrainer:
    """Automated multi-fold training manager."""

    def __init__(self, output_dir='./multi_fold_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
        self.start_time = None

    def train_single_fold(self, fold_id, timesteps=50000, reward_type='ema_sharpe', verbose=1):
        """Train on a single fold."""
        print(f"\n{'='*70}")
        print(f"🚀 TRAINING FOLD {fold_id}")
        print(f"   Timesteps: {timesteps:,}")
        print(f"   Reward: {reward_type}")
        print('='*70)

        fold_start = time.time()

        # Build command
        cmd = [
            'python', 'harlf/train_ppo.py',
            '--fold_id', str(fold_id),
            '--total_timesteps', str(timesteps),
            '--reward_type', reward_type,
            '--verbose', str(verbose),
            '--tensorboard_log', './tensorboard_logs'
        ]

        # Run training
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            elapsed = time.time() - fold_start

            if result.returncode == 0:
                print(f"\n✅ Fold {fold_id} completed in {elapsed/60:.1f} minutes")

                # Parse results from output
                metrics = self._parse_training_output(result.stdout)
                metrics['fold_id'] = fold_id
                metrics['elapsed_time'] = elapsed
                metrics['status'] = 'success'

                self.results[fold_id] = metrics

                # Save checkpoint
                self._save_checkpoint()

                return True
            else:
                print(f"\n❌ Fold {fold_id} failed!")
                print(f"Error: {result.stderr[:500]}")

                self.results[fold_id] = {
                    'fold_id': fold_id,
                    'status': 'failed',
                    'error': result.stderr[:500]
                }

                return False

        except subprocess.TimeoutExpired:
            print(f"\n⏱️  Fold {fold_id} timed out!")
            self.results[fold_id] = {
                'fold_id': fold_id,
                'status': 'timeout'
            }
            return False

        except Exception as e:
            print(f"\n❌ Fold {fold_id} error: {e}")
            self.results[fold_id] = {
                'fold_id': fold_id,
                'status': 'error',
                'error': str(e)
            }
            return False

    def _parse_training_output(self, output):
        """Parse metrics from training output."""
        metrics = {}

        # Look for final metrics (this is simplified - would need actual parsing logic)
        lines = output.split('\n')

        for line in lines:
            if 'Final Sharpe' in line:
                try:
                    metrics['final_sharpe'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'Final Value' in line:
                try:
                    metrics['final_value'] = float(line.split(':')[-1].strip().replace('$', '').replace(',', ''))
                except:
                    pass

        return metrics

    def _save_checkpoint(self):
        """Save training progress checkpoint."""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results
        }

        checkpoint_path = self.output_dir / 'training_checkpoint.json'
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def train_all_folds(self, folds, timesteps=50000, reward_type='simple'):
        """Train on all specified folds."""
        self.start_time = time.time()

        print("="*70)
        print(f"MULTI-FOLD TRAINING")
        print(f"Folds: {folds}")
        print(f"Timesteps per fold: {timesteps:,}")
        print(f"Reward type: {reward_type}")
        print("="*70)

        successful = []
        failed = []

        for fold_id in folds:
            success = self.train_single_fold(fold_id, timesteps, reward_type)

            if success:
                successful.append(fold_id)
            else:
                failed.append(fold_id)

            # Progress update
            print(f"\n📊 Progress: {len(successful) + len(failed)}/{len(folds)} folds")
            print(f"   ✅ Successful: {len(successful)}")
            print(f"   ❌ Failed: {len(failed)}")

        # Generate final report
        self._generate_final_report(successful, failed)

    def _generate_final_report(self, successful, failed):
        """Generate comprehensive final report."""
        total_time = time.time() - self.start_time

        print("\n" + "="*70)
        print("📊 MULTI-FOLD TRAINING COMPLETE")
        print("="*70)

        print(f"\n⏱️  Total Time: {total_time/60:.1f} minutes")
        print(f"✅ Successful: {len(successful)}/{len(successful) + len(failed)}")
        print(f"❌ Failed: {len(failed)}/{len(successful) + len(failed)}")

        if successful:
            print(f"\nSuccessful folds: {successful}")

        if failed:
            print(f"\nFailed folds: {failed}")

        # Save results
        results_df = pd.DataFrame(self.results).T
        csv_path = self.output_dir / 'multi_fold_results.csv'
        results_df.to_csv(csv_path)

        print(f"\n📁 Results saved to: {csv_path}")

        # Generate markdown report
        self._create_markdown_report(successful, failed, total_time)

    def _create_markdown_report(self, successful, failed, total_time):
        """Create detailed markdown report."""
        report = []
        report.append("# Multi-Fold Training Report")
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Time**: {total_time/60:.1f} minutes")
        report.append("")

        report.append("## Summary")
        report.append("")
        report.append(f"- Total folds: {len(successful) + len(failed)}")
        report.append(f"- Successful: {len(successful)}")
        report.append(f"- Failed: {len(failed)}")
        report.append(f"- Success rate: {len(successful)/(len(successful)+len(failed))*100:.1f}%")
        report.append("")

        if successful:
            report.append("## Successful Folds")
            report.append("")
            for fold_id in successful:
                metrics = self.results[fold_id]
                report.append(f"### Fold {fold_id}")
                report.append(f"- Time: {metrics.get('elapsed_time', 0)/60:.1f} min")
                if 'final_sharpe' in metrics:
                    report.append(f"- Sharpe: {metrics['final_sharpe']:.3f}")
                report.append("")

        if failed:
            report.append("## Failed Folds")
            report.append("")
            for fold_id in failed:
                report.append(f"- Fold {fold_id}: {self.results[fold_id].get('status', 'unknown')}")
            report.append("")

        report.append("## Next Steps")
        report.append("")
        report.append("1. Compare results across folds")
        report.append("2. Evaluate on test sets")
        report.append("3. Aggregate metrics")
        report.append("4. Run production deployment if results good")

        # Save report
        report_path = self.output_dir / 'MULTI_FOLD_REPORT.md'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

        print(f"📄 Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-fold training automation')
    parser.add_argument('--folds', type=int, nargs='+', default=None, help='Specific folds to train')
    parser.add_argument('--all-folds', action='store_true', help='Train on all 10 folds')
    parser.add_argument('--timesteps', type=int, default=50000, help='Timesteps per fold')
    parser.add_argument('--reward', choices=['simple', 'ema_sharpe'], default='simple',
                       help='Reward function')
    parser.add_argument('--quick', action='store_true', help='Quick test (10k steps)')
    parser.add_argument('--output_dir', type=str, default='./multi_fold_results',
                       help='Output directory')

    args = parser.parse_args()

    # Determine folds
    if args.all_folds:
        folds = list(range(10))
    elif args.folds:
        folds = args.folds
    else:
        folds = [0]  # Default to fold 0

    # Determine timesteps
    timesteps = 10000 if args.quick else args.timesteps

    # Create trainer
    trainer = MultiFoldTrainer(output_dir=args.output_dir)

    # Run training
    trainer.train_all_folds(folds=folds, timesteps=timesteps, reward_type=args.reward)

    print("\n" + "="*70)
    print("✅ ALL TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
