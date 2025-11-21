#!/bin/bash
# Multi-Fold Training Script
# Usage: bash train_multiple_folds.sh

echo "======================================================================"
echo "MULTI-FOLD TRAINING"
echo "======================================================================"

# Configuration
PYTHON_BIN="/Users/nataliamarko/miniconda3/envs/demand_forecast_env/bin/python"
TIMESTEPS=50000
REWARD_TYPE="simple"
FOLDS=(0 1 2)  # Change this to train more folds

# Create output directory
mkdir -p multi_fold_results

# Summary file
SUMMARY_FILE="multi_fold_results/training_summary.txt"
echo "Multi-Fold Training Summary" > $SUMMARY_FILE
echo "Started: $(date)" >> $SUMMARY_FILE
echo "===========================================" >> $SUMMARY_FILE

SUCCESSFUL=0
FAILED=0

# Train each fold
for FOLD in "${FOLDS[@]}"; do
    echo ""
    echo "======================================================================"
    echo "🚀 TRAINING FOLD $FOLD"
    echo "   Timesteps: $TIMESTEPS"
    echo "   Reward: $REWARD_TYPE"
    echo "======================================================================"

    # Run training
    LOG_FILE="multi_fold_results/fold${FOLD}_training.log"

    $PYTHON_BIN harlf/train_ppo.py \
        --fold_id $FOLD \
        --total_timesteps $TIMESTEPS \
        --reward_type $REWARD_TYPE \
        --verbose 1 \
        --tensorboard_log ./tensorboard_logs 2>&1 | tee $LOG_FILE

    # Check if successful
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✅ Fold $FOLD completed successfully!"
        echo "Fold $FOLD: SUCCESS" >> $SUMMARY_FILE
        SUCCESSFUL=$((SUCCESSFUL + 1))

        # Extract metrics from log
        echo "  Results:" >> $SUMMARY_FILE
        grep -A 3 "TRAIN Results:" $LOG_FILE | tail -3 >> $SUMMARY_FILE
        grep -A 3 "VAL Results:" $LOG_FILE | tail -3 >> $SUMMARY_FILE
        echo "" >> $SUMMARY_FILE
    else
        echo ""
        echo "❌ Fold $FOLD failed!"
        echo "Fold $FOLD: FAILED" >> $SUMMARY_FILE
        FAILED=$((FAILED + 1))
    fi
done

# Final summary
echo ""
echo "======================================================================"
echo "📊 MULTI-FOLD TRAINING COMPLETE"
echo "======================================================================"
echo "✅ Successful: $SUCCESSFUL/${#FOLDS[@]}"
echo "❌ Failed: $FAILED/${#FOLDS[@]}"

echo "" >> $SUMMARY_FILE
echo "===========================================" >> $SUMMARY_FILE
echo "Completed: $(date)" >> $SUMMARY_FILE
echo "Successful: $SUCCESSFUL/${#FOLDS[@]}" >> $SUMMARY_FILE
echo "Failed: $FAILED/${#FOLDS[@]}" >> $SUMMARY_FILE

echo ""
echo "📁 Results saved to: multi_fold_results/"
echo "📄 Summary: $SUMMARY_FILE"
echo ""
