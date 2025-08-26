# Prediction vs Label Logging Feature

## Overview

This document describes the new feature added to the Higgs Audio LoRA trainer that logs predictions vs labels for debugging and monitoring purposes.

## Feature Details

### What Was Added

1. **Prediction vs Label Comparison Logging**:
   - Logs first 5 predictions and labels for each sample
   - Logs last 5 predictions and labels for each sample (if sequence length > 5)
   - Limited to first 2 samples in each batch for readability
   - Limited to first 10 tokens per sequence for readability

2. **Configurable Logging Frequency**:
   - Uses the existing `--log_steps` argument to control how often predictions are logged
   - Only logs on the main process (local_rank == 0) in distributed training

3. **Safe Implementation**:
   - Wrapped in try/except block to prevent training interruption
   - Only activates every n log steps to avoid overwhelming logs
   - Does not modify the core training logic

### Where It Was Added

The feature was added to the [_compute_dual_loss](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L288-L435) method in trainer.py:
- After text loss computation in both the optimal path and fallback path
- Uses a new helper method [_log_predictions_vs_labels](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L417-L452) to handle the actual logging

### How It Works

1. **Step Counter**: A forward step counter tracks training progress
2. **Conditional Logging**: Only logs when `step_count % log_steps == 0`
3. **Prediction Extraction**: Uses `torch.argmax(logits, dim=-1)` to get predictions
4. **Selective Display**: Shows first/last 5 tokens for up to 2 samples per batch
5. **Error Handling**: Gracefully handles any issues without stopping training

### Example Log Output

```
Sample 1 Prediction vs Label Comparison:
  First 5 - Predictions: [1234, 5678, 9012, 3456, 7890]
  First 5 - Labels:      [1234, 5678, 9012, 3456, 7891]
  Last 5  - Predictions: [2345, 6789, 0123, 4567, 8901]
  Last 5  - Labels:      [2345, 6789, 0123, 4567, 8902]
```

### Benefits

1. **Debugging**: Helps identify model behavior patterns
2. **Monitoring**: Provides insight into training progress
3. **Validation**: Verifies model is learning meaningful patterns
4. **Non-Intrusive**: Doesn't affect training performance or results

### Usage

The feature automatically activates when using the trainer with the `--log_steps` argument:
```bash
torchrun --nproc_per_node=8 trainer.py \
  --train_manifest /path/train.jsonl \
  --output_dir /path/out \
  --log_steps 10  # Predictions will be logged every 10 steps
```

No additional arguments or configuration needed.