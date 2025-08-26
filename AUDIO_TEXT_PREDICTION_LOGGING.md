# Audio and Text Prediction vs Label Logging Feature

## Overview

This document describes the enhanced logging feature added to the Higgs Audio LoRA trainer that logs both text and audio predictions vs labels for debugging and monitoring purposes.

## Features Implemented

### 1. Text Prediction vs Label Logging
- **First/Last 5 Tokens**: Logs the first 5 and last 5 predictions vs labels for each sample
- **Sample Limiting**: Limited to first 2 samples per batch for readability
- **Sequence Limiting**: Limited to first 10 tokens per sequence for readability

### 2. Audio Prediction vs Label Logging
- **Codebook Focus**: Focuses on the first codebook (out of 8) for readability
- **First/Last 5 Tokens**: Logs the first 5 and last 5 predictions vs labels
- **Sequence Limiting**: Limited to first 10 tokens per sequence for readability
- **Shape Information**: Logs audio logits and labels shapes for context

### 3. Configurable Logging Frequency
- **Integrated with Existing Args**: Uses the existing `--log_steps` argument to control logging frequency
- **Distributed Training Aware**: Only logs on the main process (local_rank == 0)
- **Non-Intrusive**: Doesn't affect training performance or results

### 4. Logger Cleanup and Prettification
- **Reduced Verbosity**: Removed excessive logging while keeping essential information
- **Cleaner Output**: Commented out debug logs that were overwhelming the output
- **Focused Information**: Retained only the most important logging information

## Technical Details

### Where It Was Added
The feature was added to:
1. **[_compute_dual_loss](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L320-L477) method**: For text prediction logging after loss computation
2. **[compute_loss](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L204-L270) method**: For audio prediction logging after audio loss computation
3. **New helper methods**: 
   - [_log_predictions_vs_labels](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L479-L520) for text predictions
   - [_log_audio_predictions_vs_labels](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py#L522-L561) for audio predictions

### How It Works
1. **Step Counter**: A forward step counter tracks training progress
2. **Conditional Logging**: Only logs when `step_count % log_steps == 0`
3. **Prediction Extraction**: Uses `torch.argmax(logits, dim=-1)` to get predictions
4. **Selective Display**: Shows first/last 5 tokens for up to 2 samples per batch
5. **Error Handling**: Gracefully handles any issues without stopping training

### Example Log Output

#### Text Predictions:
```
Sample 1 Prediction vs Label Comparison:
  First 5 - Predictions: [1234, 5678, 9012, 3456, 7890]
  First 5 - Labels:      [1234, 5678, 9012, 3456, 7891]
  Last 5  - Predictions: [2345, 6789, 0123, 4567, 8901]
  Last 5  - Labels:      [2345, 6789, 0123, 4567, 8902]
```

#### Audio Predictions:
```
Audio Prediction vs Label Comparison (Codebook 0):
  First 5 - Predictions: [123, 456, 789, 012, 345]
  First 5 - Labels:      [123, 456, 789, 012, 346]
  Last 5  - Predictions: [678, 901, 234, 567, 890]
  Last 5  - Labels:      [678, 901, 234, 567, 891]
  Logits shape: torch.Size([123, 8, 1026])
  Labels shape: torch.Size([8, 123])
```

## Benefits

1. **Comprehensive Debugging**: Helps identify model behavior patterns for both text and audio
2. **Monitoring**: Provides insight into training progress for both modalities
3. **Validation**: Verifies model is learning meaningful patterns in both text and audio
4. **Non-Intrusive**: Doesn't affect training performance or results
5. **Clean Output**: Reduced verbosity makes logs more readable and focused

## Usage

The feature automatically activates when using the trainer with the `--log_steps` argument:
```bash
torchrun --nproc_per_node=8 trainer.py \
  --train_manifest /path/train.jsonl \
  --output_dir /path/out \
  --log_steps 50  # Predictions will be logged every 50 steps
```

No additional arguments or configuration needed.