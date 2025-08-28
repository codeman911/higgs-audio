# Distributed Training Timeout Fix

## Issue Description
During distributed training with `torchrun`, the training process was experiencing collective operation timeouts in the NCCL backend, specifically:
```
[Rank 6] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=47, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
```

This error was causing the entire training process to crash with a SIGABRT signal.

## Root Cause Analysis

### 1. Collective Operation Timeout
The primary issue was that collective operations (specifically ALLREDUCE) were timing out during checkpoint saving. This was happening because:

1. **Simultaneous Model Saving**: All processes were calling `model.save_pretrained()` simultaneously, which triggers collective operations in the NCCL backend
2. **Improper Synchronization**: The synchronization logic was not properly handling the distributed nature of checkpoint saving
3. **Insufficient Timeout**: The default NCCL timeout (600 seconds) was not sufficient for large model saving operations

### 2. Code Analysis
In the original implementation:
```python
# PROBLEMATIC CODE:
if self.global_step % self.args.save_steps == 0:
    # All processes were calling save_checkpoint() which internally calls model.save_pretrained()
    # This triggers collective operations that need synchronization
    checkpoint_success = self.save_checkpoint()  # Called by ALL processes
    
    if self.world_size > 1:
        torch.distributed.barrier()  # Synchronization point
```

The issue was that `save_checkpoint()` internally calls `model.save_pretrained()`, which triggers collective operations. When all 8 processes call this simultaneously, it creates a bottleneck that exceeds the default timeout.

## Solution Implemented

### 1. Increased NCCL Timeout
Increased the NCCL timeout to 30 minutes (1800 seconds) to accommodate large model saving operations:
```python
dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=1800))
```

### 2. Restructured Checkpoint Saving Logic
Modified the checkpoint saving logic to ensure only rank 0 saves checkpoints:
```python
# FIXED CODE:
if self.global_step % self.args.save_steps == 0:
    # Only rank 0 should save checkpoints in distributed training
    checkpoint_success = True
    if self.local_rank == 0:
        checkpoint_success = self.save_checkpoint()
    
    # All processes participate in synchronization
    if self.world_size > 1:
        torch.distributed.barrier()
        if self.local_rank == 0:  # Only log from main process
            if checkpoint_success:
                log_success("Checkpoint saved and synchronized across all processes")
```

### 3. Enhanced Error Handling
Added comprehensive error handling and logging to better diagnose issues:
- Detailed logging of checkpoint saving steps
- Permission checking for output directories
- Verification of saved files
- Metadata saving for checkpoint validation

## Key Changes Made

### In `trainer.py`:

1. **Increased NCCL Timeout**: Set to 30 minutes to prevent premature timeouts
2. **Restructured Checkpoint Logic**: Only rank 0 saves checkpoints, all ranks synchronize
3. **Enhanced Logging**: Added detailed logging for debugging
4. **Improved Error Handling**: Better error messages and traceback logging

## Verification

The fix has been tested with:
1. Single process checkpoint saving
2. Distributed process checkpoint saving (8 processes simulation)
3. Permission error handling
4. File verification after saving

## Expected Behavior

With this fix:
1. Training will no longer crash with NCCL collective operation timeouts
2. Checkpoints will be saved correctly by rank 0 only
3. All processes will synchronize properly after checkpoint saving
4. Error messages will be more informative for debugging

## Usage Instructions

Run training with the same command:
```bash
torchrun --nproc_per_node=8 trainer.py \
  --train_manifest ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
  --output_dir expmt_v1 \
  --base_ckpt bosonai/higgs-audio-v2-generation-3B-base \
  --batch_size 4 \
  --lr 1e-5 \
  --epochs 1 \
  --grad_accum 4 \
  --val_steps 500 \
  --save_steps 200
```

The training should now proceed without the NCCL timeout errors.