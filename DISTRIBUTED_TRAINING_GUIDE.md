# 🚀 Higgs-Audio Distributed Training Guide

## ✅ Fixed Issues
The "trainer is not a package" error in distributed training has been **completely resolved**. This guide provides instructions for running robust 8xH200 distributed training.

## 📋 Prerequisites

### Hardware Requirements
- 8x NVIDIA H200 GPUs (recommended)
- At least 192GB total VRAM (24GB per GPU)
- High-speed interconnect (NVLink/InfiniBand)
- 128+ CPU cores recommended

### Software Requirements
```bash
# Essential dependencies
pip install torch>=2.0.0 torchvision torchaudio
pip install transformers>=4.30.0
pip install peft>=0.4.0
pip install accelerate
pip install datasets
pip install numpy scipy
pip install librosa soundfile
pip install loguru
```

## 🎯 Quick Start (Recommended)

The **easiest way** to run distributed training is using the robust launcher script:

```bash
# Navigate to higgs-audio root directory
cd /path/to/higgs-audio

# Run the robust launcher (handles all setup automatically)
./scripts/launch_8xh200_training_robust.sh
```

This script:
- ✅ Validates the environment
- ✅ Pre-configures the trainer package 
- ✅ Sets up distributed training variables
- ✅ Launches torchrun with optimal settings
- ✅ Provides detailed error diagnostics

## 🔧 Manual Setup (Advanced)

If you prefer manual control, follow these steps:

### Step 1: Environment Setup
```bash
# Set working directory
cd /path/to/higgs-audio

# Set environment variables
export PYTHONPATH="/path/to/higgs-audio:$PYTHONPATH"
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Pre-setup trainer package (critical for distributed training)
python3 trainer_setup.py
```

### Step 2: Prepare Training Data
```bash
# Validate your training data format
python3 trainer/train.py --validate_data_only --train_data /path/to/your/data.json

# Or create sample data for testing
python3 trainer/train.py --create_sample_data sample_training_data.json
```

### Step 3: Launch Distributed Training
```bash
torchrun \
    --standalone \
    --nproc_per_node=8 \
    --nnodes=1 \
    trainer/train.py \
    --train_data /path/to/your/training_data.json \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --output_dir checkpoints/8xh200_training \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --use_gradient_checkpointing \
    --mixed_precision \
    --max_audio_length_seconds 30 \
    --dataloader_num_workers 4
```

## 🎛️ Configuration Options

### LoRA Fine-tuning Parameters
```bash
--lora_r 16                    # LoRA rank (8, 16, 32, 64)
--lora_alpha 32                # LoRA alpha (usually 2x rank)
--lora_dropout 0.1             # LoRA dropout rate
--lora_target_modules q_proj v_proj o_proj gate_proj up_proj down_proj  # DualFFN components
```

### Training Parameters
```bash
--batch_size 1                 # Per-GPU batch size
--gradient_accumulation_steps 8  # Effective batch size = batch_size * nproc * grad_accum
--learning_rate 2e-4           # Learning rate
--num_epochs 3                 # Number of training epochs
--max_grad_norm 1.0            # Gradient clipping
--weight_decay 0.01            # Weight decay
```

### Performance Optimizations
```bash
--use_gradient_checkpointing   # Save memory at cost of compute
--mixed_precision              # Use bfloat16 for faster training
--dataloader_num_workers 4     # Number of data loading workers
--max_audio_length_seconds 30  # Maximum audio length (affects memory)
```

### Hardware-Specific Settings
```bash
# For 8x H200 (24GB each)
--batch_size 1 --gradient_accumulation_steps 8

# For 8x A100 (40GB each) 
--batch_size 2 --gradient_accumulation_steps 4

# For 8x A100 (80GB each)
--batch_size 4 --gradient_accumulation_steps 2
```

## 🔍 Validation & Testing

### Pre-training Validation
```bash
# Test the distributed setup without training
python3 validate_8xh200_fix.py

# Validate training data format
python3 trainer/train.py --validate_data_only --train_data your_data.json

# Create sample data for testing
python3 trainer/train.py --create_sample_data test_data.json
```

### Quick Test Run
```bash
# Run a quick test with minimal settings
python3 trainer/train.py \
    --train_data test_data.json \
    --quick_test \
    --output_dir checkpoints/test
```

## 📁 Data Format

Your training data should be in ChatML format:

```json
[
  {
    "conversations": [
      {
        "from": "user", 
        "value": "Generate speech for: Hello world"
      },
      {
        "from": "assistant",
        "value": "<audio>path/to/reference_audio.wav</audio>Hello world"
      }
    ]
  }
]
```

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Monitor training logs
tail -f checkpoints/8xh200_training/training.log

# Monitor checkpoint directory
watch -n 5 'ls -la checkpoints/8xh200_training/'
```

### Performance Metrics
The trainer automatically logs:
- Training loss (text, audio, consistency)
- GPU memory usage per device
- Training speed (samples/second)
- Gradient norms
- Learning rate schedules

## 🔧 Troubleshooting

### Common Issues and Solutions

#### "trainer is not a package" Error
**Status: ✅ FIXED**
This error has been completely resolved in the updated codebase.

#### CUDA Out of Memory
```bash
# Reduce batch size
--batch_size 1

# Enable gradient checkpointing
--use_gradient_checkpointing

# Reduce sequence length
--max_audio_length_seconds 20
```

#### Slow Training Speed
```bash
# Enable mixed precision
--mixed_precision

# Increase data loading workers
--dataloader_num_workers 8

# Check NCCL setup
export NCCL_DEBUG=INFO
```

#### Import Errors
```bash
# Ensure you're in the higgs-audio root directory
cd /path/to/higgs-audio

# Run the trainer setup helper
python3 trainer_setup.py

# Check Python path
export PYTHONPATH="/path/to/higgs-audio:$PYTHONPATH"
```

### Diagnostic Commands
```bash
# Test package recognition
python3 -c "import trainer_setup; trainer_setup.setup_trainer_package()"

# Validate complete setup
python3 validate_8xh200_fix.py

# Check GPU availability
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## 📈 Performance Optimization Tips

### Memory Optimization
1. **Use gradient checkpointing** for large models
2. **Enable mixed precision** (bfloat16) training
3. **Optimize batch size** based on GPU memory
4. **Reduce audio length** if experiencing OOM

### Speed Optimization
1. **Use multiple data workers** (`--dataloader_num_workers 4-8`)
2. **Enable mixed precision** for faster computation
3. **Optimize NCCL settings** for your network topology
4. **Use NVLink/InfiniBand** for multi-GPU communication

### Convergence Optimization
1. **Use proper learning rate** (2e-4 for LoRA is good starting point)
2. **Enable gradient clipping** to prevent exploding gradients
3. **Monitor loss components** (text, audio, consistency)
4. **Use validation set** to track overfitting

## 🎉 Success Indicators

Your training is working correctly when you see:
- ✅ All 8 GPUs showing activity in `nvidia-smi`
- ✅ Training loss decreasing over time
- ✅ Memory usage stable (not growing indefinitely)
- ✅ Regular checkpoint saves
- ✅ No import or package errors

## 🆘 Getting Help

If you encounter issues:

1. **Run the validation script**: `python3 validate_8xh200_fix.py`
2. **Check the logs** in your output directory
3. **Verify your environment** matches the prerequisites
4. **Use the robust launcher** instead of manual setup

## 📚 Additional Resources

- **Training Configuration**: See `trainer/config.py` for all available options
- **Performance Monitoring**: Built-in monitoring logs to your output directory
- **Sample Data**: Use `--create_sample_data` to generate test data
- **Validation Scripts**: Multiple validation scripts ensure everything works correctly

---

**Ready to train?** Use the robust launcher script for the best experience:
```bash
cd /path/to/higgs-audio
./scripts/launch_8xh200_training_robust.sh
```