#!/bin/bash

# 🚀 Robust 8xH200 Distributed Training Launcher for Higgs-Audio
# This script provides a definitive solution for the "trainer is not a package" error
# by ensuring proper environment setup before launching distributed training.

echo "🚀 Robust 8xH200 Distributed Training Launcher"
echo "=" * 60

# Configuration
HIGGS_AUDIO_ROOT="/Users/vikram.solanki/Projects/exp/level1/higgs-audio"
TRAIN_DATA_PATH="../../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json"
OUTPUT_DIR="checkpoints/8xh200_robust_training"

# Step 1: Validate environment
echo "🔍 Step 1: Validating environment..."

if [ ! -d "$HIGGS_AUDIO_ROOT" ]; then
    echo "❌ Higgs-audio root directory not found: $HIGGS_AUDIO_ROOT"
    exit 1
fi

if [ ! -d "$HIGGS_AUDIO_ROOT/trainer" ]; then
    echo "❌ Trainer directory not found: $HIGGS_AUDIO_ROOT/trainer"
    exit 1
fi

if [ ! -d "$HIGGS_AUDIO_ROOT/boson_multimodal" ]; then
    echo "❌ boson_multimodal directory not found: $HIGGS_AUDIO_ROOT/boson_multimodal"
    echo "💡 Make sure you're running this from the correct higgs-audio repository"
    exit 1
fi

echo "✅ Environment validation passed"

# Step 2: Change to higgs-audio root directory
echo "🔄 Step 2: Changing to higgs-audio root directory..."
cd "$HIGGS_AUDIO_ROOT" || exit 1
echo "📍 Current directory: $(pwd)"

# Step 3: Pre-setup trainer package
echo "🔧 Step 3: Pre-setting up trainer package..."
python3 -c "
import sys
from pathlib import Path

# Add current directory to Python path
higgs_audio_root = Path.cwd()
if str(higgs_audio_root) not in sys.path:
    sys.path.insert(0, str(higgs_audio_root))

# Import and run trainer setup
try:
    import trainer_setup
    success = trainer_setup.setup_trainer_package()
    if success:
        print('✅ Trainer package pre-setup completed successfully')
        exit(0)
    else:
        print('❌ Trainer package pre-setup failed')
        exit(1)
except Exception as e:
    print(f'❌ Trainer package pre-setup error: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Trainer package pre-setup failed"
    exit 1
fi

# Step 4: Validate training data
echo "🔍 Step 4: Validating training data..."
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "❌ Training data not found: $TRAIN_DATA_PATH"
    echo "💡 Please check the path or create sample data"
    exit 1
fi
echo "✅ Training data found: $TRAIN_DATA_PATH"

# Step 5: Create output directory
echo "📁 Step 5: Creating output directory..."
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory ready: $OUTPUT_DIR"

# Step 6: Set environment variables for distributed training
echo "🌐 Step 6: Setting up distributed training environment..."
export PYTHONPATH="${HIGGS_AUDIO_ROOT}:${PYTHONPATH}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

echo "Environment variables set:"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Step 7: Launch distributed training with robust configuration
echo "🚀 Step 7: Launching distributed training..."
echo "Command to execute:"

# Construct the torchrun command
TORCHRUN_CMD="torchrun \
    --standalone \
    --nproc_per_node=8 \
    --nnodes=1 \
    trainer/train.py \
    --train_data $TRAIN_DATA_PATH \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --output_dir $OUTPUT_DIR \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 100 \
    --use_gradient_checkpointing \
    --mixed_precision \
    --max_audio_length_seconds 30 \
    --dataloader_num_workers 4"

echo "$TORCHRUN_CMD"
echo ""
echo "🎯 Starting training..."

# Execute the command with error handling
if $TORCHRUN_CMD; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Checkpoints saved to: $OUTPUT_DIR"
else
    echo ""
    echo "❌ Training failed with exit code $?"
    echo "💡 Check the logs above for error details"
    echo ""
    echo "🔧 Troubleshooting steps:"
    echo "1. Ensure you're running from the higgs-audio root directory"
    echo "2. Check that all dependencies are installed (torch, transformers, peft)"
    echo "3. Verify that boson_multimodal is available"
    echo "4. Check GPU availability and CUDA version"
    exit 1
fi