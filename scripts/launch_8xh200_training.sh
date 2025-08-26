#!/bin/bash

# 🚀 8xH200 Distributed Higgs-Audio Training Launch Script
# 
# Hardware Setup: 8x NVIDIA H200 GPUs (192GB total VRAM) + 128-core CPU
# Usage: bash scripts/launch_8xh200_training.sh [ARGS...]
# 
# Example:
#   bash scripts/launch_8xh200_training.sh \
#     --train_data data/voice_cloning_samples.json \
#     --output_dir outputs/higgs_lora_8xh200 \
#     --batch_size 4 \
#     --learning_rate 5e-4

set -e  # Exit on any error

# 🏠 Ensure we're in the higgs-audio root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HIGGS_AUDIO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🏠 Higgs-Audio root directory: $HIGGS_AUDIO_ROOT"

# Change to higgs-audio root directory
cd "$HIGGS_AUDIO_ROOT"

# 🔍 Verify environment setup
echo "🔍 Verifying environment..."

# Check if we're in the correct directory
if [ ! -d "boson_multimodal" ]; then
    echo "❌ Error: Not in higgs-audio root directory"
    echo "   Expected to find 'boson_multimodal' directory"
    echo "   Current directory: $(pwd)"
    exit 1
fi

if [ ! -d "trainer" ]; then
    echo "❌ Error: trainer directory not found"
    exit 1
fi

# Check if training script exists
if [ ! -f "trainer/train.py" ]; then
    echo "❌ Error: trainer/train.py not found"
    exit 1
fi

echo "✅ Directory structure verified"

# 🖥️ Set up distributed training environment variables
echo "🖥️ Setting up 8xH200 distributed training environment..."

# GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
export NCCL_P2P_DISABLE=0  # Enable P2P for better performance

# CPU optimization for 128 cores
export OMP_NUM_THREADS=16  # 128 cores / 8 GPUs = 16 threads per GPU
export MKL_NUM_THREADS=16
export NUMBA_NUM_THREADS=16

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Debugging and logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1

echo "✅ Environment variables set"

# 🔧 Check PyTorch and CUDA availability
echo "🔧 Checking PyTorch and CUDA..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(min(8, torch.cuda.device_count())):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('❌ CUDA not available!')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ PyTorch/CUDA check failed"
    exit 1
fi

echo "✅ PyTorch and CUDA verified"

# 📊 Default training arguments optimized for 8xH200
DEFAULT_ARGS=(
    "--batch_size" "4"                    # 4 samples per GPU
    "--gradient_accumulation_steps" "4"   # Effective batch size: 128
    "--learning_rate" "5e-4"              # Optimized for distributed
    "--num_epochs" "3"
    "--lora_r" "64"                       # Higher rank for better quality
    "--lora_alpha" "128"
    "--save_steps" "500"
    "--logging_steps" "50"
    "--eval_steps" "1000"
    "--use_gradient_checkpointing" "true"
    "--mixed_precision" "true"
    "--max_audio_length_seconds" "30"
    "--dataloader_num_workers" "16"       # 128 cores / 8 GPUs
)

# 🚀 Launch distributed training with torchrun
echo "🚀 Launching 8xH200 distributed training..."
echo "   Command: torchrun --nproc_per_node=8 trainer/train.py [args...]"
echo "   Working directory: $(pwd)"
echo "   Arguments: ${DEFAULT_ARGS[@]} $@"
echo ""

# Start training with comprehensive error handling
exec torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="localhost" \
    --master_port=29500 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:29500" \
    trainer/train.py \
    "${DEFAULT_ARGS[@]}" \
    "$@"

# 📝 Note: exec replaces the shell process, so the following won't execute
# unless there's an error in the exec command itself
echo "❌ Failed to launch training"
exit 1