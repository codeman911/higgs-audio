#!/bin/bash
# Launch script for distributed LoRA training on 8x H200 GPUs
# Higgs-Audio V2 Arabic+English Zero-Shot Voice Cloning

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DATA_DIR="/workspace/data"
OUTPUT_DIR="/workspace/outputs/higgs-lora-arabic-english"
CONFIG_FILE="$PROJECT_ROOT/configs/training_config.yaml"

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "=== Higgs-Audio V2 LoRA Training Setup ==="
echo "Project Root: $PROJECT_ROOT"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Config File: $CONFIG_FILE"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Step 1: Process the dataset
echo "Step 1: Processing Arabic+English dataset..."
if [ ! -d "$DATA_DIR/processed_chatml" ]; then
    echo "Processing dataset from raw format to ChatML..."
    python "$PROJECT_ROOT/scripts/data_processing/arabic_english_processor.py" \
        --dataset_dir "$DATA_DIR/raw_dataset" \
        --output_dir "$DATA_DIR/processed_chatml" \
        --audio_tokenizer_path "bosonai/higgs-audio-v2-tokenizer" \
        --max_duration 30.0 \
        --min_duration 0.5 \
        --num_workers 16
    
    echo "Dataset processing completed!"
else
    echo "Processed dataset already exists, skipping processing..."
fi

# Step 2: Verify CUDA and GPU setup
echo "Step 2: Verifying GPU setup..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Number of GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

# Step 3: Launch distributed training
echo "Step 3: Launching distributed LoRA training..."

# Option A: Using Accelerate (Recommended)
if command -v accelerate &> /dev/null; then
    echo "Using Accelerate for distributed training..."
    
    # Configure accelerate
    accelerate config default \
        --mixed_precision bf16 \
        --num_processes 8 \
        --num_machines 1 \
        --gpu_ids 0,1,2,3,4,5,6,7 \
        --main_process_port 29500
    
    # Launch training
    accelerate launch \
        --config_file ~/.cache/huggingface/accelerate/default_config.yaml \
        --main_process_port 29500 \
        "$PROJECT_ROOT/scripts/training/distributed_trainer.py" \
        --config "$CONFIG_FILE" \
        --dataset_path "$DATA_DIR/processed_chatml" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Option B: Using torchrun (Alternative)
elif command -v torchrun &> /dev/null; then
    echo "Using torchrun for distributed training..."
    
    torchrun \
        --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$PROJECT_ROOT/scripts/training/distributed_trainer.py" \
        --config "$CONFIG_FILE" \
        --dataset_path "$DATA_DIR/processed_chatml" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

# Option C: Using DeepSpeed (Alternative)
elif command -v deepspeed &> /dev/null; then
    echo "Using DeepSpeed for distributed training..."
    
    deepspeed \
        --num_gpus=8 \
        --master_port=29500 \
        "$PROJECT_ROOT/scripts/training/distributed_trainer.py" \
        --config "$CONFIG_FILE" \
        --dataset_path "$DATA_DIR/processed_chatml" \
        --output_dir "$OUTPUT_DIR" \
        --deepspeed "$PROJECT_ROOT/configs/deepspeed_config.json" \
        2>&1 | tee "$OUTPUT_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log"

else
    echo "Error: No distributed training framework found!"
    echo "Please install accelerate, torchrun, or deepspeed"
    exit 1
fi

echo "Training completed! Check logs in $OUTPUT_DIR/logs/"
echo "Model checkpoints saved in $OUTPUT_DIR/"

# Step 4: Post-training evaluation (optional)
echo "Step 4: Running post-training evaluation..."
python "$PROJECT_ROOT/scripts/evaluation/evaluate_lora_model.py" \
    --model_path "$OUTPUT_DIR/checkpoint-final" \
    --base_model_path "bosonai/higgs-audio-v2-generation-3B-base" \
    --audio_tokenizer_path "bosonai/higgs-audio-v2-tokenizer" \
    --test_data_path "$DATA_DIR/processed_chatml" \
    --output_dir "$OUTPUT_DIR/evaluation" \
    --num_samples 100

echo "=== Training Pipeline Completed Successfully ==="
