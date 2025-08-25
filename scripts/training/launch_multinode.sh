#!/bin/bash

# Multi-Node Distributed Training Launch Script
# Usage: Run this on BOTH nodes (modify NODE_RANK for each)

# === CONFIGURATION ===
export MASTER_ADDR="10.0.0.1"        # IP of Node 0 (main node)  
export MASTER_PORT="29500"            # Communication port
export WORLD_SIZE=16                  # Total GPUs (2 nodes × 8 GPUs)
export NODE_RANK=0                    # Change to 1 for second node

# Training parameters
DATASET_PATH="/path/to/lora_training_data_zr/chatml/"
OUTPUT_DIR="exp/lora_multinode"
BATCH_SIZE=6                          # Per-GPU batch size (96 total across 16 GPUs)
NUM_EPOCHS=1
LEARNING_RATE=1e-4

# === NODE 0 (Main Node) ===
if [ "$NODE_RANK" -eq 0 ]; then
    echo "🚀 Starting Node 0 (Main Node) - IP: $MASTER_ADDR"
    accelerate launch \
        --multi_gpu \
        --machine_rank 0 \
        --num_machines 2 \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_processes 16 \
        --mixed_precision bf16 \
        scripts/training/distributed_trainer.py \
        --dataset_path $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --lora_r 32 \
        --lora_alpha 64 \
        --num_workers 12 \
        --save_steps 500 \
        --warmup_steps 100 \
        --use_cached_codes

# === NODE 1 (Second Node) ===  
elif [ "$NODE_RANK" -eq 1 ]; then
    echo "🚀 Starting Node 1 (Worker Node)"
    accelerate launch \
        --multi_gpu \
        --machine_rank 1 \
        --num_machines 2 \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_processes 16 \
        --mixed_precision bf16 \
        scripts/training/distributed_trainer.py \
        --dataset_path $DATASET_PATH \
        --output_dir $OUTPUT_DIR \
        --num_epochs $NUM_EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --lora_r 32 \
        --lora_alpha 64 \
        --num_workers 12 \
        --save_steps 500 \
        --warmup_steps 100 \
        --use_cached_codes

else
    echo "❌ Invalid NODE_RANK: $NODE_RANK (must be 0 or 1)"
    exit 1
fi
