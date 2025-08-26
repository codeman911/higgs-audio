#!/bin/bash
"""
Complete Training Execution Script with Monitoring
Run this on your remote cluster after deploying the validation fixes.

Usage:
    bash run_training_with_monitoring.sh
"""

set -e  # Exit on any error

# Configuration
TRAIN_DATA="../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json"
BATCH_SIZE=4
LEARNING_RATE=5e-4
NUM_EPOCHS=3
OUTPUT_DIR="checkpoints/higgs_lora_8xh200_fixed"

echo "🎯 Higgs-Audio LoRA Training with Fixed Validation"
echo "=================================================="
echo "📅 Started at: $(date)"
echo "📁 Training data: $TRAIN_DATA"
echo "🔧 Batch size: $BATCH_SIZE"
echo "📊 Learning rate: $LEARNING_RATE"
echo "🔄 Epochs: $NUM_EPOCHS"
echo "💾 Output directory: $OUTPUT_DIR"
echo "=================================================="

# Step 1: Environment Check
echo ""
echo "🔍 Step 1: Environment Check"
echo "----------------------------"

# Check if we're in the right directory
if [ ! -f "arb_inference.py" ]; then
    echo "❌ Error: Not in higgs-audio root directory"
    echo "   Please run: cd /vs/higgs-audio"
    exit 1
fi

# Check GPU availability
echo "🖥️ GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | head -8

# Check CUDA setup
echo "🔧 CUDA Version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
echo "🐍 Python Version: $(python3 --version)"

# Step 2: Data Validation Test
echo ""
echo "🧪 Step 2: Data Validation Test"
echo "-------------------------------"

if [ -f "test_validation_fix.py" ]; then
    echo "📋 Running validation test on first 1000 samples..."
    python3 test_validation_fix.py \
        --train_data "$TRAIN_DATA" \
        --max_samples 1000 \
        --verbose || {
        echo "❌ Validation test failed!"
        echo "💡 Please check the data format or use --skip_validation"
        exit 1
    }
    echo "✅ Validation test passed!"
else
    echo "⚠️ test_validation_fix.py not found, skipping validation test"
fi

# Step 3: Check Training Data File
echo ""
echo "📂 Step 3: Training Data Check"
echo "------------------------------"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data file not found: $TRAIN_DATA"
    echo "🔍 Searching for training data..."
    find /vs -name "*train_chatml*" -type f 2>/dev/null | head -5
    exit 1
else
    echo "✅ Training data file found"
    echo "📊 File size: $(du -h "$TRAIN_DATA" | cut -f1)"
    
    # Quick JSON validation
    echo "🔍 Validating JSON format..."
    python3 -c "
import json
try:
    with open('$TRAIN_DATA', 'r') as f:
        data = json.load(f)
    sample_count = len(data) if isinstance(data, list) else 1
    print(f'✅ Valid JSON with {sample_count:,} samples')
except Exception as e:
    print(f'❌ JSON validation failed: {e}')
    exit(1)
"
fi

# Step 4: Setup Environment Variables
echo ""
echo "⚙️ Step 4: Environment Setup"
echo "----------------------------"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export NUMBA_NUM_THREADS=16
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "✅ Environment variables set for 8xH200 distributed training"

# Step 5: Create Output Directory
echo ""
echo "📁 Step 5: Output Directory Setup"
echo "---------------------------------"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
echo "✅ Created output directory: $OUTPUT_DIR"

# Step 6: Launch Training
echo ""
echo "🚀 Step 6: Launching Training"
echo "=============================="

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$OUTPUT_DIR/logs/training_${TIMESTAMP}.log"
ERROR_LOG="$OUTPUT_DIR/logs/training_errors_${TIMESTAMP}.log"

echo "📝 Training logs will be saved to: $LOG_FILE"
echo "❌ Error logs will be saved to: $ERROR_LOG"

# Determine which training script to use
if [ -f "launch_training_fixed.py" ]; then
    TRAINING_CMD="python3 launch_training_fixed.py \
        --train_data $TRAIN_DATA \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --output_dir $OUTPUT_DIR \
        --mixed_precision \
        --use_gradient_checkpointing"
    echo "🔧 Using launch_training_fixed.py (recommended)"
elif [ -f "launch_training_direct.py" ]; then
    TRAINING_CMD="python3 launch_training_direct.py \
        --train_data $TRAIN_DATA \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --output_dir $OUTPUT_DIR \
        --mixed_precision \
        --use_gradient_checkpointing"
    echo "🔧 Using launch_training_direct.py"
else
    # Direct torchrun command
    TRAINING_CMD="torchrun \
        --nproc_per_node=8 \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        trainer/train.py \
        --train_data $TRAIN_DATA \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --output_dir $OUTPUT_DIR \
        --mixed_precision \
        --use_gradient_checkpointing"
    echo "🔧 Using direct torchrun command"
fi

echo ""
echo "📋 Training Command:"
echo "$TRAINING_CMD"
echo ""

# Start training with monitoring
echo "🎬 Starting training at: $(date)"
echo "⏱️ Estimated time: ~2-4 hours for 3 epochs with 653K samples"
echo ""

# Function to monitor training
monitor_training() {
    local log_file="$1"
    local check_interval=30  # Check every 30 seconds
    
    while true; do
        if [ -f "$log_file" ]; then
            # Show recent training progress
            echo "📊 Recent training progress:"
            tail -10 "$log_file" | grep -E "(loss|epoch|step|GPU|samples)" || echo "   Waiting for training logs..."
            echo ""
        fi
        sleep $check_interval
    done
}

# Start monitoring in background
if [ "$1" != "--no-monitor" ]; then
    monitor_training "$LOG_FILE" &
    MONITOR_PID=$!
    echo "📈 Started training monitor (PID: $MONITOR_PID)"
fi

# Execute training command
echo "🏃 Executing training..."
eval "$TRAINING_CMD" 2>&1 | tee "$LOG_FILE"
TRAINING_EXIT_CODE=$?

# Stop monitoring
if [ ! -z "$MONITOR_PID" ]; then
    kill $MONITOR_PID 2>/dev/null || true
fi

# Step 7: Post-Training Analysis
echo ""
echo "📊 Step 7: Post-Training Analysis"
echo "================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "🎉 Training completed successfully!"
    
    # Check output files
    echo "📁 Output files:"
    ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "   No output files found"
    
    # Show final training metrics
    echo ""
    echo "📈 Final training metrics:"
    tail -20 "$LOG_FILE" | grep -E "(loss|accuracy|perplexity)" || echo "   No metrics found in logs"
    
    # Check model files
    echo ""
    echo "🤖 Model files:"
    find "$OUTPUT_DIR" -name "*.bin" -o -name "*.safetensors" -o -name "*.pt" | head -10
    
else
    echo "❌ Training failed with exit code: $TRAINING_EXIT_CODE"
    echo ""
    echo "🔍 Error analysis:"
    
    # Show last 50 lines of log for error analysis
    if [ -f "$LOG_FILE" ]; then
        echo "📋 Last 50 lines of training log:"
        tail -50 "$LOG_FILE"
    fi
    
    # Common error patterns
    echo ""
    echo "🕵️ Common error patterns found:"
    if [ -f "$LOG_FILE" ]; then
        echo "   CUDA errors:"
        grep -i "cuda\|gpu\|memory" "$LOG_FILE" | tail -5 || echo "   None found"
        echo "   Validation errors:"
        grep -i "validation\|missing.*message" "$LOG_FILE" | tail -5 || echo "   None found"
        echo "   Import errors:"
        grep -i "import\|module.*not.*found" "$LOG_FILE" | tail -5 || echo "   None found"
    fi
fi

echo ""
echo "📅 Training finished at: $(date)"
echo "📝 Complete logs saved to: $LOG_FILE"
echo "=================================================="

# Exit with the same code as training
exit $TRAINING_EXIT_CODE