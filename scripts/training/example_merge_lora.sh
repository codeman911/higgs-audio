#!/bin/bash

# Example: Merge LoRA weights and save merged model
# Usage: ./example_merge_lora.sh [checkpoint_path] [output_name]

set -e

# Configuration
BASE_MODEL="bosonai/higgs-audio-v2-generation-3B-base"
CHECKPOINT_PATH="${1:-./outputs/checkpoint-1000}"
OUTPUT_NAME="${2:-higgs-audio-finetuned}"
OUTPUT_DIR="./merged_models/${OUTPUT_NAME}"

echo "🚀 Merging LoRA weights for Higgs-Audio"
echo "   Base model: $BASE_MODEL"
echo "   LoRA checkpoint: $CHECKPOINT_PATH"
echo "   Output directory: $OUTPUT_DIR"
echo ""

# Run the merge script
python scripts/training/merge_lora_weights.py \
    --base_model_path "$BASE_MODEL" \
    --lora_checkpoint "$CHECKPOINT_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --verbose

echo ""
echo "✅ Merge completed! Test with:"
echo ""
echo "python -c \""
echo "from transformers import AutoTokenizer"
echo "from boson_multimodal.model.higgs_audio import HiggsAudioForCausalLM"
echo ""
echo "model = HiggsAudioForCausalLM.from_pretrained('$OUTPUT_DIR', torch_dtype='bfloat16')"
echo "tokenizer = AutoTokenizer.from_pretrained('$OUTPUT_DIR')"
echo "print(f'Model loaded: {model.__class__.__name__}')"
echo "print(f'Tokenizer vocab size: {len(tokenizer)}')"
echo "\""

# Make script executable
chmod +x "$0"
