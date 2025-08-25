# Arabic Voice Cloning Training Pipeline - Complete Execution Guide

## 🎯 Overview

This guide provides step-by-step instructions for running the complete Arabic voice cloning training pipeline using LoRA fine-tuning on Higgs Audio v2. The pipeline is optimized for zero-shot voice cloning with Arabic language support and designed for 8xH200 GPU setups.

## ✅ Key Features

- **Simple Execution**: Only requires ChatML JSON file and output directory
- **Direct Audio Paths**: No need for audio base paths - uses direct paths from ChatML
- **Zero-Shot Voice Cloning**: Aligned with original Higgs Audio implementation
- **8xH200 Optimization**: Maximum performance on high-end hardware
- **Comprehensive Monitoring**: Full checkpoint management and LoRA merging
- **Production Ready**: Complete deployment pipeline included

## 📋 Prerequisites

### Hardware Requirements
- **Minimum**: 1x NVIDIA GPU with 24GB+ VRAM
- **Recommended**: 8x NVIDIA H200 (141GB each)
- **CPU**: 128+ cores for optimal data loading
- **RAM**: 512GB+ for large dataset handling
- **Storage**: 10TB+ NVMe SSD for fast I/O

### Software Requirements
```bash
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
peft>=0.4.0
accelerate>=0.21.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
torchaudio>=2.0.0

# Training utilities
wandb>=0.15.0
loguru>=0.7.0
psutil>=5.9.0

# Higgs Audio (from boson_multimodal)
# Install according to Higgs Audio documentation
```

## 🚀 Quick Start - Single Command Training

### Step 1: Prepare Your Data
Ensure your ChatML file follows this exact format:

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant capable of generating speech in the voice of the provided reference audio."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "مثل ما قال هاك متدبس متدس ايه ما اقدر لان هذا الحديد وين ودي؟"
          },
          {
            "type": "audio",
            "audio_url": "../train-higgs-audio/datasets/zr_ar/ref_audio.wav"
          },
          {
            "type": "text",
            "text": "Please generate speech for given text in reference audio's voice: والميزة فيها يعني الكيس الماتيريال."
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "والميزة فيها يعني الكيس الماتيريال، هذا انت بيتي هو كاربون."
          },
          {
            "type": "audio",
            "audio_url": "../train-higgs-audio/datasets/zr_ar/target_audio.wav"
          }
        ]
      }
    ],
    "speaker": "speaker_001",
    "misc": {
      "duration": 6.144
    }
  }
]
```

**Important**: Audio paths in `audio_url` are used directly - no base path concatenation needed.

### Step 2: Single GPU Training
```bash
python train_arabic_voice_cloning.py \
    --data_path your_chatml_data.json \
    --output_dir ./outputs/arabic_voice_cloning
```

### Step 3: Multi-GPU Training (8xH200)
```bash
torchrun --nproc_per_node=8 train_arabic_voice_cloning.py \
    --data_path your_chatml_data.json \
    --output_dir ./outputs/arabic_voice_cloning
```

## 🔧 Advanced Configuration

### Custom Training Parameters
```bash
python train_arabic_voice_cloning.py \
    --data_path your_chatml_data.json \
    --output_dir ./outputs/arabic_voice_cloning \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --gradient_accumulation_steps 4 \
    --use_wandb \
    --wandb_project my-arabic-voice-cloning
```

### Using Configuration File
```bash
# Create custom config (optional)
cp configs/arabic_voice_cloning.yaml my_config.yaml
# Edit my_config.yaml as needed

# Run with config
python train_arabic_voice_cloning.py \
    --config my_config.yaml \
    --data_path your_chatml_data.json \
    --output_dir ./outputs/arabic_voice_cloning
```

## 📊 Architecture Deep Dive

### DualFFN Architecture Targeting
The training pipeline specifically targets Higgs Audio v2's DualFFN architecture:

```python
# LoRA targets these modules:
target_modules = [
    # Shared attention layers
    "self_attn.q_proj", "self_attn.k_proj", 
    "self_attn.v_proj", "self_attn.o_proj",
    
    # Text FFN pathway
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    
    # Audio FFN pathway (Higgs Audio specific)
    "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj",
    
    # Audio processing modules
    "audio_encoder_proj.linear",
    "audio_head.projector.linear"
]
```

### Teacher Forcing Implementation
The pipeline implements proper teacher forcing for zero-shot voice cloning:

1. **Reference Audio Processing**: Whisper + DAC dual conditioning
2. **Target Audio Generation**: Teacher forcing with proper label alignment
3. **Loss Computation**: Multi-codebook audio loss + text generation loss
4. **Voice Similarity**: Contrastive loss for voice cloning quality

### 8xH200 Optimization Features
- **Memory Management**: 95% GPU memory utilization
- **Data Loading**: 128 CPU workers (16 per GPU)
- **Mixed Precision**: BF16 for optimal H200 performance
- **Gradient Checkpointing**: Memory-efficient training
- **NCCL Backend**: High-speed inter-GPU communication

## 📈 Training Process Analysis

### Phase 1: Data Validation (Auto-runs)
```bash
# Automatic validation during startup
- ChatML format validation
- Audio file accessibility check
- Duration and quality verification
- Text length validation
```

### Phase 2: Model Initialization
```bash
# Components loaded:
- Base Higgs Audio v2 model (3B parameters)
- LoRA adapters (~50M trainable parameters)
- Audio tokenizer (DAC-based)
- Text tokenizer (LLaMA-based)
- Whisper processor for audio conditioning
```

### Phase 3: Training Loop
```bash
# Each training step:
1. Load audio files using direct paths from ChatML
2. Process reference audio (Whisper features + DAC tokens)
3. Tokenize target text and audio
4. Forward pass through DualFFN architecture
5. Compute multi-component loss
6. Backward pass with gradient accumulation
7. Update LoRA parameters only
```

### Phase 4: Checkpoint Management
```bash
# Automatic checkpoint saving every 500 steps:
- LoRA adapter weights
- Training state (optimizer, scheduler)
- Performance metrics
- Configuration backup
```

## 🔍 Monitoring and Logging

### Real-time Monitoring
The pipeline provides comprehensive monitoring:

```bash
# Console output example:
2024-08-25 15:30:45 | INFO | Step 100: Total Loss 4.234567, LR 1.23e-04, GPU 45.2GB
2024-08-25 15:30:55 | INFO | Step 110: Text Loss 3.456789, Audio Loss 2.345678
```

### Weights & Biases Integration
```bash
# Logged metrics:
- loss/total_loss
- loss/text_loss
- loss/audio_loss
- loss/contrastive_loss
- metrics/learning_rate
- metrics/gpu_memory_gb
- metrics/samples_per_second
```

### Local Logging
```bash
# Log files created:
./outputs/arabic_voice_cloning/logs/training.log  # Detailed training logs
./outputs/arabic_voice_cloning/configs/training_config.yaml  # Used configuration
```

## 💾 Checkpoint Management

### Automatic Checkpoint Saving
```bash
# Checkpoints saved to:
./outputs/arabic_voice_cloning/checkpoints/checkpoint-500/
./outputs/arabic_voice_cloning/checkpoints/checkpoint-1000/
./outputs/arabic_voice_cloning/checkpoints/checkpoint-1500/
```

### Checkpoint Validation
```bash
# Validate checkpoint integrity
python lora_merge_and_checkpoint_manager.py \
    --command validate \
    --checkpoint ./outputs/arabic_voice_cloning/checkpoints/checkpoint-1500
```

### LoRA Merging for Deployment
```bash
# Merge best checkpoint with base model
python lora_merge_and_checkpoint_manager.py \
    --command auto-merge \
    --checkpoint ./outputs/arabic_voice_cloning/checkpoints \
    --output ./deployment/merged_model
```

### Checkpoint Comparison
```bash
# Compare multiple checkpoints
python lora_merge_and_checkpoint_manager.py \
    --command compare \
    --checkpoints checkpoint-1000 checkpoint-1500 checkpoint-2000
```

## 🧪 Validation and Testing

### Pre-training Validation
```bash
# Validate data and pipeline before training
python validation_and_testing.py \
    --data_path your_chatml_data.json \
    --output_dir ./validation_results
```

### Post-training Model Testing
```bash
# Test trained model
python validation_and_testing.py \
    --data_path your_chatml_data.json \
    --model_path ./outputs/arabic_voice_cloning/checkpoints/checkpoint-final \
    --output_dir ./test_results
```

## 🎵 Inference Usage

### Using Trained Model
```python
# Load merged model for inference
from arabic_voice_cloning_inference import HiggsAudioInference

# Initialize with your trained model
inference = HiggsAudioInference(
    model_path="./deployment/merged_model",
    device="cuda"
)

# Generate speech
output_audio = inference.generate_speech(
    reference_audio="path/to/reference.wav",
    target_text="النص العربي المراد توليد الصوت له",
    speaker_name="target_speaker"
)
```

## 📊 Expected Training Results

### Training Metrics Progression
```bash
# Expected loss trajectory (3 epochs on 800h data):
Epoch 1: Total Loss 8.5 → 4.2 (Text: 6.2 → 3.1, Audio: 5.8 → 2.8)
Epoch 2: Total Loss 4.2 → 2.8 (Text: 3.1 → 2.2, Audio: 2.8 → 1.9)
Epoch 3: Total Loss 2.8 → 2.1 (Text: 2.2 → 1.8, Audio: 1.9 → 1.5)
```

### Performance Metrics
```bash
# 8xH200 Performance:
- Training Speed: ~2-3 samples/second
- GPU Memory Usage: ~95% (134GB per H200)
- CPU Usage: ~90% (all 128 cores)
- I/O Throughput: ~500MB/s per GPU
```

### Model Quality Indicators
```bash
# Quality metrics to monitor:
- Voice Similarity: >0.85 (cosine similarity)
- Arabic Pronunciation: Natural and fluent
- Zero-shot Performance: High quality without speaker-specific training
- Model Size: 3B base + 50M LoRA = efficient deployment
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Issues
```bash
# Problem: CUDA out of memory
# Solution: Reduce batch size
python train_arabic_voice_cloning.py \
    --data_path your_data.json \
    --output_dir ./outputs \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

#### 2. Audio Path Issues
```bash
# Problem: Audio files not found
# Solution: Check audio_url paths in ChatML are absolute or relative to execution directory
# The pipeline uses direct paths from ChatML without base path concatenation
```

#### 3. Slow Data Loading
```bash
# Problem: Slow training due to I/O
# Solution: Reduce number of workers or use faster storage
python train_arabic_voice_cloning.py \
    --data_path your_data.json \
    --output_dir ./outputs \
    --config configs/arabic_voice_cloning.yaml
# Edit config: training.dataloader_num_workers: 8
```

#### 4. Model Loading Failures
```bash
# Problem: Cannot load Higgs Audio model
# Solution: Ensure boson_multimodal is properly installed
pip install -e . # In boson_multimodal directory
```

### Performance Optimization Tips

#### 1. Data Preprocessing
```bash
# Pre-validate data to avoid runtime issues
python validation_and_testing.py \
    --data_path your_chatml_data.json \
    --output_dir ./validation
```

#### 2. Storage Optimization
```bash
# Use NVMe SSD for audio files
# Organize audio files for sequential access
# Consider audio compression for storage efficiency
```

#### 3. Network Optimization (Multi-GPU)
```bash
# For distributed training, ensure high-speed interconnect
# Use InfiniBand or high-speed Ethernet
# Monitor inter-GPU communication bandwidth
```

## 📚 Code Components Explanation

### 1. Dataset Loader (`arabic_voice_cloning_dataset.py`)
**Purpose**: Load and preprocess ChatML data with direct audio paths
**Key Features**:
- Direct audio path usage (no base path concatenation)
- Multi-threaded validation
- Memory-efficient audio loading
- Comprehensive error handling

```python
# Usage example:
config = ArabicVoiceCloningDatasetConfig(
    chatml_file="your_data.json",
    validate_on_init=True
)
dataset = ArabicVoiceCloningDataset(config, audio_tokenizer, text_tokenizer)
```

### 2. Training Collator (`arabic_voice_cloning_training_collator.py`)
**Purpose**: Batch processing with teacher forcing setup
**Key Features**:
- Leverages boson_multimodal infrastructure
- Proper label alignment for DualFFN
- Enhanced attention masking
- Comprehensive batch validation

### 3. LoRA Configuration (`arabic_voice_cloning_lora_config.py`)
**Purpose**: Configure LoRA for Higgs Audio DualFFN architecture
**Key Features**:
- Targets both text and audio FFN pathways
- Three targeting modes: comprehensive, audio_focused, attention_only
- Optimized for voice cloning tasks

### 4. Loss Function (`arabic_voice_cloning_loss_function.py`)
**Purpose**: Multi-component loss for voice cloning
**Key Features**:
- Text generation loss
- Multi-codebook audio loss
- Voice similarity contrastive loss
- Curriculum learning support

### 5. Distributed Trainer (`arabic_voice_cloning_distributed_trainer.py`)
**Purpose**: High-performance multi-GPU training
**Key Features**:
- NCCL backend optimization
- Mixed precision training
- Comprehensive monitoring
- 8xH200 hardware optimization

### 6. Main Training Script (`train_arabic_voice_cloning.py`)
**Purpose**: Complete training pipeline orchestration
**Key Features**:
- Configuration management
- Automatic checkpoint recovery
- Performance monitoring
- Command-line interface

### 7. Checkpoint Manager (`lora_merge_and_checkpoint_manager.py`)
**Purpose**: LoRA checkpoint management and merging
**Key Features**:
- Checkpoint validation
- Automatic merging
- Model quantization
- Deployment preparation

## 🌟 Advanced Usage Patterns

### Custom Loss Weighting
```yaml
# In config file:
loss:
  text_loss_weight: 0.8      # Reduce text focus
  audio_loss_weight: 1.2     # Increase audio focus
  contrastive_loss_weight: 0.3  # Strong voice similarity
```

### Curriculum Learning
```yaml
# Enable progressive difficulty:
loss:
  enable_curriculum_learning: true
  curriculum_steps: 5000
```

### Model Compilation (Experimental)
```yaml
# Use PyTorch 2.0 compilation:
hardware:
  compile_model: true
```

## 🚀 Production Deployment

### 1. Model Merging
```bash
# Merge best checkpoint for deployment
python lora_merge_and_checkpoint_manager.py \
    --command auto-merge \
    --checkpoint ./outputs/arabic_voice_cloning/checkpoints \
    --output ./production/model \
    --quantize  # Optional for smaller model
```

### 2. Model Serving
```python
# Production inference setup
from arabic_voice_cloning_inference import HiggsAudioInference

inference = HiggsAudioInference(
    model_path="./production/model",
    device="cuda",
    torch_dtype="bfloat16"  # For efficiency
)
```

### 3. API Integration
```python
# FastAPI server example
from fastapi import FastAPI, File, UploadFile
import torch

app = FastAPI()
inference = HiggsAudioInference("./production/model")

@app.post("/generate_speech")
async def generate_speech(
    reference_audio: UploadFile = File(...),
    target_text: str
):
    audio_output = inference.generate_speech(
        reference_audio=reference_audio.file,
        target_text=target_text
    )
    return {"audio": audio_output}
```

## 🎯 Summary

This training pipeline provides a complete solution for Arabic voice cloning using LoRA fine-tuning on Higgs Audio v2. Key advantages:

1. **Simple Usage**: Only requires ChatML JSON and output directory
2. **Direct Paths**: No audio base path complexity
3. **Production Ready**: Complete deployment pipeline
4. **Highly Optimized**: 8xH200 performance optimization
5. **Comprehensive**: Full monitoring and checkpoint management

The pipeline follows the proven patterns from the successful inference implementation while adding robust training capabilities specifically designed for zero-shot voice cloning with Arabic language support.

## 📞 Support and Troubleshooting

For issues or questions:
1. Check the validation output first
2. Review the training logs in `./outputs/arabic_voice_cloning/logs/`
3. Validate your ChatML data format
4. Ensure audio paths are accessible
5. Monitor GPU memory usage and adjust batch size if needed

The pipeline is designed to be robust and provide clear error messages to help identify and resolve issues quickly.