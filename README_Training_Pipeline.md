# Arabic Voice Cloning Training Pipeline

A complete LoRA fine-tuning training pipeline for Higgs Audio v2 to enhance zero-shot voice cloning capabilities with Arabic language support.

## 🚀 Features

- **Complete Training Pipeline**: End-to-end training system with all components
- **DualFFN Architecture Support**: Optimized for Higgs Audio v2's dual FFN layers
- **8xH200 GPU Optimization**: Maximum performance on high-end hardware
- **800 Hours Data Support**: Designed for large-scale voice cloning datasets
- **Comprehensive Validation**: Data quality and training correctness verification
- **Advanced Monitoring**: Real-time metrics and performance tracking

## 📁 Project Structure

```
arabic-voice-cloning-training/
├── configs/
│   └── arabic_voice_cloning.yaml          # Main training configuration
├── arabic_voice_cloning_dataset.py        # Dataset loading and preprocessing
├── arabic_voice_cloning_training_collator.py  # Training collator with teacher forcing
├── arabic_voice_cloning_lora_config.py    # LoRA configuration for DualFFN
├── arabic_voice_cloning_loss_function.py  # Comprehensive loss function
├── arabic_voice_cloning_distributed_trainer.py  # Multi-GPU trainer
├── train_arabic_voice_cloning.py          # Main training script
├── validation_and_testing.py              # Validation pipeline
└── README_Training_Pipeline.md            # This file
```

## 🔧 Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# For distributed training
pip install accelerate
pip install wandb  # Optional for monitoring
```

## 📊 Data Preparation

### ChatML Format
Your data should be in ChatML format with the following structure:

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
            "audio_url": "ref_audio.wav"
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
            "audio_url": "target_audio.wav"
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

### Data Validation
Run validation before training:

```bash
python validation_and_testing.py \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --output_dir validation_output
```

## 🏋️ Training

### Single GPU Training
```bash
python train_arabic_voice_cloning.py \
    --config configs/arabic_voice_cloning.yaml \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --output_dir outputs/arabic_voice_cloning
```

### Multi-GPU Training (8xH200)
```bash
torchrun --nproc_per_node=8 train_arabic_voice_cloning.py \
    --config configs/arabic_voice_cloning.yaml \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --output_dir outputs/arabic_voice_cloning
```

### Training with Custom Parameters
```bash
python train_arabic_voice_cloning.py \
    --config configs/arabic_voice_cloning.yaml \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --gradient_accumulation_steps 4 \
    --use_wandb \
    --wandb_project my-voice-cloning
```

## ⚙️ Configuration

### Key Configuration Parameters

#### Training Parameters
- `batch_size`: Batch size per GPU (default: 1)
- `gradient_accumulation_steps`: Gradient accumulation (default: 8)
- `learning_rate`: Learning rate (default: 2e-4)
- `num_epochs`: Number of training epochs (default: 3)

#### LoRA Parameters
- `r`: LoRA rank (default: 16)
- `lora_alpha`: LoRA scaling (default: 32)
- `target_modules_mode`: "comprehensive", "audio_focused", or "attention_only"

#### Loss Function
- `text_loss_weight`: Weight for text generation loss (default: 1.0)
- `audio_loss_weight`: Weight for audio generation loss (default: 1.0)
- `contrastive_loss_weight`: Weight for voice similarity loss (default: 0.1)

## 📈 Monitoring

### Weights & Biases Integration
The training pipeline includes comprehensive W&B logging:

- Loss components (text, audio, contrastive)
- Learning rate scheduling
- Hardware utilization
- Training performance metrics

### Local Logging
All training logs are saved to:
- `outputs/arabic_voice_cloning/logs/training.log`
- Console output with structured logging

## 🔍 Validation and Testing

### Pre-training Validation
```bash
python validation_and_testing.py \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --output_dir validation_output
```

### Training Pipeline Testing
```bash
# Test model loading and pipeline
python validation_and_testing.py \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --model_path bosonai/higgs-audio-v2-generation-3B-base
```

### Performance Benchmarking
```bash
# Full benchmark including performance tests
python validation_and_testing.py \
    --data_path data/arabic_voice_cloning_chatml.json \
    --audio_base_path data/audio \
    --output_dir validation_output
```

## 🏗️ Architecture Details

### DualFFN Architecture
The pipeline targets Higgs Audio v2's DualFFN architecture:
- **Shared Attention**: Text and audio tokens share attention layers
- **Dual FFN**: Separate FFN pathways for text and audio processing
- **LoRA Adaptation**: Targets both text and audio FFN modules

### LoRA Module Targeting
```python
# Comprehensive mode targets:
- "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"  # Shared attention
- "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"                                 # Text FFN
- "audio_mlp.gate_proj", "audio_mlp.up_proj", "audio_mlp.down_proj"              # Audio FFN
- "audio_encoder_proj.linear"                                                     # Audio projector
- "audio_head.projector.linear"                                                   # Audio head
```

## 📊 Performance Optimization

### 8xH200 GPU Setup
- **Memory Optimization**: 95% GPU memory utilization
- **Data Loading**: 128 CPU workers (16 per GPU)
- **Mixed Precision**: Automatic mixed precision with GradScaler
- **Flash Attention**: Enabled for memory efficiency

### Training Efficiency
- **Gradient Checkpointing**: Reduces memory usage
- **Gradient Accumulation**: Effective batch size of 64
- **Distributed Training**: NCCL backend for multi-GPU

## 🛠️ Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce batch size
python train_arabic_voice_cloning.py --batch_size 1 --gradient_accumulation_steps 16

# Enable CPU offloading
# Edit config: hardware.enable_cpu_offload: true
```

#### Data Loading Issues
```bash
# Validate data first
python validation_and_testing.py --data_path your_data.json --audio_base_path audio/

# Reduce workers if I/O bound
# Edit config: training.dataloader_num_workers: 8
```

#### Model Loading Issues
```bash
# Test model loading separately
python -c "
from arabic_voice_cloning_lora_config import create_higgs_audio_lora_model
model, config, lora_config = create_higgs_audio_lora_model()
print('Model loaded successfully')
"
```

### Hardware Requirements

#### Minimum Requirements
- **GPU**: 1x NVIDIA GPU with 24GB+ VRAM
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Storage**: 1TB+ SSD

#### Recommended (8xH200)
- **GPU**: 8x NVIDIA H200 (141GB each)
- **CPU**: 128+ cores
- **RAM**: 512GB+
- **Storage**: 10TB+ NVMe SSD

## 📝 Example Training Session

```bash
# 1. Validate data
python validation_and_testing.py \
    --data_path data/arabic_chatml.json \
    --audio_base_path data/audio

# 2. Start training
torchrun --nproc_per_node=8 train_arabic_voice_cloning.py \
    --config configs/arabic_voice_cloning.yaml \
    --data_path data/arabic_chatml.json \
    --audio_base_path data/audio \
    --use_wandb \
    --wandb_project arabic-voice-cloning

# 3. Monitor progress
# Check W&B dashboard or local logs
tail -f outputs/arabic_voice_cloning/logs/training.log
```

## 🔬 Advanced Usage

### Custom Loss Weights
```yaml
loss:
  text_loss_weight: 0.8      # Reduce text focus
  audio_loss_weight: 1.2     # Increase audio focus
  contrastive_loss_weight: 0.2  # Stronger voice similarity
```

### Curriculum Learning
```yaml
loss:
  enable_curriculum_learning: true
  curriculum_steps: 5000     # Gradually increase difficult loss components
```

### Model Compilation (Experimental)
```yaml
hardware:
  compile_model: true        # Use torch.compile for speed
```

## 📊 Expected Results

### Training Metrics
- **Text Loss**: Should decrease from ~8.0 to ~3.0
- **Audio Loss**: Should decrease from ~6.0 to ~2.0
- **Voice Similarity**: Should increase over training
- **Training Speed**: ~1-2 samples/sec on 8xH200

### Model Performance
- **Voice Cloning Quality**: High similarity to reference voice
- **Arabic Speech**: Natural and fluent Arabic speech synthesis
- **Model Size**: ~3B parameters with ~50M LoRA parameters

## 🤝 Contributing

This training pipeline follows the proven patterns from the successful inference implementation. The architecture respects Higgs Audio's DualFFN design while adding Arabic language optimization.

## 📄 License

This project follows the same license as the base Higgs Audio model.