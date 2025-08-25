# Higgs-Audio LoRA Training Pipeline

A zero-shot voice cloning training pipeline that reuses existing `boson_multimodal` components and follows the exact patterns from `generation.py` and `arb_inference.py`.

## ✨ Features

- **🎯 Zero-Shot Voice Cloning**: Train for voice cloning without speaker-specific fine-tuning
- **🔄 Reuses Existing Components**: Uses `HiggsAudioModel`, `HiggsAudioSampleCollator`, `prepare_chatml_sample` without modification
- **🏗️ DualFFN Architecture**: Robust dual-loss computation for text + audio generation paths
- **⚡ LoRA Integration**: Efficient adaptation with minimal parameter changes
- **🎤 Whisper Conditioning**: Reference audio conditioning via Whisper embeddings
- **📊 Comprehensive Monitoring**: Loss component tracking and DualFFN balance monitoring

## 🏗️ Architecture

The training pipeline leverages Higgs-Audio's DualFFN architecture:

```
ChatML Input → prepare_chatml_sample → HiggsAudioSampleCollator → HiggsAudioModel
                                                                        ↓
                                                      Shared Cross-Attention
                                                         ↙         ↘
                                                Text FFN        Audio FFN
                                                    ↓               ↓
                                               lm_head         audio_head
                                                    ↓               ↓
                                              Text Loss      Audio Loss
                                                    ↘         ↙
                                                  Total Loss
```

### Key Components

- **DualFFN Architecture**: Shared attention, separate FFN paths for text/audio
- **Multi-Codebook Audio**: 8 codebooks for rich audio representation  
- **Whisper Conditioning**: Reference audio processed via Whisper for voice characteristics
- **ChatML Format**: Same data format as inference scripts for compatibility

## 📋 Requirements

- Python 3.10+
- PyTorch with CUDA support (recommended)
- Transformers >= 4.45.1
- PEFT for LoRA adaptation
- Existing `boson_multimodal` components

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchaudio transformers peft loguru soundfile librosa
```

### 2. Create Sample Training Data

```bash
cd trainer
python train.py --create_sample_data data/sample_training_data.json
```

### 3. Basic Training

```bash
python train.py --train_data data/sample_training_data.json
```

### 4. Advanced Training

```bash
python train.py \
    --train_data data/train_samples.json \
    --val_data data/val_samples.json \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
    --batch_size 2 \
    --learning_rate 1e-4 \
    --num_epochs 5 \
    --lora_r 32 \
    --lora_alpha 64 \
    --output_dir checkpoints/voice_cloning
```

## 📁 Project Structure

```
trainer/
├── __init__.py              # Package initialization
├── config.py                # Training configuration
├── dataset.py               # Dataset wrapper using prepare_chatml_sample
├── loss.py                  # Robust dual-loss computation
├── trainer.py               # Main training class
├── train.py                 # CLI entry point
└── README.md               # This file
```

## 📊 Data Format

The training data follows the same ChatML format as `arb_inference.py`:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Generate speech in the provided voice."
    },
    {
      "role": "user", 
      "content": "Reference text spoken in the audio"
    },
    {
      "role": "assistant",
      "content": {
        "type": "audio",
        "audio_url": "path/to/reference_audio.wav"
      }
    },
    {
      "role": "user",
      "content": "Target text to generate speech for"
    }
  ],
  "speaker": "speaker_id",
  "start_index": 3
}
```

### Data Preparation Tips

1. **Reference Audio Quality**: Use clean, high-quality reference audio (24kHz preferred)
2. **Text-Audio Alignment**: Ensure reference text matches the reference audio content
3. **Audio Duration**: Keep reference audio between 0.5-30 seconds
4. **Speaker Diversity**: Include multiple speakers for better generalization

## 🎛️ Configuration Options

### Model Configuration

```python
# Model paths
model_path = "bosonai/higgs-audio-v2-generation-3B-base"
audio_tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
device = "auto"  # auto, cuda, mps, cpu
```

### Training Configuration

```python
# Basic training settings
batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-4
num_epochs = 3
max_grad_norm = 1.0
weight_decay = 0.01
```

### LoRA Configuration

```python
# LoRA adaptation settings
lora_r = 16                    # LoRA rank
lora_alpha = 32               # LoRA alpha scaling
lora_dropout = 0.1            # LoRA dropout
lora_target_modules = [       # Target modules for adaptation
    "lm_head",                # Text generation head
    "audio_head",             # Audio generation head
]
```

### Loss Configuration

```python
# Loss component weights
text_loss_weight = 1.0        # Text generation loss
audio_loss_weight = 1.0       # Audio generation loss  
consistency_loss_weight = 0.1 # Voice consistency loss
```

## 🔧 Advanced Usage

### Custom Configuration File

Create `config.json`:

```json
{
  "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
  "train_data_path": "data/my_training_data.json",
  "batch_size": 2,
  "learning_rate": 1e-4,
  "lora_r": 32,
  "lora_alpha": 64,
  "num_epochs": 10
}
```

Load with:

```python
from trainer import TrainingConfig, HiggsAudioTrainer

config = TrainingConfig.load("config.json")
trainer = HiggsAudioTrainer(config)
trainer.train()
```

### Resume Training

```bash
python train.py \
    --train_data data/train_samples.json \
    --resume_from checkpoints/checkpoint-1000
```

### Quick Testing

```bash
python train.py \
    --train_data data/train_samples.json \
    --quick_test \
    --output_dir checkpoints/test
```

### Data Validation

```bash
python train.py \
    --train_data data/train_samples.json \
    --validate_data_only
```

## 📈 Monitoring Training

### Loss Components

The trainer monitors multiple loss components:

- **Text Loss**: Cross-entropy loss for language modeling
- **Audio Loss**: Multi-codebook cross-entropy for audio generation
- **Consistency Loss**: Voice characteristic preservation across texts
- **Total Loss**: Weighted combination of all components

### DualFFN Balance Monitoring

The system automatically monitors the balance between text and audio losses:

```
✅ Good DualFFN balance (Text/Audio ratio: 1.2)
⚠️ Text loss dominance! May impact audio generation quality (ratio: 12.3)
⚠️ Audio loss dominance! May impact text understanding (ratio: 0.08)
```

### Training Logs

```
Step 100:
  Total Loss: 2.3456
  Text Loss: 1.2345
  Audio Loss: 1.0234
  Consistency Loss: 0.0877
  Text/Audio Ratio: 1.21
  ✅ Good DualFFN balance
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure boson_multimodal is in Python path
   export PYTHONPATH="${PYTHONPATH}:/path/to/higgs-audio"
   ```

2. **GPU Memory Issues**
   ```bash
   # Reduce batch size and enable gradient checkpointing
   python train.py --batch_size 1 --use_gradient_checkpointing
   ```

3. **Audio File Not Found**
   ```bash
   # Use absolute paths or set audio_base_path
   python train.py --audio_base_path /path/to/audio/files
   ```

4. **Whisper Processor Issues**
   ```bash
   # The trainer will fallback to DAC-only mode automatically
   # Check logs for Whisper processor loading status
   ```

### Performance Optimization

1. **Memory Optimization**
   - Use `--use_gradient_checkpointing` for large models
   - Reduce `--batch_size` if encountering OOM errors
   - Use `--dataloader_num_workers 0` on systems with limited CPU

2. **Speed Optimization**
   - Use CUDA with sufficient VRAM (24GB+ recommended)
   - Enable mixed precision training (automatic with torch.bfloat16)
   - Use SSD storage for faster data loading

3. **Quality Optimization**
   - Monitor DualFFN balance and adjust loss weights if needed
   - Use higher LoRA rank (32-64) for better adaptation
   - Include diverse speakers and audio qualities in training data

## 🧪 Testing and Validation

### Unit Tests

```bash
# Test dataset loading
python -c "from trainer.dataset import VoiceCloningDataset; print('Dataset import OK')"

# Test model loading  
python -c "from trainer.trainer import HiggsAudioTrainer; print('Trainer import OK')"

# Test loss computation
python -c "from trainer.loss import compute_higgs_audio_loss; print('Loss computation OK')"
```

### Integration Test

```bash
# Quick end-to-end test
python train.py \
    --create_sample_data data/test_samples.json && \
python train.py \
    --train_data data/test_samples.json \
    --quick_test \
    --output_dir checkpoints/integration_test
```

## 📚 API Reference

### TrainingConfig

Configuration dataclass for all training parameters.

```python
config = TrainingConfig(
    train_data_path="data/train.json",
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    learning_rate=2e-4,
    lora_r=16
)
```

### HiggsAudioTrainer

Main training class with DualFFN loss computation.

```python
trainer = HiggsAudioTrainer(config)
trainer.train()                              # Start training
trainer.save_checkpoint("my-checkpoint")     # Save checkpoint
trainer.load_checkpoint("my-checkpoint")     # Load checkpoint
```

### VoiceCloningDataset

Dataset wrapper using existing `boson_multimodal` functions.

```python
dataset = VoiceCloningDataset(
    data_path="data/train.json",
    tokenizer=tokenizer,
    audio_tokenizer=audio_tokenizer
)
```

## 🤝 Contributing

This implementation is designed to be minimal and robust, reusing existing `boson_multimodal` components without modification. When contributing:

1. Maintain compatibility with existing inference scripts
2. Follow the DualFFN architecture patterns
3. Preserve the ChatML data format
4. Test with various audio qualities and speakers

## 📄 License

This training pipeline follows the same license as the parent Higgs-Audio project.

## 🙏 Acknowledgments

- Built on top of the excellent Higgs-Audio v2 architecture
- Reuses `boson_multimodal` components without modification
- Follows patterns established in `generation.py` and `arb_inference.py`
- Implements robust DualFFN training with proper loss balance monitoring