# Arabic Zero-Shot Voice Cloning with Higgs Audio v2

This repository implements LoRA fine-tuning and zero-shot voice cloning for Arabic language using the Higgs Audio v2 model. The implementation includes data processing, training, inference, and validation components specifically designed for Arabic voice cloning tasks.

## Overview

The system enables:
- **Zero-shot voice cloning** for Arabic text using reference audio
- **LoRA fine-tuning** to adapt Higgs Audio v2 for Arabic language
- **ChatML format** data processing for multimodal training
- **Comprehensive validation** and testing utilities

## Architecture

### Zero-Shot Voice Cloning Mechanism

Based on analysis of the Higgs Audio codebase, the zero-shot voice cloning works through:

1. **Reference Audio Tokenization**: Audio is processed through semantic (HuBERT) and acoustic (DAC) encoders
2. **Audio-Text Conditioning**: Reference audio codes are embedded alongside text tokens
3. **Cross-Modal Attention**: The model conditions generated speech on reference audio patterns
4. **Dual FFN Architecture**: Handles both text and audio token processing simultaneously

### Data Flow

```
ChatML Input → Audio Tokenizer → Audio Codes (12 codebooks × sequence_length)
            → Text Tokenizer  → Text Tokens
            → Model → Generated Audio Codes → Audio Decoder → Output Audio
```

## Implementation Components

### 1. Zero-Shot Inference Script (`arabic_voice_cloning_inference.py`)

Main inference engine for generating Arabic speech with reference voice characteristics.

**Key Features:**
- Processes ChatML format data
- Handles Arabic text preprocessing
- Generates audio with reference voice conditioning
- Supports batch processing

**Usage:**
```bash
python arabic_voice_cloning_inference.py \
    --chatml_file data/arabic_samples.json \
    --output_dir ./generated_audio \
    --model_path bosonai/higgs-audio-v2-generation-3B-base \
    --temperature 0.3 \
    --device cuda
```

### 2. Arabic ChatML Dataset (`arabic_chatml_dataset.py`)

Dataset class for loading and processing Arabic ChatML data for training.

**Features:**
- Loads ChatML format with Arabic content
- Handles audio tokenization and text processing
- Supports train/validation/test splits
- Built-in Arabic text normalization

**Usage:**
```python
from arabic_chatml_dataset import create_arabic_chatml_datasets
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")

train_dataset, val_dataset, test_dataset = create_arabic_chatml_datasets(
    chatml_file="data/arabic_training.json",
    audio_tokenizer="bosonai/higgs-audio-v2-tokenizer",
    text_tokenizer=tokenizer,
    train_ratio=0.8,
    val_ratio=0.1
)
```

### 3. LoRA Training Script (`arabic_lora_training.py`)

Comprehensive training script for LoRA fine-tuning on Arabic data.

**Features:**
- PEFT-based LoRA implementation
- Automatic target module selection for Higgs Audio
- Mixed precision training
- Early stopping and checkpointing
- Memory optimization

**Usage:**
```bash
python arabic_lora_training.py \
    --chatml_file data/arabic_training.json \
    --output_dir ./arabic_lora_checkpoints \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 32
```

### 4. Arabic Text Preprocessing (`arabic_text_preprocessing.py`)

Comprehensive Arabic text processing utilities.

**Features:**
- Diacritics removal
- Punctuation normalization
- Number conversion (Arabic/Persian → Western)
- Abbreviation expansion
- TTS-optimized cleaning

**Usage:**
```python
from arabic_text_preprocessing import clean_arabic_for_tts, normalize_arabic_text

# For TTS/voice synthesis
clean_text = clean_arabic_for_tts("مَرْحَباً بِكُمْ فِي نِظَامِ التَّعَرُّفِ عَلَى الصَّوْتِ")

# General normalization
normalized = normalize_arabic_text(
    text="هذا نص عربي يحتوي على أرقام ١٢٣",
    remove_diacritics=True,
    normalize_numbers=True
)
```

### 5. Validation and Testing (`validation_testing.py`)

Comprehensive validation and testing utilities.

**Features:**
- ChatML data validation
- Audio quality analysis
- Model component testing
- Arabic content validation

**Usage:**
```python
from validation_testing import validate_chatml_file, test_model_components

# Validate dataset
validation_results = validate_chatml_file(
    chatml_file="data/arabic_samples.json",
    audio_base_path="data/audio/"
)

# Test model components
test_results = test_model_components("data/arabic_samples.json")
```

## Installation

1. **Install Higgs Audio dependencies:**
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio
pip install -r requirements.txt
pip install -e .
```

2. **Install additional dependencies:**
```bash
pip install peft datasets accelerate
pip install librosa soundfile
pip install loguru click
```

3. **Optional dependencies for enhanced features:**
```bash
pip install pyloudnorm  # For audio loudness analysis
pip install speechbrain  # For voice similarity evaluation
```

## Data Format

The system uses ChatML format for Arabic voice cloning data:

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
            "audio_url": "path/to/reference_audio.wav"
          },
          {
            "type": "text",
            "text": "Please generate speech for given text in reference audio's voice: والميزة فيها يعني الكيس الماتيريال"
          }
        ]
      },
      {
        "role": "assistant",
        "content": [
          {
            "type": "text",
            "text": "والميزة فيها يعني الكيس الماتيريال"
          },
          {
            "type": "audio",
            "audio_url": "path/to/target_audio.wav",
            "duration": 6.144
          }
        ]
      }
    ],
    "speaker": "sample_00073122",
    "misc": {
      "sample_id": "sample_00073122",
      "ref_transcript": "مثل ما قال هاك متدبس متدس ايه ما اقدر",
      "target_transcript": "والميزة فيها يعني الكيس الماتيريال",
      "duration": 6.144
    }
  }
]
```

## Usage Examples

### 1. Basic Inference

```bash
# Generate Arabic speech with voice cloning
python arabic_voice_cloning_inference.py \
    --chatml_file examples/arabic_samples.json \
    --output_dir ./output \
    --device cuda \
    --temperature 0.3
```

### 2. Training LoRA Model

```bash
# Train LoRA adaptation for Arabic
python arabic_lora_training.py \
    --chatml_file data/arabic_training.json \
    --output_dir ./checkpoints \
    --num_train_epochs 5 \
    --learning_rate 1e-4 \
    --use_mixed_precision
```

### 3. Data Validation

```bash
# Validate your ChatML dataset
python validation_testing.py data/arabic_samples.json data/audio/
```

### 4. Programmatic Usage

```python
from arabic_voice_cloning_inference import ArabicVoiceCloningInference

# Initialize inference engine
engine = ArabicVoiceCloningInference(
    model_path="bosonai/higgs-audio-v2-generation-3B-base",
    device="cuda"
)

# Process ChatML file
results = engine.process_chatml_file(
    chatml_file="data/samples.json",
    output_dir="./output"
)

print(f"Generated {len([r for r in results if r['status'] == 'success'])} audio files")
```

## Model Configuration

### LoRA Target Modules

The system automatically targets key components of Higgs Audio for LoRA fine-tuning:

- **Text model attention layers**: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Text model FFN layers**: `gate_proj`, `up_proj`, `down_proj`
- **Audio-specific layers**: Dual FFN components for audio processing
- **Projection layers**: Audio feature projection modules

### Audio Processing Pipeline

- **Sample Rate**: 16kHz for input processing
- **Codebooks**: 12 codebooks for audio tokenization
- **Frame Rate**: 50Hz for semantic features
- **Output Sample Rate**: 24kHz for generated audio

## Performance Considerations

### Memory Optimization

- **Gradient Checkpointing**: Reduces memory usage during training
- **Mixed Precision**: Uses bf16/fp16 for faster training
- **LoRA**: Significantly reduces trainable parameters
- **Batch Accumulation**: Effective large batch training with limited memory

### Speed Optimization

- **Static KV Cache**: Faster inference on CUDA devices
- **Parallel Processing**: Batch audio tokenization
- **Optimized Collation**: Efficient data loading

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size, enable gradient checkpointing
2. **Audio Loading Errors**: Check audio file formats and paths
3. **Arabic Text Issues**: Ensure proper UTF-8 encoding
4. **Model Loading**: Verify model paths and Hugging Face access

### Performance Tips

1. **Use CUDA**: Significantly faster than CPU inference
2. **Enable Mixed Precision**: Faster training with minimal quality loss
3. **Optimize Batch Size**: Balance memory usage and training speed
4. **Validate Data**: Use validation tools to catch issues early

## Technical Notes

### Zero-Shot Voice Cloning Mechanism

The implementation follows the exact patterns from Higgs Audio's `generation.py`:

1. **Message Structure**: System → User (ref_text) → Assistant (ref_audio) → User (target_text)
2. **Audio Conditioning**: Reference audio tokens are passed as `audio_ids` to the model
3. **Generation**: Model generates audio tokens conditioned on reference audio patterns

### Arabic Language Support

The system includes comprehensive Arabic language support:

- **Diacritics Handling**: Removal of tashkeel for better TTS
- **Punctuation**: Conversion of Arabic punctuation to standard forms
- **Numbers**: Arabic/Persian numeral conversion
- **Text Validation**: Arabic content ratio and quality checks

## License

This implementation builds upon the Higgs Audio v2 model and follows its licensing terms. Please refer to the original Higgs Audio repository for licensing details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{arabic-higgs-audio-2024,
  title={Arabic Zero-Shot Voice Cloning with Higgs Audio v2},
  author={},
  year={2024},
  note={Implementation based on Higgs Audio v2 by Boson AI}
}
```

## Contributing

Contributions are welcome! Please ensure:

1. Code follows the existing patterns
2. Arabic text processing is properly handled
3. Tests pass with the validation utilities
4. Documentation is updated accordingly

## Support

For questions and issues:

1. Check the troubleshooting section
2. Validate your data with the provided tools
3. Review the Higgs Audio documentation
4. Open an issue with detailed error information