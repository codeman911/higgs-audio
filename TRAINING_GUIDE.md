# Higgs Audio LoRA Training Guide

## 🎯 Overview

This guide explains how the **Higgs Audio v2 LoRA fine-tuning pipeline** works and how to use it for zero-shot voice cloning training. The pipeline implements **DualFFN architecture fine-tuning** with strict adherence to inference preprocessing patterns.

## 🏗️ Architecture Deep Dive

### DualFFN Architecture Explained

Higgs Audio v2 uses a **DualFFN (Dual Feed-Forward Network)** architecture that processes text and audio through separate pathways:

```mermaid
graph TB
    subgraph "Input Processing"
        A[ChatML Format] --> B[prepare_chatml_sample]
        B --> C[Text Tokens]
        B --> D[Audio Files]
        D --> E[Audio Tokenizer<br/>8 Codebooks]
        E --> F[Audio Codes<br/>Shape: 8×seq_len]
    end
    
    subgraph "Model Architecture"
        C --> G[Shared Self-Attention]
        F --> G
        G --> H{DualFFN Layer}
        H --> I[Text FFN Path<br/>mlp.{gate,up,down}_proj]
        H --> J[Audio FFN Path<br/>audio_mlp.{gate,up,down}_proj]
        I --> K[Text Logits<br/>vocab_size]
        J --> L[Audio Logits<br/>8×codebook_size]
    end
    
    subgraph "Loss Computation"
        K --> M[Text Loss<br/>CrossEntropy]
        L --> N[Audio Loss<br/>8 Codebook Losses]
        M --> O[Total Loss]
        N --> O
    end
```

### Why This Architecture Works for Voice Cloning

1. **Separate Processing Paths**: Audio and text tokens don't interfere with each other during FFN processing
2. **Shared Attention**: Cross-modal conditioning happens in attention layers
3. **Multi-Codebook Audio**: 8 separate codebooks capture different aspects of speech (fundamental frequency, harmonics, etc.)
4. **LoRA Efficiency**: Fine-tunes only the projection matrices, preserving base model knowledge

## 📁 Pipeline Components

### 1. `dataset.py` - Data Processing Engine

**Purpose**: Converts Arabic ChatML voice cloning data to model-ready tensors.

**Key Functions**:
- Uses `prepare_chatml_sample()` **unchanged** from boson_multimodal
- Processes audio with exact 24kHz sample rate matching inference
- Handles multi-codebook audio tokenization (8 codebooks)
- Creates `ChatMLDatasetSample` objects compatible with official collator

**Data Flow**:
```python
# Input: ChatML JSON format
{
    "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "Arabic text"},
            {"type": "audio", "audio_url": "ref_audio.wav"}
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Target Arabic text"},
            {"type": "audio", "audio_url": "target_audio.wav"}
        ]}
    ]
}

# Processing steps:
1. prepare_chatml_sample() → input_tokens, label_tokens, audio_contents
2. Audio tokenizer encodes → 8×seq_len audio codes
3. Librosa loads waveforms → 24kHz float32 tensors
4. ChatMLDatasetSample created with all components
```

### 2. `lora.py` - Adapter Configuration

**Purpose**: Configures LoRA (Low-Rank Adaptation) for DualFFN targeting.

**Target Modules** (Dynamically Discovered):
```python
# Text & Audio Attention Projections
"*.self_attn.{q,k,v,o}_proj"

# Text FFN (Standard LLaMA)
"*.mlp.{gate,up,down}_proj"  

# Audio FFN (DualFFN Extension)  
"*.audio_mlp.{gate,up,down}_proj"
```

**Why These Modules**:
- **Attention**: Enables cross-modal conditioning between text and audio
- **Text FFN**: Maintains text generation capabilities
- **Audio FFN**: Fine-tunes audio generation without affecting text path

### 3. `trainer.py` - DDP Training Orchestrator

**Purpose**: Distributed training with dual loss computation.

**Training Flow**:
```python
# 1. Model Setup
model = HiggsAudioModel.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")
lora_model = apply_lora(model, lora_config)  # Only adapters trainable

# 2. Dual Loss Computation
outputs = model(**batch)
text_loss = F.cross_entropy(outputs.logits, text_labels, ignore_index=-100)
audio_loss = sum([F.cross_entropy(outputs.audio_logits[:, cb, :], 
                                 audio_labels[cb, :], ignore_index=-100) 
                 for cb in range(8)]) / 8
total_loss = text_loss + audio_loss

# 3. LoRA-Only Updates
# Only LoRA parameters receive gradients, base model frozen
```

## 🚀 Usage Instructions

### Prerequisites

```bash
# Install dependencies
pip install torch torchvision torchaudio
pip install transformers peft datasets
pip install librosa soundfile
pip install accelerate

# Clone and install boson_multimodal
cd /path/to/higgs-audio
pip install -e .
```

### Data Preparation

Your training manifest should be in **ChatML format**:

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
          {"type": "text", "text": "حلو لو سميته حاجة ممكن الأهالي يسمعوه"},
          {"type": "audio", "audio_url": "/path/to/reference_audio.wav"},
          {"type": "text", "text": "Please generate speech for: وَمِنْ ثُمَّ قَالُوا أَنَّهُ لا يُوجَدُ إِلا أَنْ نَأْخُذَ سَيَنْسَتِيلْ."}
        ]
      },
      {
        "role": "assistant",
        "content": [
          {"type": "text", "text": "وَمِنْ ثُمَّ قَالُوا أَنَّهُ لا يُوجَدُ إِلا أَنْ نَأْخُذَ سَيَنْسَتِيلْ."},
          {"type": "audio", "audio_url": "/path/to/target_audio.wav"}
        ]
      }
    ],
    "speaker": "speaker_001"
  }
]
```

### Training Command

#### Single GPU Training
```bash
python trainer.py \
  --train_manifest /path/to/train.json \
  --output_dir /path/to/output \
  --base_ckpt bosonai/higgs-audio-v2-generation-3B-base \
  --batch_size 4 \
  --lr 2e-4 \
  --epochs 3 \
  --grad_accum 4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

#### Multi-GPU Training (8×H200)
```bash
torchrun --nproc_per_node=8 trainer.py \
  --train_manifest /path/to/train.json \
  --output_dir /path/to/output \
  --base_ckpt bosonai/higgs-audio-v2-generation-3B-base \
  --batch_size 2 \
  --lr 2e-4 \
  --epochs 2 \
  --grad_accum 8 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

### Parameter Explanations

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--batch_size` | Batch size per GPU | 2-4 (depends on GPU memory) |
| `--lr` | Learning rate | 1e-4 to 5e-4 |
| `--grad_accum` | Gradient accumulation steps | 4-8 (effective batch = batch_size × nproc × grad_accum) |
| `--lora_r` | LoRA rank (parameter count) | 8-32 (higher = more parameters) |
| `--lora_alpha` | LoRA scaling factor | 16-64 (usually 2× lora_r) |
| `--lora_dropout` | LoRA dropout rate | 0.05-0.1 |

### PEFT Compatibility Issue

**Critical Fix**: PEFT automatically injects a `labels` parameter that [HiggsAudioModel.forward()](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/boson_multimodal/model/higgs_audio/modeling_higgs_audio.py#L1252-L1278) doesn't expect. The DualFFN architecture uses:
- [label_ids](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/boson_multimodal/dataset/chatml_dataset.py#L25-L25) for text labels  
- [label_audio_ids](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/boson_multimodal/data_collator/higgs_audio_collator.py#L42-L42) for audio labels

**Solution**: Bypass PEFT wrapper and call underlying model directly:
```python
# Get underlying model to avoid PEFT's labels injection
if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
    actual_model = self.model.base_model.model  # PEFT wrapped
else:
    actual_model = self.model

# Clean inputs (no labels)
model_inputs = {k: v for k, v in batch.items() 
               if k not in ['label_ids', 'label_audio_ids']}

# Forward pass bypasses PEFT
outputs = actual_model(**model_inputs)
```

This is exactly how the working trainers handle PEFT compatibility.

## 🔧 Technical Details

### Memory Optimization

The pipeline uses several memory optimization techniques:

1. **LoRA Fine-tuning**: Only ~1% of parameters are trainable
2. **BF16 Mixed Precision**: Halves memory usage with minimal quality loss
3. **Gradient Accumulation**: Simulates large batch sizes without memory increase
4. **Efficient Data Loading**: 16 workers per GPU, pinned memory, persistent workers

### Loss Computation Details

```python
# Text Loss (Standard Language Modeling)
text_logits = outputs.logits  # Shape: [batch, seq_len, vocab_size]
text_labels = batch['label_ids']  # Shape: [batch, seq_len]
text_loss = F.cross_entropy(text_logits.view(-1, vocab_size), 
                           text_labels.view(-1), 
                           ignore_index=-100)

# Audio Loss (Multi-Codebook)
audio_logits = outputs.audio_logits  # Shape: [seq_len, 8, codebook_size]  
audio_labels = batch['label_audio_ids']  # Shape: [8, seq_len]
audio_loss = 0.0
for cb in range(8):  # 8 codebooks
    cb_loss = F.cross_entropy(audio_logits[:, cb, :], 
                             audio_labels[cb, :], 
                             ignore_index=-100)
    audio_loss += cb_loss
audio_loss /= 8  # Average across codebooks

total_loss = text_loss + audio_loss
```

### Batch Processing Flow

```python
# 1. Raw JSON → ChatMLDatasetSample
sample = HiggsAudioDataset.__getitem__(idx)
# Contains: input_ids, label_ids, audio_ids_concat, audio_waveforms_concat, etc.

# 2. Batch Collation → HiggsAudioBatchInput  
batch = HiggsAudioSampleCollator([sample1, sample2, ...])
# Contains: input_ids, attention_mask, audio_features, label_ids, label_audio_ids

# 3. Model Forward → HiggsAudioModelOutputWithPast
outputs = model(**batch)
# Contains: logits, audio_logits, loss, etc.
```

### Checkpoint Management

- **Automatic Saves**: Every 1000 steps during training
- **LoRA-Only**: Only adapter weights saved (~50MB vs 6GB full model)
- **Final Checkpoint**: Saved at training completion
- **Format**: PEFT-compatible, can be loaded with `PeftModel.from_pretrained()`

### Usage After Training

```python
from peft import PeftModel
from boson_multimodal.model.higgs_audio import HiggsAudioModel

# Load base model
base_model = HiggsAudioModel.from_pretrained("bosonai/higgs-audio-v2-generation-3B-base")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "/path/to/checkpoint-final")

# Use for inference (same API as base model)
outputs = model.generate(...)
```

## 📊 Performance Expectations

### Training Speed (8×H200)

- **Effective Batch Size**: 128 (2 × 8 × 8)
- **Throughput**: ~50-100 samples/second  
- **Memory Usage**: ~20-25GB per GPU
- **Training Time**: 2-4 hours for 10K samples with 2 epochs

### Quality Metrics

- **Text Loss**: Should decrease to ~2.0-3.0
- **Audio Loss**: Should decrease to ~4.0-6.0  
- **Voice Similarity**: Improved after 500-1000 steps
- **Convergence**: Usually within 1-2 epochs

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `--batch_size` to 1
   - Increase `--grad_accum` to maintain effective batch size
   - Use gradient checkpointing (automatic in LoRA)

2. **Audio Loading Errors**
   - Check audio file paths in manifest
   - Ensure audio files are accessible from all nodes
   - Verify sample rate (24kHz recommended)

3. **Slow Data Loading**
   - Reduce `num_workers` if CPU-bound
   - Use SSD storage for audio files
   - Enable `pin_memory=True`

### Performance Tuning

```bash
# For maximum throughput
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# For debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
```

## 🔍 Validation

Run the validation script to ensure everything is set up correctly:

```bash
python validate_pipeline.py
```

Expected output:
```
🚀 Higgs Audio LoRA Training Pipeline Validation
============================================================
🧪 Testing imports...
✅ dataset.py imported successfully
✅ lora.py imported successfully
✅ trainer.py imported successfully

🧪 Testing basic functionality...
✅ LoRA config creation works
✅ Collator creation function exists
✅ HiggsAudioTrainer class exists

🎉 ALL VALIDATION TESTS PASSED!
📋 Ready for training with torchrun command...
```

## 🎓 How It Works - Technical Summary

This LoRA training pipeline works by:

1. **Preserving Inference Compatibility**: Uses exact same preprocessing (`prepare_chatml_sample`) and collator (`HiggsAudioSampleCollator`) as inference
2. **DualFFN Targeting**: LoRA adapters target both text and audio FFN paths, enabling specialized voice cloning without catastrophic forgetting
3. **Multi-Codebook Audio**: Handles 8-codebook audio representation that captures different acoustic features
4. **Dual Loss Training**: Jointly optimizes text generation and audio generation with proper ignore_index masking
5. **Efficient Fine-tuning**: Only ~1% parameters trained via LoRA, making training fast and memory-efficient

The result is a voice cloning model that can generate speech in any voice provided as reference, while maintaining strong text generation capabilities.