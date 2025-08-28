# Tokenizer Initialization Fix for Unified LoRA Training Script

## Overview

This document describes the issue and fix for the missing tokenizer initialization in the `unified_lora_training.py` script. The error occurs because the `tokenizer` and `audio_tokenizer` attributes are not properly initialized in the `HiggsAudioTrainer` class, causing an `AttributeError` when the dataset is being set up.

## Problem Analysis

### Error Details
```
AttributeError: 'HiggsAudioTrainer' object has no attribute 'tokenizer'
```

### Root Cause
In the `unified_lora_training.py` script, the `load_model_and_tokenizers` method is missing the tokenizer initialization code that exists in the working `trainer.py` script. Specifically, these lines are missing:

```python
# Load tokenizers - EXACT pattern from serve_engine.py
self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_ckpt)
self.audio_tokenizer = load_higgs_audio_tokenizer(
    "bosonai/higgs-audio-v2-tokenizer", 
    device='cpu'
)

# Load Whisper processor
self.whisper_processor = AutoProcessor.from_pretrained(
    "openai/whisper-large-v3", trust_remote_code=True
)
```

## Solution

### Fix Implementation
Add the missing tokenizer initialization code to the `load_model_and_tokenizers` method in `unified_lora_training.py`, placing it after the model loading but before the LoRA application.

The fix involves adding these lines after `self.model = model.to(self.device)` and before the stage-specific model configuration:

```python
# Load tokenizers - EXACT pattern from serve_engine.py
self.tokenizer = AutoTokenizer.from_pretrained(self.args.base_ckpt)
self.audio_tokenizer = load_higgs_audio_tokenizer(
    "bosonai/higgs-audio-v2-tokenizer", 
    device='cpu'
)

# Load Whisper processor
self.whisper_processor = AutoProcessor.from_pretrained(
    "openai/whisper-large-v3", trust_remote_code=True
)
```

## Answering the Cross-Modal Question

Regarding your question about the effect of cross-modal conditioning when audio layers are frozen:

Cross-modal conditioning (specifically `use_audio_out_self_attention=True`) allows the text pathway to attend to audio context, which is essential for proper conditioning. However, when audio layers are frozen in stage 1 training:

1. **Positive Effects**: 
   - The text pathway can still learn to understand audio context through the cross-attention mechanisms
   - This helps in building better alignment between text and audio representations
   - It prepares the model for stage 2 where both pathways will be trained together

2. **Considerations**:
   - Since audio layers are frozen, the audio representations won't change during stage 1 training
   - The text pathway can only learn to better utilize the static audio representations
   - This is still beneficial as it establishes the cross-modal connections that will be refined in stage 2

This approach is good because it allows the model to learn cross-modal relationships even when one modality is frozen, which is the intended behavior for the two-stage training approach.

## Implementation Plan

1. Add the missing tokenizer initialization to `unified_lora_training.py`
2. Ensure the initialization happens at the correct point in the method
3. Verify that all required attributes are properly set before they're used in `setup_dataset`

## Testing

After implementing the fix:
1. Run the training script with stage 1 to verify the tokenizer initialization error is resolved
2. Confirm that the training proceeds without attribute errors
3. Verify that both stage 1 and stage 2 training work correctly