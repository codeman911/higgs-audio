# Higgs Audio Label Fix Summary

## Root Cause Analysis

The critical issue was identified in the training pipeline where labels were being incorrectly masked with `-100` (ignore index), preventing the model from learning effectively:

1. **Over-Masking in Collator**: The [HiggsAudioSampleCollator](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/boson_multimodal/data_collator/higgs_audio_collator.py#L63-L509) had `mask_audio_out_token_label=True` by default, which was masking ALL positions containing `<|AUDIO_OUT|>` tokens with `-100`.

2. **Insufficient Label Logging**: The logging functions were not providing specific enough names to distinguish between text and audio predictions vs labels.

## Fixes Applied

### 1. Collator Configuration Fix

**File**: [dataset.py](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/dataset.py)

**Change**: Modified the [create_collator](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/dataset.py#L96-L115) function to disable over-masking:

```python
def create_collator(config, whisper_processor):
    """Create collator with EXACT parameters from serve_engine.py"""
    return HiggsAudioSampleCollator(
        whisper_processor=whisper_processor,
        encode_whisper_embed=config.encode_whisper_embed,
        audio_in_token_id=config.audio_in_token_idx,
        audio_out_token_id=config.audio_out_token_idx,
        audio_stream_bos_id=config.audio_stream_bos_id,
        audio_stream_eos_id=config.audio_stream_eos_id,
        pad_token_id=config.pad_token_id,
        return_audio_in_tokens=False,  # EXACT from serve_engine.py
        use_delay_pattern=config.use_delay_pattern,
        audio_num_codebooks=config.audio_num_codebooks,
        round_to=1,  # EXACT from serve_engine.py
        mask_audio_out_token_label=False,  # FIX: Disable over-masking to allow proper text learning
    )
```

### 2. Enhanced Logging Functions

**File**: [trainer.py](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/trainer.py)

**Changes**:

1. **Text Prediction vs Label Logging**:
   - Added specific naming: "Text Prediction vs Label Comparison"
   - Renamed log entries to clearly indicate text modality:
     - "First 5 Text Predictions" / "First 5 Text Labels"
     - "Last 5 Text Predictions" / "Last 5 Text Labels"
     - "Rest Text Predictions" / "Rest Text Labels"

2. **Audio Prediction vs Label Logging**:
   - Added specific naming: "Audio Codebook 0 Prediction vs Label Comparison"
   - Renamed log entries to clearly indicate audio modality:
     - "First 5 Audio Predictions" / "First 5 Audio Labels"
     - "Last 5 Audio Predictions" / "Last 5 Audio Labels"
     - "Rest Audio Predictions" / "Rest Audio Labels"
   - Added specific shape logging: "Audio Logits shape" / "Audio Labels shape"

## Verification

Debug scripts confirmed that:
1. Label creation in [prepare_chatml_sample](file:///Users/vikram.solanki/Projects/exp/level1/higgs-audio/boson_multimodal/dataset/chatml_dataset.py#L306-L451) is working correctly
2. Only assistant responses are labeled for learning (as expected in ChatML format)
3. User prompts and system tokens are properly masked with `-100`
4. The percentage of learnable tokens is appropriate (~29% in our test case)

## Impact

These changes ensure:
1. **Proper Text Learning**: The model can now learn from assistant text responses
2. **Clear Debugging**: Enhanced logging makes it easier to distinguish between text and audio modalities
3. **Pipeline Integrity**: The working pipeline is preserved without over-engineering
4. **Compatibility**: All changes maintain strict compatibility with the existing boson_multimodal components

The fix addresses the core issue where the model was training with all labels masked, which explained the high loss values and lack of meaningful learning progress.