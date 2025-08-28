# Two-Stage LoRA Training Approach for Higgs Audio Model with Accurate DualFFN Implementation

## Overview

This document outlines a two-stage LoRA training approach for the Higgs Audio model to address the issue where audio loss converges quickly (from 14 to 0.05 in 2000 steps) while text loss remains high (from 14 to 8-9). The proposed solution involves:

1. **Stage 1**: Train only the text components of the DualFFN architecture while freezing audio components
2. **Stage 2**: Use the Stage 1 model as initialization for full training with both text and audio components

This approach aims to establish a strong text foundation before introducing audio training, potentially improving overall convergence and performance. Based on analysis of the actual Higgs Audio implementation, the DualFFN architecture uses separate text and audio paths within selected LLM layers, with configurable audio attention modules.

## Implementation

A unified training script `unified_lora_training.py` will be created that can handle both training stages based on a command-line argument. This script will:

1. Accept a `--training_stage` argument with values "stage1" or "stage2"
2. For Stage 1: Load the Higgs Audio model with a wrapper that freezes all audio components and configures LoRA to target only text modules
3. For Stage 2: Load the Higgs Audio model with full LoRA targeting for both text and audio modules
4. Maintain compatibility with the existing data pipeline

The implementation will make minimal changes to the existing training pipeline while providing flexible training options.

## Architecture

### Higgs Audio DualFFN Architecture Overview

The Higgs Audio model is built on top of Llama-3.2-3B with a DualFFN (Dual Feed-Forward Network) architecture that enhances the model's ability to process audio tokens. The architecture uses configurable adapter types including "dual_ffn" where selected LLM layers are replaced with dual-path layers:

```
Text Tokens ───┐
               ├──► Shared Attention Layer ───► Separate FFNs ───► Reordered Output
Audio Tokens ──┘
```

Key components in the actual implementation:
- **Shared Attention Layers**: Process both text and audio tokens together using standard Llama attention mechanisms
- **Separate FFNs**: Text FFN (`mlp`) and Audio FFN (`audio_mlp`) process their respective tokens in selected layers
- **Optional Audio Attention**: Configurable `audio_attn` modules for dedicated audio token processing
- **Layer Selection**: Configurable `audio_dual_ffn_layers` parameter determines which LLM layers use the dual-path architecture
- **Audio Tower**: Separate audio encoder (Whisper-based) for processing input audio features
- **Audio Projector**: Linear projection from audio encoder features to LLM hidden dimension
- **Audio Decoder Projector**: Maps LLM hidden states to audio logits for generation

### Actual Higgs Audio Loss Computation

Based on analysis of the Higgs Audio implementation, the model computes losses as follows:

1. **Text Loss (LLM Loss)**: Standard language modeling loss computed using cross-entropy on text logits with proper masking of non-assistant tokens (labels set to -100)
2. **Audio Loss**: Multi-codebook loss computed across 8 codebooks (based on training logs), with each codebook contributing to the total audio loss through cross-entropy
3. **Total Loss**: Combined loss that includes both text and audio components

The model's `compute_losses` method in `modeling_higgs_audio.py` returns separate `llm_loss` and `audio_loss` components, which allows for targeted optimization in our two-stage approach.

### Higgs Audio Module Structure

Based on analysis of the actual Higgs Audio implementation, the key modules that need to be considered for our two-stage training approach are:

1. **Text Modules (Always Trainable)**:
   - `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj` - Standard attention projections
   - `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` - Standard MLP layers
   - `input_layernorm`, `post_attention_layernorm` - Standard layer normalization
   - `embed_tokens` - Text token embeddings

2. **Audio Modules (Frozen in Stage 1)**:
   - `audio_mlp.gate_proj`, `audio_mlp.up_proj`, `audio_mlp.down_proj` - Audio-specific MLP layers
   - `audio_attn.q_proj`, `audio_attn.k_proj`, `audio_attn.v_proj`, `audio_attn.o_proj` - Audio-specific attention projections (when enabled)
   - `audio_input_layernorm`, `audio_post_attention_layernorm`, `audio_post_audio_attn_layer_norm` - Audio layer normalization
   - `audio_codebook_embeddings` - Audio codebook embeddings
   - `audio_out_embed_projector` - Audio output embedding projector (when enabled)
   - `audio_encoder_proj` - Audio encoder feature projector
   - `audio_decoder_proj` - Audio decoder projector

This structure is important for correctly implementing the LoRA targeting and component freezing in our two-stage approach.

## Proposed Solution: Two-Stage Training

### Stage 1: Text-Only Training

**Objective**: Establish strong text processing capabilities in the DualFFN architecture before introducing audio training by freezing all audio-specific components and training only on text loss.

**Implementation Strategy**:
1. Freeze all audio-specific components:
   - `audio_mlp` modules in layers specified by `audio_dual_ffn_layers` configuration
   - `audio_attn` modules (when enabled via `use_audio_out_self_attention` configuration)
   - Audio layer normalization components (`audio_post_attention_layernorm`, `audio_input_layernorm`, `audio_post_audio_attn_layer_norm`)
   - Audio embedding components (`audio_codebook_embeddings`, `audio_out_embed_projector`)
   - Audio decoder projector (`audio_decoder_proj`)
2. Train only text components:
   - Standard `self_attn` modules in all layers
   - Standard `mlp` modules in non-DualFFN layers
   - Text layer normalization components
   - Text embedding components (`embed_tokens`)
3. Use only text loss for optimization (ignore audio loss which will be 0 due to frozen components)

**Benefits**:
- Allows text processing to converge before audio training
- Reduces complexity during initial training phase
- May improve overall model stability
- Enables focused optimization of text understanding capabilities

### Stage 2: Full Training with Pre-trained Text Foundation

**Objective**: Leverage the strong text foundation from Stage 1 to improve audio training effectiveness.

**Implementation Strategy**:
1. Load the Stage 1 model as initialization
2. Apply standard LoRA training with both text and audio components
3. Use both text and audio losses for optimization

## Technical Implementation

### Unified Script Implementation Details

The `unified_lora_training.py` script will implement both training stages with the following key components:

1. **Argument Parsing**:
   The script will accept a new `--training_stage` argument with possible values "stage1" or "stage2". This determines which training mode to use.

2. **Conditional Model Configuration**:
   Based on the `--training_stage` argument, the script will configure the model differently:
   
   **For Stage 1 (Text-Only Training)**:
   - Freeze all audio-specific components:
     - `audio_mlp` modules in each DualFFN layer (selected by `audio_dual_ffn_layers` configuration)
     - `audio_attn` modules (when present and enabled via `use_audio_out_self_attention` configuration)
     - Audio layer normalization components (`audio_post_attention_layernorm`, `audio_input_layernorm`, `audio_post_audio_attn_layer_norm`)
     - Audio embedding components (`audio_codebook_embeddings`, `audio_out_embed_projector`)
     - Audio decoder projector (`audio_decoder_proj`)
   - This is accomplished by iterating through model layers and setting `requires_grad=False` for all audio parameters.
   
   **For Stage 2 (Full Training)**:
   - No components are frozen (all components remain trainable)
   - All model parameters maintain their default `requires_grad=True` state

3. **Conditional LoRA Targeting**:
   Based on the `--training_stage` argument, the script will apply different LoRA configurations:
   
   **For Stage 1 (Text-Only Training)**:
   - Dynamically identify text-only modules for LoRA adaptation based on the actual Higgs Audio model structure:
     - Target standard attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in `self_attn` modules
     - Target standard MLP layers (`gate_proj`, `up_proj`, `down_proj`) in `mlp` modules
     - Exclude audio-specific modules (`audio_mlp`, `audio_attn`) since they are frozen
     - Return a clean list of modules suitable for text-only LoRA training
   
   **For Stage 2 (Full Training)**:
   - Apply LoRA to both text and audio components:
     - Target standard attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) in `self_attn` modules
     - Target standard MLP layers (`gate_proj`, `up_proj`, `down_proj`) in `mlp` modules
     - Target audio-specific modules (`audio_mlp`) in DualFFN layers
     - Target audio attention modules (`audio_attn`) when enabled

4. **Loss Computation**:
   The unified trainer will maintain the standard loss computation approach for both stages:
   - Use the standard model forward pass which returns separate `llm_loss` and `audio_loss` components
   - For Stage 1: Extract only the `llm_loss` component for optimization (ignore `audio_loss` which will be 0 due to frozen components)
   - For Stage 2: Use both `llm_loss` and `audio_loss` components for optimization
   - The text loss is computed using standard cross-entropy on text logits with proper masking of non-assistant tokens
   - Audio loss is computed across 8 codebooks using cross-entropy

5. **Compatibility Features**:
   - Maintain the same data loading and preprocessing pipeline
   - Preserve all configuration options available in the standard trainer
   - Output models in the same format for seamless transition between stages

### Stage 2 Implementation Details

With the unified script, Stage 2 can be executed by specifying `--training_stage stage2`:

1. **Model Loading**: Load the model checkpoint (can be a fresh model or a Stage 1 checkpoint)
2. **Full LoRA Setup**: Apply LoRA to both text and audio components
3. **Full Loss Computation**: Use both text and audio losses
4. **No Component Freezing**: All model components remain trainable

## Training Pipeline Modifications

### New File: `unified_lora_training.py`

A new file `unified_lora_training.py` will be created that implements both training stages with minimal changes to the existing pipeline. Key features of this implementation:

1. **Unified Argument**: Accepts a `--training_stage` argument with values "stage1" or "stage2"
2. **Stage 1 Implementation**: When `--training_stage stage1` is specified, the script will automatically freeze all audio components and configure LoRA to target only text modules
3. **Stage 2 Implementation**: When `--training_stage stage2` is specified, the script will apply full LoRA training with both text and audio components
4. **Full compatibility**: Maintains all data loading, preprocessing, and other pipeline functionality

The script will accept the same command-line arguments as the standard trainer with the addition of the training stage parameter.

Usage examples:
```bash
# Stage 1: Text-only training
python unified_lora_training.py \
    --training_stage stage1 \
    --model_path /path/to/higgs/audio/model \
    --train_data_dir /path/to/training/data \
    --output_dir /path/to/stage1/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2

# Stage 2: Full training
python unified_lora_training.py \
    --training_stage stage2 \
    --model_path /path/to/stage1/output \
    --train_data_dir /path/to/training/data \
    --output_dir /path/to/stage2/output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2
```

### Integration with Existing Pipeline

The two-stage approach will maintain compatibility with the existing training pipeline:
- Use the same data loading and preprocessing
- Maintain the same model architecture
- Preserve all configuration options except those specific to stage 1
- Allow seamless transition from stage 1 to stage 2

## Expected Outcomes

### Potential Benefits

1. **Improved Text Convergence**: Establishing strong text processing before audio training
2. **Better Cross-Modal Learning**: Text foundation may improve audio learning effectiveness
3. **Stable Training**: Reduced risk of instabilities from simultaneous text/audio training
4. **Arabic Language Performance**: Better text processing for Arabic may improve overall model performance

### Risk Analysis

1. **Overfitting Risk**: Text components might overfit to text tasks without audio regularization
2. **Capacity Mismatch**: Strong text foundation might not translate to better audio learning
3. **Training Time**: Two-stage approach increases total training time
4. **Hyperparameter Sensitivity**: May require different learning rates or schedules for each stage

## Implementation Plan

### Phase 1: Unified Training Script Development

1. Create `unified_lora_training.py` based on existing trainer
2. Implement argument parsing for `--training_stage` with values "stage1" or "stage2"
3. Implement audio component freezing functionality for Stage 1 mode
4. Implement conditional LoRA targeting based on training stage
5. Maintain existing loss computation for both training modes

### Phase 2: Validation and Testing

1. Test stage 1 training with small dataset
2. Verify audio components are properly frozen
3. Confirm only text loss is being optimized
4. Evaluate text convergence improvement

### Phase 3: Stage 2 Integration

1. Implement stage 2 training with stage 1 model initialization
2. Validate seamless transition between stages
3. Compare performance with single-stage training

## Monitoring and Evaluation

### Key Metrics to Track

1. **Text Loss Convergence**: Compare convergence rate between single-stage and two-stage approaches
2. **Audio Loss Convergence**: Evaluate if stage 2 audio learning is improved
3. **Validation Performance**: Measure overall model quality on validation set
4. **Training Stability**: Monitor for any instabilities or divergences

### Diagnostic Tools

1. **Gradient Analysis**: Check gradient flow in text vs. audio components
2. **Parameter Updates**: Verify only text components are being updated in stage 1
3. **Loss Component Tracking**: Separate tracking of text and audio losses

## Conclusion

The proposed two-stage LoRA training approach addresses the observed imbalance in text and audio loss convergence by first establishing a strong text processing foundation. This approach leverages the existing Higgs Audio architecture while making minimal modifications to the training pipeline. The solution should improve overall training effectiveness, particularly for Arabic language support where the base Llama model already has capabilities but the Higgs Audio adaptation needs refinement.

The implementation of `unified_lora_training.py` will provide a clean, modular approach to both training stages that:

1. **Maintains Code Quality**: Makes minimal changes to the existing pipeline, reducing the risk of introducing bugs
2. **Ensures Compatibility**: Preserves all existing functionality while adding the new flexible training capability
3. **Provides Flexibility**: Can easily switch between stage 1 and stage 2 training with a single argument
4. **Enables Experimentation**: Allows for easy comparison between single-stage and two-stage training approaches

This approach is expected to produce a stronger text foundation that will benefit the subsequent audio training phase, potentially leading to better overall convergence and performance for both text and audio tasks.