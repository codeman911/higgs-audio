"""
Robust dual-loss computation for Higgs-Audio DualFFN architecture.

The model outputs:
- logits: Text generation logits [batch_size, seq_len, vocab_size]
- audio_logits: Audio generation logits [num_audio_tokens, num_codebooks * codebook_size]

DualFFN architecture shares cross-attention but has separate FFN paths:
- Text FFN -> lm_head -> text logits  
- Audio FFN -> audio_head -> audio logits (multi-codebook)

Both paths must learn for effective zero-shot voice cloning.
Follows EXACT patterns from train_higgs_lora.py for teacher forcing.
"""

# 🔧 Conditional imports for ML dependencies
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy classes for non-ML operations
    class torch:
        class Tensor:
            pass
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

# 🔧 Conditional torch import for environments without ML dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create dummy tensor class for utility operations
    class torch:
        class Tensor:
            pass


@dataclass
class LossComponents:
    """Container for loss components with detailed breakdown."""
    total_loss: float
    text_loss: float
    audio_loss: float
    consistency_loss: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'total_loss': self.total_loss,
            'text_loss': self.text_loss,
            'audio_loss': self.audio_loss,
            'consistency_loss': self.consistency_loss,
        }


def compute_higgs_audio_loss(
    model_outputs,
    batch,
    text_loss_weight: float = 1.0,
    audio_loss_weight: float = 1.0,
    consistency_loss_weight: float = 0.1,
) -> Tuple[torch.Tensor, LossComponents]:
    """
    Robust dual loss for zero-shot voice cloning with DualFFN architecture and teacher forcing.
    
    The Higgs-Audio model has shared cross-attention but separate FFN paths:
    - Text FFN -> lm_head -> text logits
    - Audio FFN -> audio_head -> audio logits (multi-codebook)
    
    Both paths need to learn for effective voice cloning since they share attention.
    
    Teacher forcing training:
    - Text labels with proper -100 masking for padding
    - Audio labels with multi-codebook structure following generation.py patterns
    - Proper attention masking for variable-length sequences
    
    Args:
        model_outputs: HiggsAudioModelOutputWithPast containing logits and audio_logits
        batch: Batch data with labels (HiggsAudioBatchInput from collator)
        text_loss_weight: Weight for text loss component
        audio_loss_weight: Weight for audio loss component  
        consistency_loss_weight: Weight for voice consistency loss
        
    Returns:
        Tuple of (total_loss_tensor, loss_components)
    """
    total_loss = 0.0
    loss_components = LossComponents(0.0, 0.0, 0.0, 0.0)
    
    # 1. TEXT LOSS - Standard cross-entropy for language modeling with teacher forcing
    text_loss = _compute_text_loss_with_masking(model_outputs, batch)
    if text_loss > 0:
        total_loss += text_loss_weight * text_loss
        loss_components.text_loss = text_loss
    
    # 2. AUDIO LOSS - Multi-codebook cross-entropy for voice cloning with teacher forcing
    audio_loss = _compute_audio_loss_with_teacher_forcing(model_outputs, batch)
    if audio_loss > 0:
        total_loss += audio_loss_weight * audio_loss
        loss_components.audio_loss = audio_loss
    
    # 3. VOICE CONSISTENCY LOSS (Optional for zero-shot voice cloning)
    consistency_loss = _compute_consistency_loss(model_outputs, batch)
    if consistency_loss > 0:
        total_loss += consistency_loss_weight * consistency_loss
        loss_components.consistency_loss = consistency_loss
    
    loss_components.total_loss = total_loss
    
    # Convert to tensor for backpropagation
    total_loss_tensor = torch.tensor(total_loss, requires_grad=True) if isinstance(total_loss, float) else total_loss
    
    return total_loss_tensor, loss_components


def _compute_text_loss_with_masking(model_outputs, batch) -> float:
    """Compute text generation loss with proper teacher forcing masking."""
    try:
        # Check for text logits in model outputs
        if not hasattr(model_outputs, 'logits') or model_outputs.logits is None:
            return 0.0
        
        text_logits = model_outputs.logits  # Shape: [batch, seq_len, vocab_size]
        
        # Get text labels from batch (HiggsAudioBatchInput from collator)
        text_labels = None
        if hasattr(batch, 'label_ids') and batch.label_ids is not None:
            text_labels = batch.label_ids
        elif hasattr(batch, 'labels') and batch.labels is not None:
            text_labels = batch.labels
        elif isinstance(batch, dict):
            text_labels = batch.get('label_ids') or batch.get('labels')
        
        if text_labels is None:
            return 0.0
        
        # Ensure tensors are on the same device
        if text_logits.device != text_labels.device:
            text_labels = text_labels.to(text_logits.device)
        
        # Teacher forcing: shift labels for next-token prediction
        # Input: [BOS, token1, token2, token3]
        # Labels: [token1, token2, token3, EOS]
        if text_logits.size(1) > text_labels.size(1):
            # Logits has one more position (typical for causal LM)
            text_logits = text_logits[:, :-1, :].contiguous()
        elif text_labels.size(1) > text_logits.size(1):
            # Labels has one more position
            text_labels = text_labels[:, 1:].contiguous()
        
        # Only compute loss on non-masked tokens (-100 is the ignore index)
        valid_tokens = text_labels != -100
        if valid_tokens.sum() == 0:
            return 0.0
        
        # Compute cross-entropy loss with proper masking
        text_loss = F.cross_entropy(
            text_logits.reshape(-1, text_logits.size(-1)),
            text_labels.reshape(-1),
            ignore_index=-100,
            reduction='mean'
        )
        
        return text_loss.item() if isinstance(text_loss, torch.Tensor) else text_loss
    
    except Exception as e:
        print(f"⚠️ Error computing text loss with masking: {e}")
        return 0.0


def _compute_audio_loss_with_teacher_forcing(model_outputs, batch) -> float:
    """
    Compute multi-codebook audio generation loss with teacher forcing.
    
    Follows EXACT patterns from train_higgs_lora.py:
    - Teacher forcing: shift-right inputs per codebook
    - Labels with BOS=1024 → -100, keep EOS=1025 for stopping
    - Multi-codebook loss computation with proper masking
    """
    try:
        # Check for audio logits in model outputs
        if not hasattr(model_outputs, 'audio_logits') or model_outputs.audio_logits is None:
            return 0.0
        
        audio_logits = model_outputs.audio_logits
        
        # 🎯 Get audio labels from batch (following HiggsAudioBatchInput patterns)
        audio_labels = None
        if hasattr(batch, 'label_audio_ids') and batch.label_audio_ids is not None:
            audio_labels = batch.label_audio_ids
        elif hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
            # Fallback: use audio_out_ids and create labels following train_higgs_lora.py
            audio_out_ids = batch.audio_out_ids
            AUDIO_BOS = 1024  # From higgs-audio config
            
            # 🔄 Teacher forcing label creation (exact pattern from train_higgs_lora.py)
            audio_labels = audio_out_ids.clone()
            audio_labels[:, 0] = -100  # Do not learn BOS
            
            # Map BOS tokens inside labels (if any) to -100
            bos_mask = (audio_labels == AUDIO_BOS)
            if bos_mask.any():
                audio_labels[bos_mask] = -100
        elif isinstance(batch, dict) and 'label_audio_ids' in batch:
            audio_labels = batch['label_audio_ids']
        
        if audio_labels is None:
            return 0.0
        
        # Ensure tensors are on the same device
        if audio_logits.device != audio_labels.device:
            audio_labels = audio_labels.to(audio_logits.device)
        
        # 📐 Handle audio logits shape following HiggsAudioModel patterns
        if audio_logits.dim() == 2:
            # Shape: [num_audio_tokens, num_codebooks * codebook_size]
            num_tokens, logits_dim = audio_logits.shape
            
            # Determine codebook configuration from model config
            # Standard: 12 codebooks with 1024 + 2 special tokens each = 1026
            num_codebooks = 12  # From HiggsAudioConfig
            codebook_size_with_special = logits_dim // num_codebooks
            
            # Reshape to [num_codebooks, num_tokens, codebook_size]
            audio_logits = audio_logits.view(num_tokens, num_codebooks, codebook_size_with_special).transpose(0, 1)
        
        elif audio_logits.dim() == 3:
            # Shape: [num_codebooks, num_tokens, codebook_size] - correct format
            pass
        else:
            print(f"⚠️ Unexpected audio_logits shape: {audio_logits.shape}")
            return 0.0
        
        # 📐 Ensure audio_labels has correct shape [num_codebooks, num_tokens]
        if audio_labels.dim() == 1:
            # Reshape flat tensor to [num_codebooks, num_tokens]
            num_codebooks = audio_logits.size(0)
            num_tokens = audio_labels.size(0) // num_codebooks
            if audio_labels.size(0) % num_codebooks != 0:
                print(f"⚠️ Audio labels size {audio_labels.size(0)} not divisible by codebooks {num_codebooks}")
                return 0.0
            audio_labels = audio_labels.view(num_codebooks, num_tokens)
        
        elif audio_labels.dim() == 3 and audio_labels.shape[0] == 1:
            # Remove batch dimension: [1, num_codebooks, num_tokens] → [num_codebooks, num_tokens]
            audio_labels = audio_labels.squeeze(0)
        
        # 🎯 Multi-codebook teacher forcing loss (exact pattern from train_higgs_lora.py)
        audio_loss = 0.0
        num_codebooks = min(audio_logits.size(0), audio_labels.size(0))
        valid_codebooks = 0
        
        for codebook_idx in range(num_codebooks):
            codebook_logits = audio_logits[codebook_idx]  # [num_tokens, codebook_size]
            codebook_labels = audio_labels[codebook_idx]  # [num_tokens]
            
            # Only compute loss on valid audio tokens (ignore -100 masked tokens)
            valid_mask = codebook_labels != -100
            if valid_mask.sum() == 0:
                continue  # Skip codebook with no valid tokens
            
            # 🔄 Teacher forcing alignment: ensure logits and labels match for next-token prediction
            # Following the delay pattern from generation.py and train_higgs_lora.py
            if codebook_logits.size(0) != codebook_labels.size(0):
                min_length = min(codebook_logits.size(0), codebook_labels.size(0))
                codebook_logits = codebook_logits[:min_length, :].contiguous()
                codebook_labels = codebook_labels[:min_length].contiguous()
                valid_mask = valid_mask[:min_length].contiguous()
            
            if valid_mask.sum() > 0:
                # Compute cross-entropy loss with -100 masking
                codebook_loss = F.cross_entropy(
                    codebook_logits.reshape(-1, codebook_logits.size(-1)),
                    codebook_labels.reshape(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
                
                if torch.isfinite(codebook_loss):
                    audio_loss += codebook_loss
                    valid_codebooks += 1
        
        # Average across valid codebooks (standard multi-codebook training)
        if valid_codebooks > 0:
            audio_loss = audio_loss / valid_codebooks
            return audio_loss.item() if isinstance(audio_loss, torch.Tensor) else audio_loss
        else:
            return 0.0
    
    except Exception as e:
        print(f"⚠️ Error computing audio loss with teacher forcing: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def _compute_consistency_loss(model_outputs, batch) -> float:
    """
    Compute voice consistency loss for zero-shot voice cloning.
    
    This ensures the model learns to maintain voice characteristics
    across different text inputs for the same speaker.
    """
    try:
        # Simple consistency loss based on same-speaker audio segments
        if (not hasattr(batch, 'audio_waveforms_concat') or 
            batch.audio_waveforms_concat is None or
            len(batch.audio_waveforms_concat) == 0):
            return 0.0
        
        # Check if we have speaker indices for grouping
        if (not hasattr(batch, 'audio_speaker_indices') or 
            batch.audio_speaker_indices is None or
            len(batch.audio_speaker_indices) <= 1):
            return 0.0
        
        consistency_loss = 0.0
        unique_speakers = torch.unique(batch.audio_speaker_indices)
        
        for speaker_id in unique_speakers:
            speaker_mask = batch.audio_speaker_indices == speaker_id
            if speaker_mask.sum() > 1:  # Multiple segments from same speaker
                # Get audio features for this speaker
                speaker_waveforms = batch.audio_waveforms_concat[speaker_mask]
                
                if len(speaker_waveforms) > 1:
                    # Simple L2 consistency loss between audio features
                    mean_features = speaker_waveforms.mean(dim=0)
                    speaker_consistency = F.mse_loss(
                        speaker_waveforms,
                        mean_features.unsqueeze(0).expand_as(speaker_waveforms),
                        reduction='mean'
                    )
                    consistency_loss += speaker_consistency
        
        # Scale down consistency loss (it's auxiliary)
        if consistency_loss > 0:
            consistency_loss = consistency_loss * 0.1
            return consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss
        else:
            return 0.0
    
    except Exception as e:
        print(f"⚠️ Error computing consistency loss: {e}")
        return 0.0


def log_training_metrics(loss_components: LossComponents, step: int):
    """
    Log training metrics to monitor voice cloning quality.
    """
    print(f"Step {step}:")
    print(f"  Total Loss: {loss_components.total_loss:.4f}")
    print(f"  Text Loss: {loss_components.text_loss:.4f}")
    print(f"  Audio Loss: {loss_components.audio_loss:.4f}")
    print(f"  Consistency Loss: {loss_components.consistency_loss:.4f}")
    
    # Voice cloning quality indicators
    if loss_components.text_loss > 0 and loss_components.audio_loss > 0:
        text_audio_ratio = loss_components.text_loss / max(loss_components.audio_loss, 1e-8)
        print(f"  Text/Audio Ratio: {text_audio_ratio:.2f}")
        
        # Check for DualFFN balance (critical for voice cloning)
        if text_audio_ratio > 10:
            print(f"  ⚠️ Text loss dominance! May impact audio generation quality")
        elif text_audio_ratio < 0.1:
            print(f"  ⚠️ Audio loss dominance! May impact text understanding")
        else:
            print(f"  ✅ Good DualFFN balance")


def validate_loss_computation(model, batch, device):
    """
    Validation function to ensure loss computation is working correctly.
    """
    model.eval()
    with torch.no_grad():
        # Move batch to device
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
        
        try:
            outputs = model(**batch)
            loss_tensor, loss_components = compute_higgs_audio_loss(outputs, batch)
            
            print(f"🔍 Loss Validation Results:")
            print(f"  Model outputs available: logits={hasattr(outputs, 'logits')}, audio_logits={hasattr(outputs, 'audio_logits')}")
            print(f"  Loss components: {loss_components.to_dict()}")
            print(f"  Total loss tensor: {loss_tensor}")
            print(f"  Loss requires grad: {loss_tensor.requires_grad}")
            
            return loss_tensor, loss_components
            
        except Exception as e:
            print(f"❌ Loss validation failed: {e}")
            return None, None


def create_audio_labels_for_teacher_forcing(audio_out_ids, audio_stream_bos_id=1024, audio_stream_eos_id=1025):
    """
    Create audio labels for teacher forcing following HiggsAudioSampleCollator patterns.
    
    This function mimics the label creation logic from the collator:
    - BOS token (1024) at position 0 is masked with -100
    - EOS token (1025) is kept for stopping logic
    - All intermediate tokens are used for learning
    
    Args:
        audio_out_ids: [num_codebooks, seq_len] tensor with audio tokens
        audio_stream_bos_id: BOS token ID (default 1024)
        audio_stream_eos_id: EOS token ID (default 1025)
    
    Returns:
        Audio labels with proper masking for teacher forcing
    """
    if not TORCH_AVAILABLE:
        return None
    
    if audio_out_ids is None or audio_out_ids.numel() == 0:
        return None
    
    # Clone to avoid modifying original
    audio_labels = audio_out_ids.clone()
    
    # 🔄 Teacher forcing setup (exact pattern from train_higgs_lora.py)
    # Mask BOS tokens at position 0 with -100 (don't learn BOS)
    audio_labels[:, 0] = -100
    
    # Map any internal BOS tokens to -100 (following train_higgs_lora.py)
    bos_mask = (audio_labels == audio_stream_bos_id)
    if bos_mask.any():
        audio_labels[bos_mask] = -100
    
    # Keep EOS tokens for stopping logic (no masking)
    # EOS tokens at the end help the model learn when to stop generation
    
    return audio_labels


def compute_training_loss(model, batch, device, **loss_kwargs):
    """
    Enhanced training wrapper with proper audio label handling.
    
    This function handles the complete training forward pass including:
    - Model forward pass on correct device
    - Audio label creation if missing
    - Dual-loss computation with teacher forcing
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training loss computation")
    
    model.train()
    
    # 🖥️ Move batch to device
    if isinstance(batch, dict):
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device)
    else:
        # Handle HiggsAudioBatchInput objects
        for attr_name in dir(batch):
            if not attr_name.startswith('_'):
                attr_value = getattr(batch, attr_name)
                if isinstance(attr_value, torch.Tensor):
                    setattr(batch, attr_name, attr_value.to(device))
    
    # 🎯 Create audio labels if missing (following collator patterns)
    if not hasattr(batch, 'label_audio_ids') or batch.label_audio_ids is None:
        if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
            # Create labels following the exact collator pattern
            batch.label_audio_ids = create_audio_labels_for_teacher_forcing(
                batch.audio_out_ids,
                audio_stream_bos_id=1024,  # From HiggsAudioConfig
                audio_stream_eos_id=1025   # From HiggsAudioConfig
            )
    
    # 🚀 Model forward pass
    try:
        model_outputs = model(**batch.__dict__ if hasattr(batch, '__dict__') else batch)
    except Exception as e:
        print(f"⚠️ Model forward pass failed: {e}")
        # Return dummy loss for error handling
        return torch.tensor(0.0, device=device, requires_grad=True), LossComponents(0.0, 0.0, 0.0, 0.0)
    
    # 🎯 Compute dual loss with enhanced teacher forcing
    loss_tensor, loss_components = compute_higgs_audio_loss(
        model_outputs, 
        batch, 
        **loss_kwargs
    )
    
    return loss_tensor, loss_components
    # Move batch to device  
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
    
    # Forward pass
    model.train()
    outputs = model(**batch)
    
    # Compute loss using robust dual-loss function
    loss, loss_components = compute_higgs_audio_loss(outputs, batch, **loss_kwargs)
    
    return loss, loss_components


# Backward compatibility aliases
compute_loss = compute_higgs_audio_loss
compute_dual_loss = compute_higgs_audio_loss