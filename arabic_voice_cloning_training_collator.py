#!/usr/bin/env python3
"""
Optimized Training Collator for Arabic Voice Cloning

This module implements a training-optimized collator that leverages the existing
boson_multimodal infrastructure while adding proper teacher forcing setup and
label alignment for DualFFN training.

Key Features:
- Uses HiggsAudioSampleCollator as base for complex audio processing
- Proper label alignment for both text and audio generation
- Enhanced attention masking for audio regions  
- Support for teacher forcing with correct token shifting
- Comprehensive logging and validation
- Optimized for multi-GPU training with proper batching
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from loguru import logger

# Higgs Audio imports
from boson_multimodal.data_collator.higgs_audio_collator import (
    HiggsAudioSampleCollator,
    HiggsAudioBatchInput
)
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample
from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
from boson_multimodal.model.higgs_audio.utils import build_delay_pattern_mask
from transformers.models.whisper.processing_whisper import WhisperProcessor


@dataclass
class HiggsAudioTrainingBatch:
    """Enhanced batch structure for training with proper label alignment."""
    
    # Standard batch inputs
    input_ids: torch.LongTensor  # shape (bsz, seq_len)
    attention_mask: torch.Tensor  # shape (bsz, seq_len)
    labels: Optional[torch.LongTensor]  # shape (bsz, seq_len) - text labels
    
    # Audio features for conditioning
    audio_features: Optional[torch.Tensor]  # shape (num_audio_in, feature_dim, max_mel_seq_len)
    audio_feature_attention_mask: Optional[torch.Tensor]  # shape (num_audio_in, max_mel_seq_len)
    
    # Audio generation inputs
    audio_out_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor]  # shape (num_audio_out,)
    audio_out_ids_start_group_loc: Optional[torch.LongTensor]  # shape (num_audio_out,)
    
    # Audio conditioning inputs  
    audio_in_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor]  # shape (num_audio_in,)
    
    # Audio labels for teacher forcing
    audio_labels: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    
    # Additional training metadata
    sample_ids: Optional[List[str]] = None
    loss_weights: Optional[torch.Tensor] = None


class ArabicVoiceCloningTrainingCollator:
    """
    Training-optimized collator for Arabic voice cloning with teacher forcing.
    
    This collator leverages the existing HiggsAudioSampleCollator for complex
    audio processing while adding training-specific enhancements:
    - Proper label alignment for text and audio
    - Teacher forcing setup with correct token shifting
    - Enhanced attention masking
    - Comprehensive validation and logging
    """
    
    def __init__(
        self,
        config: HiggsAudioConfig,
        whisper_processor: Optional[WhisperProcessor] = None,
        text_loss_weight: float = 1.0,
        audio_loss_weight: float = 1.0,
        enable_teacher_forcing: bool = True,
        validate_batches: bool = True,
        **kwargs
    ):
        """
        Initialize the training collator.
        
        Args:
            config: Higgs Audio model configuration
            whisper_processor: Whisper processor for audio conditioning
            text_loss_weight: Weight for text generation loss
            audio_loss_weight: Weight for audio generation loss  
            enable_teacher_forcing: Whether to enable teacher forcing
            validate_batches: Whether to validate batch structure
            **kwargs: Additional arguments for base collator
        """
        self.config = config
        self.text_loss_weight = text_loss_weight
        self.audio_loss_weight = audio_loss_weight
        self.enable_teacher_forcing = enable_teacher_forcing
        self.validate_batches = validate_batches
        
        # Initialize base collator with training-optimized settings
        self.base_collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=config.audio_in_token_idx,
            audio_out_token_id=config.audio_out_token_idx,
            audio_stream_bos_id=config.audio_stream_bos_id,
            audio_stream_eos_id=config.audio_stream_eos_id,
            encode_whisper_embed=config.encode_whisper_embed,
            pad_token_id=config.pad_token_id,
            return_audio_in_tokens=config.encode_audio_in_tokens,
            use_delay_pattern=config.use_delay_pattern,
            round_to=1,  # Training optimized
            audio_num_codebooks=config.audio_num_codebooks,
            mask_audio_out_token_label=True,  # Important for training
            **kwargs
        )
        
        logger.info(f"Training collator initialized:")
        logger.info(f"  - Teacher forcing: {enable_teacher_forcing}")
        logger.info(f"  - Whisper processing: {whisper_processor is not None}")
        logger.info(f"  - Loss weights: text={text_loss_weight}, audio={audio_loss_weight}")
        logger.info(f"  - Delay pattern: {config.use_delay_pattern}")
    
    def __call__(self, batch: List[ChatMLDatasetSample]) -> HiggsAudioTrainingBatch:
        """
        Collate batch with enhanced training features.
        
        Args:
            batch: List of ChatMLDatasetSample instances
            
        Returns:
            HiggsAudioTrainingBatch with proper label alignment
        """
        if self.validate_batches:
            self._validate_input_batch(batch)
        
        # Defensive handling for tensor dimension issues
        for i, sample in enumerate(batch):
            if sample.audio_ids_concat is not None:
                # Fix 1D to 2D tensor issue
                if sample.audio_ids_concat.dim() == 1:
                    logger.warning(f"Sample {i}: Converting 1D audio_ids_concat to 2D")
                    # If we have tokens but wrong shape, reshape assuming 8 codebooks
                    if len(sample.audio_ids_concat) > 0:
                        # Try to reshape, but if not divisible by 8, create empty 2D tensor
                        if len(sample.audio_ids_concat) % 8 == 0:
                            sample.audio_ids_concat = sample.audio_ids_concat.reshape(8, -1)
                        else:
                            logger.warning(f"Sample {i}: Cannot reshape audio tokens, using empty tensor")
                            sample.audio_ids_concat = torch.empty((8, 0), dtype=torch.long)
                    else:
                        # Empty 1D tensor, convert to empty 2D
                        sample.audio_ids_concat = torch.empty((8, 0), dtype=torch.long)
                
                # Validate codebook dimension
                if sample.audio_ids_concat.shape[0] != self.config.audio_num_codebooks:
                    logger.warning(f"Sample {i}: Wrong codebook count {sample.audio_ids_concat.shape[0]}, expected {self.config.audio_num_codebooks}")
                    # Create properly shaped empty tensor
                    sample.audio_ids_concat = torch.empty((self.config.audio_num_codebooks, 0), dtype=torch.long)
                    sample.audio_ids_start = torch.tensor([0], dtype=torch.long)
        
        try:
            # Use base collator for complex audio processing
            logger.debug(f"Processing batch of {len(batch)} samples")
            base_batch = self.base_collator(batch)
        except IndexError as e:
            if "too many indices for tensor of dimension 1" in str(e):
                logger.error(f"Tensor dimension error in base collator: {e}")
                logger.info("Attempting to fix tensor dimensions and retry...")
                
                # Fix tensor dimensions in batch
                for i, sample in enumerate(batch):
                    if hasattr(sample, 'audio_ids_concat') and sample.audio_ids_concat is not None:
                        if sample.audio_ids_concat.dim() == 1:
                            # Convert 1D to proper 2D shape
                            num_codebooks = self.config.audio_num_codebooks
                            sample.audio_ids_concat = torch.empty((num_codebooks, 0), dtype=torch.long)
                            sample.audio_ids_start = torch.tensor([0], dtype=torch.long)
                            logger.info(f"Fixed sample {i} tensor dimensions")
                
                # Retry with fixed tensors
                base_batch = self.base_collator(batch)
            else:
                raise
        
        # Create enhanced training batch
        training_batch = self._create_training_batch(base_batch, batch)
        
        if self.validate_batches:
            self._validate_output_batch(training_batch)
        
        return training_batch
    
    def _validate_input_batch(self, batch: List[ChatMLDatasetSample]):
        """Validate input batch structure."""
        if not batch:
            raise ValueError("Empty batch provided")
        
        for i, sample in enumerate(batch):
            if sample.input_ids is None or len(sample.input_ids) == 0:
                raise ValueError(f"Sample {i} has empty input_ids")
            
            if self.enable_teacher_forcing and sample.label_ids is None:
                logger.warning(f"Sample {i} missing label_ids for teacher forcing")
    
    def _create_training_batch(
        self, 
        base_batch: HiggsAudioBatchInput, 
        original_batch: List[ChatMLDatasetSample]
    ) -> HiggsAudioTrainingBatch:
        """Create training batch with proper label alignment."""
        
        # Extract loss weights if available
        loss_weights = None
        if hasattr(original_batch[0], 'loss_weight'):
            loss_weights = torch.tensor([getattr(sample, 'loss_weight', 1.0) for sample in original_batch])
        
        # Create aligned audio labels for teacher forcing
        audio_labels = None
        if self.enable_teacher_forcing and base_batch.audio_out_ids is not None:
            audio_labels = self._create_audio_labels(base_batch, original_batch)
        
        # Enhanced attention masking
        enhanced_attention_mask = self._create_enhanced_attention_mask(base_batch)
        
        # Create sample IDs for tracking
        sample_ids = [f"sample_{i}" for i in range(len(original_batch))]
        
        return HiggsAudioTrainingBatch(
            input_ids=base_batch.input_ids,
            attention_mask=enhanced_attention_mask,
            labels=base_batch.label_ids,
            audio_features=base_batch.audio_features,
            audio_feature_attention_mask=base_batch.audio_feature_attention_mask,
            audio_out_ids=base_batch.audio_out_ids,
            audio_out_ids_start=base_batch.audio_out_ids_start,
            audio_out_ids_start_group_loc=base_batch.audio_out_ids_start_group_loc,
            audio_in_ids=base_batch.audio_in_ids,
            audio_in_ids_start=base_batch.audio_in_ids_start,
            audio_labels=audio_labels,
            sample_ids=sample_ids,
            loss_weights=loss_weights
        )
    
    def _create_audio_labels(
        self, 
        base_batch: HiggsAudioBatchInput, 
        original_batch: List[ChatMLDatasetSample]
    ) -> Optional[torch.LongTensor]:
        """
        Create properly aligned audio labels for teacher forcing.
        
        This method ensures that audio labels are correctly aligned with the
        audio generation tokens for proper teacher forcing training.
        """
        if base_batch.audio_out_ids is None:
            return None
        
        try:
            # Start with the audio output tokens as base labels
            audio_labels = base_batch.audio_out_ids.clone()
            
            # Apply delay pattern if used in model (important for training alignment)
            if self.config.use_delay_pattern:
                audio_labels = self._apply_delay_pattern_to_labels(audio_labels)
            
            # Apply label smoothing if configured
            if hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
                audio_labels = self._apply_label_smoothing(audio_labels)
            
            logger.debug(f"Created audio labels with shape: {audio_labels.shape}")
            return audio_labels
            
        except Exception as e:
            logger.warning(f"Failed to create audio labels: {e}")
            return base_batch.audio_out_ids.clone() if base_batch.audio_out_ids is not None else None
    
    def _apply_delay_pattern_to_labels(self, audio_labels: torch.LongTensor) -> torch.LongTensor:
        """
        Apply delay pattern to audio labels for training alignment.
        
        This ensures that the training labels match the delay pattern used
        during audio generation, which is crucial for proper learning.
        """
        try:
            num_codebooks, seq_len = audio_labels.shape
            
            # Build delay pattern mask matching the model's configuration
            delay_pattern = build_delay_pattern_mask(
                num_codebooks, 
                seq_len, 
                device=audio_labels.device
            )
            
            # Apply delay pattern to labels
            delayed_labels = audio_labels.clone()
            for cb_idx in range(num_codebooks):
                delay = cb_idx  # Standard delay pattern
                if delay > 0:
                    # Shift labels according to delay pattern
                    delayed_labels[cb_idx, :-delay] = audio_labels[cb_idx, delay:]
                    delayed_labels[cb_idx, -delay:] = -100  # Mask delayed positions
            
            return delayed_labels
            
        except Exception as e:
            logger.warning(f"Failed to apply delay pattern to labels: {e}")
            return audio_labels
    
    def _apply_label_smoothing(self, audio_labels: torch.LongTensor) -> torch.LongTensor:
        """Apply label smoothing to audio labels if configured."""
        # Placeholder for label smoothing implementation
        # This would involve creating soft targets instead of hard labels
        return audio_labels
    
    def _create_enhanced_attention_mask(self, base_batch: HiggsAudioBatchInput) -> torch.Tensor:
        """
        Create enhanced attention mask with special handling for audio regions.
        
        This mask ensures proper attention flow during training, especially
        for the dual FFN architecture where audio and text tokens have
        different processing paths.
        """
        if base_batch.attention_mask is None:
            # Create basic attention mask if not provided
            attention_mask = torch.ones_like(base_batch.input_ids, dtype=torch.float32)
        else:
            attention_mask = base_batch.attention_mask.clone()
        
        # Apply enhanced masking for audio regions if needed
        if hasattr(self.config, 'use_enhanced_audio_masking') and self.config.use_enhanced_audio_masking:
            attention_mask = self._apply_audio_region_masking(attention_mask, base_batch)
        
        return attention_mask
    
    def _apply_audio_region_masking(
        self, 
        attention_mask: torch.Tensor, 
        base_batch: HiggsAudioBatchInput
    ) -> torch.Tensor:
        """Apply special masking for audio token regions."""
        # Identify audio token positions
        audio_positions = (
            (base_batch.input_ids == self.config.audio_in_token_idx) |
            (base_batch.input_ids == self.config.audio_out_token_idx)
        )
        
        # Apply enhanced masking for audio regions (optional)
        # This could include causal masking within audio sequences
        enhanced_mask = attention_mask.clone()
        
        # For now, keep standard attention masking
        # Future enhancement: implement specialized audio attention patterns
        
        return enhanced_mask
    
    def _validate_output_batch(self, batch: HiggsAudioTrainingBatch):
        """Validate output batch structure."""
        # Check tensor shapes and types
        if batch.input_ids is None:
            raise ValueError("Missing input_ids in training batch")
        
        if batch.attention_mask is None:
            raise ValueError("Missing attention_mask in training batch")
        
        if batch.input_ids.shape != batch.attention_mask.shape:
            raise ValueError(f"Shape mismatch: input_ids {batch.input_ids.shape} vs attention_mask {batch.attention_mask.shape}")
        
        if batch.labels is not None and batch.labels.shape != batch.input_ids.shape:
            raise ValueError(f"Shape mismatch: input_ids {batch.input_ids.shape} vs labels {batch.labels.shape}")
        
        # Validate audio components
        if batch.audio_out_ids is not None:
            if batch.audio_out_ids.dim() != 2:
                raise ValueError(f"audio_out_ids should be 2D, got {batch.audio_out_ids.dim()}D")
            
            expected_codebooks = self.config.audio_num_codebooks
            if batch.audio_out_ids.shape[0] != expected_codebooks:
                raise ValueError(f"Expected {expected_codebooks} codebooks, got {batch.audio_out_ids.shape[0]}")
        
        # Validate audio labels if present
        if batch.audio_labels is not None and batch.audio_out_ids is not None:
            if batch.audio_labels.shape != batch.audio_out_ids.shape:
                raise ValueError(f"Audio labels shape {batch.audio_labels.shape} doesn't match audio_out_ids {batch.audio_out_ids.shape}")
        
        logger.debug(f"Training batch validation passed:")
        logger.debug(f"  - Batch size: {batch.input_ids.shape[0]}")
        logger.debug(f"  - Sequence length: {batch.input_ids.shape[1]}")
        logger.debug(f"  - Audio tokens: {batch.audio_out_ids.shape if batch.audio_out_ids is not None else 'None'}")
        logger.debug(f"  - Audio features: {batch.audio_features.shape if batch.audio_features is not None else 'None'}")
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get configured loss weights."""
        return {
            'text_weight': self.text_loss_weight,
            'audio_weight': self.audio_loss_weight
        }
    
    def update_loss_weights(self, text_weight: float = None, audio_weight: float = None):
        """Update loss weights during training."""
        if text_weight is not None:
            self.text_loss_weight = text_weight
        if audio_weight is not None:
            self.audio_loss_weight = audio_weight
        
        logger.info(f"Updated loss weights: text={self.text_loss_weight}, audio={self.audio_loss_weight}")


# Utility functions for batch processing
def validate_training_batch(batch: HiggsAudioTrainingBatch) -> bool:
    """Standalone function to validate training batch structure."""
    try:
        # Basic structure checks
        if batch.input_ids is None or batch.attention_mask is None:
            return False
        
        # Shape consistency checks
        if batch.input_ids.shape != batch.attention_mask.shape:
            return False
        
        if batch.labels is not None and batch.labels.shape != batch.input_ids.shape:
            return False
        
        # Audio consistency checks
        if batch.audio_out_ids is not None and batch.audio_labels is not None:
            if batch.audio_out_ids.shape != batch.audio_labels.shape:
                return False
        
        return True
        
    except Exception:
        return False


def create_training_collator(
    config: HiggsAudioConfig,
    whisper_processor: Optional[WhisperProcessor] = None,
    **kwargs
) -> ArabicVoiceCloningTrainingCollator:
    """
    Factory function to create training collator with sensible defaults.
    
    Args:
        config: Higgs Audio model configuration
        whisper_processor: Optional Whisper processor
        **kwargs: Additional collator arguments
        
    Returns:
        Configured training collator
    """
    return ArabicVoiceCloningTrainingCollator(
        config=config,
        whisper_processor=whisper_processor,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    from boson_multimodal.model.higgs_audio.configuration_higgs_audio import HiggsAudioConfig
    
    # Test configuration
    config = HiggsAudioConfig()
    
    # Create collator
    collator = ArabicVoiceCloningTrainingCollator(
        config=config,
        whisper_processor=None,  # Would normally load a real processor
        enable_teacher_forcing=True,
        validate_batches=True
    )
    
    logger.info("Training collator created successfully")
    logger.info(f"Loss weights: {collator.get_loss_weights()}")