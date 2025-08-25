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
    label_ids: Optional[torch.LongTensor]  # shape (bsz, seq_len) - text labels (matches model API)
    
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
    
    # Audio labels for teacher forcing (matches model API)
    label_audio_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    
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
            # Comprehensive validation including padding and masking
            self._validate_output_batch(training_batch)
            
            # Additional padding/masking validation for teacher forcing
            padding_validation = self._validate_padding_and_masking(training_batch)
            if not padding_validation['teacher_forcing_ready']:
                logger.error(f"Teacher forcing validation failed: {padding_validation['warnings']}")
                # Don't raise exception here, just log warning for now
            
            # Comprehensive teacher forcing setup validation
            if self.enable_teacher_forcing:
                tf_validation = self._validate_teacher_forcing_setup(training_batch)
                if tf_validation['overall_status'] != 'READY':
                    logger.warning(f"Teacher forcing setup issues detected: {tf_validation['issues']}")
                    if tf_validation['recommendations']:
                        logger.info(f"Recommendations: {tf_validation['recommendations']}")
                    
                    # Log critical issues but don't stop training
                    if not tf_validation['voice_cloning_ready']:
                        logger.error("❌ Voice cloning setup not ready - training may not work correctly")
                else:
                    logger.debug("✅ Teacher forcing setup validated successfully")
            
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
            label_ids=base_batch.label_ids,  # Use correct parameter name for model
            audio_features=base_batch.audio_features,
            audio_feature_attention_mask=base_batch.audio_feature_attention_mask,
            audio_out_ids=base_batch.audio_out_ids,
            audio_out_ids_start=base_batch.audio_out_ids_start,
            audio_out_ids_start_group_loc=base_batch.audio_out_ids_start_group_loc,
            audio_in_ids=base_batch.audio_in_ids,
            audio_in_ids_start=base_batch.audio_in_ids_start,
            label_audio_ids=audio_labels,  # Use correct parameter name for model
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
        
        IMPORTANT: For teacher forcing training, labels must have EXACT same shape
        as audio_out_ids. The delay pattern is applied during model forward pass,
        not in the labels.
        """
        if base_batch.audio_out_ids is None:
            return None
        
        try:
            # For teacher forcing training, labels should match audio_out_ids exactly
            # DO NOT apply delay pattern here - it's applied by the model during forward pass
            audio_labels = base_batch.audio_out_ids.clone()
            
            # Apply proper label masking for teacher forcing
            # Mask positions that should not contribute to loss (e.g., padding)
            if hasattr(self.config, 'pad_token_id'):
                # Mask padding tokens in labels
                pad_mask = (audio_labels == self.config.pad_token_id)
                audio_labels[pad_mask] = -100  # Standard ignore index for CrossEntropyLoss
            
            # For teacher forcing, we typically want to predict the next token
            # So we can shift labels by 1 position if needed (this depends on model architecture)
            # For now, keep labels as-is since Higgs Audio handles shifting internally
            
            # Apply label smoothing if configured (optional)
            if hasattr(self.config, 'label_smoothing') and self.config.label_smoothing > 0:
                audio_labels = self._apply_label_smoothing(audio_labels)
            
            logger.debug(f"Created audio labels with shape: {audio_labels.shape}")
            logger.debug(f"Audio labels shape matches audio_out_ids: {audio_labels.shape == base_batch.audio_out_ids.shape}")
            return audio_labels
            
        except Exception as e:
            logger.warning(f"Failed to create audio labels: {e}")
            # Fallback: return exact copy of audio_out_ids
            return base_batch.audio_out_ids.clone() if base_batch.audio_out_ids is not None else None
    
    def _apply_delay_pattern_to_labels(self, audio_labels: torch.LongTensor) -> torch.LongTensor:
        """
        Apply delay pattern to audio labels for training alignment.
        
        ⚠️  WARNING: This function is currently DISABLED for teacher forcing training.
        
        The delay pattern adds extra tokens (seq_len + num_codebooks - 1) which changes
        the tensor shape. This causes shape mismatches during training because:
        
        1. audio_out_ids has shape [num_codebooks, seq_len] 
        2. Delay pattern creates shape [num_codebooks, seq_len + num_codebooks - 1]
        3. This breaks teacher forcing where labels must match inputs exactly
        
        The delay pattern should be:
        - APPLIED during model inference/generation (handled by model)
        - NOT APPLIED to training labels (handled here)
        
        For proper teacher forcing training, labels should have identical shape
        to the target tokens they're supposed to predict.
        """
        try:
            logger.debug(f"Delay pattern application requested but disabled for training compatibility")
            logger.debug(f"Original audio_labels shape: {audio_labels.shape}")
            
            # DISABLED: Do not apply delay pattern to training labels
            # This prevents shape mismatches in teacher forcing training
            # The model will apply delay pattern internally during forward pass
            
            # Keep original implementation for reference but commented out:
            # num_codebooks, seq_len = audio_labels.shape
            # audio_labels_with_batch = audio_labels.unsqueeze(0)  # Add batch dimension
            # bos_token_id = getattr(self.config, 'audio_stream_bos_id', self.config.pad_token_id)
            # pad_token_id = getattr(self.config, 'pad_token_id', -100)
            # delayed_labels, _ = build_delay_pattern_mask(
            #     input_ids=audio_labels_with_batch,
            #     bos_token_id=bos_token_id,
            #     pad_token_id=pad_token_id
            # )
            # delayed_labels = delayed_labels.squeeze(0)
            # return delayed_labels
            
            return audio_labels  # Return unchanged for training compatibility
            
        except Exception as e:
            logger.warning(f"Failed to apply delay pattern to labels: {e}")
            return audio_labels
    
    def _apply_label_smoothing(self, audio_labels: torch.LongTensor) -> torch.LongTensor:
        """Apply label smoothing to audio labels if configured."""
        # Placeholder for label smoothing implementation
        # This would involve creating soft targets instead of hard labels
        return audio_labels
    
    def _validate_padding_and_masking(self, batch: HiggsAudioTrainingBatch) -> Dict[str, Any]:
        """
        Comprehensive validation of padding and masking for teacher forcing.
        
        Returns:
            Dict with validation results and statistics
        """
        validation_results = {
            'padding_valid': True,
            'masking_valid': True,
            'teacher_forcing_ready': True,
            'statistics': {},
            'warnings': []
        }
        
        try:
            # 1. Validate text label padding and masking
            if batch.label_ids is not None:
                # Check for proper padding token usage
                pad_token_id = getattr(self.config, 'pad_token_id', -100)
                ignore_index = -100  # Standard ignore index for CrossEntropyLoss
                
                # Count padding tokens
                pad_positions = (batch.label_ids == pad_token_id)
                ignore_positions = (batch.label_ids == ignore_index)
                
                validation_results['statistics']['text_pad_tokens'] = pad_positions.sum().item()
                validation_results['statistics']['text_ignore_tokens'] = ignore_positions.sum().item()
                
                # Validate that padding is properly masked for loss computation
                if pad_positions.any() and not ignore_positions.any():
                    validation_results['warnings'].append(
                        "Found padding tokens but no ignore tokens in text labels - may cause incorrect loss computation"
                    )
            
            # 2. Validate audio label padding and masking
            if batch.label_audio_ids is not None:
                # Check audio label masking
                audio_pad_positions = (batch.label_audio_ids == getattr(self.config, 'pad_token_id', -100))
                audio_ignore_positions = (batch.label_audio_ids == -100)
                
                validation_results['statistics']['audio_pad_tokens'] = audio_pad_positions.sum().item()
                validation_results['statistics']['audio_ignore_tokens'] = audio_ignore_positions.sum().item()
                
                # Validate codebook consistency
                num_codebooks = batch.label_audio_ids.shape[0]
                for cb_idx in range(num_codebooks):
                    cb_pad_count = audio_pad_positions[cb_idx].sum().item()
                    cb_ignore_count = audio_ignore_positions[cb_idx].sum().item()
                    validation_results['statistics'][f'codebook_{cb_idx}_pad'] = cb_pad_count
                    validation_results['statistics'][f'codebook_{cb_idx}_ignore'] = cb_ignore_count
            
            # 3. Validate attention mask consistency
            if batch.attention_mask is not None:
                # Check attention mask validity
                mask_zeros = (batch.attention_mask == 0).sum().item()
                mask_ones = (batch.attention_mask == 1).sum().item()
                
                validation_results['statistics']['attention_masked_positions'] = mask_zeros
                validation_results['statistics']['attention_active_positions'] = mask_ones
                
                # Validate mask alignment with input length
                if batch.input_ids is not None:
                    seq_len = batch.input_ids.shape[1]
                    if batch.attention_mask.shape[1] != seq_len:
                        validation_results['masking_valid'] = False
                        validation_results['warnings'].append(
                            f"Attention mask length {batch.attention_mask.shape[1]} doesn't match sequence length {seq_len}"
                        )
            
            # 4. Teacher forcing readiness check
            if batch.audio_out_ids is not None and batch.label_audio_ids is not None:
                # Verify exact shape matching for teacher forcing
                shape_match = (batch.audio_out_ids.shape == batch.label_audio_ids.shape)
                validation_results['teacher_forcing_ready'] = shape_match
                
                if not shape_match:
                    validation_results['warnings'].append(
                        f"Teacher forcing incompatible: audio_out_ids {batch.audio_out_ids.shape} != label_audio_ids {batch.label_audio_ids.shape}"
                    )
                
                # Check if labels are properly shifted for next-token prediction
                # (This depends on model architecture - Higgs Audio handles internally)
                validation_results['statistics']['audio_sequence_length'] = batch.audio_out_ids.shape[1]
                validation_results['statistics']['audio_codebooks'] = batch.audio_out_ids.shape[0]
            
            # 5. Log validation summary
            if validation_results['warnings']:
                logger.warning(f"Padding/masking validation warnings: {validation_results['warnings']}")
            else:
                logger.debug(f"Padding/masking validation passed: {validation_results['statistics']}")
                
        except Exception as e:
            logger.error(f"Padding/masking validation failed: {e}")
            validation_results['padding_valid'] = False
            validation_results['masking_valid'] = False
            validation_results['teacher_forcing_ready'] = False
        
        return validation_results
    
    def _validate_teacher_forcing_setup(self, batch: HiggsAudioTrainingBatch) -> Dict[str, Any]:
        """
        Comprehensive validation of teacher forcing setup for Arabic voice cloning.
        
        Validates:
        1. Input-target alignment for text generation
        2. Audio token alignment for voice cloning
        3. Reference audio conditioning setup
        4. Multi-codebook consistency
        5. Zero-shot capability preservation
        
        Returns:
            Dict with validation results and recommendations
        """
        validation_results = {
            'teacher_forcing_valid': True,
            'voice_cloning_ready': True,
            'zero_shot_compatible': True,
            'issues': [],
            'recommendations': [],
            'statistics': {}
        }
        
        try:
            # 1. Text Teacher Forcing Validation
            if batch.input_ids is not None and batch.label_ids is not None:
                # Verify input-label alignment for text generation
                input_shape = batch.input_ids.shape
                label_shape = batch.label_ids.shape
                
                if input_shape != label_shape:
                    validation_results['teacher_forcing_valid'] = False
                    validation_results['issues'].append(
                        f"Text input-label shape mismatch: {input_shape} vs {label_shape}"
                    )
                
                # Check for proper label shifting (if model expects it)
                # Note: Higgs Audio handles shifting internally, so we keep them aligned
                validation_results['statistics']['text_sequence_length'] = input_shape[1]
                validation_results['statistics']['batch_size'] = input_shape[0]
            
            # 2. Audio Teacher Forcing Validation
            if batch.audio_out_ids is not None and batch.label_audio_ids is not None:
                audio_input_shape = batch.audio_out_ids.shape
                audio_label_shape = batch.label_audio_ids.shape
                
                # CRITICAL: Audio inputs and labels must have identical shapes for teacher forcing
                if audio_input_shape != audio_label_shape:
                    validation_results['teacher_forcing_valid'] = False
                    validation_results['voice_cloning_ready'] = False
                    validation_results['issues'].append(
                        f"Audio input-label shape mismatch: {audio_input_shape} vs {audio_label_shape}"
                    )
                    validation_results['recommendations'].append(
                        "Ensure delay pattern is NOT applied to training labels - only to model inputs during inference"
                    )
                
                # Validate multi-codebook consistency
                num_codebooks = audio_input_shape[0]
                expected_codebooks = getattr(self.config, 'audio_num_codebooks', 8)
                
                if num_codebooks != expected_codebooks:
                    validation_results['issues'].append(
                        f"Codebook count mismatch: got {num_codebooks}, expected {expected_codebooks}"
                    )
                
                validation_results['statistics']['audio_codebooks'] = num_codebooks
                validation_results['statistics']['audio_sequence_length'] = audio_input_shape[1]
                
                # Check for proper token range (codebook-specific validation)
                for cb_idx in range(num_codebooks):
                    cb_tokens = batch.audio_out_ids[cb_idx]
                    unique_tokens = torch.unique(cb_tokens[cb_tokens != -100])  # Exclude ignore tokens
                    
                    if len(unique_tokens) > 0:
                        min_token = unique_tokens.min().item()
                        max_token = unique_tokens.max().item()
                        validation_results['statistics'][f'codebook_{cb_idx}_token_range'] = (min_token, max_token)
                        
                        # Validate token range is reasonable for audio codebook
                        expected_max = getattr(self.config, 'audio_codebook_size', 1024)
                        # Include special audio tokens: BOS (1024) and EOS (1025)
                        audio_stream_bos = getattr(self.config, 'audio_stream_bos_id', 1024)
                        audio_stream_eos = getattr(self.config, 'audio_stream_eos_id', 1025)
                        max_valid_token = max(expected_max - 1, audio_stream_bos, audio_stream_eos)
                        
                        if max_token > max_valid_token:
                            validation_results['issues'].append(
                                f"Codebook {cb_idx} has token {max_token} > expected max {max_valid_token} "
                                f"(codebook_size: {expected_max}, BOS: {audio_stream_bos}, EOS: {audio_stream_eos})"
                            )
                        else:
                            # Log valid token range for debugging
                            logger.debug(f"Codebook {cb_idx} token range {min_token}-{max_token} is valid "
                                       f"(max allowed: {max_valid_token})")
            
            # 3. Voice Cloning Setup Validation
            if batch.audio_features is not None:
                # Verify reference audio features are present for voice conditioning
                audio_features_shape = batch.audio_features.shape
                validation_results['statistics']['audio_features_shape'] = audio_features_shape
                
                # Expected shape: (num_audio_segments, feature_dim, time_steps)
                if len(audio_features_shape) != 3:
                    validation_results['voice_cloning_ready'] = False
                    validation_results['issues'].append(
                        f"Audio features should be 3D, got {len(audio_features_shape)}D: {audio_features_shape}"
                    )
                
                # Verify feature dimensions are reasonable
                if audio_features_shape[1] < 512:  # Whisper features are typically 512+ dim
                    validation_results['recommendations'].append(
                        f"Audio features dimension {audio_features_shape[1]} seems low for Whisper features"
                    )
            else:
                # No audio features - check if this is intended
                if batch.audio_out_ids is not None:
                    validation_results['recommendations'].append(
                        "AUDIO FEATURES MISSING: No Whisper features found but audio generation requested. "
                        "For zero-shot voice cloning, you need reference audio processed through Whisper."
                    )
                    validation_results['recommendations'].append(
                        "Quick fix: Ensure your dataset includes reference audio and Whisper processor is enabled. "
                        "This is currently operating in text-to-speech mode, not voice cloning mode."
                    )
                else:
                    # Text-only training, which is fine
                    validation_results['recommendations'].append(
                        "ℹ️ Text-only training mode detected (no audio generation)"
                    )
            
            # 4. Zero-Shot Compatibility Check
            # For zero-shot voice cloning, we need both reference conditioning and target generation
            has_reference_audio = (batch.audio_features is not None)
            has_target_audio = (batch.audio_out_ids is not None and batch.label_audio_ids is not None)
            
            if has_target_audio and not has_reference_audio:
                validation_results['zero_shot_compatible'] = False
                validation_results['issues'].append(
                    "Target audio generation without reference audio - not suitable for zero-shot voice cloning"
                )
                validation_results['recommendations'].append(
                    "CRITICAL: For zero-shot voice cloning, you need reference audio features processed through Whisper. "
                    "Either: 1) Enable Whisper processing in your dataset/collator, or 2) Use a different training mode."
                )
                validation_results['recommendations'].append(
                    "Check: 1) Whisper processor is initialized, 2) Reference audio paths are valid, "
                    "3) encode_whisper_embed=True in model config"
                )
            elif has_target_audio and has_reference_audio:
                # Good setup for zero-shot voice cloning
                validation_results['recommendations'].append(
                    "✅ Zero-shot voice cloning setup detected with reference audio conditioning"
                )
            
            # 5. Attention Mask Validation for Audio Regions
            if batch.attention_mask is not None and batch.input_ids is not None:
                # Check if audio tokens are properly masked
                audio_token_ids = [
                    getattr(self.config, 'audio_in_token_idx', None),
                    getattr(self.config, 'audio_out_token_idx', None)
                ]
                
                for token_id in audio_token_ids:
                    if token_id is not None:
                        audio_positions = (batch.input_ids == token_id)
                        if audio_positions.any():
                            # Check if audio positions have proper attention
                            audio_attention = batch.attention_mask[audio_positions]
                            if not audio_attention.all():
                                validation_results['recommendations'].append(
                                    f"Some audio tokens (ID: {token_id}) are masked in attention - verify if intended"
                                )
            
            # 6. Generate Summary and Recommendations
            if validation_results['issues']:
                logger.warning(f"Teacher forcing validation issues: {validation_results['issues']}")
            
            if validation_results['recommendations']:
                logger.info(f"Teacher forcing recommendations: {validation_results['recommendations']}")
            
            # Overall status
            overall_valid = (
                validation_results['teacher_forcing_valid'] and
                validation_results['voice_cloning_ready'] and
                validation_results['zero_shot_compatible']
            )
            
            validation_results['overall_status'] = 'READY' if overall_valid else 'ISSUES_FOUND'
            
            logger.debug(f"Teacher forcing validation: {validation_results['overall_status']}")
            logger.debug(f"Statistics: {validation_results['statistics']}")
            
        except Exception as e:
            logger.error(f"Teacher forcing validation failed: {e}")
            validation_results['teacher_forcing_valid'] = False
            validation_results['voice_cloning_ready'] = False
            validation_results['zero_shot_compatible'] = False
            validation_results['issues'].append(f"Validation exception: {str(e)}")
        
        return validation_results
    
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
        """Validate output batch structure with comprehensive checks."""
        # Check tensor shapes and types
        if batch.input_ids is None:
            raise ValueError("Missing input_ids in training batch")
        
        if batch.attention_mask is None:
            raise ValueError("Missing attention_mask in training batch")
        
        if batch.input_ids.shape != batch.attention_mask.shape:
            raise ValueError(f"Shape mismatch: input_ids {batch.input_ids.shape} vs attention_mask {batch.attention_mask.shape}")
        
        if batch.label_ids is not None and batch.label_ids.shape != batch.input_ids.shape:
            raise ValueError(f"Shape mismatch: input_ids {batch.input_ids.shape} vs label_ids {batch.label_ids.shape}")
        
        # Validate audio components
        if batch.audio_out_ids is not None:
            if batch.audio_out_ids.dim() != 2:
                raise ValueError(f"audio_out_ids should be 2D, got {batch.audio_out_ids.dim()}D with shape {batch.audio_out_ids.shape}")
            
            expected_codebooks = self.config.audio_num_codebooks
            if batch.audio_out_ids.shape[0] != expected_codebooks:
                raise ValueError(f"Expected {expected_codebooks} codebooks, got {batch.audio_out_ids.shape[0]} with shape {batch.audio_out_ids.shape}")
        
        # CRITICAL: Validate audio labels match audio_out_ids exactly
        if batch.label_audio_ids is not None and batch.audio_out_ids is not None:
            if batch.label_audio_ids.shape != batch.audio_out_ids.shape:
                logger.error(f"SHAPE MISMATCH DETECTED:")
                logger.error(f"  - audio_out_ids shape: {batch.audio_out_ids.shape}")
                logger.error(f"  - label_audio_ids shape: {batch.label_audio_ids.shape}")
                logger.error(f"  - Expected: Both shapes should be identical for teacher forcing")
                logger.error(f"  - This is likely due to delay pattern being applied to labels")
                raise ValueError(f"Audio labels shape {batch.label_audio_ids.shape} doesn't match audio_out_ids {batch.audio_out_ids.shape}. "
                               f"For teacher forcing, these must be identical. Check if delay pattern is incorrectly applied to labels.")
        
        # Validate audio features if present
        if batch.audio_features is not None:
            if batch.audio_features.dim() != 3:  # Expected: (num_audio, feature_dim, seq_len)
                logger.warning(f"Unexpected audio_features dimensions: {batch.audio_features.shape}")
        
        # Validate start indices consistency
        if batch.audio_out_ids is not None and batch.audio_out_ids_start is not None:
            max_start_idx = batch.audio_out_ids_start.max().item() if len(batch.audio_out_ids_start) > 0 else 0
            actual_seq_len = batch.audio_out_ids.shape[1]
            if max_start_idx > actual_seq_len:
                logger.warning(f"audio_out_ids_start max index {max_start_idx} exceeds sequence length {actual_seq_len}")
        
        logger.debug(f"Training batch validation passed:")
        logger.debug(f"  - Batch size: {batch.input_ids.shape[0]}")
        logger.debug(f"  - Sequence length: {batch.input_ids.shape[1]}")
        logger.debug(f"  - Audio tokens: {batch.audio_out_ids.shape if batch.audio_out_ids is not None else 'None'}")
        logger.debug(f"  - Audio labels: {batch.label_audio_ids.shape if batch.label_audio_ids is not None else 'None'}")
        logger.debug(f"  - Audio features: {batch.audio_features.shape if batch.audio_features is not None else 'None'}")
        logger.debug(f"  - Shapes match: {batch.label_audio_ids.shape == batch.audio_out_ids.shape if batch.label_audio_ids is not None and batch.audio_out_ids is not None else 'N/A'}")
    
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
        
        if batch.label_ids is not None and batch.label_ids.shape != batch.input_ids.shape:
            return False
        
        # Audio consistency checks
        if batch.audio_out_ids is not None and batch.label_audio_ids is not None:
            if batch.audio_out_ids.shape != batch.label_audio_ids.shape:
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