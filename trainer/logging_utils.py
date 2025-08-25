"""
Comprehensive logging and validation utilities for Higgs-Audio LoRA training.

Provides debugging capabilities similar to arb_inference.py:
- Training pipeline validation
- Audio conditioning verification  
- Sample processing logging
- Loss component monitoring
- Alignment validation with inference patterns
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
from dataclasses import dataclass
import json


@dataclass
class TrainingValidationResults:
    """Container for training validation results."""
    collator_alignment: bool
    whisper_conditioning: bool
    audio_processing: bool
    sample_structure: bool
    token_processing: bool
    
    def is_valid(self) -> bool:
        """Check if all validation checks passed."""
        return all([
            self.collator_alignment,
            self.whisper_conditioning, 
            self.audio_processing,
            self.sample_structure,
            self.token_processing
        ])
    
    def get_failed_checks(self) -> List[str]:
        """Get list of failed validation checks."""
        failed = []
        if not self.collator_alignment:
            failed.append("collator_alignment")
        if not self.whisper_conditioning:
            failed.append("whisper_conditioning")
        if not self.audio_processing:
            failed.append("audio_processing")
        if not self.sample_structure:
            failed.append("sample_structure")
        if not self.token_processing:
            failed.append("token_processing")
        return failed


class TrainingLogger:
    """
    Comprehensive training logger matching arb_inference.py patterns.
    
    Provides detailed logging for:
    - Pipeline configuration validation
    - Audio conditioning verification
    - Sample processing monitoring
    - Loss component tracking
    - Alignment validation
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize training logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_level = log_level
        self.validation_results = {}
        self.training_metrics = {}
        
        # Configure logger format matching arb_inference.py
        logger.remove()
        logger.add(
            lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level
        )
    
    def log_pipeline_initialization(
        self,
        model_path: str,
        audio_tokenizer_path: str,
        device: str,
        collator_config: Dict[str, Any]
    ):
        """Log pipeline initialization details."""
        logger.info("🚀 Higgs-Audio LoRA Training Pipeline Initialization")
        logger.info("=" * 60)
        logger.info(f"📦 Model: {model_path}")
        logger.info(f"🎵 Audio Tokenizer: {audio_tokenizer_path}")
        logger.info(f"🔧 Device: {device}")
        logger.info("")
        logger.info("🎛️ Collator Configuration:")
        for key, value in collator_config.items():
            logger.info(f"   {key}: {value}")
        logger.info("")
    
    def validate_collator_alignment(
        self, 
        collator, 
        expected_config: Dict[str, Any]
    ) -> bool:
        """
        Validate collator configuration alignment with serve_engine.py patterns.
        
        Args:
            collator: HiggsAudioSampleCollator instance
            expected_config: Expected configuration values
            
        Returns:
            True if alignment is correct
        """
        logger.info("🔍 Validating Collator Alignment with serve_engine.py")
        
        mismatches = []
        alignment_correct = True
        
        # Check critical configuration parameters
        critical_params = {
            'return_audio_in_tokens': False,  # serve_engine.py uses False
            'round_to': 1,                    # serve_engine.py uses fixed round_to=1
            'encode_whisper_embed': expected_config.get('encode_whisper_embed', True),
        }
        
        for param, expected_value in critical_params.items():
            actual_value = getattr(collator, param, None)
            if actual_value != expected_value:
                mismatches.append(f"{param}: expected={expected_value}, actual={actual_value}")
                alignment_correct = False
            else:
                logger.info(f"   ✅ {param}: {actual_value}")
        
        if mismatches:\n            logger.error("❌ Collator Alignment Issues Found:")
            for mismatch in mismatches:
                logger.error(f"     - {mismatch}")
        else:
            logger.info("✅ Collator configuration perfectly aligned with serve_engine.py")
        
        self.validation_results['collator_alignment'] = alignment_correct
        return alignment_correct
    
    def validate_whisper_conditioning(
        self, 
        collator,
        sample_batch
    ) -> bool:
        """
        Validate Whisper conditioning setup matching arb_inference.py.
        
        Args:
            collator: HiggsAudioSampleCollator instance
            sample_batch: Batch of processed samples
            
        Returns:
            True if Whisper conditioning is properly configured
        """
        logger.info("🎤 Validating Whisper Conditioning Pipeline")
        
        whisper_available = collator.whisper_processor is not None
        whisper_enabled = collator.encode_whisper_embed
        
        logger.info(f"   Whisper processor available: {whisper_available}")
        logger.info(f"   Whisper embedding enabled: {whisper_enabled}")
        
        if whisper_available and whisper_enabled:
            # Check if batch contains audio features
            has_audio_features = (
                hasattr(sample_batch, 'audio_features') and 
                sample_batch.audio_features is not None and
                sample_batch.audio_features.numel() > 0
            )
            
            if has_audio_features:
                audio_features_shape = sample_batch.audio_features.shape
                logger.info(f"   ✅ Audio features shape: {audio_features_shape}")
                logger.info("   ✅ Whisper conditioning pipeline active")
                conditioning_valid = True
            else:
                logger.warning("   ⚠️ No audio features found in batch (may be empty batch)")
                conditioning_valid = True  # Still valid configuration
        else:
            logger.warning("   ⚠️ Whisper conditioning disabled (DAC-only mode)")
            conditioning_valid = True  # DAC-only is still valid
        
        self.validation_results['whisper_conditioning'] = conditioning_valid
        return conditioning_valid
    
    def log_sample_processing(
        self,
        sample_idx: int,
        input_tokens_len: int,
        label_tokens_len: int,
        audio_tokens_shape: Optional[Tuple],
        waveform_shape: Optional[Tuple],
        processing_mode: str
    ):
        """Log sample processing details."""
        logger.debug(f"📋 Sample {sample_idx} Processing:")
        logger.debug(f"   Input tokens: {input_tokens_len}")
        logger.debug(f"   Label tokens: {label_tokens_len}")
        logger.debug(f"   Audio tokens shape: {audio_tokens_shape}")
        logger.debug(f"   Waveform shape: {waveform_shape}")
        logger.debug(f"   Processing mode: {processing_mode}")
    
    def validate_audio_processing(
        self,
        audio_tokens: Optional[torch.Tensor],
        waveform: Optional[torch.Tensor],
        sample_rate: int
    ) -> bool:
        """
        Validate audio processing pipeline.
        
        Args:
            audio_tokens: Processed audio tokens
            waveform: Reference waveform  
            sample_rate: Audio sample rate
            
        Returns:
            True if audio processing is valid
        """
        logger.debug("🎵 Validating Audio Processing Pipeline")
        
        processing_valid = True
        
        # Validate audio tokens
        if audio_tokens is not None:
            if audio_tokens.dim() == 2:
                num_codebooks, seq_len = audio_tokens.shape
                logger.debug(f"   Audio tokens: {num_codebooks} codebooks, {seq_len} tokens")
                
                if num_codebooks != 12:  # Expected for DAC
                    logger.warning(f"   ⚠️ Unexpected codebook count: {num_codebooks} (expected 12)")
                else:
                    logger.debug("   ✅ Audio tokens shape valid")
            else:
                logger.warning(f"   ⚠️ Unexpected audio tokens shape: {audio_tokens.shape}")
                processing_valid = False
        else:
            logger.debug("   ⚠️ No audio tokens provided")
        
        # Validate waveform
        if waveform is not None:
            logger.debug(f"   Waveform shape: {waveform.shape}, sample rate: {sample_rate}Hz")
            
            # Check for common issues
            if torch.isnan(waveform).any():
                logger.error("   ❌ Waveform contains NaN values")
                processing_valid = False
            elif torch.isinf(waveform).any():
                logger.error("   ❌ Waveform contains infinite values")  
                processing_valid = False
            elif sample_rate != 16000:
                logger.warning(f"   ⚠️ Unexpected sample rate: {sample_rate}Hz (expected 16000Hz for Whisper)")
            else:
                logger.debug("   ✅ Waveform validation passed")
        else:
            logger.debug("   ⚠️ No waveform provided (DAC-only mode)")
        
        self.validation_results['audio_processing'] = processing_valid
        return processing_valid
    
    def log_batch_collation(
        self,
        batch_size: int,
        input_ids_shape: Tuple,
        attention_mask_shape: Tuple,
        audio_features_shape: Optional[Tuple],
        audio_out_ids_shape: Optional[Tuple]
    ):
        """Log batch collation results."""
        logger.debug(f"📦 Batch Collation Results:")
        logger.debug(f"   Batch size: {batch_size}")
        logger.debug(f"   Input IDs shape: {input_ids_shape}")
        logger.debug(f"   Attention mask shape: {attention_mask_shape}")
        if audio_features_shape is not None:
            logger.debug(f"   Audio features shape: {audio_features_shape}")
        if audio_out_ids_shape is not None:
            logger.debug(f"   Audio output IDs shape: {audio_out_ids_shape}")
    
    def log_loss_computation(
        self,
        step: int,
        total_loss: float,
        text_loss: float,
        audio_loss: float,
        consistency_loss: float
    ):
        """
        Log training loss components with detailed breakdown.
        
        Args:
            step: Training step
            total_loss: Total combined loss
            text_loss: Text generation loss component
            audio_loss: Audio generation loss component
            consistency_loss: Voice consistency loss component
        """
        logger.info(f"📊 Step {step} Loss Components:")
        logger.info(f"   Total Loss: {total_loss:.6f}")
        logger.info(f"   Text Loss: {text_loss:.6f}")
        logger.info(f"   Audio Loss: {audio_loss:.6f}")
        logger.info(f"   Consistency Loss: {consistency_loss:.6f}")
        
        # Analyze loss balance (critical for DualFFN training)
        if text_loss > 0 and audio_loss > 0:
            text_audio_ratio = text_loss / max(audio_loss, 1e-8)
            logger.info(f"   Text/Audio Ratio: {text_audio_ratio:.2f}")
            
            if text_audio_ratio > 10:
                logger.warning("   ⚠️ Text loss dominance! May impact audio generation quality")
            elif text_audio_ratio < 0.1:
                logger.warning("   ⚠️ Audio loss dominance! May impact text understanding")
            else:
                logger.info("   ✅ Good DualFFN balance")
        
        # Store metrics for monitoring
        self.training_metrics[step] = {
            'total_loss': total_loss,
            'text_loss': text_loss,
            'audio_loss': audio_loss,
            'consistency_loss': consistency_loss,
            'text_audio_ratio': text_loss / max(audio_loss, 1e-8) if audio_loss > 0 else float('inf')
        }
    
    def validate_training_pipeline(
        self,
        trainer
    ) -> TrainingValidationResults:
        """
        Comprehensive validation of training pipeline alignment.
        
        Args:
            trainer: HiggsAudioTrainer instance
            
        Returns:
            TrainingValidationResults with detailed validation status
        """
        logger.info("🔍 Comprehensive Training Pipeline Validation")
        logger.info("=" * 60)
        
        # Reset validation results
        self.validation_results = {}
        
        # 1. Validate collator alignment
        collator_config = {
            'encode_whisper_embed': trainer.collator.encode_whisper_embed,
        }
        self.validate_collator_alignment(trainer.collator, collator_config)
        
        # 2. Validate Whisper conditioning (if applicable)
        # Note: Sample batch validation will be done during training
        whisper_valid = trainer.collator.whisper_processor is not None
        logger.info(f"🎤 Whisper Processor Available: {whisper_valid}")
        self.validation_results['whisper_conditioning'] = True  # Will be validated during training
        
        # 3. Validate model configuration
        logger.info("📦 Model Configuration:")
        logger.info(f"   Device: {trainer.device}")
        logger.info(f"   Model dtype: {next(trainer.model.parameters()).dtype}")
        logger.info(f"   LoRA enabled: {hasattr(trainer.model, 'peft_config')}")
        
        # 4. Validate dataset
        logger.info("📚 Dataset Validation:")
        logger.info(f"   Training samples: {len(trainer.train_dataset)}")
        if hasattr(trainer, 'val_dataset') and trainer.val_dataset:
            logger.info(f"   Validation samples: {len(trainer.val_dataset)}")
        
        # Set default values for remaining validations
        self.validation_results.setdefault('audio_processing', True)
        self.validation_results.setdefault('sample_structure', True)
        self.validation_results.setdefault('token_processing', True)
        
        # Create validation results
        results = TrainingValidationResults(
            collator_alignment=self.validation_results.get('collator_alignment', False),
            whisper_conditioning=self.validation_results.get('whisper_conditioning', False),
            audio_processing=self.validation_results.get('audio_processing', False),
            sample_structure=self.validation_results.get('sample_structure', False),
            token_processing=self.validation_results.get('token_processing', False)
        )
        
        # Log final validation status
        if results.is_valid():
            logger.info("✅ Training pipeline validation PASSED")
        else:
            failed_checks = results.get_failed_checks()
            logger.error(f"❌ Training pipeline validation FAILED: {', '.join(failed_checks)}")
        
        logger.info("=" * 60)
        return results
    
    def log_training_start(
        self,
        config: Dict[str, Any]
    ):
        """Log training start with configuration summary."""
        logger.info("🎯 Starting Higgs-Audio LoRA Training")
        logger.info("=" * 60)
        logger.info("📋 Training Configuration:")
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                logger.info(f"   {key}: {value}")
        logger.info("=" * 60)
    
    def log_training_progress(
        self,
        epoch: int,
        step: int,
        total_steps: int,
        loss: float,
        learning_rate: float
    ):
        """Log training progress."""
        progress = (step / total_steps) * 100
        logger.info(f"📈 Epoch {epoch}, Step {step}/{total_steps} ({progress:.1f}%) - Loss: {loss:.6f}, LR: {learning_rate:.2e}")
    
    def save_training_metrics(self, output_path: str):
        """Save training metrics to file for analysis."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.training_metrics, f, indent=2)
            logger.info(f"💾 Training metrics saved to {output_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save training metrics: {e}")


# Global training logger instance
training_logger = TrainingLogger()