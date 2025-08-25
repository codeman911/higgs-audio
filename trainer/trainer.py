"""
Main training class for Higgs-Audio LoRA training pipeline.

Follows the exact patterns from generation.py and arb_inference.py:
- HiggsAudioModel loading and setup
- HiggsAudioSampleCollator configuration  
- Whisper processor integration
- LoRA integration with PEFT
- Robust dual-loss computation for DualFFN architecture
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger

# Add parent directory to path for boson_multimodal imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import existing boson_multimodal components (conditional for better error handling)
try:
    from boson_multimodal.model.higgs_audio import HiggsAudioModel
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    BOSON_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import boson_multimodal components: {e}")
    logger.error(f"   Make sure you're running from the higgs-audio directory")
    logger.error(f"   Current working directory: {os.getcwd()}")
    logger.error(f"   Python path includes: {[p for p in sys.path if 'higgs' in p]}")
    BOSON_AVAILABLE = False
    
    # Create dummy classes to prevent import errors
    class HiggsAudioModel:
        pass
    class HiggsAudioSampleCollator:
        pass
    def load_higgs_audio_tokenizer(*args, **kwargs):
        return None

# Import our custom components
from .config import TrainingConfig
from .dataset import VoiceCloningDataset
from .loss import compute_training_loss, log_training_metrics, validate_loss_computation
from .logging_utils import training_logger
from .audio_validation import audio_validator


class HiggsAudioTrainer:
    """
    Trainer with robust dual-loss function for zero-shot voice cloning.
    
    Reuses existing boson_multimodal components without over-engineering.
    Follows the exact model loading and collator setup patterns from generation.py.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the Higgs-Audio trainer.
        
        Args:
            config: Training configuration containing all hyperparameters
        """
        # Check if boson_multimodal is available
        if not BOSON_AVAILABLE:
            raise ImportError(
                "boson_multimodal package not found. Please ensure you're running from the higgs-audio directory "
                "and the boson_multimodal package is in your Python path."
            )
        
        self.config = config
        self.device = self._setup_device()
        
        # Initialize comprehensive logging
        training_logger.log_pipeline_initialization(
            model_path=config.model_path,
            audio_tokenizer_path=config.audio_tokenizer_path,
            device=self.device,
            collator_config={
                'return_audio_in_tokens': False,  # serve_engine.py alignment
                'round_to': 1,                    # serve_engine.py alignment
                'encode_whisper_embed': 'auto',   # Will be determined
            }
        )
        
        # Validate configuration for training if needed
        # Note: Skip validation for utility operations like sample data creation
        self.config_validated = False
        
        # Setup model and tokenizers (exact match with generation.py)
        self._setup_model()
        self._setup_tokenizers()
        self._setup_lora()
        self._setup_collator()
        self._setup_data()
        
        # Comprehensive pipeline validation
        validation_results = training_logger.validate_training_pipeline(self)
        if not validation_results.is_valid():
            failed_checks = validation_results.get_failed_checks()
            logger.warning(f"⚠️ Some validation checks failed: {', '.join(failed_checks)}")
            logger.warning("Training will continue but may have issues")
        
        logger.info("✅ Higgs-Audio LoRA Trainer initialized successfully")
    
    def _setup_device(self) -> str:
        """Setup compute device matching generation.py patterns."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda:0"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = self.config.device
        
        logger.info(f"🔧 Using device: {device}")
        return device
    
    def _setup_model(self):
        """Load model exactly like generation.py."""
        logger.info(f"📦 Loading Higgs-Audio model from {self.config.model_path}")
        
        # Load model with exact same pattern as generation.py
        self.model = HiggsAudioModel.from_pretrained(
            self.config.model_path,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )
        
        # Switch to training mode
        self.model.train()
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _setup_tokenizers(self):
        """Setup tokenizers matching generation.py patterns."""
        logger.info("🔤 Loading tokenizers")
        
        # Load text tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        self.model_config = AutoConfig.from_pretrained(self.config.model_path)
        
        # Load audio tokenizer (exact match with generation.py)
        audio_tokenizer_device = "cpu" if self.device == "mps" else self.device
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.config.audio_tokenizer_path,
            device=audio_tokenizer_device
        )
        
        logger.info("✅ Tokenizers loaded successfully")
    
    def _setup_lora(self):
        """Setup LoRA adaptation focusing on DualFFN heads."""
        logger.info("🔗 Setting up LoRA adaptation")
        
        # LoRA configuration focusing on DualFFN output heads
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            task_type=TaskType.FEATURE_EXTRACTION
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        # Log LoRA statistics
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"✅ LoRA setup complete")
        logger.info(f"   Target modules: {self.config.lora_target_modules}")
        logger.info(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        logger.info(f"   LoRA rank: {self.config.lora_r}, alpha: {self.config.lora_alpha}")
    
    def _setup_collator(self):
        """Setup collator matching generation.py patterns."""
        logger.info("🎛️ Setting up data collator")
        
        # Try to load Whisper processor (matching arb_inference.py pattern)
        whisper_processor = None
        whisper_models = [
            "openai/whisper-large-v3",
            "openai/whisper-base",
            "openai/whisper-tiny"
        ]
        
        for model_name in whisper_models:
            try:
                whisper_processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                logger.info(f"✅ Loaded Whisper processor: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
        
        if whisper_processor is None:
            logger.warning("⚠️ No Whisper processor available - using DAC-only mode")
        
        # Force enable Whisper embeddings for better voice cloning (matching arb_inference.py)
        encode_whisper_embed = whisper_processor is not None
        if encode_whisper_embed:
            logger.info("🎤 Whisper embedding enabled for reference audio conditioning")
        
        # CRITICAL: Setup collator with exact serve_engine.py configuration for training/inference alignment
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            audio_in_token_id=self.model_config.audio_in_token_idx,
            audio_out_token_id=self.model_config.audio_out_token_idx,
            audio_stream_bos_id=self.model_config.audio_stream_bos_id,
            audio_stream_eos_id=self.model_config.audio_stream_eos_id,
            encode_whisper_embed=encode_whisper_embed,
            pad_token_id=self.model_config.pad_token_id,
            return_audio_in_tokens=False,  # CRITICAL: serve_engine.py uses False
            use_delay_pattern=self.model_config.use_delay_pattern,
            round_to=1,  # CRITICAL: serve_engine.py uses fixed round_to=1
            audio_num_codebooks=self.model_config.audio_num_codebooks,
        )
        
        logger.info("✅ Data collator setup complete")
        logger.info(f"   Whisper embedding: {encode_whisper_embed}")
        logger.info(f"   Audio codebooks: {self.model_config.audio_num_codebooks}")
        
        # Validate collator alignment with serve_engine.py
        expected_config = {'encode_whisper_embed': encode_whisper_embed}
        training_logger.validate_collator_alignment(self.collator, expected_config)
    
    def _setup_data(self):
        """Setup training and validation datasets."""
        logger.info("📚 Setting up datasets")
        
        # Training dataset
        self.train_dataset = VoiceCloningDataset(
            data_path=self.config.train_data_path,
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            validate_audio_paths=True,
        )
        
        # Validation dataset (if provided)
        self.val_dataset = None
        if os.path.exists(self.config.val_data_path):
            self.val_dataset = VoiceCloningDataset(
                data_path=self.config.val_data_path,
                tokenizer=self.tokenizer,
                audio_tokenizer=self.audio_tokenizer,
                validate_audio_paths=True,
            )
            logger.info(f"📊 Validation dataset: {len(self.val_dataset)} samples")
        
        # Training dataloader
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.dataloader_pin_memory,
        )
        
        logger.info(f"✅ Training dataset: {len(self.train_dataset)} samples")
        logger.info(f"   Batch size: {self.config.batch_size}")
        logger.info(f"   Steps per epoch: {len(self.train_dataloader)}")
    
    def train(self):
        """Main training loop with robust dual-loss computation."""
        # Validate configuration before training
        if not self.config_validated:
            logger.info("🔍 Validating configuration for training...")
            self.config.validate_for_training()
            self.config_validated = True
        
        logger.info("🎯 Starting training")
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs * len(self.train_dataloader)
        )
        
        # Training state
        global_step = 0
        best_loss = float('inf')
        
        # Enable gradient checkpointing if configured
        if self.config.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("🔧 Gradient checkpointing enabled")
        
        try:
            for epoch in range(self.config.num_epochs):
                logger.info(f"\n🚀 Epoch {epoch + 1}/{self.config.num_epochs}")
                
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    try:
                        # Training step
                        optimizer.zero_grad()
                        
                        # Forward pass with robust dual-loss
                        loss, loss_components = compute_training_loss(
                            self.model,
                            batch,
                            self.device,
                            text_loss_weight=self.config.text_loss_weight,
                            audio_loss_weight=self.config.audio_loss_weight,
                            consistency_loss_weight=self.config.consistency_loss_weight,
                        )
                        
                        # Backward pass with gradient clipping
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            max_norm=self.config.max_grad_norm
                        )
                        optimizer.step()
                        scheduler.step()
                        
                        # Update statistics
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        # Logging
                        if global_step % self.config.logging_steps == 0:
                            log_training_metrics(loss_components, global_step)
                            
                            # DualFFN balance monitoring (critical for voice cloning)
                            self._monitor_dualffn_balance(loss_components)
                            
                            # Audio quality validation (if we have audio outputs)
                            if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
                                self._validate_training_audio_quality(batch, global_step)
                        
                        # Validation
                        if global_step % self.config.eval_steps == 0 and global_step > 0:
                            val_loss = self._validate()
                            if val_loss < best_loss:
                                best_loss = val_loss
                                self.save_checkpoint(f"best-checkpoint")
                                logger.info(f"💎 New best validation loss: {val_loss:.4f}")
                        
                        # Checkpointing
                        if global_step % self.config.save_steps == 0 and global_step > 0:
                            self.save_checkpoint(f"checkpoint-{global_step}")
                        
                        global_step += 1
                        
                    except Exception as e:
                        logger.error(f"❌ Error in training step {global_step}: {e}")
                        continue
                
                # Epoch summary
                avg_epoch_loss = epoch_loss / max(num_batches, 1)
                logger.info(f"📈 Epoch {epoch + 1} complete: Avg Loss = {avg_epoch_loss:.4f}")
        
        except KeyboardInterrupt:
            logger.info("⏸️ Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
        finally:
            # Save final checkpoint
            self.save_checkpoint("final-checkpoint")
            logger.info("🏁 Training completed")
    
    def _monitor_dualffn_balance(self, loss_components):
        """Monitor DualFFN balance for voice cloning quality."""
        if loss_components.text_loss > 0 and loss_components.audio_loss > 0:
            balance_ratio = loss_components.text_loss / loss_components.audio_loss
            
            if balance_ratio > 10:
                logger.warning(f"⚠️ DualFFN imbalance! Text dominance: {balance_ratio:.2f}")
                logger.warning("   This may impact audio generation quality")
            elif balance_ratio < 0.1:
                logger.warning(f"⚠️ DualFFN imbalance! Audio dominance: {balance_ratio:.2f}")
                logger.warning("   This may impact text understanding")
    
    def _validate_training_audio_quality(self, batch, step: int):
        """Validate audio quality during training for debugging."""
        try:
            # Validate audio tokens from the batch
            if hasattr(batch, 'audio_out_ids') and batch.audio_out_ids is not None:
                audio_tokens = batch.audio_out_ids
                
                # Validate token sequences
                validation_results = audio_validator.validate_audio_tokens(
                    audio_tokens=audio_tokens,
                    audio_id=f"step_{step}_batch"
                )
                
                if not validation_results['valid']:
                    logger.warning(f"🎵 Step {step}: Audio token validation issues: {', '.join(validation_results['issues'])}")
                
                # Log token diversity metrics
                metrics = validation_results['metrics']
                if 'avg_unique_tokens' in metrics:
                    avg_unique = metrics['avg_unique_tokens']
                    seq_len = metrics.get('sequence_length', 0)
                    logger.debug(f"🎵 Step {step}: Token diversity: {avg_unique:.1f} unique tokens (seq_len: {seq_len})")
            
            # Validate reference audio waveforms if present
            if hasattr(batch, 'audio_features') and batch.audio_features is not None:
                training_logger.validate_whisper_conditioning(self.collator, batch)
                
        except Exception as e:
            logger.warning(f"⚠️ Audio quality validation error at step {step}: {e}")
    
    def _validate(self) -> float:
        """Run validation and return average loss."""
        if self.val_dataset is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Use small batch for validation
            shuffle=False,
            collate_fn=self.collator,
        )
        
        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    loss, _ = validate_loss_computation(self.model, batch, self.device)
                    if loss is not None:
                        total_loss += loss.item()
                        num_batches += 1
                except Exception as e:
                    logger.warning(f"Validation batch failed: {e}")
                    continue
        
        self.model.train()
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"🔍 Validation: {num_batches} batches, Avg Loss = {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.output_dir) / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save LoRA model
            self.model.save_pretrained(checkpoint_path)
            
            # Save training config
            config_path = checkpoint_path / "training_config.json"
            self.config.save(str(config_path))
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        try:
            checkpoint_dir = Path(checkpoint_path)
            
            # Load model weights
            self.model.load_state_dict(torch.load(checkpoint_dir / "pytorch_model.bin"))
            
            # Load config if available
            config_path = checkpoint_dir / "training_config.json"
            if config_path.exists():
                loaded_config = TrainingConfig.load(str(config_path))
                logger.info(f"📋 Loaded training config from checkpoint")
            
            self.model.train()
            logger.info(f"✅ Checkpoint loaded: {checkpoint_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise
    
    def generate_sample(self, text: str, ref_audio_path: str, output_path: str):
        """Generate a sample for testing (basic implementation)."""
        logger.info(f"🎵 Generating sample: '{text[:50]}...'")
        logger.info(f"   Reference audio: {ref_audio_path}")
        logger.info(f"   Output: {output_path}")
        
        # This would implement generation logic similar to arb_inference.py
        # For now, just log the intent
        logger.info("🚧 Sample generation not implemented yet - use arb_inference.py for testing")


# Utility functions for easy training setup
def create_trainer_from_config_file(config_path: str) -> HiggsAudioTrainer:
    """Create trainer from configuration file."""
    config = TrainingConfig.load(config_path)
    return HiggsAudioTrainer(config)


def create_trainer_with_defaults(
    train_data_path: str,
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
    **kwargs
) -> HiggsAudioTrainer:
    """Create trainer with default configuration and overrides."""
    config = TrainingConfig(
        train_data_path=train_data_path,
        model_path=model_path,
        audio_tokenizer_path=audio_tokenizer_path,
        **kwargs
    )
    return HiggsAudioTrainer(config)