"""
Minimal DDP trainer with dual loss computation.
Strictly mirrors inference forward pass and loss patterns.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
import logging
from tqdm import tqdm

# Import exact components from boson_multimodal
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Import our components
from dataset import HiggsAudioDataset, create_collator
from lora import apply_lora, create_lora_config, save_lora_adapters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HiggsAudioTrainer:
    """Minimal trainer for DualFFN LoRA fine-tuning."""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.load_model_and_tokenizers()
        self.setup_dataset()
        self.setup_training()
        self.verify_output_directory()
    
    def setup_distributed(self):
        """Setup DDP training."""
        if "LOCAL_RANK" in os.environ:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.device = torch.device(f"cuda:{self.local_rank}")
    
    def load_model_and_tokenizers(self):
        """Load model and tokenizers exactly as inference does."""
        
        # Load configuration
        self.config = HiggsAudioConfig.from_pretrained(self.args.base_ckpt)
        
        # Force enable Whisper embeddings (from inference patterns)
        self.config.encode_whisper_embed = True
        
        # CRITICAL FIX: Enable cross-modal conditioning (audio attention)
        # This is essential for text to learn from audio context
        if not getattr(self.config, 'use_audio_out_self_attention', None):
            logger.info("ENABLING cross-modal conditioning (use_audio_out_self_attention=True)")
            self.config.use_audio_out_self_attention = True
        
        # Load model with exact inference initialization
        model = HiggsAudioModel.from_pretrained(
            self.args.base_ckpt,
            config=self.config,
            torch_dtype=torch.bfloat16
        )
        self.model = model.to(self.device)
        
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
        
        # Apply LoRA
        lora_config = create_lora_config(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout
        )
        self.model = apply_lora(self.model, lora_config)
        
        # Wrap with DDP
        if self.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def setup_dataset(self):
        """Setup dataset and dataloader."""
        
        # Create full dataset
        full_dataset = HiggsAudioDataset(
            manifest_path=self.args.train_manifest,
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer
        )
        
        # Split dataset into train and validation (95% train, 5% validation)
        total_size = len(full_dataset)
        val_size = int(total_size * 0.05)
        train_size = total_size - val_size
        
        # Create indices for train and validation splits
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create train and validation datasets
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Create collator with EXACT parameters
        self.collator = create_collator(self.config, self.whisper_processor)
        
        # Setup distributed sampler for training
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        # Setup distributed sampler for validation
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank,
            shuffle=False
        ) if self.world_size > 1 else None
        
        # Optimize DataLoader settings for performance
        # Using moderate num_workers to balance CPU utilization and overhead
        # Keeping persistent_workers=False to prevent deadlocks in distributed training
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collator,
            num_workers=8,  # Moderate value for balanced performance
            pin_memory=True,
            persistent_workers=False  # Disabled to prevent deadlock
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=4,  # Lower value for validation
            pin_memory=True,
            persistent_workers=False  # Disabled to prevent deadlock
        )
    
    def setup_training(self):
        """Setup optimizer and scheduler."""
        
        # Optimizer - target only LoRA parameters
        optimizer_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimizer_params.append(param)
        
        self.optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=self.args.lr,
            weight_decay=self.args.wd,
            betas=(0.9, 0.95)
        )
        
        # Scheduler
        total_steps = len(self.train_dataloader) * self.args.epochs // self.args.grad_accum
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.args.warmup,
            num_training_steps=total_steps
        )
        
        # Loss function - EXACT pattern from model implementation
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.audio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def compute_loss(self, batch):
        """Compute dual loss with complete PEFT bypass via custom forward implementation."""
        
        # Ensure batch is a dictionary
        if not isinstance(batch, dict):
            raise ValueError(f"Expected batch to be dict, got {type(batch)}")
        
        # Get the true underlying HiggsAudioModel
        base_model = self._get_base_higgs_model()
        
        # Prepare inputs - only remove the generic 'labels' key that PEFT might inject
        # Keep label_ids, label_audio_ids, and audio_out_ids which are required by the model
        clean_inputs = {}
        for k, v in batch.items():
            # Only exclude generic 'labels' that PEFT auto-injects
            # Keep all model-specific label parameters
            if k not in ['labels']:  # Remove ONLY the generic PEFT-injected 'labels'
                clean_inputs[k] = v
        
        # CRITICAL FIX: Ensure all tensors have consistent dtype for mixed precision training
        # The error occurs because audio embeddings are Float32 but final_embedding is BFloat16
        target_dtype = torch.bfloat16
        
        # Check for potential dtype issues and cast tensors to target dtype
        dtype_corrected_inputs = {}
        for k, v in clean_inputs.items():
            if isinstance(v, torch.Tensor):
                # Cast all tensor inputs to the target dtype to prevent dtype mismatches
                if v.dtype == torch.float32:
                    # Cast float32 tensors to bfloat16 for consistency
                    dtype_corrected_inputs[k] = v.to(dtype=target_dtype)
                elif v.dtype == torch.bfloat16:
                    # Already correct dtype
                    dtype_corrected_inputs[k] = v
                else:
                    # For non-float tensors (like int64 for indices), keep as is
                    dtype_corrected_inputs[k] = v
            else:
                dtype_corrected_inputs[k] = v
        
        # Extract labels for fallback loss computation (if needed)
        text_labels = batch.get('label_ids')
        audio_labels = batch.get('label_audio_ids')
        
        # BYPASS PEFT: Call the forward method directly on the base model
        # This completely avoids PEFT's parameter injection
        try:
            with torch.autocast(device_type='cuda', dtype=target_dtype):
                # Direct call to HiggsAudioModel.forward, bypassing all wrappers
                outputs = base_model.forward(**dtype_corrected_inputs)
        except Exception as e:
            logger.error(f"Direct forward call failed: {e}")
            raise
        
        return self._compute_dual_loss(outputs, text_labels, audio_labels)
    
    def _get_base_higgs_model(self):
        """Extract the actual HiggsAudioModel from all wrapper layers."""
        model = self.model
        path = []
        
        # Iteratively unwrap until we find HiggsAudioModel
        max_depth = 20
        for depth in range(max_depth):
            # Check if model is a tensor (error case)
            if isinstance(model, torch.Tensor):
                break
                
            model_type = type(model).__name__ if hasattr(model, '__class__') else str(type(model))
            path.append(model_type)
            
            # Found the target model
            if model_type == 'HiggsAudioModel':
                return model
            
            # Try different unwrapping attributes in order of likelihood
            if hasattr(model, 'module') and model.module is not None and not isinstance(model.module, torch.Tensor):  # DDP wrapper
                model = model.module
                continue
            elif hasattr(model, 'base_model') and model.base_model is not None and not isinstance(model.base_model, torch.Tensor):  # PEFT wrapper
                model = model.base_model
                continue
            elif hasattr(model, 'model') and model.model is not None and not isinstance(model.model, torch.Tensor):  # Generic wrapper
                model = model.model
                continue
            elif (hasattr(model, 'base_model') and 
                  hasattr(model.base_model, 'model') and 
                  model.base_model.model is not None and 
                  not isinstance(model.base_model.model, torch.Tensor)):
                model = model.base_model.model
                continue
            else:
                # No more wrappers found, check if this is the right model
                break
        
        return model
    
    def _compute_dual_loss(self, outputs, text_labels, audio_labels):
        """Compute dual loss from model outputs and labels."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}
        
        # Extract outputs
        text_logits = getattr(outputs, 'logits', None)
        audio_logits = getattr(outputs, 'audio_logits', None)
        
        # CRITICAL INSIGHT: The model already provides correctly aligned labels!
        # Use the model's expanded_labels instead of the original batch labels
        model_expanded_labels = getattr(outputs, 'expanded_labels', None)
        
        # ENHANCED LOGGING: Add detailed diagnostics for audio training
        if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
            logger.info(f"=== AUDIO TRAINING DIAGNOSTICS (Step {self.global_step}) ===")
            logger.info(f"Audio logits shape: {audio_logits.shape if audio_logits is not None else 'None'}")
            logger.info(f"Audio labels shape: {audio_labels.shape if audio_labels is not None else 'None'}")
            logger.info(f"Audio logits numel: {audio_logits.numel() if audio_logits is not None else 0}")
            logger.info(f"Audio labels numel: {audio_labels.numel() if audio_labels is not None else 0}")
            
            # Check for empty logits
            if audio_logits is not None and audio_logits.numel() == 0:
                logger.warning("❌ CRITICAL: Audio logits are empty! This will cause zero audio loss.")
                logger.warning("Check: 1) audio_out_ids in batch, 2) audio_out_mask generation")
            
            # Check label masking
            if audio_labels is not None and audio_labels.numel() > 0:
                masked_count = (audio_labels == -100).sum().item()
                total_count = audio_labels.numel()
                mask_percentage = (masked_count / max(total_count, 1)) * 100
                logger.info(f"Audio labels masking: {masked_count}/{total_count} ({mask_percentage:.1f}%) masked")
                
                if mask_percentage > 90:
                    logger.warning("⚠️  HIGH AUDIO LABEL MASKING: Over 90% of audio labels are masked!")
                    logger.warning("This may prevent effective audio learning.")
                
                # Log some sample audio label values for debugging
                if total_count > 0:
                    sample_labels = audio_labels.flatten()[:10].tolist()
                    logger.info(f"Sample audio labels (first 10): {sample_labels}")
            
            # Log text diagnostics as well
            if text_logits is not None and model_expanded_labels is not None:
                text_masked_count = (model_expanded_labels == -100).sum().item()
                text_total_count = model_expanded_labels.numel()
                text_mask_percentage = (text_masked_count / max(text_total_count, 1)) * 100
                logger.info(f"Text labels masking: {text_masked_count}/{text_total_count} ({text_mask_percentage:.1f}%) masked")
        
        # Text loss with OPTIMAL teacher forcing alignment
        if text_logits is not None and model_expanded_labels is not None:
            # BEST CASE: Use model's expanded_labels which are already correctly aligned!
            
            # CRITICAL FIX: The model's expanded_labels are already properly aligned with logits
            # No need to remove the last logit - they should have the same sequence length
            shift_logits = text_logits.contiguous()  # [batch, seq_len, vocab]
            shift_labels = model_expanded_labels.contiguous()  # [batch, seq_len]
            
            # Validate alignment
            if shift_logits.size(1) == shift_labels.size(1):
                text_loss = self.text_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq_len, vocab]
                    shift_labels.view(-1)                          # [batch*seq_len]
                )
                total_loss = total_loss + text_loss
                loss_dict['text_loss'] = text_loss.item()
                
        elif text_logits is not None and text_labels is not None:
            # FALLBACK: Use manual alignment if expanded_labels not available
            
            # STANDARD teacher forcing shift for autoregressive models
            shift_logits = text_logits[..., :-1, :].contiguous()  # Remove last logit
            shift_labels = text_labels[..., 1:].contiguous()      # Remove first label
            
            if shift_logits.size(1) == shift_labels.size(1):
                text_loss = self.text_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss = total_loss + text_loss
                loss_dict['text_loss'] = text_loss.item()
                
        # Audio loss - handle multi-codebook structure
        if audio_logits is not None and audio_labels is not None:
            if audio_logits.numel() > 0:
                audio_loss = self._compute_audio_loss(audio_logits, audio_labels)
                total_loss = total_loss + audio_loss
                loss_dict['audio_loss'] = audio_loss.item()
                
                # CRITICAL FIX: Add validation for near-zero audio loss
                if audio_loss.item() < 0.001:  # Near zero threshold
                    logger.warning(f"⚠️  NEAR-ZERO AUDIO LOSS DETECTED: {audio_loss.item():.6f}")
                    logger.warning("This indicates potential issues with audio learning.")
                    logger.warning("Possible causes: 1) Over-masking 2) Empty logits 3) Data issues")
            else:
                loss_dict['audio_loss'] = 0.0
                logger.warning("⚠️  Audio logits are empty - audio loss set to 0.0")
        else:
            loss_dict['audio_loss'] = 0.0
            if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
                logger.warning("⚠️  Audio logits or labels are None - audio loss set to 0.0")
        
        # Final loss summary
        loss_dict['total_loss'] = total_loss.item()
        
        # Log only the essential information
        if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
            # Log losses
            logger.info(f"Step {self.global_step} - Losses: Text={loss_dict.get('text_loss', 0.0):.4f}, Audio={loss_dict.get('audio_loss', 0.0):.4f}, Total={loss_dict.get('total_loss', 0.0):.4f}")
            
            # Log text predictions if we have text logits
            if text_logits is not None and model_expanded_labels is not None:
                self._log_minimal_text_predictions(text_logits, model_expanded_labels)
            
            # Log audio predictions if we have audio logits
            if audio_logits is not None and audio_labels is not None and audio_logits.numel() > 0:
                self._log_minimal_audio_predictions(audio_logits, audio_labels)
        
        return total_loss, loss_dict

    def _log_minimal_text_predictions(self, logits, labels):
        """Log minimal text predictions vs target labels with detokenization for Arabic text."""
        try:
            # Get predictions (argmax of logits)
            predictions = torch.argmax(logits, dim=-1)
            
            # Log for first sample in batch
            batch_size = logits.size(0)
            if batch_size > 0:
                # Get sequence for first sample
                sample_idx = 0
                sample_logits = logits[sample_idx]
                sample_predictions = predictions[sample_idx]
                sample_labels = labels[sample_idx]
                
                # Find valid (non-masked) positions
                valid_mask = sample_labels != -100
                if valid_mask.sum() > 0:
                    # Get valid predictions and labels
                    valid_predictions = sample_predictions[valid_mask]
                    valid_labels = sample_labels[valid_mask]
                    
                    # Convert to lists for easier handling
                    pred_tokens = valid_predictions.tolist()
                    label_tokens = valid_labels.tolist()
                    
                    # Detokenize to Arabic text
                    try:
                        predicted_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        target_text = self.tokenizer.decode(label_tokens, skip_special_tokens=True)
                        
                        logger.info(f"Arabic Text - Predicted: '{predicted_text}' | Target: '{target_text}'")
                    except Exception as decode_error:
                        pass
                        
        except Exception as e:
            pass

    def _compute_audio_loss(self, audio_logits, audio_labels):
        """Compute audio loss across multiple codebooks with proper sequence alignment."""
        if audio_logits is None or audio_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        # Handle empty sequences
        if audio_logits.numel() == 0 or audio_labels.numel() == 0:
            logger.warning("⚠️  Audio logits or labels are empty - returning zero loss")
            return torch.tensor(0.0, device=self.device)
        
        audio_loss = torch.tensor(0.0, device=self.device)
        
        # Handle different tensor shapes for multi-codebook audio
        if len(audio_logits.shape) == 3 and audio_logits.size(1) == 8:
            # Shape: [seq_len, 8_codebooks, vocab_size]
            num_codebooks = 8
            seq_len_logits = audio_logits.size(0)
            
            # Audio labels should be [8_codebooks, seq_len]
            if audio_labels.dim() == 2 and audio_labels.size(0) == 8:
                seq_len_labels = audio_labels.size(1)
                
                # Apply teacher forcing shift for audio (similar to text)
                # Audio logits predict next audio token
                if seq_len_logits > seq_len_labels:
                    # Trim logits to match labels
                    audio_logits = audio_logits[:seq_len_labels, :, :]
                    seq_len_logits = seq_len_labels
                elif seq_len_labels > seq_len_logits:
                    # Trim labels to match logits  
                    audio_labels = audio_labels[:, :seq_len_logits]
                    seq_len_labels = seq_len_logits
                
                # CRITICAL FIX: Enhanced validation for audio label quality
                total_tokens = audio_labels.numel()
                valid_tokens = (audio_labels != -100).sum().item()
                mask_ratio = 1.0 - (valid_tokens / max(total_tokens, 1))
                
                if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
                    logger.info(f"Audio label quality - Valid: {valid_tokens}/{total_tokens} ({(valid_tokens/max(total_tokens, 1))*100:.1f}%)")
                
                if mask_ratio > 0.95:  # Over 95% masked
                    logger.warning(f"⚠️  CRITICAL: Audio labels are {mask_ratio*100:.1f}% masked! This will prevent learning.")
                    if self.local_rank == 0:
                        logger.warning("Suggested actions:")
                        logger.warning("1. Check dataset for proper audio label generation")
                        logger.warning("2. Verify collator mask_audio_out_token_label=False")
                        logger.warning("3. Ensure audio files exist and are accessible")
                
                valid_codebooks = 0
                codebook_losses = []
                
                for cb in range(num_codebooks):
                    cb_logits = audio_logits[:, cb, :]  # [seq_len, vocab_size]
                    cb_labels = audio_labels[cb, :]     # [seq_len]
                    
                    # Only compute loss on valid tokens (ignore -100)
                    valid_mask = cb_labels != -100
                    valid_token_count = valid_mask.sum().item()
                    
                    if valid_token_count > 0:
                        cb_loss = self.audio_loss_fn(cb_logits, cb_labels)
                        if torch.isfinite(cb_loss):
                            audio_loss += cb_loss
                            valid_codebooks += 1
                            codebook_losses.append(cb_loss.item())
                            
                            # Log individual codebook loss for debugging
                            if self.global_step % (self.args.log_steps * 10) == 0 and self.local_rank == 0:
                                logger.info(f"  Codebook {cb} loss: {cb_loss.item():.4f} ({valid_token_count} valid tokens)")
                    elif self.global_step % (self.args.log_steps * 10) == 0 and self.local_rank == 0:
                        logger.info(f"  Codebook {cb}: No valid tokens (all masked)")
                
                if valid_codebooks > 0:
                    audio_loss = audio_loss / valid_codebooks
                    
                    # CRITICAL FIX: Add validation for near-zero audio loss
                    avg_loss = audio_loss.item()
                    if avg_loss < 0.001:  # Near zero threshold
                        logger.warning(f"⚠️  NEAR-ZERO AUDIO LOSS DETECTED: {avg_loss:.6f}")
                        logger.warning("This indicates potential issues with audio learning.")
                        logger.warning("Possible causes: 1) Over-masking 2) Empty logits 3) Data issues")
                        
                        # Add detailed diagnostics
                        if codebook_losses:
                            logger.info(f"  Individual codebook losses: {[f'{loss:.6f}' for loss in codebook_losses]}")
                        
                        # Suggest corrective actions
                        if self.local_rank == 0:
                            logger.warning("Suggested corrective actions:")
                            logger.warning("1. Verify audio files exist and are readable")
                            logger.warning("2. Check that audio_label_contents is properly extracted in dataset")
                            logger.warning("3. Ensure mask_audio_out_token_label=False in collator")
                            logger.warning("4. Validate audio tokenizer is working correctly")
                else:
                    logger.warning("⚠️  No valid codebooks found for audio loss computation")
                    return torch.tensor(0.0, device=self.device)
        else:
            # Fallback: treat as single output
            logger.warning(f"⚠️  Unexpected audio logits shape: {audio_logits.shape}")
            audio_loss = self.audio_loss_fn(
                audio_logits.view(-1, audio_logits.size(-1)),
                audio_labels.view(-1)
            )
        
        return audio_loss

    def _log_minimal_audio_predictions(self, audio_logits, audio_labels):
        """Log minimal audio predictions vs target labels - first and last 5 values."""
        try:
            # Get predictions for first codebook (most significant)
            if len(audio_logits.shape) == 3 and audio_logits.size(1) >= 1:
                # Get first codebook logits and labels
                first_codebook_logits = audio_logits[:, 0, :]  # [seq_len, vocab_size]
                first_codebook_labels = audio_labels[0, :]     # [seq_len]
                
                # Get predictions (argmax)
                predictions = torch.argmax(first_codebook_logits, dim=-1)
                
                # Find valid positions
                valid_mask = first_codebook_labels != -100
                if valid_mask.sum() > 0:
                    valid_predictions = predictions[valid_mask]
                    valid_labels = first_codebook_labels[valid_mask]
                    
                    # Get first 5 and last 5 predictions and labels
                    pred_tokens = valid_predictions.tolist()
                    label_tokens = valid_labels.tolist()
                    
                    # First 5 values
                    first_5_pred = pred_tokens[:5]
                    first_5_label = label_tokens[:5]
                    
                    # Last 5 values (if we have at least 5 tokens)
                    last_5_pred = pred_tokens[-5:] if len(pred_tokens) >= 5 else pred_tokens
                    last_5_label = label_tokens[-5:] if len(label_tokens) >= 5 else label_tokens
                    
                    logger.info(f"Audio Tokens - First 5 Pred: {first_5_pred} | First 5 Target: {first_5_label}")
                    logger.info(f"Audio Tokens - Last 5 Pred: {last_5_pred} | Last 5 Target: {last_5_label}")
                        
        except Exception as e:
            pass

    def train_step(self, batch):
        """Single training step with robust batch handling."""
        
        # Handle different batch types
        if hasattr(batch, '__dict__'):
            # Convert HiggsAudioBatchInput to dict
            batch_dict = {}
            for key, value in batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    # DTYPE FIX: Ensure consistent dtype for mixed precision training
                    if value.dtype == torch.float32:
                        # Cast float32 to bfloat16 for consistency
                        batch_dict[key] = value.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    else:
                        batch_dict[key] = value.to(self.device, non_blocking=True)
                else:
                    batch_dict[key] = value
            batch = batch_dict
        elif isinstance(batch, dict):
            # Handle regular dict batch with dtype consistency
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # DTYPE FIX: Ensure consistent dtype for mixed precision training
                    if v.dtype == torch.float32:
                        # Cast float32 to bfloat16 for consistency
                        processed_batch[k] = v.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                    else:
                        processed_batch[k] = v.to(self.device, non_blocking=True)
                else:
                    processed_batch[k] = v
            batch = processed_batch
        else:
            # Unknown batch type - try to handle gracefully
            try:
                processed_batch = {}
                for k, v in batch.__dict__.items():
                    if not k.startswith('_'):
                        if isinstance(v, torch.Tensor):
                            # DTYPE FIX: Ensure consistent dtype
                            if v.dtype == torch.float32:
                                processed_batch[k] = v.to(device=self.device, dtype=torch.bfloat16, non_blocking=True)
                            else:
                                processed_batch[k] = v.to(self.device, non_blocking=True)
                        else:
                            processed_batch[k] = v
                batch = processed_batch
            except Exception as e:
                raise
        
        # Validate batch has required keys
        required_keys = ['input_ids']
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise ValueError(f"Batch missing required keys: {missing_keys}")
        
        # Compute loss
        try:
            loss, loss_dict = self.compute_loss(batch)
        except Exception as e:
            raise
        
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.args.grad_accum
        
        # Backward pass
        scaled_loss.backward()
        
        return loss_dict
    
    def validate(self):
        """Validation loop."""
        self.model.eval()
        total_val_loss = 0.0
        total_text_loss = 0.0
        total_audio_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                if hasattr(batch, '__dict__'):
                    batch_dict = {}
                    for key, value in batch.__dict__.items():
                        if isinstance(value, torch.Tensor):
                            batch_dict[key] = value.to(self.device, non_blocking=True)
                        else:
                            batch_dict[key] = value
                    batch = batch_dict
                
                # Compute validation loss
                try:
                    val_loss, val_loss_dict = self.compute_loss(batch)
                    total_val_loss += val_loss.item()
                    if 'text_loss' in val_loss_dict:
                        total_text_loss += val_loss_dict['text_loss']
                    if 'audio_loss' in val_loss_dict:
                        total_audio_loss += val_loss_dict['audio_loss']
                    val_steps += 1
                except Exception as e:
                    continue
        
        self.model.train()
        
        if val_steps > 0:
            avg_val_loss = total_val_loss / val_steps
            avg_text_loss = total_text_loss / val_steps
            avg_audio_loss = total_audio_loss / val_steps
            return avg_val_loss, avg_text_loss, avg_audio_loss
        else:
            return 0.0, 0.0, 0.0
    
    def train(self):
        """Main training loop."""
        
        self.model.train()
        self.global_step = 0
        accum_step = 0
        
        # Log training configuration only once
        if self.local_rank == 0:
            logger.info(f"Starting training - Output: {self.args.output_dir}, Save steps: {self.args.save_steps}")
        
        for epoch in range(self.args.epochs):
            # Check if sampler has set_epoch method before calling it
            if (self.world_size > 1 and 
                hasattr(self.train_dataloader, 'sampler') and
                hasattr(self.train_dataloader.sampler, 'set_epoch') and 
                callable(getattr(self.train_dataloader.sampler, 'set_epoch', None))):
                self.train_dataloader.sampler.set_epoch(epoch)
            
            # Create progress bar for this epoch
            if self.local_rank == 0:
                pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}")
            else:
                pbar = self.train_dataloader
            
            for batch in pbar:
                # Training step
                loss_dict = self.train_step(batch)
                accum_step += 1
                
                # Gradient accumulation
                if accum_step % self.args.grad_accum == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Update
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Logging - only essential information
                    if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
                        # Check if pbar has set_postfix method before calling it
                        if hasattr(pbar, 'set_postfix') and callable(getattr(pbar, 'set_postfix', None)):
                            pbar.set_postfix({"loss": loss_dict.get('total_loss', 0.0)})
                    
                    # Validation
                    if self.global_step % self.args.val_steps == 0 and self.local_rank == 0:
                        val_loss, val_text_loss, val_audio_loss = self.validate()
                        logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}, Text: {val_text_loss:.4f}, Audio: {val_audio_loss:.4f}")
                    
                    # Checkpointing
                    if self.global_step % self.args.save_steps == 0:
                        try:
                            # Only save from main process in distributed training
                            if self.local_rank == 0:
                                # Ensure output directory exists
                                os.makedirs(self.args.output_dir, exist_ok=True)
                                logger.info(f"Created output directory: {self.args.output_dir}")
                                
                                # Check write permissions
                                if not os.access(self.args.output_dir, os.W_OK):
                                    logger.error(f"No write permission for output directory: {self.args.output_dir}")
                                    raise PermissionError(f"No write permission for output directory: {self.args.output_dir}")
                                    
                                # Create checkpoint directory
                                checkpoint_dir = f"{self.args.output_dir}/checkpoint-{self.global_step}"
                                os.makedirs(checkpoint_dir, exist_ok=True)
                                logger.info(f"Created checkpoint directory: {checkpoint_dir}")
                                
                                # Verify checkpoint directory is writable
                                test_file = os.path.join(checkpoint_dir, ".write_test")
                                try:
                                    with open(test_file, "w") as f:
                                        f.write("test")
                                    os.remove(test_file)
                                except Exception as perm_error:
                                    logger.error(f"No write permission for checkpoint directory: {checkpoint_dir}, Error: {perm_error}")
                                    raise PermissionError(f"No write permission for checkpoint directory: {checkpoint_dir}")
                                
                                # Save LoRA adapters
                                model_to_save = self.model.module if self.world_size > 1 else self.model
                                logger.info("Attempting to save LoRA adapters...")
                                save_lora_adapters(model_to_save, checkpoint_dir)
                                logger.info(f"Successfully saved checkpoint to: {checkpoint_dir}")
                                
                                # Verify checkpoint files were created
                                if os.path.exists(checkpoint_dir):
                                    files = os.listdir(checkpoint_dir)
                                    logger.info(f"Checkpoint directory contains files: {files}")
                                    if not files or (len(files) == 1 and '.write_test' in files):
                                        logger.warning(f"Checkpoint directory may be incomplete: {checkpoint_dir}")
                                else:
                                    logger.error(f"Checkpoint directory was not created: {checkpoint_dir}")
                                    raise FileNotFoundError(f"Checkpoint directory was not created: {checkpoint_dir}")
                                
                            # In distributed training, synchronize all processes
                            if self.world_size > 1:
                                torch.distributed.barrier()
                                logger.info("Checkpoint saved and synchronized across all processes")
                                
                        except Exception as e:
                            logger.error(f"Failed to save checkpoint at step {self.global_step}")
                            logger.error(f"Error type: {type(e).__name__}")
                            logger.error(f"Error message: {str(e)}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def verify_output_directory(self):
        """Verify that the output directory is properly configured and writable."""
        try:
            # Ensure output directory exists
            os.makedirs(self.args.output_dir, exist_ok=True)
            
            # Test write access by creating a temporary file
            test_file = os.path.join(self.args.output_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
        except Exception as e:
            # Try to create a default output directory
            default_output_dir = "./output"
            try:
                os.makedirs(default_output_dir, exist_ok=True)
                self.args.output_dir = default_output_dir
            except Exception as fallback_error:
                pass


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training")
    
    # Data arguments
    parser.add_argument("--train_manifest", type=str, required=True,
                        help="Path to training manifest JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for model and checkpoints")
    
    # Model arguments
    parser.add_argument("--base_ckpt", type=str, default="bosonai/higgs-audio-v2-generation-3B-base",
                        help="Base model checkpoint path")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--wd", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--grad_accum", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--warmup", type=int, default=100,
                        help="Warmup steps")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Logging and validation arguments
    parser.add_argument("--log_steps", type=int, default=30,
                        help="Log every N steps")
    parser.add_argument("--val_steps", type=int, default=500,
                        help="Validate every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize trainer
    trainer = HiggsAudioTrainer(args)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()