"""
Minimal DDP trainer with dual loss computation.
Strictly mirrors inference forward pass and loss patterns.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
import logging
from tqdm import tqdm
from typing import Any, Dict, Union

# Import exact components from boson_multimodal
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Import our components
from dataset import HiggsAudioDataset, create_collator
from lora import apply_lora, create_lora_config, save_lora_adapters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HiggsAudioTrainer:
    """Minimal trainer for DualFFN LoRA fine-tuning."""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.load_model_and_tokenizers()
        self.setup_dataset()
        self.setup_training()
    
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
        
        # Create dataloaders with optimal settings for 8xH200
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collator,
            num_workers=16,  # 128 cores / 8 GPUs = 16 per GPU
            pin_memory=True,
            persistent_workers=True
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True
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
            logger.error(f"Base model type: {type(base_model)}")
            logger.error(f"Available methods: {[m for m in dir(base_model) if 'forward' in m.lower()]}")
            
            # Additional dtype debugging
            logger.error("Input tensor dtypes:")
            for k, v in clean_inputs.items():
                if isinstance(v, torch.Tensor):
                    logger.error(f"  {k}: {v.dtype}")
            raise
        
        try:
            return self._compute_dual_loss(outputs, text_labels, audio_labels)
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            # Log batch info for debugging
            logger.error(f"Batch keys: {list(batch.keys())}")
            logger.error(f"Batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")
            raise
    
    def _validate_model_inputs(self, original_batch, clean_inputs):
        """Validate that all required model inputs are present and have correct dimensions."""
        required_inputs = ['label_ids', 'audio_out_ids', 'label_audio_ids']
        missing_inputs = []
        
        for input_name in required_inputs:
            if input_name not in clean_inputs or clean_inputs[input_name] is None:
                missing_inputs.append(input_name)
            elif hasattr(clean_inputs[input_name], 'numel') and clean_inputs[input_name].numel() == 0:
                logger.warning(f"⚠️  {input_name} is present but empty")
                missing_inputs.append(f"{input_name} (empty)")
        
        if missing_inputs:
            logger.error(f"❌ CRITICAL: Missing required inputs: {missing_inputs}")
            logger.error("This can cause audio loss to become zero!")
            # Log available inputs for debugging
            available_inputs = [k for k in clean_inputs.keys() if clean_inputs[k] is not None]
            logger.error(f"Available inputs: {available_inputs}")
            
            # Add detailed debugging information
            self._log_detailed_batch_info(original_batch, clean_inputs)
    
    def _log_detailed_batch_info(self, original_batch, clean_inputs):
        """Log detailed information about the batch for debugging zero loss conditions."""
        logger.error("=" * 80)
        logger.error("DETAILED BATCH DEBUGGING INFO")
        logger.error("=" * 80)
        
        # Log original batch structure
        logger.error("Original batch keys and types:")
        for key, value in original_batch.items():
            if isinstance(value, torch.Tensor):
                logger.error(f"  {key}: Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})")
            else:
                logger.error(f"  {key}: {type(value)}")
        
        # Log clean inputs structure
        logger.error("Clean inputs keys and types:")
        for key, value in clean_inputs.items():
            if isinstance(value, torch.Tensor):
                logger.error(f"  {key}: Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})")
            else:
                logger.error(f"  {key}: {type(value)}")
        
        # Check for specific problematic patterns
        if 'labels' in original_batch:
            logger.error("⚠️  Generic 'labels' found in original batch - this should be filtered out!")
            
        # Check for dtype consistency
        dtypes = set()
        for key, value in clean_inputs.items():
            if isinstance(value, torch.Tensor):
                dtypes.add((key, value.dtype))
        logger.error(f"Input dtypes: {dtypes}")
        
        logger.error("=" * 80)
    
    def _get_base_higgs_model(self):
        """Extract the actual HiggsAudioModel from all wrapper layers."""
        model = self.model
        path = []
        
        # Iteratively unwrap until we find HiggsAudioModel
        max_depth = 20
        for depth in range(max_depth):
            # Check if model is a tensor (error case)
            if isinstance(model, torch.Tensor):
                logger.error("Model became a tensor during unwrapping - this should not happen")
                break
                
            model_type = type(model).__name__ if hasattr(model, '__class__') else str(type(model))
            path.append(model_type)
            
            # Found the target model
            if model_type == 'HiggsAudioModel':
                # logger.info(f"Found HiggsAudioModel at depth {depth}: {' -> '.join(path)}")
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
        
        # logger.warning(f"Could not find HiggsAudioModel, using: {type(model).__name__}")
        # logger.warning(f"Unwrapping path: {' -> '.join(path)}")
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
        
        if text_logits is not None:
            logger.info(f"Text logits shape: {text_logits.shape}")
        if audio_logits is not None:
            logger.info(f"Audio logits shape: {audio_logits.shape}")
            if audio_logits.numel() == 0:
                logger.error("❌ Audio logits are empty! This means no audio output positions were detected.")
                logger.error("Check: 1) audio_out_ids in batch, 2) audio_out_mask generation, 3) audio output tokens in data")
            else:
                logger.info(f"✓ Audio logits non-empty: {audio_logits.shape}")
        else:
            logger.warning("⚠️  Audio logits are None")
        if model_expanded_labels is not None:
            logger.info(f"✓ Model expanded labels shape: {model_expanded_labels.shape}")
        else:
            logger.warning("⚠️  Model expanded_labels is None - falling back to manual alignment")
        
        # Text loss with OPTIMAL teacher forcing alignment
        if text_logits is not None and model_expanded_labels is not None:
            # BEST CASE: Use model's expanded_labels which are already correctly aligned!
            logger.info("✓ Using model's expanded_labels (optimal path)")
            
            # CRITICAL FIX: The model's expanded_labels are already properly aligned with logits
            # No need to remove the last logit - they should have the same sequence length
            shift_logits = text_logits.contiguous()  # [batch, seq_len, vocab]
            shift_labels = model_expanded_labels.contiguous()  # [batch, seq_len]
            
            logger.info(f"Final logits shape: {shift_logits.shape}")
            logger.info(f"Final labels shape: {shift_labels.shape}")
            
            # Validate alignment
            if shift_logits.size(1) == shift_labels.size(1):
                # Count valid (non-ignore) tokens for loss computation
                valid_mask = shift_labels != -100
                num_valid_tokens = valid_mask.sum().item()
                logger.info(f"✓ Perfect alignment! Computing loss on {num_valid_tokens} valid tokens")
                
                # ADD DETAILED TEXT PREDICTION LOGGING
                self._log_text_predictions(shift_logits, shift_labels)
                
                text_loss = self.text_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq_len, vocab]
                    shift_labels.view(-1)                          # [batch*seq_len]
                )
                total_loss = total_loss + text_loss
                loss_dict['text_loss'] = text_loss.item()
                logger.info(f"✓ Text loss: {text_loss.item():.4f}")
            else:
                logger.error(f"❌ Model expanded_labels alignment failed: {shift_logits.size(1)} vs {shift_labels.size(1)}")
                logger.error("This should never happen with properly functioning model!")
                
        elif text_logits is not None and text_labels is not None:
            # FALLBACK: Use manual alignment if expanded_labels not available
            logger.warning("⚠️  Using fallback text loss computation with original labels")
            
            # Check if we need to handle audio token expansion manually
            input_seq_len = text_labels.size(1)
            logits_seq_len = text_logits.size(1)
            expansion_factor = logits_seq_len / input_seq_len
            
            logger.info(f"Expansion analysis: logits {logits_seq_len} / labels {input_seq_len} = {expansion_factor:.2f}x")
            
            if expansion_factor > 1.5:  # Significant expansion likely due to audio tokens
                logger.warning("Detected significant sequence expansion - likely due to audio tokens")
                logger.warning("Manual alignment may be inaccurate. Check model input filtering.")
            
            # STANDARD teacher forcing shift for autoregressive models
            shift_logits = text_logits[..., :-1, :].contiguous()  # Remove last logit
            shift_labels = text_labels[..., 1:].contiguous()      # Remove first label
            
            logger.info(f"Fallback logits shape: {shift_logits.shape}")
            logger.info(f"Fallback labels shape: {shift_labels.shape}")
            
            if shift_logits.size(1) == shift_labels.size(1):
                valid_mask = shift_labels != -100
                num_valid_tokens = valid_mask.sum().item()
                logger.info(f"✓ Fallback alignment OK! Computing loss on {num_valid_tokens} valid tokens")
                
                # ADD DETAILED TEXT PREDICTION LOGGING
                self._log_text_predictions(shift_logits, shift_labels)
                
                text_loss = self.text_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss = total_loss + text_loss
                loss_dict['text_loss'] = text_loss.item()
                logger.info(f"✓ Fallback text loss: {text_loss.item():.4f}")
            else:
                logger.error(f"❌ Fallback alignment failed: {shift_logits.size(1)} vs {shift_labels.size(1)}")
                logger.error("Skipping text loss - check data preprocessing and model inputs")
        else:
            logger.warning("⚠️  Skipping text loss computation - insufficient data")
        
        # Audio loss - handle multi-codebook structure
        if audio_logits is not None and audio_labels is not None:
            if audio_logits.numel() > 0:
                logger.info("✓ Computing audio loss on non-empty logits")
                audio_loss = self._compute_audio_loss(audio_logits, audio_labels)
                total_loss = total_loss + audio_loss
                loss_dict['audio_loss'] = audio_loss.item()
                logger.info(f"✓ Audio loss: {audio_loss.item():.4f}")
            else:
                logger.warning("⚠️  Audio logits are empty - skipping audio loss")
                loss_dict['audio_loss'] = 0.0
        else:
            logger.warning("⚠️  Skipping audio loss computation - missing audio_logits or audio_labels")
            loss_dict['audio_loss'] = 0.0
        
        # Final loss summary
        loss_dict['total_loss'] = total_loss.item()
        
        if total_loss.item() > 0:
            logger.info(f"✓ TRAINING SUCCESSFUL - Total loss: {total_loss.item():.4f}")
            if 'text_loss' in loss_dict and loss_dict['text_loss'] > 0:
                logger.info(f"  ✓ Text contribution: {loss_dict['text_loss']:.4f}")
            if 'audio_loss' in loss_dict and loss_dict['audio_loss'] > 0:
                logger.info(f"  ✓ Audio contribution: {loss_dict['audio_loss']:.4f}")
        else:
            logger.error("❌ CRITICAL: Total loss is ZERO! Training will not work.")
            logger.error("Check: 1) Model inputs, 2) Label alignment, 3) Data preprocessing")
        
        return total_loss, loss_dict

    def _log_text_predictions(self, logits, labels):
        """Log detailed text predictions vs target labels with detokenization for Arabic text."""
        try:
            # Get predictions (argmax of logits)
            predictions = torch.argmax(logits, dim=-1)
            
            # Log for first sample in batch
            batch_size = logits.size(0)
            if batch_size > 0:
                logger.info("=== ARABIC TEXT PREDICTION ANALYSIS ===")
                
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
                    
                    # Limit to first 20 tokens for readability
                    max_tokens = min(20, len(valid_predictions))
                    valid_predictions = valid_predictions[:max_tokens]
                    valid_labels = valid_labels[:max_tokens]
                    
                    # Convert to lists for easier handling
                    pred_tokens = valid_predictions.tolist()
                    label_tokens = valid_labels.tolist()
                    
                    # Detokenize to Arabic text
                    try:
                        predicted_text = self.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                        target_text = self.tokenizer.decode(label_tokens, skip_special_tokens=True)
                        
                        logger.info(f"Predicted Arabic Text: {predicted_text}")
                        logger.info(f"Target Arabic Text:    {target_text}")
                        
                        # Show token-level comparison
                        logger.info("Token-level comparison (first 10 tokens):")
                        for i in range(min(10, len(pred_tokens))):
                            logger.info(f"  Position {i}: Predicted={pred_tokens[i]} vs Target={label_tokens[i]}")
                            
                    except Exception as decode_error:
                        logger.warning(f"Failed to decode tokens to text: {decode_error}")
                        logger.info(f"Predicted tokens: {pred_tokens}")
                        logger.info(f"Target tokens:    {label_tokens}")
                else:
                    logger.warning("No valid (non-masked) tokens found for text prediction analysis")
                    
        except Exception as e:
            logger.warning(f"Failed to log text predictions: {e}")

    def _compute_audio_loss(self, audio_logits, audio_labels):
        """Compute audio loss across multiple codebooks with proper sequence alignment."""
        if audio_logits is None or audio_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        # Log shapes for debugging
        logger.info(f"Audio logits shape: {audio_logits.shape}")
        logger.info(f"Audio labels shape: {audio_labels.shape}")
        
        # Handle empty sequences
        if audio_logits.numel() == 0 or audio_labels.numel() == 0:
            logger.info("Empty audio sequences, returning zero loss")
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
                
                logger.info(f"Aligned - Audio logits: {audio_logits.shape}, Audio labels: {audio_labels.shape}")
                
                valid_codebooks = 0
                for cb in range(num_codebooks):
                    cb_logits = audio_logits[:, cb, :]  # [seq_len, vocab_size]
                    cb_labels = audio_labels[cb, :]     # [seq_len]
                    
                    # Only compute loss on valid tokens (ignore -100)
                    valid_mask = cb_labels != -100
                    if valid_mask.sum() > 0:
                        cb_loss = self.audio_loss_fn(cb_logits, cb_labels)
                        if torch.isfinite(cb_loss):
                            audio_loss += cb_loss
                            valid_codebooks += 1
                
                if valid_codebooks > 0:
                    audio_loss = audio_loss / valid_codebooks
                    logger.info(f"Audio loss computed across {valid_codebooks} codebooks")
                    
                    # ADD DETAILED AUDIO PREDICTION LOGGING
                    self._log_audio_predictions(audio_logits, audio_labels)
            else:
                logger.warning(f"Unexpected audio labels shape: {audio_labels.shape}")
        else:
            # Fallback: treat as single output
            logger.info("Using fallback audio loss computation")
            audio_loss = self.audio_loss_fn(
                audio_logits.view(-1, audio_logits.size(-1)),
                audio_labels.view(-1)
            )
        
        return audio_loss

    def _log_audio_predictions(self, audio_logits, audio_labels):
        """Log detailed audio predictions vs target labels."""
        try:
            logger.info("=== AUDIO PREDICTION ANALYSIS ===")
            
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
                    
                    # Limit to first 10 tokens for readability
                    max_tokens = min(10, len(valid_predictions))
                    valid_predictions = valid_predictions[:max_tokens]
                    valid_labels = valid_labels[:max_tokens]
                    
                    pred_tokens = valid_predictions.tolist()
                    label_tokens = valid_labels.tolist()
                    
                    logger.info("First codebook audio token predictions:")
                    logger.info(f"  Predicted tokens: {pred_tokens}")
                    logger.info(f"  Target tokens:    {label_tokens}")
                    
                    # Show differences
                    differences = []
                    for i, (pred, target) in enumerate(zip(pred_tokens, label_tokens)):
                        if pred != target:
                            differences.append(f"Pos {i}: {pred} vs {target}")
                    
                    if differences:
                        logger.info(f"  Differences: {', '.join(differences)}")
                    else:
                        logger.info("  Perfect match for first 10 tokens!")
                        
                else:
                    logger.warning("No valid (non-masked) audio tokens found for prediction analysis")
                    
        except Exception as e:
            logger.warning(f"Failed to log audio predictions: {e}")

    def _log_predictions_vs_labels_detailed(self, logits, labels):
        """Log first and last 10 text predictions vs labels for a few samples."""
        try:
            # Get predictions (argmax of logits)
            predictions = torch.argmax(logits, dim=-1)
            
            # Log for first few samples in batch (up to 2 samples)
            batch_size = min(2, logits.size(0))
            
            for i in range(batch_size):
                logger.info(f"Sample {i+1} Text Prediction vs Label Comparison:")
                
                # Get sequence length for this sample
                seq_len = min(logits.size(1), 20)  # Limit to 20 tokens for better debugging
                
                # Log first 10 predictions and labels
                if seq_len >= 10:
                    pred_first_10 = predictions[i, :10].tolist()
                    label_first_10 = labels[i, :10].tolist()
                    logger.info(f"  First 10 Text Predictions: {pred_first_10}")
                    logger.info(f"  First 10 Text Labels:      {label_first_10}")
                
                # Log last 10 predictions and labels (if sequence is long enough)
                if seq_len > 10:
                    pred_last_10 = predictions[i, -10:].tolist()
                    label_last_10 = labels[i, -10:].tolist()
                    logger.info(f"  Last 10 Text Predictions:  {pred_last_10}")
                    logger.info(f"  Last 10 Text Labels:       {label_last_10}")
                elif seq_len > 0:
                    # If sequence is shorter than 20, just log what we have
                    pred_rest = predictions[i, 10:seq_len].tolist()
                    label_rest = labels[i, 10:seq_len].tolist()
                    logger.info(f"  Rest Text Predictions:     {pred_rest}")
                    logger.info(f"  Rest Text Labels:          {label_rest}")
                
                # Log some statistics about masking
                sample_labels = labels[i]
                masked_count = (sample_labels == -100).sum().item()
                unmasked_count = (sample_labels != -100).sum().item()
                total_count = sample_labels.numel()
                logger.info(f"  Label Stats: {masked_count} masked, {unmasked_count} unmasked, {total_count} total")
        except Exception as e:
            logger.warning(f"Failed to log text predictions vs labels: {e}")
    
    def _log_audio_predictions_vs_labels_detailed(self, audio_logits, audio_labels):
        """Log first and last 10 audio predictions vs labels for a few samples."""
        try:
            # Handle different tensor shapes for multi-codebook audio
            if len(audio_logits.shape) == 3 and audio_logits.size(1) == 8:
                # Shape: [seq_len, 8_codebooks, vocab_size]
                seq_len = min(audio_logits.size(0), 20)  # Limit to 20 tokens for better debugging
                num_codebooks = min(audio_logits.size(1), 3)  # Limit to 3 codebooks for readability
                
                # Get predictions (argmax of logits) for first codebook
                predictions = torch.argmax(audio_logits[:, 0, :], dim=-1)  # First codebook
                labels = audio_labels[0, :]  # First codebook labels
                
                logger.info("Audio Codebook 0 Prediction vs Label Comparison:")
                
                # Log first 10 predictions and labels
                if seq_len >= 10:
                    pred_first_10 = predictions[:10].tolist()
                    label_first_10 = labels[:10].tolist()
                    logger.info(f"  First 10 Audio Predictions: {pred_first_10}")
                    logger.info(f"  First 10 Audio Labels:      {label_first_10}")
                
                # Log last 10 predictions and labels (if sequence is long enough)
                if seq_len > 10:
                    pred_last_10 = predictions[-10:].tolist()
                    label_last_10 = labels[-10:].tolist()
                    logger.info(f"  Last 10 Audio Predictions:  {pred_last_10}")
                    logger.info(f"  Last 10 Audio Labels:       {label_last_10}")
                elif seq_len > 0:
                    # If sequence is shorter than 20, just log what we have
                    pred_rest = predictions[10:seq_len].tolist()
                    label_rest = labels[10:seq_len].tolist()
                    logger.info(f"  Rest Audio Predictions:     {pred_rest}")
                    logger.info(f"  Rest Audio Labels:          {label_rest}")
                
                # Also log some stats about the logits
                logger.info(f"  Audio Logits shape: {audio_logits.shape}")
                logger.info(f"  Audio Labels shape: {audio_labels.shape}")
                
                # Log statistics about audio masking
                masked_count = (audio_labels == -100).sum().item()
                unmasked_count = (audio_labels != -100).sum().item()
                total_count = audio_labels.numel()
                logger.info(f"  Audio Label Stats: {masked_count} masked, {unmasked_count} unmasked, {total_count} total")
        except Exception as e:
            logger.warning(f"Failed to log audio predictions vs labels: {e}")
    
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
            logger.warning(f"Unknown batch type: {type(batch)}")
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
                logger.error(f"Failed to convert batch: {e}")
                raise
        
        # Validate batch has required keys
        required_keys = ['input_ids']
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            logger.error(f"Missing required keys in batch: {missing_keys}")
            logger.error(f"Available keys: {list(batch.keys())}")
            raise ValueError(f"Batch missing required keys: {missing_keys}")
        
        # Compute loss
        try:
            loss, loss_dict = self.compute_loss(batch)
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            logger.error(f"Batch keys: {list(batch.keys())}")
            logger.error(f"Batch shapes: {[(k, v.shape if isinstance(v, torch.Tensor) else type(v)) for k, v in batch.items()]}")
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
                    # Only log validation errors if needed for debugging
                    # logger.error(f"Validation loss computation failed: {e}")
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
                    
                    # Logging
                    if self.global_step % self.args.log_steps == 0 and self.local_rank == 0:
                        log_msg = f"Step {self.global_step} - Train Loss: {loss_dict}"
                        # Check if pbar has set_postfix method before calling it
                        if hasattr(pbar, 'set_postfix') and callable(getattr(pbar, 'set_postfix', None)):
                            pbar.set_postfix({"loss": loss_dict.get('total_loss', 0.0)})
                        logger.info(log_msg)
                        
                        # ADD DETAILED PREDICTION LOGGING EVERY 10 LOG STEPS
                        if self.global_step % (self.args.log_steps * 10) == 0:
                            logger.info("=== DETAILED PREDICTION LOGGING ===")
                            logger.info(f"Current losses - Total: {loss_dict.get('total_loss', 0.0):.4f}, "
                                      f"Text: {loss_dict.get('text_loss', 0.0):.4f}, "
                                      f"Audio: {loss_dict.get('audio_loss', 0.0):.4f}")
                    
                    # Validation
                    if self.global_step % self.args.val_steps == 0 and self.local_rank == 0:
                        val_loss, val_text_loss, val_audio_loss = self.validate()
                        logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}, Text: {val_text_loss:.4f}, Audio: {val_audio_loss:.4f}")
                    
                    # Checkpointing
                    if self.global_step % self.args.save_steps == 0 and self.local_rank == 0:
                        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{self.global_step}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_lora_adapters(self.model.module if self.world_size > 1 else self.model, 
                                         checkpoint_dir)
                        logger.info(f"Saved checkpoint at step {self.global_step}")
    
    def _take_corrective_action(self, batch, alert):
        """Take corrective action when near-zero audio loss is detected."""
        logger.warning("⚠️  INITIATING CORRECTIVE ACTION FOR NEAR-ZERO AUDIO LOSS")
        
        # Record the recovery attempt
        # if hasattr(self, 'audio_loss_monitor'):
        #     self.audio_loss_monitor.record_recovery_attempt(getattr(self, 'global_step', 0))
        
        # Log detailed information about the current state
        logger.warning("CORRECTIVE ACTION DETAILS:")
        logger.warning(f"  Alert type: {alert['type']}")
        logger.warning(f"  Alert message: {alert['message']}")
        logger.warning(f"  Current step: {getattr(self, 'global_step', 0)}")
        
        # Check batch for issues
        logger.warning("BATCH VALIDATION:")
        if isinstance(batch, dict):
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.warning(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.warning(f"  {key}: {type(value)}")
        
        # Suggest corrective actions
        logger.warning("RECOMMENDED CORRECTIVE ACTIONS:")
        logger.warning("  1. Verify that audio_out_ids and label_audio_ids are present in batch")
        logger.warning("  2. Check that audio data files exist and are accessible")
        logger.warning("  3. Validate that audio tokenization is working correctly")
        logger.warning("  4. Ensure that audio masking is not over-masking tokens")
        logger.warning("  5. Check for dtype consistency between model and batch data")
        logger.warning("  6. Verify that PEFT wrappers are not filtering required inputs")
        
        # If this is a critical alert, consider stopping training
        if alert['type'] == 'critical':
            logger.error("❌ CRITICAL LOSS CONDITION DETECTED!")
            logger.error("Training may be ineffective with current audio loss.")
            logger.error("Consider stopping training and investigating the root cause.")
            
            # In a real implementation, you might want to:
            # 1. Save a checkpoint before stopping
            # 2. Send an alert to monitoring systems
            # 3. Potentially adjust training parameters
            # 4. Or stop training entirely
            
            # For now, we'll just log the critical condition


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
    parser.add_argument("--batch_size", type=int, default=1,
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
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout")
    
    # Logging and validation arguments
    parser.add_argument("--log_steps", type=int, default=10,
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
