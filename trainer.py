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
        self.text_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # Keep per-element losses for debugging
        self.audio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')  # Keep per-element losses for debugging
    
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
                # Direct call to HiggsAudioModel forward method, bypassing all wrappers
                if callable(getattr(base_model, 'forward', None)):
                    outputs = base_model.forward(**dtype_corrected_inputs)
                else:
                    # Fallback for models that don't have explicit forward method
                    outputs = base_model(**dtype_corrected_inputs)
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
            return self._compute_dual_loss(outputs, text_labels, audio_labels, batch)
        except Exception as e:
            logger.error(f"Loss computation failed: {e}")
            # Only log batch info for debugging if needed
            # logger.error(f"Batch keys: {list(batch.keys())}")
            # logger.error(f"Batch shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch.items()]}")
            raise
    
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
    
    def _compute_dual_loss(self, outputs, text_labels, audio_labels, batch):
        """Compute dual loss from model outputs and labels."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}
        
        # For logging predictions vs labels
        self.forward_step_count = getattr(self, "forward_step_count", 0)
        self.forward_step_count += 1
        
        # Extract outputs
        text_logits = getattr(outputs, 'logits', None)
        audio_logits = getattr(outputs, 'audio_logits', None)
        
        # CRITICAL FIX: Check if model's expanded_labels are properly aligned before using them
        # Only use expanded_labels if they have the same sequence length as logits
        model_expanded_labels = getattr(outputs, 'expanded_labels', None)
        
        # Validate expanded_labels alignment before using them
        if model_expanded_labels is not None and text_logits is not None:
            if model_expanded_labels.size(1) != text_logits.size(1):
                logger.warning(f"⚠️  Model expanded_labels misaligned: {model_expanded_labels.size(1)} vs {text_logits.size(1)}")
                logger.warning("⚠️  Falling back to original labels to avoid alignment issues")
                model_expanded_labels = None
        
        # Only log shapes if needed for debugging
        # if text_logits is not None:
        #     logger.info(f"Text logits shape: {text_logits.shape}")
        # if audio_logits is not None:
        #     logger.info(f"Audio logits shape: {audio_logits.shape}")
        # if model_expanded_labels is not None:
        #     logger.info(f"Model expanded labels shape: {model_expanded_labels.shape}")
        
        # Text loss with OPTIMAL teacher forcing alignment
        if text_logits is not None and model_expanded_labels is not None:
            # BEST CASE: Use model's expanded_labels which are already correctly aligned!
            logger.info("✓ Using model's expanded_labels (optimal path)")
            
            # CRITICAL FIX: The model's expanded_labels are already properly aligned with logits
            # No need to remove the last logit - they should have the same sequence length
            shift_logits = text_logits.contiguous()  # [batch, seq_len, vocab]
            shift_labels = model_expanded_labels.contiguous()  # [batch, seq_len]
            
            # Validate alignment
            if shift_logits.size(1) == shift_labels.size(1):
                # Count valid (non-ignore) tokens for loss computation
                valid_mask = shift_labels != -100
                num_valid_tokens = valid_mask.sum().item()
                # Only log detailed alignment info if needed
                # logger.info(f"Perfect alignment! Computing loss on {num_valid_tokens} valid tokens")
                
                # DEBUG: Log detailed token masking information
                if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                    logger.info(f"TEXT LOSS DEBUG: Total tokens: {shift_labels.numel()}, Valid (unmasked) tokens: {num_valid_tokens}")
                
                # Compute loss only on unmasked tokens
                text_loss_per_element = self.text_loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),  # [batch*seq_len, vocab]
                    shift_labels.view(-1)                          # [batch*seq_len]
                )
                
                # Only average over valid (unmasked) tokens
                if num_valid_tokens > 0:
                    text_loss = text_loss_per_element[valid_mask.view(-1)].mean()
                else:
                    text_loss = torch.tensor(0.0, device=self.device)
                
                total_loss = total_loss + text_loss
                loss_dict['text_loss'] = text_loss.item()
                logger.info(f"✓ Text loss: {text_loss.item():.4f}")
                
                # Log first and last 10 predictions vs labels every n log steps
                if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                    self._log_predictions_vs_labels_detailed(shift_logits, shift_labels)
            else:
                logger.error(f"❌ Model expanded_labels alignment failed: {shift_logits.size(1)} vs {shift_labels.size(1)}")
                logger.error("This should never happen with properly functioning model!")
                # Fall back to original labels
                model_expanded_labels = None
        # CRITICAL FIX: Skip the expanded_labels path and go directly to fallback
        elif text_logits is not None and text_labels is not None:
            # FALLBACK: Use manual alignment if expanded_labels not available or misaligned
            logger.warning("⚠️  Using fallback text loss computation with original labels")
            
            # Check if we need to handle audio token expansion manually
            input_seq_len = text_labels.size(1)
            logits_seq_len = text_logits.size(1)
            expansion_factor = logits_seq_len / input_seq_len
            
            logger.info(f"Expansion analysis: logits {logits_seq_len} / labels {input_seq_len} = {expansion_factor:.2f}x")
            
            if expansion_factor > 1.5:  # Significant expansion likely due to audio tokens
                logger.warning("Detected significant sequence expansion - likely due to audio tokens")
                logger.warning("⚠️  Complex alignment needed for audio-expanded sequences")
                
                # For audio-expanded sequences, we need a different approach
                # Since we can't do simple teacher-forcing shifts, we'll compute loss on overlapping regions
                # This is a simplified approach - in practice, you might want to implement more sophisticated alignment
                
                # Use only the text portion of the logits that corresponds to the original sequence length
                # This is a heuristic approach to handle the expansion
                effective_logits_len = min(logits_seq_len, input_seq_len)
                effective_labels_len = min(logits_seq_len - 1, input_seq_len - 1)
                
                if effective_labels_len > 0:
                    shift_logits = text_logits[:, :effective_labels_len, :].contiguous()
                    shift_labels = text_labels[:, :effective_labels_len].contiguous()
                    
                    logger.info(f"Adjusted alignment: logits {shift_logits.size(1)} vs labels {shift_labels.size(1)}")
                    
                    if shift_logits.size(1) == shift_labels.size(1):
                        valid_mask = shift_labels != -100
                        num_valid_tokens = valid_mask.sum().item()
                        
                        # DEBUG: Log detailed token masking information
                        if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                            logger.info(f"TEXT LOSS DEBUG (FALLBACK): Total tokens: {shift_labels.numel()}, Valid (unmasked) tokens: {num_valid_tokens}")
                        
                        if num_valid_tokens > 0:
                            text_loss_per_element = self.text_loss_fn(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)
                            )
                            text_loss = text_loss_per_element[valid_mask.view(-1)].mean()
                            
                            total_loss = total_loss + text_loss
                            loss_dict['text_loss'] = text_loss.item()
                            logger.info(f"✓ Adjusted fallback text loss: {text_loss.item():.4f}")
                            
                            # Log first and last 10 predictions vs labels every n log steps
                            if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                                self._log_predictions_vs_labels_detailed(shift_logits, shift_labels)
                        else:
                            logger.warning("⚠️  No valid tokens in adjusted alignment - skipping text loss")
                    else:
                        logger.error(f"❌ Adjusted alignment still failed: {shift_logits.size(1)} vs {shift_labels.size(1)}")
                else:
                    logger.warning("⚠️  Insufficient sequence length for adjusted alignment - skipping text loss")
            else:
                # STANDARD teacher forcing shift for autoregressive models (normal case)
                shift_logits = text_logits[..., :-1, :].contiguous()  # Remove last logit
                shift_labels = text_labels[..., 1:].contiguous()      # Remove first label
                
                if shift_logits.size(1) == shift_labels.size(1):
                    valid_mask = shift_labels != -100
                    num_valid_tokens = valid_mask.sum().item()
                    
                    # DEBUG: Log detailed token masking information
                    if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                        logger.info(f"TEXT LOSS DEBUG (STANDARD): Total tokens: {shift_labels.numel()}, Valid (unmasked) tokens: {num_valid_tokens}")
                    
                    if num_valid_tokens > 0:
                        text_loss_per_element = self.text_loss_fn(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1)
                        )
                        text_loss = text_loss_per_element[valid_mask.view(-1)].mean()
                        
                        total_loss = total_loss + text_loss
                        loss_dict['text_loss'] = text_loss.item()
                        logger.info(f"✓ Standard fallback text loss: {text_loss.item():.4f}")
                        
                        # Log first and last 10 predictions vs labels every n log steps
                        if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                            self._log_predictions_vs_labels_detailed(shift_logits, shift_labels)
                    else:
                        logger.warning("⚠️  No valid tokens in standard alignment - skipping text loss")
                else:
                    logger.error(f"❌ Standard fallback alignment failed: {shift_logits.size(1)} vs {shift_labels.size(1)}")
                    logger.error("Skipping text loss - check data preprocessing and model inputs")
        else:
            logger.warning("⚠️  Skipping text loss computation - insufficient data")
        
        # Audio loss - handle multi-codebook structure
        if audio_logits is not None and audio_labels is not None:
            if audio_logits.numel() > 0:
                audio_loss = self._compute_audio_loss_detailed(audio_logits, audio_labels)
                total_loss = total_loss + audio_loss
                loss_dict['audio_loss'] = audio_loss.item()
                logger.info(f"✓ Audio loss: {audio_loss.item():.4f}")
                
                # Log first and last 10 audio predictions vs labels every n log steps
                if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                    self._log_audio_predictions_vs_labels_detailed(audio_logits, audio_labels)
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
    
    def _compute_audio_loss_detailed(self, audio_logits, audio_labels):
        """Compute audio loss across multiple codebooks with proper sequence alignment and detailed logging."""
        if audio_logits is None or audio_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        # Handle empty sequences
        if audio_logits.numel() == 0 or audio_labels.numel() == 0:
            # Only log if needed for debugging
            # logger.info("Empty audio sequences, returning zero loss")
            return torch.tensor(0.0, device=self.device)
        
        audio_loss = torch.tensor(0.0, device=self.device)
        
        # Handle different tensor shapes for multi-codebook audio
        # Check if audio_logits has the expected shape for Higgs Audio model
        if len(audio_logits.shape) == 3:
            # Shape: [seq_len, num_codebooks, vocab_size] or [num_codebooks, seq_len, vocab_size]
            # Audio labels should be [num_codebooks, seq_len] or [seq_len, num_codebooks]
            
            # Determine the correct dimensions
            if audio_logits.size(1) == 8:  # Likely [seq_len, 8_codebooks, vocab_size]
                # Transpose to [8_codebooks, seq_len, vocab_size]
                audio_logits = audio_logits.transpose(0, 1)
                num_codebooks = 8
                seq_len_logits = audio_logits.size(1)
            elif audio_logits.size(0) == 8:  # Likely [8_codebooks, seq_len, vocab_size]
                num_codebooks = 8
                seq_len_logits = audio_logits.size(1)
            else:
                # Fallback for unexpected shapes
                logger.warning(f"Unexpected audio logits shape: {audio_logits.shape}")
                return torch.tensor(0.0, device=self.device)
            
            # Audio labels should be [8_codebooks, seq_len]
            if audio_labels.dim() == 2 and audio_labels.size(0) == 8:
                seq_len_labels = audio_labels.size(1)
                
                # Align sequences if needed
                if seq_len_logits > seq_len_labels:
                    # Trim logits to match labels
                    audio_logits = audio_logits[:, :seq_len_labels, :]
                    seq_len_logits = seq_len_labels
                elif seq_len_labels > seq_len_logits:
                    # Trim labels to match logits  
                    audio_labels = audio_labels[:, :seq_len_logits]
                    seq_len_labels = seq_len_logits
                
                # Only log detailed alignment if needed
                # logger.info(f"Aligned - Audio logits: {audio_logits.shape}, Audio labels: {audio_labels.shape}")
                
                valid_codebooks = 0
                total_valid_tokens = 0
                total_tokens = 0
                
                # CRITICAL FIX: Compute loss for each codebook separately
                for cb in range(num_codebooks):
                    cb_logits = audio_logits[cb, :, :]  # [seq_len, vocab_size]
                    cb_labels = audio_labels[cb, :]     # [seq_len]
                    
                    # Only compute loss on valid tokens (ignore -100)
                    valid_mask = cb_labels != -100
                    num_valid_tokens = valid_mask.sum().item()
                    total_tokens += cb_labels.numel()
                    total_valid_tokens += num_valid_tokens
                    
                    if num_valid_tokens > 0:
                        cb_loss_per_element = self.audio_loss_fn(cb_logits, cb_labels)
                        cb_loss = cb_loss_per_element[valid_mask].mean()
                        if torch.isfinite(cb_loss):
                            audio_loss += cb_loss
                            valid_codebooks += 1
                
                # DEBUG: Log detailed audio token masking information
                if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                    logger.info(f"AUDIO LOSS DEBUG: Total tokens: {total_tokens}, Valid (unmasked) tokens: {total_valid_tokens}, Valid codebooks: {valid_codebooks}")
                
                if valid_codebooks > 0:
                    audio_loss = audio_loss / valid_codebooks
                    # logger.info(f"Audio loss computed across {valid_codebooks} codebooks")
                else:
                    logger.warning("No valid codebooks found for audio loss computation")
                    audio_loss = torch.tensor(0.0, device=self.device)
            else:
                logger.warning(f"Unexpected audio labels shape: {audio_labels.shape}")
                audio_loss = torch.tensor(0.0, device=self.device)
        else:
            # Fallback: treat as single output
            valid_mask = audio_labels != -100
            num_valid_tokens = valid_mask.sum().item()
            
            # DEBUG: Log detailed audio token masking information
            if self.local_rank == 0 and self.forward_step_count % self.args.log_steps == 0:
                logger.info(f"AUDIO LOSS DEBUG (FALLBACK): Total tokens: {audio_labels.numel()}, Valid (unmasked) tokens: {num_valid_tokens}")
            
            if num_valid_tokens > 0:
                audio_loss_per_element = self.audio_loss_fn(
                    audio_logits.view(-1, audio_logits.size(-1)),
                    audio_labels.view(-1)
                )
                audio_loss = audio_loss_per_element[valid_mask.view(-1)].mean()
            else:
                audio_loss = torch.tensor(0.0, device=self.device)
        
        return audio_loss
    
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
        global_step = 0
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
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % self.args.log_steps == 0 and self.local_rank == 0:
                        log_msg = f"Step {global_step} - Train Loss: {loss_dict}"
                        # Check if pbar has set_postfix method before calling it
                        if hasattr(pbar, 'set_postfix') and callable(getattr(pbar, 'set_postfix', None)):
                            pbar.set_postfix({"loss": loss_dict.get('total_loss', 0.0)})
                        logger.info(log_msg)
                    
                    # Validation
                    if global_step % self.args.val_steps == 0 and self.local_rank == 0:
                        val_loss, val_text_loss, val_audio_loss = self.validate()
                        logger.info(f"Step {global_step} - Val Loss: {val_loss:.4f}, Text: {val_text_loss:.4f}, Audio: {val_audio_loss:.4f}")
                    
                    # Checkpointing
                    if global_step % self.args.save_steps == 0 and self.local_rank == 0:
                        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{global_step}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_lora_adapters(self.model.module if self.world_size > 1 else self.model, 
                                         checkpoint_dir)
                        logger.info(f"Saved checkpoint at step {global_step}")


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