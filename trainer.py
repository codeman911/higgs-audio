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
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoProcessor, get_cosine_schedule_with_warmup
import logging

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
        self.model = HiggsAudioModel.from_pretrained(
            self.args.base_ckpt,
            config=self.config,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
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
        
        # Create dataset
        dataset = HiggsAudioDataset(
            manifest_path=self.args.train_manifest,
            tokenizer=self.tokenizer,
            audio_tokenizer=self.audio_tokenizer
        )
        
        # Create collator with EXACT parameters
        self.collator = create_collator(self.config, self.whisper_processor)
        
        # Setup distributed sampler
        sampler = DistributedSampler(
            dataset, 
            num_replicas=self.world_size, 
            rank=self.local_rank,
            shuffle=True
        ) if self.world_size > 1 else None
        
        # Create dataloader with optimal settings for 8xH200
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self.collator,
            num_workers=16,  # 128 cores / 8 GPUs = 16 per GPU
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
        total_steps = len(self.dataloader) * self.args.epochs // self.args.grad_accum
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
        
        logger.info(f"Batch keys: {list(batch.keys())}")
        
        # CRITICAL SOLUTION: Instead of calling model.forward(), we'll call the underlying
        # model's forward method directly, completely bypassing PEFT's wrapper methods
        
        # Get the true underlying HiggsAudioModel
        base_model = self._get_base_higgs_model()
        
        # Prepare clean inputs - remove ALL label keys to prevent any injection
        clean_inputs = {}
        for k, v in batch.items():
            # Only include non-label keys for model input
            if k not in ['label_ids', 'label_audio_ids', 'labels', 'audio_out_ids']:
                clean_inputs[k] = v
        
        logger.info(f"Clean inputs: {list(clean_inputs.keys())}")
        
        # Extract labels for manual loss computation
        text_labels = batch.get('label_ids')
        audio_labels = batch.get('label_audio_ids')
        
        # BYPASS PEFT: Call the forward method directly on the base model
        # This completely avoids PEFT's parameter injection
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                # Direct call to HiggsAudioModel.forward, bypassing all wrappers
                outputs = base_model.forward(**clean_inputs)
        except Exception as e:
            logger.error(f"Direct forward call failed: {e}")
            logger.error(f"Base model type: {type(base_model)}")
            logger.error(f"Available methods: {[m for m in dir(base_model) if 'forward' in m.lower()]}")
            raise
        
        return self._compute_dual_loss(outputs, text_labels, audio_labels)
    
    def _get_base_higgs_model(self):
        """Extract the actual HiggsAudioModel from all wrapper layers."""
        model = self.model
        path = []
        
        # Iteratively unwrap until we find HiggsAudioModel
        max_depth = 20
        for depth in range(max_depth):
            model_type = type(model).__name__
            path.append(model_type)
            
            # Found the target model
            if model_type == 'HiggsAudioModel':
                logger.info(f"Found HiggsAudioModel at depth {depth}: {' -> '.join(path)}")
                return model
            
            # Try different unwrapping attributes
            if hasattr(model, 'module'):  # DDP wrapper
                model = model.module
                continue
            elif hasattr(model, 'base_model'):  # PEFT wrapper
                model = model.base_model
                continue
            elif hasattr(model, 'model'):  # Generic wrapper
                model = model.model
                continue
            else:
                # No more wrappers found, check if this is the right model
                break
        
        logger.warning(f"Could not find HiggsAudioModel, using: {type(model).__name__}")
        logger.warning(f"Unwrapping path: {' -> '.join(path)}")
        return model
    
    def _compute_dual_loss(self, outputs, text_labels, audio_labels):
        """Compute dual loss from model outputs and labels."""
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_dict = {}
        
        # Extract outputs
        text_logits = getattr(outputs, 'logits', None)
        audio_logits = getattr(outputs, 'audio_logits', None)
        
        if text_logits is not None:
            logger.info(f"Text logits shape: {text_logits.shape}")
        if audio_logits is not None:
            logger.info(f"Audio logits shape: {audio_logits.shape}")
        
        # Text loss
        if text_logits is not None and text_labels is not None:
            text_loss = self.text_loss_fn(
                text_logits.view(-1, text_logits.size(-1)),
                text_labels.view(-1)
            )
            total_loss = total_loss + text_loss
            loss_dict['text_loss'] = text_loss.item()
            logger.info(f"Text loss: {text_loss.item():.4f}")
        else:
            logger.warning("Skipping text loss computation")
        
        # Audio loss - handle multi-codebook structure
        if audio_logits is not None and audio_labels is not None:
            audio_loss = self._compute_audio_loss(audio_logits, audio_labels)
            total_loss = total_loss + audio_loss
            loss_dict['audio_loss'] = audio_loss.item()
            logger.info(f"Audio loss: {audio_loss.item():.4f}")
        else:
            logger.warning("Skipping audio loss computation")
        
        loss_dict['total_loss'] = total_loss.item()
        logger.info(f"Total loss: {total_loss.item():.4f}")
        
        return total_loss, loss_dict
    
    def _compute_audio_loss(self, audio_logits, audio_labels):
        """Compute audio loss across multiple codebooks."""
        if audio_logits is None or audio_labels is None:
            return torch.tensor(0.0, device=self.device)
            
        audio_loss = torch.tensor(0.0, device=self.device)
        
        # Handle different tensor shapes for multi-codebook audio
        if len(audio_logits.shape) == 3 and audio_logits.size(1) == 8:
            # Shape: [seq_len, 8_codebooks, vocab_size]
            num_codebooks = 8
            for cb in range(num_codebooks):
                cb_logits = audio_logits[:, cb, :]  # [seq_len, vocab_size]
                if audio_labels.dim() > 1 and audio_labels.size(0) == 8:
                    cb_labels = audio_labels[cb, :]  # [seq_len]
                else:
                    cb_labels = audio_labels  # Fallback
                
                cb_loss = self.audio_loss_fn(cb_logits, cb_labels)
                audio_loss += cb_loss
            
            audio_loss = audio_loss / num_codebooks
        else:
            # Fallback: treat as single output
            audio_loss = self.audio_loss_fn(
                audio_logits.view(-1, audio_logits.size(-1)),
                audio_labels.view(-1)
            )
        
        return audio_loss
    
    def train_step(self, batch):
        """Single training step with robust batch handling."""
        
        # Handle different batch types
        if hasattr(batch, '__dict__'):
            # Convert HiggsAudioBatchInput to dict
            batch_dict = {}
            for key, value in batch.__dict__.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(self.device, non_blocking=True)
                else:
                    batch_dict[key] = value
            batch = batch_dict
        elif isinstance(batch, dict):
            # Handle regular dict batch
            batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        else:
            # Unknown batch type - try to handle gracefully
            logger.warning(f"Unknown batch type: {type(batch)}")
            try:
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.__dict__.items() if not k.startswith('_')}
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
    
    def train(self):
        """Main training loop."""
        
        self.model.train()
        step = 0
        accum_step = 0
        
        for epoch in range(self.args.epochs):
            if self.world_size > 1 and hasattr(self.dataloader.sampler, 'set_epoch'):
                self.dataloader.sampler.set_epoch(epoch)
            
            for batch in self.dataloader:
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
                    
                    step += 1
                    
                    # Logging
                    if step % 10 == 0 and self.local_rank == 0:
                        logger.info(f"Step {step}: {loss_dict}")
                    
                    # Checkpointing
                    if step % 1000 == 0 and self.local_rank == 0:
                        checkpoint_dir = f"{self.args.output_dir}/checkpoint-{step}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        save_lora_adapters(self.model.module if self.world_size > 1 else self.model, 
                                         checkpoint_dir)
                        logger.info(f"Saved checkpoint at step {step}")
        
        # Final checkpoint
        if self.local_rank == 0:
            final_dir = f"{self.args.output_dir}/final"
            os.makedirs(final_dir, exist_ok=True)
            save_lora_adapters(self.model.module if self.world_size > 1 else self.model, 
                             final_dir)
            logger.info("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    
    # Paths
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--base_ckpt", type=str, required=True)
    
    # Training
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--grad_accum", type=int, default=8)
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    args = parser.parse_args()
    
    # Train
    trainer = HiggsAudioTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()