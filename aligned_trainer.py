#!/usr/bin/env python3
"""
Optimized Higgs Audio Trainer with Zero-Shot Voice Cloning Support
Based on Hugging Face Trainer for better performance while maintaining custom functionality
"""

import os
import json
import logging
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer, 
    AutoProcessor,
    TrainingArguments, 
    Trainer,
    get_cosine_schedule_with_warmup
)
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Import exact components from boson_multimodal
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer

# Import our aligned components
from aligned_dataset import AlignedHiggsAudioDataset, create_aligned_collator
from aligned_lora import apply_aligned_lora, create_aligned_lora_config, HiggsAudioModelWrapper

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class AlignedHiggsAudioTrainer(Trainer):
    """Custom trainer for Higgs Audio with optimized loss computation"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config = self.model.config
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation using model's internal loss when available"""
        # Convert ExtendedHiggsAudioBatchInput to dict if needed
        if hasattr(inputs, 'to_dict'):
            inputs_dict = inputs.to_dict()
        else:
            inputs_dict = dict(inputs)
        
        # Handle labels parameter conversion to label_ids (match train-higgs-audio approach)
        if 'labels' in inputs_dict:
            inputs_dict['label_ids'] = inputs_dict.pop('labels')
        
        # Ensure all inputs are on the correct device
        for key, value in inputs_dict.items():
            if isinstance(value, torch.Tensor):
                inputs_dict[key] = value.to(model.device)
        
        # Forward pass - model will compute loss internally if labels are provided
        outputs = model(**inputs_dict)
        
        # Extract loss from model outputs
        if isinstance(outputs, dict) and "loss" in outputs:
            loss = outputs["loss"]
        elif hasattr(outputs, "loss"):
            loss = outputs.loss
        else:
            # Fallback if no loss computed by model
            raise ValueError("Model did not compute loss. Check input labels.")
        
        return (loss, outputs) if return_outputs else loss


class HiggsAudioTrainingPipeline:
    """Main training pipeline for Higgs Audio with zero-shot voice cloning support"""
    
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
            # Initialize process group with timeout settings to prevent hanging
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=1800))  # 30 minutes timeout
            torch.cuda.set_device(self.local_rank)
        else:
            self.local_rank = 0
            self.world_size = 1
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info(f"Distributed setup complete - Local rank: {self.local_rank}, World size: {self.world_size}")
    
    def load_model_and_tokenizers(self):
        """Load model and tokenizers exactly as inference does."""
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info("Loading model and tokenizers...")
        
        # Load configuration
        self.config = HiggsAudioConfig.from_pretrained(self.args.base_ckpt)
        
        # Force enable Whisper embeddings (from inference patterns)
        self.config.encode_whisper_embed = True
        
        # CRITICAL FIX: Enable cross-modal conditioning (audio attention)
        # This is essential for text to learn from audio context
        if not getattr(self.config, 'use_audio_out_self_attention', None):
            # Only log on global rank 0
            if dist.get_rank() == 0:
                logger.info("Enabling cross-modal conditioning (use_audio_out_self_attention=True)")
            self.config.use_audio_out_self_attention = True
        
        # Load model with exact inference initialization
        model = HiggsAudioModel.from_pretrained(
            self.args.base_ckpt,
            config=self.config,
            torch_dtype=torch.bfloat16
        )
        model = model.to(self.device)
        
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
        
        # Wrap with our optimized model wrapper FIRST
        model = HiggsAudioModelWrapper(model)
        
        # THEN apply LoRA to the wrapped model
        lora_config = create_aligned_lora_config(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout
        )
        self.model = apply_aligned_lora(model, lora_config)
        
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info("Model and tokenizers loaded successfully")
    
    def setup_dataset(self):
        """Setup dataset and dataloader."""
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info("Setting up dataset...")
        
        # Create full dataset
        full_dataset = AlignedHiggsAudioDataset(
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
        from torch.utils.data import Subset
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        
        # Create collator with EXACT parameters
        self.collator = create_aligned_collator(self.config, self.whisper_processor)
        
        # Setup distributed sampler for training
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=self.world_size, 
            rank=dist.get_rank() if self.world_size > 1 else 0,
            shuffle=True
        ) if self.world_size > 1 else None
        
        # Setup distributed sampler for validation
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=self.world_size, 
            rank=dist.get_rank() if self.world_size > 1 else 0,
            shuffle=False
        ) if self.world_size > 1 else None
        
        # Optimize DataLoader settings for performance
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collator,
            num_workers=4,  # Reduced from 8 to prevent FS saturation
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=2,  # Reduced from 4
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2  # Prefetch 2 batches per worker
        )
        
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info(f"Dataset setup complete - Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    def setup_training(self):
        """Setup training arguments for Hugging Face Trainer."""
        if dist.get_rank() == 0:
            logger.info("Setting up training...")
        
        # Create Hugging Face TrainingArguments
        self.training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.grad_accum,
            learning_rate=self.args.lr,
            warmup_steps=self.args.warmup,
            logging_steps=self.args.log_steps,
            save_steps=self.args.save_steps,
            eval_steps=self.args.val_steps if self.args.val_steps > 0 else None,
            eval_strategy="steps" if self.args.val_steps > 0 else "no",
            save_total_limit=3,
            load_best_model_at_end=False,  # Disable for DDP compatibility
            metric_for_best_model=None,
            fp16=False,  # Use bfloat16 instead
            bf16=True,   # Enable bfloat16 mixed precision
            dataloader_pin_memory=False,  # Disable for better performance
            remove_unused_columns=False,  # Keep all columns for custom processing
            report_to=[],  # Disable reporting for simplicity
            logging_dir=f"{self.args.output_dir}/logs",
            ddp_find_unused_parameters=False,  # Optimize DDP
            dataloader_num_workers=4,
            dataloader_prefetch_factor=2,
            # DDP settings
            local_rank=self.local_rank,
        )
        
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info("Training setup complete")
    
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
            
            if self.local_rank == 0:
                logger.info(f"Output directory verified: {self.args.output_dir}")
        except Exception as e:
            # Try to create a default output directory
            default_output_dir = "./output"
            try:
                os.makedirs(default_output_dir, exist_ok=True)
                self.args.output_dir = default_output_dir
                if self.local_rank == 0:
                    logger.warning(f"Using default output directory: {default_output_dir}")
            except Exception as fallback_error:
                raise
    
    def train(self):
        """Start training using Hugging Face Trainer."""
        # Create the Hugging Face Trainer
        trainer = AlignedHiggsAudioTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataloader.dataset,
            eval_dataset=self.val_dataloader.dataset if self.args.val_steps > 0 else None,
            tokenizer=self.tokenizer,
            data_collator=self.collator,
        )
        
        # Only log on global rank 0
        if dist.get_rank() == 0:
            logger.info("=" * 60)
            logger.info("  TRAINING STARTED")
            logger.info("=" * 60)
            logger.info(f"Output directory: {self.args.output_dir}")
            logger.info(f"Save steps: {self.args.save_steps}")
            logger.info(f"Log steps: {self.args.log_steps}")
            logger.info(f"Validation steps: {self.args.val_steps}")
            logger.info(f"World size: {self.world_size}")
            logger.info(f"Starting training at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Start training
        trainer.train()
        
        # Save model on global rank 0 only
        if dist.get_rank() == 0:
            trainer.save_model()
            logger.info(f"Model saved to {self.args.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Higgs-Audio LoRA Training with Zero-Shot Voice Cloning")
    
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
    parser.add_argument("--wd", type=float, default=0.01,
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
                        help="Validate every N steps (0 to disable)")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize training pipeline
    pipeline = HiggsAudioTrainingPipeline(args)
    
    # Start training
    pipeline.train()


if __name__ == "__main__":
    main()