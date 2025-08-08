#!/usr/bin/env python3
"""
Distributed Training Pipeline for Higgs-Audio V2 LoRA Fine-tuning
Optimized for 8x H200 GPUs with DeepSpeed and Accelerate integration.
"""

import os
import sys
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import deepspeed
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
from dataclasses import dataclass, field
import yaml

# Robust import handling for both CLI and module usage
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from scripts.training.lora_integration import HiggsAudioLoRAConfig, create_lora_model
except ImportError:
    # Fallback for different project structures
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
    from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
    from scripts.training.lora_integration import HiggsAudioLoRAConfig, create_lora_model


@dataclass
class TrainingConfig:
    """Training configuration for distributed LoRA fine-tuning"""
    
    # Model and data paths
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    dataset_path: str = "/workspace/data/processed_chatml"
    output_dir: str = "/workspace/outputs/higgs-lora-arabic-english"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size_per_device: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Audio-specific settings
    max_audio_length: int = 1500  # Max audio tokens
    max_text_length: int = 512    # Max text tokens
    
    # Distributed training
    deepspeed_config: Optional[str] = None
    use_deepspeed: bool = True
    use_accelerate: bool = True
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Wandb logging
    use_wandb: bool = True
    wandb_project: str = "higgs-audio-lora-arabic-english"
    wandb_run_name: Optional[str] = None
    
    # Data processing
    train_split_ratio: float = 0.9
    val_split_ratio: float = 0.1
    num_workers: int = 8
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Optimization
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    
    # Language-specific settings
    arabic_weight: float = 1.0
    english_weight: float = 1.0
    mixed_weight: float = 1.5  # Higher weight for code-switching samples


class ArabicEnglishDataset(torch.utils.data.Dataset):
    """Dataset class for Arabic+English ChatML samples"""
    
    def __init__(
        self,
        chatml_samples: List[Dict],
        tokenizer,
        audio_tokenizer,
        max_text_length: int = 512,
        max_audio_length: int = 1500,
    ):
        self.samples = chatml_samples
        self.tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.max_text_length = max_text_length
        self.max_audio_length = max_audio_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Convert back to ChatML format
            from boson_multimodal.data_types import ChatMLSample, Message, AudioContent, TextContent
            
            messages = []
            for msg_data in sample['messages']:
                content = msg_data['content']
                if isinstance(content, str):
                    message_content = content
                else:
                    # Handle multimodal content
                    message_content = []
                    for c in content:
                        if c['type'] == 'text':
                            message_content.append(TextContent(text=c['text']))
                        elif c['type'] == 'audio':
                            message_content.append(AudioContent(
                                audio_url=c.get('audio_url', ''),
                                raw_audio=c.get('raw_audio', ''),
                                duration=c.get('duration'),
                                offset=c.get('offset')
                            ))
                
                messages.append(Message(
                    role=msg_data['role'],
                    content=message_content
                ))
            
            chatml_sample = ChatMLSample(
                messages=messages,
                start_index=sample.get('start_index', 0),
                speaker=sample.get('speaker'),
                misc=sample.get('misc', {})
            )
            
            # Prepare the sample for training
            dataset_sample = prepare_chatml_sample(
                chatml_sample,
                self.tokenizer,
                self.audio_tokenizer,
                max_text_length=self.max_text_length,
                max_audio_length=self.max_audio_length
            )
            
            return dataset_sample
            
        except Exception as e:
            logging.warning(f"Error processing sample {idx}: {e}")
            # Return a dummy sample to avoid breaking the batch
            return self._create_dummy_sample()
    
    def _create_dummy_sample(self):
        """Create a dummy sample for error cases"""
        return ChatMLDatasetSample(
            input_ids=torch.tensor([1, 2, 3]),  # Dummy tokens
            label_ids=torch.tensor([-100, -100, -100]),
            audio_ids_concat=torch.tensor([[]]),
            audio_ids_start=torch.tensor([]),
            audio_waveforms_concat=torch.tensor([]),
            audio_waveforms_start=torch.tensor([]),
            audio_sample_rate=torch.tensor([]),
            audio_speaker_indices=torch.tensor([])
        )


class HiggsAudioDistributedTrainer:
    """Distributed trainer for Higgs-Audio LoRA fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.accelerator = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_dataloader = None
        self.val_dataloader = None
        
        # Setup logging
        self._setup_logging()
        
        # Initialize distributed training
        self._setup_distributed()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _setup_distributed(self):
        """Setup distributed training with Accelerate"""
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        deepspeed_plugin = None
        if self.config.deepspeed_config:
            try:
                from accelerate import DeepSpeedPlugin
                if os.path.exists(self.config.deepspeed_config):
                    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config=self.config.deepspeed_config)
                else:
                    self.logger.warning(f"DeepSpeed config not found: {self.config.deepspeed_config}. Continuing without DeepSpeed.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize DeepSpeed plugin: {e}. Continuing without DeepSpeed.")
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision="bf16" if self.config.bf16 else ("fp16" if self.config.fp16 else "no"),
            log_with="wandb" if self.config.use_wandb else None,
            project_dir=self.config.output_dir,
            kwargs_handlers=[ddp_kwargs],
            deepspeed_plugin=deepspeed_plugin,
        )
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Setup wandb
        if self.config.use_wandb and self.accelerator.is_main_process:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=self.config.__dict__
            )
    
    def load_datasets(self):
        """Load and prepare datasets"""
        self.logger.info("Loading datasets...")
        
        # Load training data from unified pipeline output
        # Files are directly in the dataset_path directory
        train_path = os.path.join(self.config.dataset_path, "train_chatml_samples.json")
        val_path = os.path.join(self.config.dataset_path, "val_chatml_samples.json")
        
        # Alternative paths in case they're in a chatml subdirectory
        if not os.path.exists(train_path):
            train_path = os.path.join(self.config.dataset_path, "chatml", "train_chatml_samples.json")
        if not os.path.exists(val_path):
            val_path = os.path.join(self.config.dataset_path, "chatml", "val_chatml_samples.json")
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found at: {train_path}")
        if not os.path.exists(val_path):
            raise FileNotFoundError(f"Validation data not found at: {val_path}")
        
        self.logger.info(f"Loading training data from: {train_path}")
        self.logger.info(f"Loading validation data from: {val_path}")
        
        # Load ChatML samples
        with open(train_path, 'r', encoding='utf-8') as f:
            train_samples = json.load(f)
        
        with open(val_path, 'r', encoding='utf-8') as f:
            val_samples = json.load(f)
        
        self.logger.info(f"Loaded {len(train_samples)} training samples")
        self.logger.info(f"Loaded {len(val_samples)} validation samples")
        
        # Load processing statistics for monitoring
        stats_path = os.path.join(self.config.dataset_path, "processing_stats.json")
        if not os.path.exists(stats_path):
            stats_path = os.path.join(self.config.dataset_path, "chatml", "processing_stats.json")
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            
            manifest_meta = stats.get('manifest_metadata', {})
            self.logger.info(f"Dataset statistics:")
            self.logger.info(f"  • Total duration: {manifest_meta.get('total_duration_hours', 0):.2f} hours")
            self.logger.info(f"  • Total samples: {manifest_meta.get('total_samples', 0):,}")
            self.logger.info(f"  • Directories processed: {manifest_meta.get('directories_processed', 0)}")
        
        return train_samples, val_samples
    
    def create_dataloaders(self, train_samples, val_samples):
        """Create data loaders"""
        from transformers import AutoTokenizer
        from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
        
        # Load tokenizers
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        audio_tokenizer = load_higgs_audio_tokenizer(
            self.config.audio_tokenizer_path,
            device=self.accelerator.device
        )
        
        # Create datasets
        train_dataset = ArabicEnglishDataset(
            train_samples,
            tokenizer,
            audio_tokenizer,
            max_text_length=self.config.max_text_length,
            max_audio_length=self.config.max_audio_length
        )
        
        val_dataset = ArabicEnglishDataset(
            val_samples,
            tokenizer,
            audio_tokenizer,
            max_text_length=self.config.max_text_length,
            max_audio_length=self.config.max_audio_length
        )
        
        # Create data collator
        collator = HiggsAudioSampleCollator(
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            padding=True,
            max_length=self.config.max_text_length
        )
        
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset) if self.accelerator.num_processes > 1 else None
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.accelerator.num_processes > 1 else None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=collator,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size_per_device,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        return train_dataloader, val_dataloader, tokenizer, audio_tokenizer
    
    def setup_model_and_optimizer(self, tokenizer, audio_tokenizer):
        """Setup model and optimizer"""
        self.logger.info("Setting up model and optimizer...")
        
        # Create LoRA configuration
        lora_config = HiggsAudioLoRAConfig(
            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            freeze_base_model=True,
            freeze_audio_tower=True,
            freeze_audio_encoder_proj=False,
            enable_audio_lora=True,
            enable_multilingual_lora=True
        )
        
        # Create LoRA model
        trainer = create_lora_model(
            model_path=self.config.model_path,
            lora_config=lora_config,
            device=str(self.accelerator.device)
        )
        
        model = trainer.prepare_for_training()
        
        # Setup optimizer
        if self.config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        return model, optimizer, trainer
    
    def setup_scheduler(self, optimizer, num_training_steps):
        """Setup learning rate scheduler"""
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        
        if self.config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler_type}")
        
        return scheduler
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        train_samples, val_samples = self.load_datasets()
        
        # Create data loaders
        train_dataloader, val_dataloader, tokenizer, audio_tokenizer = self.create_dataloaders(
            train_samples, val_samples
        )
        
        # Setup model and optimizer
        model, optimizer, lora_trainer = self.setup_model_and_optimizer(tokenizer, audio_tokenizer)
        
        # Calculate training steps
        steps_per_epoch = max(1, (len(train_dataloader) + self.config.gradient_accumulation_steps - 1) // self.config.gradient_accumulation_steps)
        num_training_steps = steps_per_epoch * self.config.num_epochs
        
        # Setup scheduler
        scheduler = self.setup_scheduler(optimizer, num_training_steps)
        
        # Prepare for distributed training
        model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, scheduler
        )
        
        # Training loop
        global_step = 0
        best_val_loss = float('inf')
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            model.train()
            train_loss = 0.0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")):
                with self.accelerator.accumulate(model):
                    # Forward pass
                    outputs = model(**batch)
                    
                    # Compute loss
                    loss = lora_trainer.compute_loss(batch, outputs)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item()
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = train_loss / self.config.logging_steps
                    self.logger.info(f"Step {global_step}: Loss = {avg_loss:.4f}")
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({
                            "train_loss": avg_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "global_step": global_step
                        })
                    
                    train_loss = 0.0
                
                # Validation
                if global_step % self.config.eval_steps == 0:
                    val_loss = self.validate(model, val_dataloader, lora_trainer)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.save_checkpoint(model, global_step, "best")
                    
                    if self.config.use_wandb and self.accelerator.is_main_process:
                        wandb.log({"val_loss": val_loss, "global_step": global_step})
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(model, global_step)
        
        # Final save
        self.save_checkpoint(model, global_step, "final")
        self.logger.info("Training completed!")
    
    def validate(self, model, val_dataloader, lora_trainer):
        """Validation loop"""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                outputs = model(**batch)
                loss = lora_trainer.compute_loss(batch, outputs)
                val_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = val_loss / num_batches
        self.logger.info(f"Validation loss: {avg_val_loss:.4f}")
        
        model.train()
        return avg_val_loss
    
    def save_checkpoint(self, model, global_step, suffix=""):
        """Save model checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        save_dir = Path(self.config.output_dir) / f"checkpoint-{global_step}"
        if suffix:
            save_dir = Path(self.config.output_dir) / f"checkpoint-{suffix}"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA adapters
        unwrapped_model = self.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_dir)
        
        # Save training config
        with open(save_dir / "training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {save_dir}")


def main():
    """Main training function"""
    import argparse
    
    def str2bool(v):
        if isinstance(v, bool):
            return v
        return v.lower() in ("yes", "true", "t", "1")
    
    parser = argparse.ArgumentParser(description="Distributed LoRA training for Higgs-Audio")
    parser.add_argument("--config", type=str, help="Path to training config YAML file")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to processed dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    # Training hyperparameters
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Alias for batch_size_per_device")
    parser.add_argument("--batch_size_per_device", type=int, default=None, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--max_grad_norm", type=float, default=None, help="Max gradient norm")
    # Precision
    parser.add_argument("--bf16", type=str2bool, nargs='?', const=True, default=None, help="Use bfloat16")
    parser.add_argument("--fp16", type=str2bool, nargs='?', const=True, default=None, help="Use float16")
    # Logging / eval / save
    parser.add_argument("--logging_steps", type=int, default=None, help="Logging frequency in steps")
    parser.add_argument("--eval_steps", type=int, default=None, help="Evaluation frequency in steps")
    parser.add_argument("--save_steps", type=int, default=None, help="Checkpoint save frequency in steps")
    # Dataloader
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers")
    # Wandb
    parser.add_argument("--use_wandb", type=str2bool, nargs='?', const=True, default=None, help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    # DeepSpeed
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to DeepSpeed JSON config")
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig()
    
    # Override with command line arguments
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs is not None:
        config.num_epochs = args.num_epochs
    # Batch size alias support
    if args.batch_size_per_device is not None:
        config.batch_size_per_device = args.batch_size_per_device
    elif args.batch_size is not None:
        config.batch_size_per_device = args.batch_size
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        config.warmup_ratio = args.warmup_ratio
    if args.max_grad_norm is not None:
        config.max_grad_norm = args.max_grad_norm
    if args.bf16 is not None:
        config.bf16 = args.bf16
    if args.fp16 is not None:
        config.fp16 = args.fp16
    if args.logging_steps is not None:
        config.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.use_wandb is not None:
        config.use_wandb = args.use_wandb
    if args.wandb_project is not None:
        config.wandb_project = args.wandb_project
    if args.wandb_run_name is not None:
        config.wandb_run_name = args.wandb_run_name
    if args.deepspeed_config is not None:
        config.deepspeed_config = args.deepspeed_config
    
    # Start training
    trainer = HiggsAudioDistributedTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
