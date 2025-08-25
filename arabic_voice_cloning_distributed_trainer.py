#!/usr/bin/env python3
"""
High-Performance Distributed Trainer for Arabic Voice Cloning

Optimized for 8xH200 GPU setup with advanced memory management and monitoring.
"""

import os
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup, AutoTokenizer
import wandb
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Import our custom modules
from arabic_voice_cloning_dataset import *
from arabic_voice_cloning_training_collator import *
from arabic_voice_cloning_lora_config import *
from arabic_voice_cloning_loss_function import *
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training."""
    
    # Paths
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    data_path: str = "path/to/chatml_data.json"
    output_dir: str = "./checkpoints"
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Optimization
    use_mixed_precision: bool = True
    dataloader_num_workers: int = 16
    gradient_checkpointing: bool = False  # Disabled for Higgs Audio compatibility
    
    # Distributed
    local_rank: int = -1
    world_size: int = 8
    backend: str = "nccl"
    
    # Monitoring
    save_steps: int = 500
    logging_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "higgs-audio-arabic-voice-cloning"
    
    def __post_init__(self):
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if self.world_size != int(os.environ.get('WORLD_SIZE', 1)):
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


class ArabicVoiceCloningDistributedTrainer:
    """High-performance distributed trainer for Arabic voice cloning."""
    
    def __init__(
        self,
        training_config: DistributedTrainingConfig,
        dataset_config: ArabicVoiceCloningDatasetConfig,
        lora_config: HiggsAudioLoRATrainingConfig,
        loss_config: LossConfig
    ):
        self.training_config = training_config
        self.dataset_config = dataset_config
        self.lora_config = lora_config
        self.loss_config = loss_config
        
        self._setup_distributed()
        self._setup_device()
        self._initialize_components()
        
        logger.info(f"Trainer initialized: {self.device}, World size: {self.training_config.world_size}")
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if self.training_config.world_size > 1:
            dist.init_process_group(
                backend=self.training_config.backend,
                init_method='env://',
                world_size=self.training_config.world_size,
                rank=self.training_config.local_rank
            )
        self.is_main_process = self.training_config.local_rank in [-1, 0]
    
    def _setup_device(self):
        """Setup device and CUDA optimization."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.training_config.local_rank}")
            torch.cuda.set_device(self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            logger.info(f"GPU {self.training_config.local_rank}: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
    
    def _initialize_components(self):
        """Initialize model, data, and training components."""
        # Load model with LoRA
        self.model, self.model_config, _ = create_higgs_audio_lora_model(
            model_path=self.training_config.model_path,
            custom_config=self.lora_config,
            device_map="cpu",  # Use CPU first, then move to device
            torch_dtype=torch.bfloat16,
            enable_gradient_checkpointing=self.training_config.gradient_checkpointing
        )
        self.model = self.model.to(self.device)
        
        # Load tokenizers
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.training_config.model_path)
        audio_device = "cpu" if self.device.type == "mps" else self.device
        self.audio_tokenizer = load_higgs_audio_tokenizer(
            self.training_config.audio_tokenizer_path,
            device=audio_device
        )
        
        # Setup DDP
        if self.training_config.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.training_config.local_rank],
                output_device=self.training_config.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True
            )
        
        # Setup data
        self._setup_data_pipeline()
        
        # Setup training components
        self._setup_training_components()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _setup_data_pipeline(self):
        """Setup data loading."""
        self.dataset = ArabicVoiceCloningDataset(
            config=self.dataset_config,
            audio_tokenizer=self.audio_tokenizer,
            text_tokenizer=self.text_tokenizer
        )
        
        if self.training_config.world_size > 1:
            self.sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.training_config.world_size,
                rank=self.training_config.local_rank,
                shuffle=True
            )
        else:
            self.sampler = None
        
        # Setup collator first
        try:
            from transformers import AutoProcessor
            whisper_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3")
        except:
            whisper_processor = None
        
        self.collator = ArabicVoiceCloningTrainingCollator(
            config=self.model_config,
            whisper_processor=whisper_processor,
            enable_teacher_forcing=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.training_config.batch_size,
            sampler=self.sampler,
            shuffle=(self.sampler is None),
            num_workers=0,  # Force single-process to avoid CUDA multiprocessing errors
            pin_memory=True,
            drop_last=True,
            persistent_workers=False,  # No workers = no persistence needed
            collate_fn=self.collator  # Use our custom collator
        )
        
        self.effective_batch_size = (
            self.training_config.batch_size * 
            self.training_config.gradient_accumulation_steps * 
            self.training_config.world_size
        )
        
        logger.info(f"Data pipeline: {len(self.dataset)} samples, effective batch size: {self.effective_batch_size}")
    
    def _setup_training_components(self):
        """Setup optimizer, scheduler, and loss."""
        # Loss function
        self.loss_fn = create_loss_function(
            config=self.model_config,
            vocab_size=len(self.text_tokenizer),
            loss_config=self.loss_config
        ).to(self.device)
        
        # Optimizer
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        from torch.optim import AdamW
        self.optimizer = AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.dataloader) * self.training_config.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.training_config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision
        if self.training_config.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        
        param_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Training setup: {param_count:,} trainable parameters, {total_steps} total steps")
    
    def _setup_monitoring(self):
        """Setup monitoring."""
        if self.is_main_process and self.training_config.use_wandb:
            try:
                wandb.init(
                    project=self.training_config.wandb_project,
                    config={
                        "training_config": self.training_config.__dict__,
                        "effective_batch_size": self.effective_batch_size,
                        "total_parameters": sum(p.numel() for p in self.model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    }
                )
                logger.info("Weights & Biases initialized")
            except:
                logger.warning("Failed to initialize wandb")
                self.training_config.use_wandb = False
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        self.model.train()
        
        total_steps = len(self.dataloader) * self.training_config.num_epochs
        progress_bar = tqdm(total=total_steps, desc="Training", disable=not self.is_main_process)
        
        for epoch in range(self.training_config.num_epochs):
            self.current_epoch = epoch
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
            
            for batch_idx, batch in enumerate(self.dataloader):
                loss_dict = self._training_step(batch)
                
                if self.current_step % self.training_config.logging_steps == 0:
                    self._log_metrics(loss_dict)
                
                if self.current_step % self.training_config.save_steps == 0:
                    self._save_checkpoint()
                
                self.current_step += 1
                progress_bar.update(1)
        
        progress_bar.close()
        self._save_checkpoint(is_final=True)
        logger.info("Training completed!")
    
    def _training_step(self, batch) -> Optional[Dict[str, Any]]:
        """Execute training step."""
        try:
            # The batch is already collated by the DataLoader's collate_fn
            training_batch = self._move_batch_to_device(batch)
            
            with autocast(enabled=self.training_config.use_mixed_precision):
                outputs = self.model(
                    input_ids=training_batch.input_ids,
                    attention_mask=training_batch.attention_mask,
                    audio_features=training_batch.audio_features,
                    audio_feature_attention_mask=training_batch.audio_feature_attention_mask,
                    audio_out_ids=training_batch.audio_out_ids,
                    audio_out_ids_start=training_batch.audio_out_ids_start,
                    audio_in_ids=training_batch.audio_in_ids,
                    audio_in_ids_start=training_batch.audio_in_ids_start,
                )
                
                loss_dict = self.loss_fn(
                    text_logits=outputs.logits,
                    audio_logits=getattr(outputs, 'audio_logits', None),
                    batch=training_batch,
                    audio_features=training_batch.audio_features,
                    step=self.current_step
                )
                
                total_loss = loss_dict['losses']['total_loss']
                scaled_loss = total_loss / self.training_config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            
            # Optimization step
            if (self.current_step + 1) % self.training_config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
            
            return loss_dict
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            return None
    
    def _move_batch_to_device(self, batch):
        """Move batch to device."""
        for field_name in batch.__dataclass_fields__:
            field_value = getattr(batch, field_name)
            if isinstance(field_value, torch.Tensor):
                setattr(batch, field_name, field_value.to(self.device))
        return batch
    
    def _log_metrics(self, loss_dict: Optional[Dict[str, Any]]):
        """Log metrics."""
        if not loss_dict or not self.is_main_process:
            return
        
        metrics = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
        }
        
        for name, loss in loss_dict['losses'].items():
            if isinstance(loss, torch.Tensor):
                metrics[f"loss/{name}"] = loss.item()
        
        for name, metric in loss_dict['metrics'].items():
            if isinstance(metric, torch.Tensor):
                metrics[f"metrics/{name}"] = metric.item()
        
        if torch.cuda.is_available():
            metrics["gpu_memory_gb"] = torch.cuda.memory_allocated(self.device) / 1e9
        
        if self.current_step % (self.training_config.logging_steps * 5) == 0:
            logger.info(f"Step {self.current_step}: Total Loss {metrics.get('loss/total_loss', 0):.6f}, "
                       f"LR {metrics['learning_rate']:.6e}, GPU {metrics.get('gpu_memory_gb', 0):.1f}GB")
        
        if self.training_config.use_wandb:
            wandb.log(metrics, step=self.current_step)
    
    def _save_checkpoint(self, is_final: bool = False):
        """Save checkpoint."""
        if not self.is_main_process:
            return
        
        checkpoint_dir = Path(self.training_config.output_dir) / f"checkpoint-{self.current_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        
        model_to_save.save_pretrained(checkpoint_dir)
        self.text_tokenizer.save_pretrained(checkpoint_dir)
        
        state = {
            "step": self.current_step,
            "epoch": self.current_epoch,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        
        logger.info(f"Checkpoint saved: {checkpoint_dir}")
    
    def cleanup(self):
        """Cleanup resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.training_config.world_size > 1:
            dist.destroy_process_group()
        if self.training_config.use_wandb and self.is_main_process:
            wandb.finish()


def create_distributed_trainer(
    data_path: str,
    output_dir: str,
    **kwargs
) -> ArabicVoiceCloningDistributedTrainer:
    """Factory function to create trainer."""
    training_config = DistributedTrainingConfig(
        data_path=data_path,
        output_dir=output_dir,
        **kwargs
    )
    
    dataset_config = ArabicVoiceCloningDatasetConfig(
        chatml_file=data_path,
        validate_on_init=True
    )
    
    lora_config = HiggsAudioLoRATrainingConfig(
        r=16,
        lora_alpha=32,
        target_modules_mode="comprehensive"
    )
    
    loss_config = LossConfig(
        text_loss_weight=1.0,
        audio_loss_weight=1.0,
        contrastive_loss_weight=0.1
    )
    
    return ArabicVoiceCloningDistributedTrainer(
        training_config=training_config,
        dataset_config=dataset_config,
        lora_config=lora_config,
        loss_config=loss_config
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to ChatML data with direct audio paths")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    
    args = parser.parse_args()
    
    trainer = create_distributed_trainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs
    )
    
    try:
        trainer.train()
        trainer.cleanup()
        logger.info("✅ Training completed successfully")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise