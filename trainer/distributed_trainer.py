"""
Distributed Higgs-Audio Trainer for 8xH200 Setup

This module implements distributed training optimized for 8x NVIDIA H200 GPUs
with 128-core CPU, featuring:
- DistributedDataParallel (DDP) training
- Optimized data loading and memory management
- Performance monitoring across GPUs
- Gradient synchronization and checkpointing
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional

# 🏠 ENHANCED: Robust import system for root directory execution
current_file = Path(__file__).resolve()
trainer_dir = current_file.parent
higgs_audio_root = trainer_dir.parent

if str(higgs_audio_root) not in sys.path:
    sys.path.insert(0, str(higgs_audio_root))

# 🔧 Conditional imports for ML dependencies
try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from loguru import logger

# Import trainer components
try:
    from .trainer import HiggsAudioTrainer
    from .config import DistributedTrainingConfig
    from .loss import compute_training_loss, LossComponents
except ImportError:
    # Fallback imports for direct execution
    from trainer import HiggsAudioTrainer
    from config import DistributedTrainingConfig
    from loss import compute_training_loss, LossComponents


class DistributedHiggsAudioTrainer(HiggsAudioTrainer):
    """
    Enhanced trainer for 8xH200 distributed setup with DDP and optimized data loading.
    
    Hardware Optimization:
    - 8x NVIDIA H200 GPUs (192GB total VRAM)
    - 128-core CPU for data preprocessing  
    - NVLink/InfiniBand interconnect
    
    Features:
    - DistributedDataParallel (DDP) training
    - Optimized data loading and memory management
    - Performance monitoring across GPUs
    - Gradient synchronization and checkpointing
    """
    
    def __init__(self, config: DistributedTrainingConfig):
        """Initialize distributed trainer with 8xH200 optimizations."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for distributed training")
        
        # Initialize distributed backend first
        self._setup_distributed_environment()
        
        # Initialize base trainer
        super().__init__(config)
        
        # Setup distributed model and data loading
        self._setup_distributed_model()
        self._setup_distributed_data_loading()
        
        # Initialize performance monitoring
        self.performance_monitor = MultiGPUPerformanceMonitor(
            local_rank=self.local_rank,
            world_size=self.world_size
        )
        
        logger.info(f"✅ DistributedHiggsAudioTrainer initialized on {self.world_size}xH200 GPUs")
    
    def _setup_distributed_environment(self):
        """Initialize distributed training environment for 8xH200."""
        # Check if running in distributed environment
        if 'LOCAL_RANK' in os.environ:
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.world_size = int(os.environ['WORLD_SIZE']) 
            self.global_rank = int(os.environ['RANK'])
            
            # Set CUDA device
            torch.cuda.set_device(self.local_rank)
            
            # Initialize process group
            torch.distributed.init_process_group(
                backend='nccl',  # Optimized for H200
                init_method='env://',
                world_size=self.world_size,
                rank=self.global_rank
            )
            
            logger.info(f"🌐 Distributed training initialized: rank {self.global_rank}/{self.world_size}")
            logger.info(f"🖥️ Local GPU: {self.local_rank}, Device: {torch.cuda.get_device_name(self.local_rank)}")
        else:
            # Single GPU fallback
            self.local_rank = 0
            self.world_size = 1
            self.global_rank = 0
            logger.warning("⚠️ No distributed environment detected, using single GPU")
    
    def _setup_distributed_model(self):
        """Wrap model with DistributedDataParallel for 8xH200."""
        if self.world_size > 1:
            # Move model to local GPU
            self.model = self.model.to(f'cuda:{self.local_rank}')
            
            # Wrap with DDP for gradient synchronization across H200s
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,  # Optimize for performance
                gradient_as_bucket_view=True,  # Memory optimization for H200
            )
            
            logger.info(f"🔗 Model wrapped with DistributedDataParallel on GPU {self.local_rank}")
    
    def _setup_distributed_data_loading(self):
        """Setup distributed data sampling optimized for 128-core CPU."""
        if self.world_size > 1:
            # Create distributed sampler
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
                drop_last=True,  # Ensure consistent batch sizes across GPUs
            )
            
            # Update dataloader with optimized settings for 128-core CPU
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size_per_gpu,
                sampler=self.train_sampler,
                collate_fn=self.collator,
                num_workers=self.config.dataloader_num_workers,
                pin_memory=True,
                prefetch_factor=getattr(self.config, 'prefetch_factor', 2),
                persistent_workers=getattr(self.config, 'persistent_workers', True),
            )
            
            logger.info(f"📊 Distributed data loading setup: {len(self.train_dataloader)} steps per epoch")
            logger.info(f"🚀 CPU workers per GPU: {self.config.dataloader_num_workers} (Total: {self.config.dataloader_num_workers * self.world_size}/128 cores)")
    
    def train(self):
        """Enhanced training loop with distributed synchronization and performance monitoring."""
        # Validate configuration before training
        if not self.config_validated:
            logger.info("🔍 Validating distributed training configuration...")
            if hasattr(self.config, 'validate_for_distributed_training'):
                self.config.validate_for_distributed_training()
            else:
                self.config.validate_for_training()
            self.config_validated = True
        
        # Only log from rank 0 to avoid spam
        if self.global_rank == 0:
            logger.info(f"🎆 Starting distributed training on {self.world_size}xH200 GPUs")
            logger.info(f"📊 Effective batch size: {self.config.batch_size_per_gpu * self.world_size * self.config.gradient_accumulation_steps}")
        
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
            if hasattr(self.model, 'module'):  # DDP wrapped
                self.model.module.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_enable()
            
            if self.global_rank == 0:
                logger.info("🔧 Gradient checkpointing enabled")
        
        try:
            for epoch in range(self.config.num_epochs):
                if self.global_rank == 0:
                    logger.info(f"\n🚀 Epoch {epoch + 1}/{self.config.num_epochs}")
                
                # Set epoch for distributed sampler
                if hasattr(self, 'train_sampler'):
                    self.train_sampler.set_epoch(epoch)
                
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_idx, batch in enumerate(self.train_dataloader):
                    try:
                        # Training step with distributed synchronization
                        optimizer.zero_grad()
                        
                        # Forward pass with enhanced loss computation
                        loss, loss_components = compute_training_loss(
                            self.model,
                            batch,
                            f'cuda:{self.local_rank}',
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
                        
                        # Performance monitoring and logging (only rank 0)
                        if self.global_rank == 0 and global_step % self.config.logging_steps == 0:
                            self.performance_monitor.log_training_metrics(global_step, loss_components)
                            self._monitor_dualffn_balance(loss_components)
                        
                        # Distributed metrics logging
                        if global_step % (self.config.logging_steps * 5) == 0:
                            self.performance_monitor.log_distributed_metrics()
                        
                        # Validation (only on rank 0)
                        if self.global_rank == 0 and global_step % self.config.eval_steps == 0 and global_step > 0:
                            val_loss = self._validate_distributed()
                            if val_loss < best_loss:
                                best_loss = val_loss
                                self.save_checkpoint(f"best-checkpoint")
                                logger.info(f"💎 New best validation loss: {val_loss:.4f}")
                        
                        # Checkpointing (only on rank 0)
                        if self.global_rank == 0 and global_step % self.config.save_steps == 0 and global_step > 0:
                            self.save_checkpoint(f"checkpoint-{global_step}")
                        
                        global_step += 1
                        
                    except Exception as e:
                        if self.global_rank == 0:
                            logger.error(f"❌ Error in training step {global_step}: {e}")
                        continue
                
                # Epoch summary (only rank 0)
                if self.global_rank == 0:
                    avg_epoch_loss = epoch_loss / max(num_batches, 1)
                    logger.info(f"📊 Epoch {epoch + 1} complete: Avg Loss = {avg_epoch_loss:.4f}")
        
        except KeyboardInterrupt:
            if self.global_rank == 0:
                logger.info("⏸️ Training interrupted by user")
        except Exception as e:
            if self.global_rank == 0:
                logger.error(f"❌ Distributed training failed: {e}")
            raise
        finally:
            # Save final checkpoint (only rank 0)
            if self.global_rank == 0:
                self.save_checkpoint("final-checkpoint")
                logger.info("🏁 Distributed training completed")
            
            # Cleanup distributed environment
            if self.world_size > 1:
                torch.distributed.destroy_process_group()
    
    def _validate_distributed(self) -> float:
        """Validation with distributed synchronization."""
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            return float('inf')
        
        # Simple validation implementation
        self.model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            # Sample a few validation batches
            for i in range(min(10, len(self.val_dataset))):
                try:
                    sample = self.val_dataset[i]
                    batch = self.collator([sample])
                    
                    loss, _ = compute_training_loss(
                        self.model,
                        batch,
                        f'cuda:{self.local_rank}'
                    )
                    
                    val_loss += loss.item()
                    num_val_batches += 1
                except Exception:
                    continue
        
        self.model.train()
        return val_loss / max(num_val_batches, 1)
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save checkpoint with distributed training state."""
        checkpoint_path = os.path.join(self.config.output_dir, f"{checkpoint_name}.pt")
        
        # Extract model state (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
        
        checkpoint = {
            'model_state_dict': model_state,
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': getattr(self, 'global_step', 0),
            'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else vars(self.config),
            'world_size': self.world_size,
            'local_rank': self.local_rank,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")


class MultiGPUPerformanceMonitor:
    """Performance monitoring for 8xH200 distributed training."""
    
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size
        self.step_times = []
        self.memory_usage = []
        self.last_step_time = time.time()
    
    def log_training_metrics(self, step: int, loss_components: LossComponents):
        """Log training metrics with GPU-specific information."""
        if self.local_rank == 0 and TORCH_AVAILABLE:  # Only log from rank 0
            # GPU memory usage
            memory_allocated = torch.cuda.memory_allocated(self.local_rank) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(self.local_rank) / 1024**3   # GB
            
            # Training speed
            current_time = time.time()
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)
            
            if len(self.step_times) > 100:
                self.step_times = self.step_times[-100:]
            
            avg_step_time = sum(self.step_times) / len(self.step_times)
            samples_per_second = self.world_size / avg_step_time  # Assuming batch_size=1 per GPU
            
            self.last_step_time = current_time
            
            logger.info(f"📊 Step {step} Performance:")
            logger.info(f"   GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
            logger.info(f"   Training Speed: {samples_per_second:.1f} samples/sec across {self.world_size} GPUs")
            logger.info(f"   Loss Components: text={loss_components.text_loss:.4f}, audio={loss_components.audio_loss:.4f}")
    
    def log_distributed_metrics(self):
        """Log distributed training specific metrics."""
        if not TORCH_AVAILABLE or not torch.distributed.is_initialized():
            return
        
        # Gather metrics from all ranks
        memory_stats = torch.cuda.memory_stats(self.local_rank)
        peak_memory = memory_stats.get('allocated_bytes.all.peak', 0) / 1024**3
        
        # All-reduce to get average across GPUs
        peak_memory_tensor = torch.tensor(peak_memory, device=f'cuda:{self.local_rank}')
        torch.distributed.all_reduce(peak_memory_tensor)
        avg_peak_memory = peak_memory_tensor.item() / self.world_size
        
        if self.local_rank == 0:
            logger.info(f"🔧 Distributed Training Stats:")
            logger.info(f"   Average Peak Memory: {avg_peak_memory:.1f}GB per GPU")
            logger.info(f"   Total Memory Usage: {avg_peak_memory * self.world_size:.1f}GB")
            logger.info(f"   World Size: {self.world_size} GPUs")


# Utility function for easy distributed trainer creation
def create_8xh200_trainer(
    train_data_path: str,
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base",
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer",
    **kwargs
) -> DistributedHiggsAudioTrainer:
    """Create optimized distributed trainer for 8xH200 setup."""
    from .config import get_8x_h200_config
    
    # Use 8x H200 config as base, override with provided params
    config = get_8x_h200_config()
    config.train_data_path = train_data_path
    config.model_path = model_path
    config.audio_tokenizer_path = audio_tokenizer_path
    
    # Override with any additional parameters
    for key, value in kwargs.items():
        setattr(config, key, value)
    
    return DistributedHiggsAudioTrainer(config)