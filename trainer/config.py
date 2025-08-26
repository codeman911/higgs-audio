"""
Training Configuration for Higgs-Audio LoRA Pipeline

Follows the design document patterns and matches the existing model architecture.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

# 🔧 Conditional imports for ML dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingConfig:
    """
    Configuration class for Higgs-Audio LoRA training.
    
    All settings are optimized for zero-shot voice cloning with DualFFN architecture.
    """
    
    # Model Configuration (Exact Match with generation.py)
    model_path: str = "bosonai/higgs-audio-v2-generation-3B-base"
    audio_tokenizer_path: str = "bosonai/higgs-audio-v2-tokenizer"
    device: str = "auto"
    device_id: Optional[int] = None
    
    # Data Configuration
    train_data_path: str = "data/train_samples.json"
    val_data_path: str = "data/val_samples.json"
    max_audio_duration: float = 30.0
    min_audio_duration: float = 0.5
    target_sample_rate: int = 24000
    
    # Training Configuration
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # LoRA Configuration (Focus on DualFFN heads)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "lm_head",        # Text generation head
        "audio_head",     # Audio generation head
    ])
    lora_bias: str = "none"
    
    # Loss Configuration (DualFFN Text + Audio)
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.1
    
    # Generation Configuration (for validation)
    max_new_tokens: int = 512
    temperature: float = 0.3
    top_p: float = 0.9
    
    # Logging and Checkpointing
    output_dir: str = "checkpoints"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Hardware Optimization
    use_gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Setup directories and basic configuration. Validation is done separately."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup device
        if self.device == "auto":
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    self.device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            else:
                # If torch is not available, keep as "auto" and handle later
                pass
        
        # Adjust batch size for MPS (limited memory)
        if self.device == "mps" and self.batch_size > 1:
            print(f"⚠️ Reducing batch size from {self.batch_size} to 1 for MPS compatibility")
            self.batch_size = 1
    
    def validate_for_training(self):
        """Validate configuration specifically for training. Call this before starting training."""
        # Validate training data path
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        
        # Validate other requirements
        if self.batch_size <= 0:
            raise ValueError(f"Invalid batch size: {self.batch_size}")
        
        if self.learning_rate <= 0:
            raise ValueError(f"Invalid learning rate: {self.learning_rate}")
        
        if self.num_epochs <= 0:
            raise ValueError(f"Invalid number of epochs: {self.num_epochs}")
        
        print(f"✅ Configuration validation passed for training")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'model_path': self.model_path,
            'audio_tokenizer_path': self.audio_tokenizer_path,
            'device': self.device,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'text_loss_weight': self.text_loss_weight,
            'audio_loss_weight': self.audio_loss_weight,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class DistributedTrainingConfig(TrainingConfig):
    """
    Enhanced configuration for 8xH200 GPU distributed training.
    
    Hardware Specifications:
    - 8x NVIDIA H200 GPUs (24GB VRAM each = 192GB total)
    - 128-core CPU for data preprocessing
    - High-bandwidth memory and NVLink/InfiniBand interconnect
    
    Optimizations:
    - Distributed data parallel training with gradient synchronization
    - Memory-efficient batch processing and gradient accumulation
    - CPU-optimized data loading with 128 cores
    - Mixed precision training for H200 architecture
    """
    
    # 🖥️ Hardware-specific settings
    world_size: int = 8  # 8xH200 GPUs
    local_rank: int = -1  # Set by torchrun automatically
    batch_size_per_gpu: int = 4  # 4 samples per GPU
    effective_batch_size: int = 128  # 4 * 8 * 4 (grad_accum) = 128
    
    # 🚀 CPU optimization for 128 cores
    dataloader_num_workers: int = 16  # 128 cores / 8 GPUs = 16 workers per GPU
    cpu_data_preprocessing: bool = True  # Use CPU for data preprocessing
    prefetch_factor: int = 4  # Prefetch 4 batches per worker
    persistent_workers: bool = True  # Keep workers alive between epochs
    
    # 💾 Memory optimization for H200 (24GB each)
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True  # bfloat16 for H200
    max_audio_length_seconds: int = 30  # Prevent OOM on long audio
    cpu_offload: bool = False  # H200 has sufficient VRAM
    
    # 🎯 LoRA settings optimized for distributed training
    lora_r: int = 64  # Higher rank for better quality with more GPUs
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    
    # 📊 Training stability for distributed setup
    gradient_clipping: float = 1.0
    warmup_steps: int = 1000  # Longer warmup for distributed training
    
    # 💾 Checkpoint and logging for multi-GPU
    save_steps: int = 500
    eval_steps: int = 1000
    logging_steps: int = 50
    checkpoint_total_limit: int = 3
    
    # 🌐 Distributed training specific
    find_unused_parameters: bool = False  # Optimize for performance
    gradient_as_bucket_view: bool = True  # Memory optimization
    ddp_timeout_seconds: int = 1800  # 30 minutes timeout for large models
    
    # 🎵 Audio generation settings for validation
    validation_max_new_tokens: int = 512
    validation_temperature: float = 0.3
    validation_batch_size: int = 1  # Smaller for validation
    
    def __post_init__(self):
        """Enhanced post-init for distributed training setup."""
        super().__post_init__()
        
        # Auto-calculate effective batch size
        self.effective_batch_size = (
            self.batch_size_per_gpu * 
            self.world_size * 
            self.gradient_accumulation_steps
        )
        
        # Adjust settings based on distributed environment
        if TORCH_AVAILABLE:
            try:
                if torch.distributed.is_initialized():
                    self.world_size = torch.distributed.get_world_size()
                    self.local_rank = torch.distributed.get_rank()
                    self.effective_batch_size = (
                        self.batch_size_per_gpu * 
                        self.world_size * 
                        self.gradient_accumulation_steps
                    )
                    print(f"🌐 Distributed training detected: rank {self.local_rank}/{self.world_size}")
                    print(f"📊 Effective batch size: {self.effective_batch_size}")
            except Exception:
                pass  # Distributed not initialized, keep defaults
        
        # Hardware validation
        if self.world_size == 8:
            expected_vram = 8 * 24  # 8 GPUs * 24GB each
            print(f"🖥️ 8xH200 Configuration:")
            print(f"   Total VRAM: {expected_vram}GB (8x24GB)")
            print(f"   CPU cores utilized: {self.dataloader_num_workers * self.world_size}/{128}")
            print(f"   Batch size per GPU: {self.batch_size_per_gpu}")
            print(f"   Effective batch size: {self.effective_batch_size}")
    
    def validate_for_distributed_training(self):
        """Validate configuration specifically for distributed training."""
        # Call parent validation first
        self.validate_for_training()
        
        # Distributed-specific validations
        if self.world_size <= 1:
            print("⚠️ World size is 1, this is not distributed training")
        
        if self.batch_size_per_gpu <= 0:
            raise ValueError(f"Invalid batch size per GPU: {self.batch_size_per_gpu}")
        
        if self.dataloader_num_workers > 32:
            print(f"⚠️ Very high number of workers ({self.dataloader_num_workers}), may cause overhead")
        
        # Memory estimation for H200
        estimated_memory_per_gpu = (
            self.batch_size_per_gpu * 
            self.max_audio_length_seconds * 
            24000 * 4 / (1024**3)  # Rough estimation: 30s * 24kHz * 4 bytes
        )
        
        if estimated_memory_per_gpu > 20:  # Leave 4GB buffer on 24GB H200
            print(f"⚠️ Estimated memory usage per GPU: {estimated_memory_per_gpu:.1f}GB")
            print("   Consider reducing batch_size_per_gpu or max_audio_length_seconds")
        
        print(f"✅ Distributed training configuration validation passed")
        print(f"   Target hardware: 8xH200 GPUs (192GB total VRAM)")
        print(f"   Estimated memory per GPU: {estimated_memory_per_gpu:.1f}GB")


# Predefined configurations for different use cases
def get_default_config() -> TrainingConfig:
    """Get default configuration without validation."""
    return TrainingConfig()

def get_quick_test_config() -> TrainingConfig:
    """Get quick test configuration."""
    return TrainingConfig(
        batch_size=1,
        gradient_accumulation_steps=2,
        num_epochs=1,
        lora_r=8,
        lora_alpha=16,
        logging_steps=5,
        save_steps=50,
    )

def get_8xh200_config() -> DistributedTrainingConfig:
    """Get optimized configuration for 8xH200 GPU training."""
    return DistributedTrainingConfig(
        # Hardware optimization
        world_size=8,
        batch_size_per_gpu=4,
        gradient_accumulation_steps=4,
        dataloader_num_workers=16,
        
        # Training settings
        learning_rate=5e-4,  # Slightly higher for distributed
        num_epochs=3,
        warmup_steps=1000,
        
        # LoRA settings
        lora_r=64,
        lora_alpha=128,
        lora_dropout=0.1,
        
        # Memory optimization
        use_gradient_checkpointing=True,
        use_mixed_precision=True,
        max_audio_length_seconds=30,
        
        # Distributed specific
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        persistent_workers=True,
        
        # Checkpointing
        save_steps=500,
        logging_steps=50,
        eval_steps=1000,
    )

def get_distributed_test_config() -> DistributedTrainingConfig:
    """Get test configuration for distributed training validation."""
    return DistributedTrainingConfig(
        world_size=8,
        batch_size_per_gpu=1,  # Smaller for testing
        gradient_accumulation_steps=2,
        num_epochs=1,
        lora_r=32,
        lora_alpha=64,
        save_steps=50,
        logging_steps=10,
        eval_steps=100,
    )