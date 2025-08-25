"""
Training Configuration for Higgs-Audio LoRA Pipeline

Follows the design document patterns and matches the existing model architecture.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


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
        """Setup directories and validate configuration."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate paths
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"Training data not found: {self.train_data_path}")
        
        # Setup device
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Adjust batch size for MPS (limited memory)
        if self.device == "mps" and self.batch_size > 1:
            print(f"⚠️ Reducing batch size from {self.batch_size} to 1 for MPS compatibility")
            self.batch_size = 1
    
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


# Predefined configurations for different use cases
DEFAULT_CONFIG = TrainingConfig()

QUICK_TEST_CONFIG = TrainingConfig(
    batch_size=1,
    gradient_accumulation_steps=2,
    num_epochs=1,
    lora_r=8,
    lora_alpha=16,
    logging_steps=5,
    save_steps=50,
)

PRODUCTION_CONFIG = TrainingConfig(
    batch_size=2,
    gradient_accumulation_steps=16,
    num_epochs=5,
    learning_rate=1e-4,
    lora_r=32,
    lora_alpha=64,
    weight_decay=0.05,
)