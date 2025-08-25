#!/usr/bin/env python3
"""
Main Training Script for Arabic Voice Cloning with Higgs Audio v2

This script provides a complete training pipeline for Arabic voice cloning using
LoRA fine-tuning on Higgs Audio v2 with DualFFN architecture.

Usage:
    # Single GPU training
    python train_arabic_voice_cloning.py --config configs/arabic_voice_cloning.yaml
    
    # Multi-GPU distributed training (8xH200)
    torchrun --nproc_per_node=8 train_arabic_voice_cloning.py --config configs/arabic_voice_cloning.yaml

Features:
- Comprehensive configuration management
- Multi-GPU distributed training support
- Automatic checkpoint recovery
- Performance monitoring and logging
- Hardware optimization for H200 GPUs
"""

import os
import sys
import yaml
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
from loguru import logger
import torch

# Configure warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Import our training components
from arabic_voice_cloning_distributed_trainer import (
    ArabicVoiceCloningDistributedTrainer,
    DistributedTrainingConfig,
    create_distributed_trainer
)
from arabic_voice_cloning_dataset import ArabicVoiceCloningDatasetConfig
from arabic_voice_cloning_lora_config import HiggsAudioLoRATrainingConfig
from arabic_voice_cloning_loss_function import LossConfig


class ConfigManager:
    """Configuration manager for training pipeline."""
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
            **kwargs: Additional configuration overrides
        """
        self.config = self._load_config(config_path, **kwargs)
        self._validate_config()
        self._setup_paths()
    
    def _load_config(self, config_path: Optional[str], **kwargs) -> Dict[str, Any]:
        """Load configuration from file and command line arguments."""
        # Default configuration
        default_config = {
            # Model and data paths
            "model_path": "bosonai/higgs-audio-v2-generation-3B-base",
            "audio_tokenizer_path": "bosonai/higgs-audio-v2-tokenizer",
            "data_path": "data/arabic_voice_cloning_chatml.json",
            "output_dir": "./outputs/arabic_voice_cloning",
            
            # Training configuration
            "training": {
                "num_epochs": 3,
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "max_grad_norm": 1.0,
                "use_mixed_precision": True,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 16,
                "save_steps": 500,
                "eval_steps": 250,
                "logging_steps": 10,
                "save_total_limit": 3
            },
            
            # Dataset configuration
            "dataset": {
                "max_audio_duration": 30.0,
                "target_sample_rate": 16000,
                "min_audio_duration": 0.5,
                "max_text_length": 512,
                "min_text_length": 5,
                "validate_on_init": True,
                "teacher_forcing": True,
                "return_labels": True
            },
            
            # LoRA configuration
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules_mode": "comprehensive",
                "bias": "none",
                "use_rslora": False,
                "modules_to_save": ["audio_head", "lm_head"],
                "enable_lora_for_audio_head": True,
                "enable_lora_for_audio_projector": True
            },
            
            # Loss function configuration
            "loss": {
                "text_loss_weight": 1.0,
                "audio_loss_weight": 1.0,
                "contrastive_loss_weight": 0.1,
                "consistency_loss_weight": 0.05,
                "l2_regularization": 0.0,
                "label_smoothing": 0.0,
                "audio_label_smoothing": 0.0,
                "enable_curriculum_learning": True,
                "curriculum_steps": 10000
            },
            
            # Distributed training
            "distributed": {
                "backend": "nccl",
                "world_size": 1,
                "local_rank": -1
            },
            
            # Monitoring and logging
            "monitoring": {
                "use_wandb": True,
                "wandb_project": "higgs-audio-arabic-voice-cloning",
                "wandb_run_name": None,
                "log_level": "INFO"
            },
            
            # Hardware optimization
            "hardware": {
                "max_memory_usage": 0.95,
                "enable_cpu_offload": False,
                "prefetch_factor": 4,
                "pin_memory": True,
                "compile_model": False,
                "use_flash_attention": True
            }
        }
        
        # Load from YAML file if provided
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
            default_config = self._deep_update(default_config, file_config)
            logger.info(f"Configuration loaded from {config_path}")
        
        # Override with command line arguments
        if kwargs:
            default_config = self._deep_update(default_config, kwargs)
            logger.info(f"Configuration overridden with: {kwargs}")
        
        return default_config
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict) -> Dict:
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
        return base_dict
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_paths = ["data_path", "output_dir"]
        for path_key in required_paths:
            if not self.config.get(path_key):
                raise ValueError(f"Required configuration '{path_key}' is missing")
        
        # Validate training parameters
        training_config = self.config.get("training", {})
        if training_config.get("learning_rate", 0) <= 0:
            raise ValueError("Learning rate must be positive")
        
        if training_config.get("batch_size", 0) <= 0:
            raise ValueError("Batch size must be positive")
        
        # Validate LoRA parameters
        lora_config = self.config.get("lora", {})
        if lora_config.get("r", 0) <= 0:
            raise ValueError("LoRA rank must be positive")
        
        logger.info("Configuration validation passed")
    
    def _setup_paths(self):
        """Setup and create necessary directories."""
        # Create output directory
        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "configs").mkdir(exist_ok=True)
        
        # Save configuration
        config_file = output_dir / "configs" / "training_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Output directory setup: {output_dir}")
    
    def get_training_config(self) -> DistributedTrainingConfig:
        """Get training configuration object."""
        training_params = self.config["training"].copy()
        training_params.update({
            "model_path": self.config["model_path"],
            "audio_tokenizer_path": self.config["audio_tokenizer_path"],
            "data_path": self.config["data_path"],
            "output_dir": str(Path(self.config["output_dir"]) / "checkpoints"),
        })
        
        # Add distributed parameters
        training_params.update(self.config.get("distributed", {}))
        
        # Add monitoring parameters
        training_params.update(self.config.get("monitoring", {}))
        
        # Add hardware parameters
        training_params.update(self.config.get("hardware", {}))
        
        return DistributedTrainingConfig(**training_params)
    
    def get_dataset_config(self) -> ArabicVoiceCloningDatasetConfig:
        """Get dataset configuration object."""
        dataset_params = self.config["dataset"].copy()
        dataset_params.update({
            "chatml_file": self.config["data_path"],
            "num_workers": self.config["training"]["dataloader_num_workers"],
            "prefetch_factor": self.config["hardware"]["prefetch_factor"]
        })
        
        return ArabicVoiceCloningDatasetConfig(**dataset_params)
    
    def get_lora_config(self) -> HiggsAudioLoRATrainingConfig:
        """Get LoRA configuration object."""
        return HiggsAudioLoRATrainingConfig(**self.config["lora"])
    
    def get_loss_config(self) -> LossConfig:
        """Get loss configuration object."""
        return LossConfig(**self.config["loss"])
    
    def save_config(self, path: str):
        """Save current configuration to file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


def setup_logging(log_level: str = "INFO", output_dir: Optional[str] = None):
    """Setup logging configuration."""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File logging
    if output_dir:
        log_file = Path(output_dir) / "logs" / "training.log"
        logger.add(
            log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="100 MB",
            retention="7 days"
        )
        logger.info(f"Logging to file: {log_file}")


def check_environment():
    """Check and log environment information."""
    logger.info("Environment Information:")
    logger.info(f"  Python version: {sys.version}")
    logger.info(f"  PyTorch version: {torch.__version__}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  CUDA version: {torch.version.cuda}")
        logger.info(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Check for distributed training environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    
    if world_size > 1:
        logger.info(f"  Distributed training: World size {world_size}, Local rank {local_rank}")
    else:
        logger.info("  Single GPU/CPU training")


def load_checkpoint(trainer: ArabicVoiceCloningDistributedTrainer, checkpoint_path: str):
    """Load checkpoint and resume training."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return
    
    try:
        # Load model checkpoint
        if hasattr(trainer.model, 'module'):
            trainer.model.module.load_adapter(checkpoint_path)
        else:
            trainer.model.load_adapter(checkpoint_path)
        
        # Load training state
        state_file = Path(checkpoint_path) / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location=trainer.device)
            trainer.current_step = state.get("step", 0)
            trainer.current_epoch = state.get("epoch", 0)
            trainer.optimizer.load_state_dict(state["optimizer_state_dict"])
            
            if trainer.scheduler and state.get("scheduler_state_dict"):
                trainer.scheduler.load_state_dict(state["scheduler_state_dict"])
            
            if trainer.scaler and state.get("scaler_state_dict"):
                trainer.scaler.load_state_dict(state["scaler_state_dict"])
            
            logger.info(f"Checkpoint loaded: Step {trainer.current_step}, Epoch {trainer.current_epoch}")
        else:
            logger.warning("Training state not found, starting from beginning")
            
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Arabic Voice Cloning Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data_path", type=str, help="Path to training data (ChatML JSON with direct audio paths)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--resume_from", type=str, help="Resume training from checkpoint")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, help="Log every N steps")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--dry_run", action="store_true", help="Dry run without training")
    
    args = parser.parse_args()
    
    # Prepare configuration overrides from command line
    config_overrides = {}
    if args.data_path:
        config_overrides["data_path"] = args.data_path
    if args.output_dir:
        config_overrides["output_dir"] = args.output_dir
    
    # Training overrides
    training_overrides = {}
    if args.batch_size:
        training_overrides["batch_size"] = args.batch_size
    if args.learning_rate:
        training_overrides["learning_rate"] = args.learning_rate
    if args.num_epochs:
        training_overrides["num_epochs"] = args.num_epochs
    if args.gradient_accumulation_steps:
        training_overrides["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    if args.save_steps:
        training_overrides["save_steps"] = args.save_steps
    if args.logging_steps:
        training_overrides["logging_steps"] = args.logging_steps
    
    if training_overrides:
        config_overrides["training"] = training_overrides
    
    # Monitoring overrides
    monitoring_overrides = {}
    if args.use_wandb:
        monitoring_overrides["use_wandb"] = True
    if args.wandb_project:
        monitoring_overrides["wandb_project"] = args.wandb_project
    if args.log_level:
        monitoring_overrides["log_level"] = args.log_level
    
    if monitoring_overrides:
        config_overrides["monitoring"] = monitoring_overrides
    
    try:
        # Initialize configuration
        config_manager = ConfigManager(args.config, **config_overrides)
        
        # Setup logging
        setup_logging(
            log_level=config_manager.config["monitoring"]["log_level"],
            output_dir=config_manager.config["output_dir"]
        )
        
        logger.info("🚀 Starting Arabic Voice Cloning Training")
        logger.info("=" * 60)
        
        # Check environment
        check_environment()
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = ArabicVoiceCloningDistributedTrainer(
            training_config=config_manager.get_training_config(),
            dataset_config=config_manager.get_dataset_config(),
            lora_config=config_manager.get_lora_config(),
            loss_config=config_manager.get_loss_config()
        )
        
        # Resume from checkpoint if specified
        if args.resume_from:
            logger.info(f"Resuming from checkpoint: {args.resume_from}")
            load_checkpoint(trainer, args.resume_from)
        
        # Log training configuration
        logger.info("Training Configuration:")
        logger.info(f"  Dataset size: {len(trainer.dataset)} samples")
        logger.info(f"  Batch size per GPU: {trainer.training_config.batch_size}")
        logger.info(f"  Effective batch size: {trainer.effective_batch_size}")
        logger.info(f"  Number of epochs: {trainer.training_config.num_epochs}")
        logger.info(f"  Learning rate: {trainer.training_config.learning_rate}")
        logger.info(f"  Steps per epoch: {len(trainer.dataloader)}")
        
        total_steps = len(trainer.dataloader) * trainer.training_config.num_epochs
        logger.info(f"  Total training steps: {total_steps}")
        
        # Dry run check
        if args.dry_run:
            logger.info("🔍 Dry run completed successfully")
            return
        
        # Start training
        logger.info("🏋️ Starting training...")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        if torch.cuda.is_available():
            start_time.record()
        
        trainer.train()
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000 / 60  # Convert to minutes
            logger.info(f"⏱️ Total training time: {training_time:.2f} minutes")
        
        # Cleanup
        trainer.cleanup()
        
        logger.info("✅ Training completed successfully!")
        logger.info("=" * 60)
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        if 'trainer' in locals():
            trainer.cleanup()
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        if 'trainer' in locals():
            trainer.cleanup()
        raise


if __name__ == "__main__":
    main()