#!/usr/bin/env python3
"""
Higgs-Audio LoRA Training Entry Point

Zero-shot voice cloning training pipeline that follows the exact patterns
from generation.py and arb_inference.py, using existing boson_multimodal components.

Usage Examples:

    # Basic training with default settings
    python train.py --train_data data/train_samples.json
    
    # Advanced training with custom configuration
    python train.py \
        --train_data data/train_samples.json \
        --val_data data/val_samples.json \
        --model_path bosonai/higgs-audio-v2-generation-3B-base \
        --audio_tokenizer_path bosonai/higgs-audio-v2-tokenizer \
        --batch_size 2 \
        --learning_rate 1e-4 \
        --num_epochs 5 \
        --lora_r 32 \
        --output_dir checkpoints/voice_cloning
    
    # Quick test run
    python train.py \
        --train_data data/train_samples.json \
        --quick_test \
        --output_dir checkpoints/test
    
    # Resume from checkpoint
    python train.py \
        --train_data data/train_samples.json \
        --resume_from checkpoints/checkpoint-1000
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Conditional imports for ML dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import utility functions directly (no ML dependencies)
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import create_sample_data, validate_dataset_format
    DATASET_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    DATASET_UTILS_AVAILABLE = False

# Conditional imports for training components
try:
    from trainer import HiggsAudioTrainer, TrainingConfig
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Higgs-Audio LoRA Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--train_data",
        type=str,
        help="Path to training data JSON file (ChatML format)"
    )
    
    # Optional data arguments
    parser.add_argument(
        "--val_data",
        type=str,
        help="Path to validation data JSON file (optional)"
    )
    
    parser.add_argument(
        "--audio_base_path",
        type=str,
        default="",
        help="Base path for resolving relative audio file paths"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="bosonai/higgs-audio-v2-generation-3B-base",
        help="Path or name of the Higgs-Audio model"
    )
    
    parser.add_argument(
        "--audio_tokenizer_path",
        type=str,
        default="bosonai/higgs-audio-v2-tokenizer",
        help="Path or name of the audio tokenizer"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    
    # LoRA arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (dimensionality of adaptation)"
    )
    
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (scaling parameter)"
    )
    
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout rate"
    )
    
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["lm_head", "audio_head"],
        help="Target modules for LoRA adaptation"
    )
    
    # Loss weights
    parser.add_argument(
        "--text_loss_weight",
        type=float,
        default=1.0,
        help="Weight for text loss component"
    )
    
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=1.0,
        help="Weight for audio loss component"
    )
    
    parser.add_argument(
        "--consistency_loss_weight",
        type=float,
        default=0.1,
        help="Weight for voice consistency loss component"
    )
    
    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints and logs"
    )
    
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Number of steps between logging"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between checkpoint saves"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=100,
        help="Number of steps between evaluations"
    )
    
    # Hardware and performance
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help="Number of dataloader workers"
    )
    
    # Convenience flags
    parser.add_argument(
        "--quick_test",
        action="store_true",
        help="Use quick test configuration (small LoRA rank, fewer epochs)"
    )
    
    parser.add_argument(
        "--resume_from",
        type=str,
        help="Path to checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--validate_data_only",
        action="store_true",
        help="Only validate the dataset format and exit"
    )
    
    parser.add_argument(
        "--create_sample_data",
        type=str,
        help="Create sample training data at the specified path and exit"
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create training configuration from command line arguments."""
    
    # Apply quick test settings if requested
    if args.quick_test:
        print("🚀 Using quick test configuration")
        args.batch_size = 1
        args.num_epochs = 1
        args.lora_r = 8
        args.lora_alpha = 16
        args.logging_steps = 5
        args.save_steps = 50
        args.eval_steps = 25
    
    # Create configuration without validation (validation happens later if needed)
    if not TRAINER_AVAILABLE:
        raise ImportError("Training components not available. Install dependencies: torch, transformers, peft")
    
    config = TrainingConfig(
        # Data
        train_data_path=args.train_data,
        val_data_path=args.val_data or "data/val_samples.json",
        
        # Model
        model_path=args.model_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=args.device,
        
        # Training
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        
        # LoRA
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        
        # Loss weights
        text_loss_weight=args.text_loss_weight,
        audio_loss_weight=args.audio_loss_weight,
        consistency_loss_weight=args.consistency_loss_weight,
        
        # Output
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        
        # Hardware
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    
    return config


def validate_environment():
    """Validate training environment and dependencies."""
    print("🔍 Validating environment...")
    
    # Check PyTorch
    if TORCH_AVAILABLE:
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Check MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            print(f"   MPS available: True")
    else:
        print("   ❌ PyTorch not found")
        return False
    
    # Check key dependencies
    try:
        import transformers
        print(f"   Transformers: {transformers.__version__}")
    except ImportError:
        print("   ❌ Transformers not found")
        return False
    
    try:
        import peft
        print(f"   PEFT: {peft.__version__}")
    except ImportError:
        print("   ❌ PEFT not found")
        return False
    
    try:
        if TRAINER_AVAILABLE:
            print("   ✅ Trainer components available")
        else:
            print("   ❌ Trainer components not available")
            return False
    except Exception as e:
        print(f"   ❌ Trainer import failed: {e}")
        return False
    
    print("✅ Environment validation passed")
    return True


def main():
    """Main training entry point."""
    args = parse_arguments()
    
    print("🎵 Higgs-Audio LoRA Training Pipeline")
    print("=" * 50)
    
    # Handle utility commands first (these don't need full validation)
    if args.create_sample_data:
        if not DATASET_UTILS_AVAILABLE:
            print("❌ Dataset utilities not available")
            sys.exit(1)
        print(f"📝 Creating sample training data at {args.create_sample_data}")
        create_sample_data(args.create_sample_data, num_samples=10)
        print("✅ Sample data created successfully")
        return
    
    if args.validate_data_only:
        if not DATASET_UTILS_AVAILABLE:
            print("❌ Dataset utilities not available")
            sys.exit(1)
        print(f"🔍 Validating dataset: {args.train_data}")
        if validate_dataset_format(args.train_data):
            print("✅ Dataset validation passed")
            return
        else:
            print("❌ Dataset validation failed")
            sys.exit(1)
    
    # For training operations, validate environment and data
    if not validate_environment():
        print("❌ Environment validation failed")
        sys.exit(1)
    
    # Validate training data exists for training operations
    if not args.train_data:
        print("❌ --train_data is required for training operations")
        sys.exit(1)
        
    if not os.path.exists(args.train_data):
        print(f"❌ Training data not found: {args.train_data}")
        sys.exit(1)
    
    if not validate_dataset_format(args.train_data):
        print("❌ Training data validation failed")
        sys.exit(1)
    
    # Create configuration
    try:
        if not TRAINER_AVAILABLE:
            print("❌ Training components not available. Install dependencies: torch, transformers, peft")
            sys.exit(1)
            
        config = create_config_from_args(args)
        print("📋 Training Configuration:")
        config_dict = config.to_dict()
        for key, value in config_dict.items():
            print(f"   {key}: {value}")
        
        # Save configuration
        os.makedirs(config.output_dir, exist_ok=True)
        config_path = os.path.join(config.output_dir, "training_config.json")
        config.save(config_path)
        print(f"💾 Configuration saved to {config_path}")
        
    except Exception as e:
        print(f"❌ Configuration creation failed: {e}")
        sys.exit(1)
    
    # Initialize trainer
    try:
        if not TRAINER_AVAILABLE:
            print("❌ Training components not available")
            sys.exit(1)
            
        print("\n🚀 Initializing trainer...")
        trainer = HiggsAudioTrainer(config)
        
        # Resume from checkpoint if requested
        if args.resume_from:
            print(f"📂 Resuming from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start training
    try:
        print("\n🎯 Starting training...")
        trainer.train()
        print("\n🏁 Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏸️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()