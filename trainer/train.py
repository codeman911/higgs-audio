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

# 📁 ENHANCED: Robust path setup for 'python3 trainer/train.py' execution from higgs-audio root
trainer_dir = Path(__file__).parent
higgs_audio_root = trainer_dir.parent  # Go up from trainer/ to higgs-audio/

# 🎯 CRITICAL: Ensure higgs-audio root is in Python path
sys.path.insert(0, str(higgs_audio_root))  # Add higgs-audio root for boson_multimodal
sys.path.insert(0, str(trainer_dir))       # Add trainer directory for local imports

print(f"✅ Enhanced import system initialized:")
print(f"   Higgs-audio root: {higgs_audio_root}")
print(f"   Trainer directory: {trainer_dir}")
print(f"   Working directory: {Path.cwd()}")

# Try to find and add boson_multimodal to path if needed
def setup_boson_multimodal_path():
    """Enhanced function to ensure boson_multimodal is available for training execution."""
    try:
        import boson_multimodal
        print(f"✅ boson_multimodal already available")
        return True
    except ImportError:
        # 🔍 Try to locate boson_multimodal in common paths
        possible_paths = [
            higgs_audio_root,  # Should be the correct location
            higgs_audio_root.parent,  # One level up
            Path.cwd(),  # Current working directory
            Path.cwd().parent,  # Parent of working directory
        ]
        
        print(f"🔍 Searching for boson_multimodal...")
        for path in possible_paths:
            boson_path = path / "boson_multimodal"
            print(f"   Checking: {boson_path} - {'EXISTS' if boson_path.exists() else 'NOT FOUND'}")
            
            if boson_path.exists() and boson_path.is_dir():
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
                try:
                    import boson_multimodal
                    print(f"✅ Found and loaded boson_multimodal from: {path}")
                    return True
                except ImportError as e:
                    print(f"❌ Failed to import from {path}: {e}")
                    continue
        
        print(f"❌ Could not locate boson_multimodal in any expected location")
        print(f"   Expected at: {higgs_audio_root / 'boson_multimodal'}")
        print(f"   Please ensure you're running from higgs-audio root: python3 trainer/train.py")
        return False

# Setup boson_multimodal path
BOSON_MULTIMODAL_AVAILABLE = setup_boson_multimodal_path()

# Import utility functions directly (no ML dependencies)
try:
    from utils import create_sample_data, validate_dataset_format
    DATASET_UTILS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    DATASET_UTILS_AVAILABLE = False

# Conditional imports for training components
TRAINER_AVAILABLE = False
TRAINER_IMPORT_ERROR = None

if BOSON_MULTIMODAL_AVAILABLE:
    try:
        # Try different import patterns to find the trainer module
        try:
            from trainer.trainer import HiggsAudioTrainer
            from trainer.config import TrainingConfig
        except ImportError:
            try:
                from trainer import HiggsAudioTrainer
                from config import TrainingConfig
            except ImportError:
                # Direct imports as last resort
                from .trainer import HiggsAudioTrainer
                from .config import TrainingConfig
        TRAINER_AVAILABLE = True
    except ImportError as e:
        TRAINER_IMPORT_ERROR = str(e)
        TRAINER_AVAILABLE = False
else:
    TRAINER_IMPORT_ERROR = "boson_multimodal not available - run from higgs-audio root directory"


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
        default=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Target modules for LoRA adaptation (DualFFN architecture components)"
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
    
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training (bfloat16)"
    )
    
    parser.add_argument(
        "--max_audio_length_seconds",
        type=int,
        default=30,
        help="Maximum audio length in seconds"
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
        mixed_precision=args.mixed_precision,
        max_audio_duration=args.max_audio_length_seconds,
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
    
    # Enhanced boson_multimodal diagnosis
    print(f"   🔍 Diagnosing boson_multimodal availability...")
    if BOSON_MULTIMODAL_AVAILABLE:
        import boson_multimodal
        print(f"      ✅ boson_multimodal found at: {Path(boson_multimodal.__file__).parent}")
    else:
        print(f"      ❌ boson_multimodal not available")
        print(f"      💡 SOLUTION: Run from the higgs-audio root directory:")
        print(f"         cd /vs/higgs-audio")
        print(f"         python3 trainer/train.py --train_data ...")
        print(f"")
        print(f"      📍 Current working directory: {Path.cwd()}")
        print(f"      📍 Script location: {Path(__file__).parent}")
        print(f"      📍 Expected higgs-audio root: {parent_dir}")
        
        # Check if we can find higgs-audio root
        if (parent_dir / "boson_multimodal").exists():
            print(f"      ✅ Found boson_multimodal at expected location: {parent_dir / 'boson_multimodal'}")
            print(f"      💡 Issue: Python path not set correctly")
        else:
            print(f"      ❌ boson_multimodal not found at expected location")
            print(f"      💡 Make sure you're in the correct higgs-audio repository")
    
    # Enhanced trainer import diagnosis
    if TRAINER_AVAILABLE:
        print("   ✅ Trainer components available")
        return True
    else:
        print("   ❌ Trainer components not available")
        if TRAINER_IMPORT_ERROR:
            print(f"      Error details: {TRAINER_IMPORT_ERROR}")
        
        if not BOSON_MULTIMODAL_AVAILABLE:
            print(f"      🎯 PRIMARY ISSUE: boson_multimodal not available")
            print(f"      🔧 QUICK FIX: Run this exact command:")
            print(f"         cd /vs/higgs-audio && python3 trainer/train.py --train_data ../../ms-swift/lora_training_data_zr/chatml_fixed/val_chatml_samples.json")
        else:
            print(f"      🎯 SECONDARY ISSUE: Trainer modules not importing correctly")
            print(f"      🔧 Check trainer module dependencies")
        
        return False


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