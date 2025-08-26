#!/usr/bin/env python3
"""
Direct Training Launcher for Higgs-Audio LoRA Training

This launcher is designed to be simple and direct:
- Uses your input data as-is (no conversion unless necessary)
- Minimal validation
- No dummy file creation unless audio files are completely missing
- Direct path to training

Usage:
    python3 launch_training_direct.py \\
        --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \\
        --batch_size 4 \\
        --learning_rate 5e-4
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def setup_environment():
    """Setup environment paths."""
    script_dir = Path(__file__).parent.absolute()
    
    # Ensure we're in higgs-audio root
    if script_dir.name != "higgs-audio":
        print(f"❌ Please run from higgs-audio root directory")
        print(f"   Current: {script_dir}")
        sys.exit(1)
    
    return script_dir


def validate_basic_format(data_path: str) -> bool:
    """Basic validation - just check if file exists and is valid JSON."""
    try:
        if not os.path.exists(data_path):
            print(f"❌ Data file not found: {data_path}")
            return False
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        print(f"✅ Data file valid: {len(data)} samples loaded")
        return len(data) > 0
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False


def launch_distributed_training(data_path: str, args: argparse.Namespace) -> bool:
    """Launch the distributed training directly."""
    try:
        print("🚀 Launching 8xH200 distributed training...")
        
        # Build torchrun command
        cmd = [
            "torchrun",
            "--nproc_per_node=8",
            "--nnodes=1", 
            "--node_rank=0",
            "--master_addr=localhost",
            "--master_port=29500",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:29500",
            "trainer/train.py",
            "--train_data", data_path,
            "--batch_size", str(args.batch_size),
            "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
            "--learning_rate", str(args.learning_rate),
            "--num_epochs", str(args.num_epochs),
            "--lora_r", str(args.lora_r),
            "--lora_alpha", str(args.lora_alpha),
            "--save_steps", str(args.save_steps),
            "--logging_steps", str(args.logging_steps),
            "--output_dir", args.output_dir,
            "--dataloader_num_workers", str(args.dataloader_num_workers),
        ]
        
        # Add optional flags
        if args.mixed_precision:
            cmd.append("--mixed_precision")
        if args.use_gradient_checkpointing:
            cmd.append("--use_gradient_checkpointing")
        if args.max_audio_length_seconds:
            cmd.extend(["--max_audio_length_seconds", str(args.max_audio_length_seconds)])
        
        print(f"📋 Training command: {' '.join(cmd)}")
        
        # Set up environment variables for 8xH200
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "OMP_NUM_THREADS": "16",
            "MKL_NUM_THREADS": "16", 
            "NUMBA_NUM_THREADS": "16",
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "NCCL_P2P_DISABLE": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL"
        })
        
        # Launch training
        result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            return True
        else:
            print(f"❌ Training failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to launch training: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Direct Higgs-Audio Training Launcher")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data file")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="checkpoints/higgs_lora_8xh200")
    parser.add_argument("--dataloader_num_workers", type=int, default=16)
    parser.add_argument("--max_audio_length_seconds", type=int, default=30)
    
    # Flags
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument("--skip_validation", action="store_true")
    
    args = parser.parse_args()
    
    print("🎯 Direct Higgs-Audio Training Launcher")
    print("=" * 50)
    
    # Setup environment
    higgs_root = setup_environment()
    print(f"📁 Working directory: {higgs_root}")
    
    # Basic validation
    if not args.skip_validation:
        print(f"🔍 Validating data file: {args.train_data}")
        if not validate_basic_format(args.train_data):
            print("❌ Data validation failed")
            sys.exit(1)
    else:
        print("⏭️ Skipping validation as requested")
    
    # Launch training directly
    print(f"🚀 Starting training with data: {args.train_data}")
    success = launch_distributed_training(args.train_data, args)
    
    if success:
        print("🎉 Training completed successfully!")
    else:
        print("❌ Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()