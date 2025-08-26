#!/usr/bin/env python3
"""
Fixed Training Launcher for Higgs-Audio LoRA Training

This launcher includes the corrected validation logic that matches
arb_inference.py patterns exactly.

Usage:
    python3 launch_training_fixed.py \
        --train_data ../ms-swift/lora_training_data_zr/chatml_fixed/train_chatml_samples.json \
        --batch_size 4 \
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


def validate_dataset_with_fixed_logic(data_path: str) -> bool:
    """
    Use the FIXED validation logic that matches arb_inference.py patterns.
    """
    try:
        print(f"🔍 Validating training data format: {data_path}")
        
        if not os.path.exists(data_path):
            print(f"❌ Training data file not found: {data_path}")
            return False
            
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        print(f"📊 Found {len(data)} samples to validate")
        
        valid_samples = 0
        for idx, sample in enumerate(data):
            if _validate_sample_structure_fixed(sample, idx):
                valid_samples += 1
                
        print(f"✅ Validation complete: {valid_samples}/{len(data)} samples valid")
        
        if valid_samples == 0:
            print("❌ No valid samples found!")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False


def _validate_sample_structure_fixed(sample: dict, idx: int) -> bool:
    """FIXED validation using exact arb_inference.py process_chatml_sample logic."""
    try:
        # Check required fields
        if "messages" not in sample:
            print(f"  Sample {idx}: Missing 'messages' field")
            return False
        
        messages = sample["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            print(f"  Sample {idx}: Invalid 'messages' field")
            return False
        
        # Use EXACT same logic as arb_inference.py process_chatml_sample
        ref_audio_path = None
        ref_text = None
        target_text = None
        
        for message in messages:
            if not isinstance(message, dict):
                print(f"  Sample {idx}: Invalid message format")
                return False
            
            if "role" not in message or "content" not in message:
                print(f"  Sample {idx}: Missing 'role' or 'content'")
                return False
            
            if message["role"] == "user":
                content = message["content"]
                if isinstance(content, list):
                    # Look for text and audio content (EXACT arb_inference.py logic)
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "audio":
                                if ref_audio_path is None:  # First audio is reference
                                    ref_audio_path = item.get("audio_url")
                    
                    if len(text_parts) >= 2:
                        ref_text = text_parts[0]  # First text is reference
                        # Look for target text
                        for text_part in text_parts[1:]:
                            if "Please generate speech" in text_part:
                                # Extract target text after the instruction
                                target_text = text_part.split(":")[-1].strip()
                                break
                        if target_text is None and len(text_parts) > 1:
                            target_text = text_parts[-1]  # Last text as fallback
                elif isinstance(content, str):
                    # Simple string content
                    if ref_text is None:
                        ref_text = content
                    else:
                        target_text = content
                        
            elif message["role"] == "assistant":
                content = message["content"]
                if isinstance(content, dict) and content.get("type") == "audio":
                    if ref_audio_path is None:
                        ref_audio_path = content.get("audio_url")
        
        # Validate that we found all required components (EXACT arb_inference.py logic)
        if not all([ref_audio_path, ref_text, target_text]):
            print(f"  Sample {idx}: Missing required components: ref_audio={ref_audio_path is not None}, ref_text={ref_text is not None}, target_text={target_text is not None}")
            return False
        
        if idx < 3:  # Show first few samples for debugging
            print(f"  Sample {idx}: ✅ Valid structure")
            print(f"    - ref_audio: {ref_audio_path}")
            print(f"    - ref_text: '{ref_text[:50]}...'")
            print(f"    - target_text: '{target_text[:50]}...'")
        elif idx % 10000 == 0:  # Progress indicator for large datasets
            print(f"  Sample {idx}: ✅ Valid structure (progress update)")
        
        return True
        
    except Exception as e:
        print(f"  Sample {idx}: Validation error: {e}")
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
    parser = argparse.ArgumentParser(description="FIXED Higgs-Audio Training Launcher")
    
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
    parser.add_argument("--skip_validation", action="store_true", 
                       help="Skip validation and proceed directly to training")
    
    args = parser.parse_args()
    
    print("🎯 FIXED Higgs-Audio Training Launcher")
    print("=" * 50)
    print("🔧 This version includes the corrected validation logic")
    print("   that matches arb_inference.py patterns exactly.")
    print("=" * 50)
    
    # Setup environment
    higgs_root = setup_environment()
    print(f"📁 Working directory: {higgs_root}")
    
    # Validation with FIXED logic
    if not args.skip_validation:
        print(f"🔍 Validating data file with FIXED logic: {args.train_data}")
        if not validate_dataset_with_fixed_logic(args.train_data):
            print("❌ Data validation failed")
            print("💡 The validation logic has been fixed to match arb_inference.py")
            print("   If validation still fails, your data may need format conversion.")
            sys.exit(1)
        print("✅ Data validation passed with FIXED logic!")
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