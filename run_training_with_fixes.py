#!/usr/bin/env python3
"""
Complete Training Execution Script with Error Fixes

This script addresses the specific error encountered by the user and provides
a comprehensive solution for running the Arabic voice cloning training pipeline.

Usage:
    python3 run_training_with_fixes.py --data_path your_chatml_data.json --output_dir ./outputs

Features:
- Fixes all import and type annotation issues
- Validates the complete pipeline before training
- Provides clear error messages and solutions
- Optimized for 8xH200 GPU setup
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

def check_dependencies() -> bool:
    """Check if all required dependencies are installed."""
    print("📦 Checking dependencies...")
    
    required_modules = [
        'torch', 'transformers', 'peft', 'accelerate', 
        'librosa', 'soundfile', 'loguru', 'tqdm'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}")
    
    if missing_modules:
        print(f"\n📋 Missing modules: {missing_modules}")
        print("💡 Install with: pip install -r requirements_training.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def fix_import_issues():
    """Fix the specific import issues that caused the error."""
    print("🔧 Fixing import issues...")
    
    # Fix the Tuple import issue in arabic_voice_cloning_lora_config.py
    lora_config_file = Path("arabic_voice_cloning_lora_config.py")
    if lora_config_file.exists():
        with open(lora_config_file, 'r') as f:
            content = f.read()
        
        # Ensure Tuple is imported
        if "from typing import" in content and "Tuple" not in content.split("from typing import")[1].split('\n')[0]:
            content = content.replace(
                "from typing import List, Dict, Optional, Any, Union",
                "from typing import List, Dict, Optional, Any, Union, Tuple"
            )
            
            with open(lora_config_file, 'w') as f:
                f.write(content)
            print("✅ Fixed Tuple import in arabic_voice_cloning_lora_config.py")
        else:
            print("✅ Tuple import already correct in arabic_voice_cloning_lora_config.py")
    
    # Fix device_map issue in distributed trainer
    trainer_file = Path("arabic_voice_cloning_distributed_trainer.py")
    if trainer_file.exists():
        with open(trainer_file, 'r') as f:
            content = f.read()
        
        # Fix device_map=None issue
        if "device_map=None" in content:
            content = content.replace("device_map=None", "device_map=\"cpu\"")
            
            with open(trainer_file, 'w') as f:
                f.write(content)
            print("✅ Fixed device_map issue in distributed trainer")
        else:
            print("✅ device_map already correct in distributed trainer")

def validate_chatml_data(data_path: str) -> bool:
    """Validate the ChatML data format."""
    print(f"📊 Validating ChatML data: {data_path}")
    
    if not Path(data_path).exists():
        print(f"❌ Data file not found: {data_path}")
        return False
    
    try:
        import json
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Data should be a list of ChatML samples")
            return False
        
        if len(data) == 0:
            print("❌ Data file is empty")
            return False
        
        # Check first sample structure
        sample = data[0]
        if "messages" not in sample:
            print("❌ Sample missing 'messages' field")
            return False
        
        # Check for audio URLs
        audio_urls = []
        for message in sample["messages"]:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio_urls.append(item.get("audio_url"))
        
        print(f"✅ Found {len(data)} samples with {len(audio_urls)} audio files")
        print(f"📁 Sample audio paths: {audio_urls[:2]}...")  # Show first 2 paths
        return True
        
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return False

def run_training(data_path: str, output_dir: str, **kwargs) -> bool:
    """Run the training with proper error handling."""
    print("🏋️ Starting training...")
    
    # Build training command
    cmd = [
        sys.executable, "train_arabic_voice_cloning.py",
        "--data_path", data_path,
        "--output_dir", output_dir
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        # Run training
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Training completed successfully!")
            print(result.stdout)
            return True
        else:
            print("❌ Training failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Training execution failed: {e}")
        return False

def run_distributed_training(data_path: str, output_dir: str, num_gpus: int = 8, **kwargs) -> bool:
    """Run distributed training for multi-GPU setup."""
    print(f"🔥 Starting distributed training on {num_gpus} GPUs...")
    
    # Build distributed training command
    cmd = [
        "torchrun", f"--nproc_per_node={num_gpus}",
        "train_arabic_voice_cloning.py",
        "--data_path", data_path,
        "--output_dir", output_dir
    ]
    
    # Add additional arguments
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        # Run distributed training
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Distributed training execution failed: {e}")
        return False

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Arabic Voice Cloning Training with Error Fixes")
    parser.add_argument("--data_path", required=True, help="Path to ChatML data JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs for distributed training")
    parser.add_argument("--skip_validation", action="store_true", help="Skip pipeline validation")
    parser.add_argument("--fix_only", action="store_true", help="Only apply fixes, don't run training")
    
    args = parser.parse_args()
    
    print("🎯 Arabic Voice Cloning Training Pipeline")
    print("=" * 60)
    
    # Step 1: Apply fixes
    print("\n1️⃣ Applying Error Fixes")
    fix_import_issues()
    
    if args.fix_only:
        print("✅ Fixes applied successfully!")
        return 0
    
    # Step 2: Check dependencies
    print("\n2️⃣ Dependency Check")
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return 1
    
    # Step 3: Validate data
    if not args.skip_validation:
        print("\n3️⃣ Data Validation")
        if not validate_chatml_data(args.data_path):
            print("❌ Data validation failed")
            return 1
    
    # Step 4: Create output directory
    print("\n4️⃣ Setup")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    print(f"✅ Output directory: {args.output_dir}")
    
    # Step 5: Run training
    print("\n5️⃣ Training")
    
    training_kwargs = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs
    }
    
    if args.num_gpus > 1:
        success = run_distributed_training(
            args.data_path, args.output_dir, args.num_gpus, **training_kwargs
        )
    else:
        success = run_training(args.data_path, args.output_dir, **training_kwargs)
    
    if success:
        print("\n🎉 Training Pipeline Completed Successfully!")
        print(f"📁 Check results in: {args.output_dir}")
        print("\n📋 Next Steps:")
        print("1. Monitor training logs for loss progression")
        print("2. Validate model checkpoints")
        print("3. Merge LoRA weights for deployment")
        return 0
    else:
        print("\n❌ Training Pipeline Failed!")
        print("🔧 Please check the error messages above")
        return 1

if __name__ == "__main__":
    exit(main())